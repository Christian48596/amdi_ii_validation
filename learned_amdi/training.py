from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from amdi.energy import EnergyParameters
from amdi.graph import GraphParameters
from .backend import HaarAMDIBackend
from .config import FeatureScales, NetworkConfig, PPOConfig, RewardConfig
from .datasets import noisy_case
from .features import estimate_scales, raw_feature_rows
from .networks import SharedLocalActor, VariableSetCritic
from .ppo import PPOTrainer
from .reference import uniform_reference_trajectory
from .runner import collect_trajectory


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def build_backend(noisy: np.ndarray, cfg: dict) -> HaarAMDIBackend:
    return HaarAMDIBackend(
        noisy,
        initial_threshold=float(cfg["amdi"]["initial_threshold"]),
        min_level=int(cfg["amdi"]["min_level"]),
        eparams=EnergyParameters(
            alpha=float(cfg["amdi"]["alpha"]),
            beta=float(cfg["amdi"]["beta"]),
            tau=float(cfg["amdi"]["tau"]),
            smooth_l1=False,
        ),
        gparams=GraphParameters(
            sigma_c=float(cfg["amdi"]["sigma_c"]),
            state_dependent=True,
            refinement_decay=float(cfg["amdi"]["refinement_decay"]),
        ),
        h=float(cfg["amdi"]["h"]),
        zeta=float(cfg["amdi"]["zeta"]),
        refine_fraction=float(cfg["amdi"]["refine_fraction"]),
        coarsen_fraction=float(cfg["amdi"]["coarsen_fraction"]),
    )


def make_reference(noisy: np.ndarray, backend: HaarAMDIBackend, n_steps: int):
    return uniform_reference_trajectory(
        noisy, eparams=backend.eparams, gparams=backend.gparams, h=backend.h, n_steps=n_steps
    )[0]


def estimate_training_scales(cfg: dict, seeds: list[int]) -> FeatureScales:
    rows=[]
    n=int(cfg["data"]["train_n"]); sigma=float(cfg["data"]["noise_sigma"])
    n_steps=int(cfg["training"]["n_steps"])
    for s in seeds:
        _, noisy = noisy_case("random_multiscale", n, s, sigma, 10000+s)
        backend=build_backend(noisy,cfg)
        for step in range(min(n_steps,3)):
            backend.propagate_for_adaptation()
            rec=backend.candidate_records()
            rows.append(raw_feature_rows(backend.current_field(),rec,backend.active_leaf_cells(),backend.max_level,step+1,n_steps))
            # Complete the deterministic AMDI step with deterministic adaptation so that
            # feature scales are estimated on the same post-propagation states
            # seen by the learned policy.
            backend.deterministic_adapt(backend.last_diagnostics)
    return estimate_scales(np.vstack(rows), quantile=0.95)


def train(cfg: dict, out_dir: str | Path, device: torch.device) -> dict:
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    seed=int(cfg["training"]["seed"]); seed_everything(seed)
    train_seeds=[int(v) for v in cfg["data"]["train_image_seeds"]]
    val_seeds=[int(v) for v in cfg["data"]["validation_image_seeds"]]
    scales=estimate_training_scales(cfg,train_seeds)
    net=NetworkConfig(tuple(cfg["network"]["actor_hidden"]),tuple(cfg["network"]["critic_hidden"]),cfg["network"]["activation"])
    actor=SharedLocalActor(net.actor_hidden,net.activation)
    critic=VariableSetCritic(net.critic_hidden,net.activation)
    pcfg=PPOConfig(**cfg["ppo"]); rcfg=RewardConfig(**cfg["reward"])
    trainer=PPOTrainer(actor,critic,pcfg,device)
    n=int(cfg["data"]["train_n"]); sigma=float(cfg["data"]["noise_sigma"]); n_steps=int(cfg["training"]["n_steps"])
    log=[]; best_val=-np.inf; best_state=None
    rng=np.random.default_rng(seed)
    for update in range(pcfg.updates):
        trajectories=[]
        chosen=rng.choice(train_seeds,size=pcfg.trajectories_per_update,replace=True)
        for s in chosen:
            _, noisy=noisy_case("random_multiscale",n,int(s),sigma,10000+int(s)+update)
            backend=build_backend(noisy,cfg); ref=make_reference(noisy,backend,n_steps)
            traj,_=collect_trajectory(backend,ref,actor,critic,scales,rcfg,n_steps,device,False)
            trajectories.append(traj)
        metrics=trainer.update(trajectories)
        # Deterministic validation return on frozen validation seeds.
        vals=[]
        for s in val_seeds:
            _, noisy=noisy_case("random_multiscale",n,s,sigma,20000+s)
            backend=build_backend(noisy,cfg); ref=make_reference(noisy,backend,n_steps)
            traj,_=collect_trajectory(backend,ref,actor,critic,scales,rcfg,n_steps,device,True)
            vals.append(traj.total_reward)
        v=float(np.mean(vals))
        row={"update":update+1,"validation_return":v,**metrics}; log.append(row)
        if v>best_val:
            best_val=v
            best_state={"actor":{k:v.detach().cpu().clone() for k,v in actor.state_dict().items()},"critic":{k:v.detach().cpu().clone() for k,v in critic.state_dict().items()}}
    assert best_state is not None
    torch.save({**best_state,"scales":scales.__dict__,"config":cfg,"best_validation_return":best_val},out/"best_policy.pt")
    (out/"training_log.json").write_text(json.dumps(log,indent=2),encoding="utf-8")
    return {"best_validation_return":best_val,"n_updates":pcfg.updates,"scales":scales.__dict__}
