from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from amdi.metrics import all_metrics
from .config import FeatureScales, NetworkConfig, RewardConfig
from .datasets import noisy_case
from .networks import SharedLocalActor, VariableSetCritic
from .runner import collect_trajectory, run_deterministic_trajectory
from .training import build_backend, make_reference


def load_policy(path: str | Path, device: torch.device):
    ckpt=torch.load(path,map_location=device,weights_only=False)
    cfg=ckpt["config"]
    net=NetworkConfig(tuple(cfg["network"]["actor_hidden"]),tuple(cfg["network"]["critic_hidden"]),cfg["network"]["activation"])
    actor=SharedLocalActor(net.actor_hidden,net.activation).to(device); actor.load_state_dict(ckpt["actor"])
    critic=VariableSetCritic(net.critic_hidden,net.activation).to(device); critic.load_state_dict(ckpt["critic"])
    scales=FeatureScales(**ckpt["scales"])
    return actor,critic,scales,cfg


def evaluate_case(checkpoint: str | Path, *, kind: str, n: int, image_seed: int, noise_sigma: float, noise_seed: int, device: torch.device) -> list[dict]:
    actor,critic,scales,cfg=load_policy(checkpoint,device)
    n_steps=int(cfg["training"]["n_steps"]); rcfg=RewardConfig(**cfg["reward"])
    truth,noisy=noisy_case(kind,n,image_seed,noise_sigma,noise_seed)
    base=build_backend(noisy,cfg); ref=make_reference(noisy,base,n_steps)
    rows=[]
    # Uniform reference: algorithmic target, full occupancy.
    rows.append({"method":"Uniform AMDI reference",**all_metrics(np.clip(ref[-1],0,1),truth,active=n*n,full=n*n),"reference_error":0.0,"switching_mean":0.0})
    # Deterministic AMDI baseline.  The separate modular deterministic control
    # is exercised only by the protocol-alignment regression, not treated as
    # a fourth numerical method in manuscript comparisons.
    b=build_backend(noisy,cfg); t0=time.perf_counter()
    diag=run_deterministic_trajectory(b,ref,n_steps,"baseline")
    runtime=time.perf_counter()-t0
    rows.append({"method":"AMDI",**all_metrics(np.clip(b.current_field(),0,1),truth,active=b.n_active,full=n*n),"reference_error":diag[-1]["reference_error"],"switching_mean":float(np.mean([d["switching"] for d in diag])),"runtime_s":runtime})
    b=build_backend(noisy,cfg); t0=time.perf_counter(); _,diag=collect_trajectory(b,ref,actor,critic,scales,rcfg,n_steps,device,True); runtime=time.perf_counter()-t0
    rows.append({"method":"Learned AMDI",**all_metrics(np.clip(b.current_field(),0,1),truth,active=b.n_active,full=n*n),"reference_error":diag[-1]["reference_error"],"switching_mean":float(np.mean([d["switching"] for d in diag])),"runtime_s":runtime})
    for r in rows:
        r.update({"kind":kind,"n":n,"image_seed":image_seed,"noise_seed":noise_seed,"noise_sigma":noise_sigma})
    return rows
