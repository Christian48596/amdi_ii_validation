#!/usr/bin/env python3
"""Verify exact protocol alignment between deterministic AMDI pathways."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from amdi.synthetic import add_gaussian_noise, four_region_image
from learned_amdi.config import load_json
from learned_amdi.features import normalized_error
from learned_amdi.io import ensure_dir, write_csv, write_json
from learned_amdi.runner import run_deterministic_trajectory
from learned_amdi.training import build_backend, make_reference

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(ROOT/'configs'/'development.json')); args=ap.parse_args()
    cfg=load_json(args.config); n=int(cfg['data']['train_n']); steps=int(cfg['training']['n_steps']); sigma=float(cfg['data']['noise_sigma'])
    truth=four_region_image(n); noisy=add_gaussian_noise(truth,sigma,seed=23)
    b0=build_backend(noisy,cfg); ref=make_reference(noisy,b0,steps)
    b1=build_backend(noisy,cfg); d1=run_deterministic_trajectory(b1,ref,steps,'baseline')
    b2=build_backend(noisy,cfg); d2=run_deterministic_trajectory(b2,ref,steps,'control')
    rows=[]
    for a,b in zip(d1,d2):
        rows += [
            {"pathway":"deterministic AMDI baseline",**a},
            {"pathway":"modular deterministic control",**b},
        ]
    final_diff=normalized_error(b2.current_field(),b1.current_field(),1e-12)
    tree_distance=b1.tree.distance(b2.tree)
    pass_alignment=bool(final_diff <= 1e-14 and tree_distance == 0 and abs(b1.occupancy-b2.occupancy) <= 1e-15)
    summary={
        "final_relative_difference":float(final_diff),
        "C_rel_baseline":b1.occupancy,
        "C_rel_control":b2.occupancy,
        "tree_distance":tree_distance,
        "pass":pass_alignment,
        "interpretation":"The modular deterministic control reproduces the deterministic AMDI propagate-then-adapt pathway exactly." if pass_alignment else "Protocol alignment failed: do not run publication training.",
    }
    out=ensure_dir(ROOT/'results'/'02_protocol_alignment_audit'); write_csv(out/'trajectory.csv',rows); write_json(out/'summary.json',summary); print(json.dumps(summary,indent=2))
    if not pass_alignment:
        raise SystemExit(2)
if __name__=='__main__': main()
