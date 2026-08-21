#!/usr/bin/env python3
"""Verify exact agreement between the modular wrapper and deterministic AMDI baseline."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from amdi.energy import EnergyParameters
from amdi.graph import GraphParameters
from amdi.haar import adaptive_tree_from_detail_threshold, full_coefficients
from amdi.solver import adaptive_frozen_denoise
from amdi.synthetic import add_gaussian_noise, four_region_image
from learned_amdi.config import load_json
from learned_amdi.io import ensure_dir, write_json
from learned_amdi.training import build_backend

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(ROOT/'configs'/'development.json')); args=ap.parse_args()
    cfg=load_json(args.config); n=int(cfg['data']['train_n']); steps=int(cfg['training']['n_steps']); sigma=float(cfg['data']['noise_sigma'])
    truth=four_region_image(n); noisy=add_gaussian_noise(truth,sigma,seed=17)
    backend=build_backend(noisy,cfg)
    for _ in range(steps): backend.deterministic_baseline_step()
    image_b=backend.current_field(); tree_b=backend.tree; coeff_b=backend.coeffs

    full=full_coefficients(noisy,max_level=int(np.log2(n)))
    initial=adaptive_tree_from_detail_threshold(full,2,int(np.log2(n)),threshold=float(cfg['amdi']['initial_threshold']),min_level=int(cfg['amdi']['min_level']))
    ep=EnergyParameters(alpha=float(cfg['amdi']['alpha']),beta=float(cfg['amdi']['beta']),tau=float(cfg['amdi']['tau']),smooth_l1=False)
    gp=GraphParameters(sigma_c=float(cfg['amdi']['sigma_c']),state_dependent=True,refinement_decay=float(cfg['amdi']['refinement_decay']))
    image_a,tree_a,coeff_a,_=adaptive_frozen_denoise(noisy,initial,full,ep,gp,h=float(cfg['amdi']['h']),outer_iterations=steps,adapt=True,zeta=float(cfg['amdi']['zeta']),refine_fraction=float(cfg['amdi']['refine_fraction']),coarsen_fraction=float(cfg['amdi']['coarsen_fraction']))
    keys=set(coeff_a)|set(coeff_b); coeff_err=max([abs(coeff_a.get(k,0)-coeff_b.get(k,0)) for k in keys] or [0.0])
    result={"image_relative_error":float(np.linalg.norm(image_a-image_b)/max(np.linalg.norm(image_a),1e-15)),"tree_identical":bool(tree_a.refined==tree_b.refined),"max_coefficient_abs_error":float(coeff_err),"basis_size_direct":tree_a.basis_size(),"basis_size_wrapper":tree_b.basis_size()}
    result['pass']=bool(result['image_relative_error']<1e-12 and result['tree_identical'] and coeff_err<1e-12)
    out=ensure_dir(ROOT/'results'/'01_deterministic_backend_regression'); write_json(out/'regression.json',result); print(json.dumps(result,indent=2))
    if not result['pass']: raise SystemExit('deterministic AMDI wrapper regression FAILED')
if __name__=='__main__': main()
