#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from amdi.synthetic import add_gaussian_noise,four_region_image
from learned_amdi.config import load_json
from learned_amdi.io import ensure_dir,write_json
from learned_amdi.reference import uniform_reference_trajectory
from learned_amdi.training import build_backend

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(ROOT/'configs'/'development.json')); a=ap.parse_args(); cfg=load_json(a.config)
 n=int(cfg['data']['train_n']); noisy=add_gaussian_noise(four_region_image(n),float(cfg['data']['noise_sigma']),seed=29); b=build_backend(noisy,cfg); traj,h=uniform_reference_trajectory(noisy,eparams=b.eparams,gparams=b.gparams,h=b.h,n_steps=int(cfg['training']['n_steps']))
 energies=np.asarray([r['energy'] for r in h]); result={"n_states":len(traj),"basis_size":h[0]['basis_size'],"expected_full_basis":n*n,"energy_monotone":bool(np.all(np.diff(energies)<=1e-10)),"all_safeguards_accepted":bool(all(r.get('safeguard_accepted',True) for r in h[1:]))}
 result['pass']=bool(result['basis_size']==n*n and result['energy_monotone']); out=ensure_dir(ROOT/'results'/'03_uniform_reference_check'); write_json(out/'reference_check.json',result); print(json.dumps(result,indent=2));
 if not result['pass']: raise SystemExit('reference check failed')
if __name__=='__main__': main()
