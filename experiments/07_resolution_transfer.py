#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from learned_amdi.evaluation import evaluate_case,load_policy
from learned_amdi.io import ensure_dir,write_csv

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=str(ROOT/'checkpoints'/'selected'/'best_policy.pt')); ap.add_argument('--resolutions',default='32,64,128'); ap.add_argument('--device',default='cpu'); a=ap.parse_args(); device=torch.device(a.device); _,_,_,cfg=load_policy(a.checkpoint,device); rows=[]
 for n in [int(x) for x in a.resolutions.split(',')]:
  for seed in cfg['data']['test_image_seeds'][:4]: rows += evaluate_case(a.checkpoint,kind='random_multiscale',n=n,image_seed=int(seed),noise_sigma=float(cfg['data']['noise_sigma']),noise_seed=50000+int(seed)+n,device=device)
 out=ensure_dir(ROOT/'results'/'07_resolution_transfer'); write_csv(out/'resolution_transfer_runs.csv',rows)
 summary=[]
 for n in sorted({r['n'] for r in rows}):
  for method in sorted({r['method'] for r in rows}):
   rr=[r for r in rows if r['n']==n and r['method']==method]; summary.append({"n":n,"method":method,"RMSE_mean":float(np.mean([r['RMSE'] for r in rr])),"SSIM_mean":float(np.mean([r['SSIM'] for r in rr])),"C_rel_mean":float(np.mean([r['C_rel'] for r in rr])),"reference_error_mean":float(np.mean([r['reference_error'] for r in rr]))})
 write_csv(out/'resolution_transfer_summary.csv',summary); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
