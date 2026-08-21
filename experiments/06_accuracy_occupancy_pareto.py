#!/usr/bin/env python3
"""Train/evaluate several occupancy weights and construct the learned Pareto front."""
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from learned_amdi.config import load_json,save_json
from learned_amdi.evaluation import evaluate_case
from learned_amdi.io import ensure_dir,write_csv
from learned_amdi.training import train

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(ROOT/'configs'/'publication.json')); ap.add_argument('--weights',default='0.03,0.08,0.15,0.25,0.30,0.60'); ap.add_argument('--train-missing',action='store_true'); ap.add_argument('--device',default='cpu'); a=ap.parse_args(); base=load_json(a.config); device=torch.device(a.device); out=ensure_dir(ROOT/'results'/'06_accuracy_occupancy_pareto'); rows=[]
 for w in [float(x) for x in a.weights.split(',')]:
  tag=f'occ_{w:.4g}'.replace('.','p'); ck=ROOT/'checkpoints'/'pareto'/tag/'best_policy.pt'; cfg=copy.deepcopy(base); cfg['reward']['lambda_occ']=w
  if not ck.exists():
   if not a.train_missing: continue
   train(cfg,ck.parent,device)
  for seed in cfg['data']['test_image_seeds'][:4]:
   rr=evaluate_case(ck,kind='random_multiscale',n=int(cfg['data']['train_n']),image_seed=int(seed),noise_sigma=float(cfg['data']['noise_sigma']),noise_seed=40000+int(seed),device=device)
   learned=[r for r in rr if r['method']=='Learned AMDI'][0]; rows.append({"lambda_occ":w,**learned})
 write_csv(out/'pareto_runs.csv',rows)
 agg=[]
 for w in sorted({r['lambda_occ'] for r in rows}):
  rr=[r for r in rows if r['lambda_occ']==w]; agg.append({"lambda_occ":w,"reference_error_mean":float(np.mean([r['reference_error'] for r in rr])),"C_rel_mean":float(np.mean([r['C_rel'] for r in rr])),"RMSE_mean":float(np.mean([r['RMSE'] for r in rr])),"SSIM_mean":float(np.mean([r['SSIM'] for r in rr]))})
 write_csv(out/'pareto_summary.csv',agg); print(json.dumps(agg,indent=2))
if __name__=='__main__': main()
