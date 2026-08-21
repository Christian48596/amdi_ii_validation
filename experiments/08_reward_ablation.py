#!/usr/bin/env python3
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from learned_amdi.config import load_json
from learned_amdi.evaluation import evaluate_case
from learned_amdi.io import ensure_dir,write_csv
from learned_amdi.training import train

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(ROOT/'configs'/'publication.json')); ap.add_argument('--train-missing',action='store_true'); ap.add_argument('--device',default='cpu'); a=ap.parse_args(); base=load_json(a.config); device=torch.device(a.device)
 variants={"full":{},"no_switching":{"lambda_sw":0.0},"no_occupancy":{"lambda_occ":0.0}}
 rows=[]
 for name,mods in variants.items():
  cfg=copy.deepcopy(base); cfg['reward'].update(mods); ck=ROOT/'checkpoints'/'ablation'/name/'best_policy.pt'
  if not ck.exists():
   if not a.train_missing: continue
   train(cfg,ck.parent,device)
  for seed in cfg['data']['test_image_seeds'][:4]:
   rr=evaluate_case(ck,kind='random_multiscale',n=int(cfg['data']['train_n']),image_seed=int(seed),noise_sigma=float(cfg['data']['noise_sigma']),noise_seed=60000+int(seed),device=device)
   learned=[r for r in rr if r['method']=='Learned AMDI'][0]; rows.append({"ablation":name,**learned})
 out=ensure_dir(ROOT/'results'/'08_reward_ablation'); write_csv(out/'ablation_runs.csv',rows)
 summary=[]
 for name in sorted({r['ablation'] for r in rows}):
  rr=[r for r in rows if r['ablation']==name]; summary.append({"ablation":name,"reference_error_mean":float(np.mean([r['reference_error'] for r in rr])),"C_rel_mean":float(np.mean([r['C_rel'] for r in rr])),"switching_mean":float(np.mean([r['switching_mean'] for r in rr])),"RMSE_mean":float(np.mean([r['RMSE'] for r in rr]))})
 write_csv(out/'ablation_summary.csv',summary); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
