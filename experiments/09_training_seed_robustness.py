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
 ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(ROOT/'configs'/'publication.json')); ap.add_argument('--training-seeds',default='20260811,20260817,20260823'); ap.add_argument('--train-missing',action='store_true'); ap.add_argument('--device',default='cpu'); a=ap.parse_args(); base=load_json(a.config); device=torch.device(a.device); rows=[]
 for tseed in [int(x) for x in a.training_seeds.split(',')]:
  cfg=copy.deepcopy(base); cfg['training']['seed']=tseed; ck=ROOT/'checkpoints'/'robustness'/str(tseed)/'best_policy.pt'
  if not ck.exists():
   if not a.train_missing: continue
   train(cfg,ck.parent,device)
  for seed in cfg['data']['test_image_seeds'][:4]:
   rr=evaluate_case(ck,kind='random_multiscale',n=int(cfg['data']['train_n']),image_seed=int(seed),noise_sigma=float(cfg['data']['noise_sigma']),noise_seed=70000+int(seed),device=device); learned=[r for r in rr if r['method']=='Learned AMDI'][0]; rows.append({"training_seed":tseed,**learned})
 out=ensure_dir(ROOT/'results'/'09_training_seed_robustness'); write_csv(out/'robustness_runs.csv',rows); print(f'wrote {len(rows)} rows')
if __name__=='__main__': main()
