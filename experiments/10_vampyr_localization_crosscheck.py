#!/usr/bin/env python3
"""Compare deterministic AMDI, learned-AMDI, and VAMPyR regional localization."""
from __future__ import annotations
import argparse,sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from amdi.haar import refinement_level_map
from amdi.synthetic import add_gaussian_noise,four_region_image
from amdi.vampyr_adapter import adaptive_project,end_node_table,tree_summary
from learned_amdi.config import RewardConfig
from learned_amdi.evaluation import load_policy
from learned_amdi.io import ensure_dir,write_csv,write_json
from learned_amdi.runner import collect_trajectory,run_deterministic_trajectory
from learned_amdi.training import build_backend,make_reference

def f(r):
 x,y=float(r[0]),float(r[1])
 if x<0.5 and y<0.5:return 0.25
 if x>=0.5 and y<0.5:return 0.15+0.7*(x-0.5)/0.5
 if x<0.5 and y>=0.5:return 0.9 if ((x-0.25)**2+(y-0.75)**2)<=0.12**2 else 0.15
 return float(np.clip(0.5+0.28*np.sin(18*np.pi*x)*np.sin(18*np.pi*y),0,1))

def rname(x,y):
 if x<0.5 and y<0.5:return 'constant'
 if x>=0.5 and y<0.5:return 'gradient'
 if x<0.5 and y>=0.5:return 'edge'
 return 'texture'

def means(levels):
 n=levels.shape[0]; regs={'constant':(slice(0,n//2),slice(0,n//2)),'gradient':(slice(0,n//2),slice(n//2,n)),'edge':(slice(n//2,n),slice(0,n//2)),'texture':(slice(n//2,n),slice(n//2,n))}
 return {k:(float(levels[s].mean()),int(levels[s].max())) for k,s in regs.items()}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint',default=str(ROOT/'checkpoints'/'selected'/'best_policy.pt')); ap.add_argument('--vampyr-precision',type=float,default=1e-3); ap.add_argument('--vampyr-max-depth',type=int,default=8); ap.add_argument('--device',default='cpu'); a=ap.parse_args(); device=torch.device(a.device)
 actor,critic,scales,cfg=load_policy(a.checkpoint,device); n=int(cfg['data']['train_n']); sigma=float(cfg['data']['noise_sigma']); steps=int(cfg['training']['n_steps']); truth=four_region_image(n); noisy=add_gaussian_noise(truth,sigma,seed=137); b0=build_backend(noisy,cfg); ref=make_reference(noisy,b0,steps)
 b1=build_backend(noisy,cfg); run_deterministic_trajectory(b1,ref,steps,'baseline'); m1=means(refinement_level_map(b1.tree,truth.shape))
 b2=build_backend(noisy,cfg); collect_trajectory(b2,ref,actor,critic,scales,RewardConfig(**cfg['reward']),steps,device,True); m2=means(refinement_level_map(b2.tree,truth.shape))
 out=ensure_dir(ROOT/'results'/'10_vampyr_localization_crosscheck')
 try: vt=adaptive_project(f,dim=2,order=5,precision=a.vampyr_precision,max_depth=a.vampyr_max_depth)
 except RuntimeError as e:
  write_json(out/'vampyr_skipped.json',{'reason':str(e)}); print(e); return
 vrows=end_node_table(vt); vreg={k:[] for k in m1}
 for row in vrows:
  x,y=row['center'][:2]; lo=np.asarray(row['lower'][:2]); hi=np.asarray(row['upper'][:2]); area=max(float(np.prod(hi-lo)),1e-300); vreg[rname(x,y)].append(-0.5*np.log2(area))
 rows=[]
 for name in m1:
  rows.append({'region':name,'amdi_mean_level':m1[name][0],'amdi_max_level':m1[name][1],'learned_mean_level':m2[name][0],'learned_max_level':m2[name][1],'vampyr_mean_effective_level':float(np.mean(vreg[name])) if vreg[name] else np.nan,'vampyr_max_effective_level':float(np.max(vreg[name])) if vreg[name] else np.nan})
 write_csv(out/'regional_localization.csv',rows); write_json(out/'vampyr_tree_summary.json',tree_summary(vt)); print(rows)
if __name__=='__main__': main()
