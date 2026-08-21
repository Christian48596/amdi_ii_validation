#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from learned_amdi.config import load_json
from learned_amdi.training import train

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(ROOT/'configs'/'publication.json')); ap.add_argument('--out',default=str(ROOT/'checkpoints'/'training_reproduction')); ap.add_argument('--device',default='cpu'); a=ap.parse_args(); cfg=load_json(a.config); result=train(cfg,a.out,torch.device(a.device)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
