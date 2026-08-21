from __future__ import annotations

import csv, json
from pathlib import Path


def ensure_dir(path):
    p=Path(path); p.mkdir(parents=True,exist_ok=True); return p


def write_json(path,data):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(data,indent=2,sort_keys=True),encoding="utf-8")


def write_csv(path,rows):
    rows=list(rows); p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        p.write_text("",encoding="utf-8"); return
    keys=[]
    for row in rows:
        for k in row:
            if k not in keys: keys.append(k)
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
