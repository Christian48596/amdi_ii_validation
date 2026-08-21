#!/usr/bin/env python3
from __future__ import annotations
import json, platform, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from learned_amdi.io import ensure_dir, write_json

def version(name):
    try:
        m=__import__(name); return getattr(m,'__version__','unknown')
    except Exception as e: return f'unavailable: {e}'

def main():
    out=ensure_dir(ROOT/'results'/'00_environment_report')
    data={"python":sys.version,"platform":platform.platform(),"numpy":version('numpy'),"scipy":version('scipy'),"skimage":version('skimage'),"torch":version('torch'),"matplotlib":version('matplotlib'),"pandas":version('pandas')}
    try:
        from vampyr import vampyr2d as vp
        data['vampyr']='import OK'
        try: data['mrcpp_version']=str(vp.MRA())
        except Exception: pass
    except Exception as e: data['vampyr']=f'unavailable: {e}'
    write_json(out/'environment.json',data); print(json.dumps(data,indent=2))
if __name__=='__main__': main()
