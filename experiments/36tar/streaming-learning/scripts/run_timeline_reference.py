"""Portable original timeline reference with bounded outer watchdog."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--out',type=Path,required=True);p.add_argument('--wall-seconds',type=int,default=90)
    a,_=p.parse_known_args()
    if a.out.exists() or not 1<=a.wall_seconds<=180:p.error('Fresh output and bounded wall time required')
    out=a.out.resolve();out.parent.mkdir(parents=True,exist_ok=True)
    cap=Path(__file__).resolve().parents[1];log=out.with_name(out.name+'.console.log')
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    with log.open('x',encoding='utf-8') as stream:
        try:
            result=subprocess.run([sys.executable,'-m','scripts.timeline_reference',*sys.argv[1:]],cwd=cap/'code/qwen35-streaming-s4',
                                  env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=a.wall_seconds)
            code,timed=result.returncode,False
        except subprocess.TimeoutExpired:code,timed=124,True
    if out.exists():(out/'watchdog_status.json').write_bytes(json.dumps({'exit_code':code,'timed_out':timed,'wall_seconds_including_load':a.wall_seconds},indent=2).encode('utf-8'))
    sys.stdout.reconfigure(encoding='utf-8');print(log.read_text(encoding='utf-8')[-5000:]);return code
if __name__=='__main__':sys.exit(main())
