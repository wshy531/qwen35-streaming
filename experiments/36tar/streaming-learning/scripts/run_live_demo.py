"""Bounded camera launcher; capture child watches parent death independently."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--wall-seconds',type=int,default=90)
    args,_ = parser.parse_known_args()
    if not 1<=args.wall_seconds<=180 or args.out.exists():
        parser.error('Fresh output and wall budget 1..180 seconds required')
    out = args.out.resolve()
    out.parent.mkdir(parents=True,exist_ok=True)
    log = out.with_name(out.name+'.console.log')
    cap = Path(__file__).resolve().parents[1]
    env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    with log.open('x',encoding='utf-8') as stream:
        try:
            result = subprocess.run([sys.executable,'-m','scripts.live_demo',*sys.argv[1:]],
                cwd=cap/'code/qwen35-streaming-s4',env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=args.wall_seconds)
            code,timed = result.returncode,False
        except subprocess.TimeoutExpired:
            code,timed = 124,True
    if out.exists():
        (out/'watchdog_status.json').write_text(json.dumps({'exit_code':code,'timed_out':timed,
            'wall_seconds_including_load':args.wall_seconds},indent=2),encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    print(log.read_text(encoding='utf-8')[-5000:])
    return code


if __name__=='__main__':
    sys.exit(main())
