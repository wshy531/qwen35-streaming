"""Bounded offline child, fresh output, 300-second external watchdog."""
import argparse,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(add_help=False);p.add_argument('--out',type=Path,required=True);p.add_argument('--wall-seconds',type=int,default=300)
a,_=p.parse_known_args();out=a.out.resolve()
if out.exists() or not 1<=a.wall_seconds<=300:p.error('Fresh output and watchdog <=300s required')
out.parent.mkdir(parents=True,exist_ok=True);log=out.with_name(out.name+'.console.log')
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
with log.open('x',encoding='utf-8') as f:
    try:
        r=subprocess.run([sys.executable,str(Path(__file__).with_name('benchmark.py')),*sys.argv[1:]],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=a.wall_seconds)
        code,timed=r.returncode,False
    except subprocess.TimeoutExpired:code,timed=124,True
if out.exists():(out/'watchdog_status.json').write_bytes(json.dumps({'exit_code':code,'timed_out':timed,'wall_seconds':a.wall_seconds}).encode('utf-8'))
sys.stdout.reconfigure(encoding='utf-8');print(log.read_text(encoding='utf-8')[-7500:]);sys.exit(code)
