"""Run publication-only CPU checks in an isolated child; no assets or GPU."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CAP=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
temp_root=out/'test_temp';temp_root.mkdir()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
    PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(CAP/'src'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
    MEMORY_EVAL_TEST_TMP=str(temp_root),TMP=str(temp_root),TEMP=str(temp_root))
start=time.monotonic()
with (out/'tests.log').open('x',encoding='utf-8') as log:
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(CAP/'tests'),'-v'],cwd=CAP,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
if result.returncode:raise SystemExit('CPU checks failed; see tests.log')
verify=subprocess.run([sys.executable,str(CAP/'tools/verify.py'),'--out',str(out/'archive_verification.json')],cwd=CAP,env=env,capture_output=True,text=True,encoding='utf-8',timeout=60)
if verify.returncode:raise SystemExit('Archive check failed: '+verify.stderr)
receipt=dict(status='passed',seconds=time.monotonic()-start,pretrained_loads=0,GPU_used=False,
    scope='delivery contracts and archived result replay; no media revalidation or model quality rerun')
(out/'summary.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False))
