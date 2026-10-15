"""Serial bounded real-video checks, using existing proof-gated entrypoints."""
from pathlib import Path
import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[2]
CORE=ROOT/'code/qwen35-streaming-s4'
MODEL=Path(os.environ.get('QWEN35_MODEL_DIR','D:/AI_project/models/Qwen3.5-0.8B'))
DATA=Path(os.environ.get('QWEN35_CLOSEOUT_DATA_DIR','D:/AI_project/data/ucf101_capture_validation'))

def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def contracts(name,out):
    out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(CORE))
    if name=='temporal':
        directory=ROOT/'diagnostics/temporal_frame_memory_v1'
        sys.path.insert(0,str(directory))
        suite=unittest.defaultTestLoader.loadTestsFromName('test_memory.MemoryTests')
    else:
        import torch
        from transformers import AutoProcessor
        from tests.test_online import small_model
        torch.set_num_threads(1);torch.manual_seed(42)
        directory=ROOT/'diagnostics/time_file_v1'
        sys.path.insert(0,str(directory))
        importlib.import_module('file_demo')  # Register the entry alias used by its contracts.
        module=importlib.import_module('contracts')
        module.Contracts.processor=AutoProcessor.from_pretrained(MODEL,local_files_only=True)
        module.Contracts.model=small_model(module.Contracts.processor)
        suite=unittest.defaultTestLoader.loadTestsFromTestCase(module.Contracts)
    started=time.monotonic()
    with (out/'unittest.log').open('w',encoding='utf-8',newline='\n') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    row={'suite':name,'tests':result.testsRun,'passed':result.wasSuccessful(),'failures':len(result.failures),
         'errors':len(result.errors),'skipped':len(result.skipped),'wall_seconds':time.monotonic()-started}
    write(out/'summary.json',row);print(json.dumps(row),flush=True)
    return 0 if result.wasSuccessful() and not result.skipped else 1

if len(sys.argv)>1 and sys.argv[1]=='--contracts':
    raise SystemExit(contracts(sys.argv[2],Path(sys.argv[3]).resolve()))

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('out',type=Path,help='Fresh controller output; its name prefixes all child outputs')
parser.add_argument('--cpu-run',type=Path,required=True,help='Successful complete CPU suite directory')
args=parser.parse_args()
assert json.loads((args.cpu_run/'summary.json').read_text(encoding='utf-8'))['all_passed']
out=args.out.resolve()
prefix=out.name
for suffix in ('capture','quality_cpu','quality_gpu','generic_cpu','generic_gpu','sparse_file_cpu','sparse_file_gpu'):
    if (ROOT/'runs'/f'{prefix}_{suffix}').exists():
        raise SystemExit(f'Output already exists: {prefix}_{suffix}')
out.mkdir(parents=True,exist_ok=False)
(out/'source_snapshot').mkdir()
for p in Path(__file__).parent.iterdir():
    if p.suffix in ('.py','.md'):(out/'source_snapshot'/p.name).write_bytes(p.read_bytes())
rows=[]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',QWEN35_MODEL_DIR=str(MODEL))

def job(name,args,dest,timeout=600):
    started=time.monotonic()
    with (out/(name+'.console.log')).open('x',encoding='utf-8') as stream:
        try:
            p=subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
            code=p.returncode
        except subprocess.TimeoutExpired:code=124
    row={'job':name,'exit_code':code,'passed':code==0,'wall_seconds':time.monotonic()-started,'output':str(dest.relative_to(ROOT))}
    rows.append(row);write(out/'jobs.json',rows);print(json.dumps(row),flush=True)
    return code==0

for name in ('temporal','time_file'):
    dest=out/name
    job(name,[Path(__file__),'--contracts',name,dest],dest)
cap=ROOT/'runs'/f'{prefix}_capture'
job('capture',['diagnostics/capture_dataset_v1/validate.py','--data-dir',DATA,'--out',cap],cap,240)
quality_cpu=ROOT/'runs'/f'{prefix}_quality_cpu'
quality_gpu=ROOT/'runs'/f'{prefix}_quality_gpu'
quality=['diagnostics/ucf101_quality_v1/run.py','--model-dir',MODEL,'--data-dir',DATA]
if job('quality_cpu',[*quality,'--stage','preflight','--out',quality_cpu],quality_cpu):
    job('quality_gpu',[*quality,'--stage','gpu','--preflight',quality_cpu,'--out',quality_gpu],quality_gpu)
video=DATA/'v_ShavingBeard_g02_c05.avi'
common=['--model-dir',MODEL,'--video',video,'--question','What is the main action shown in the video? Answer in one short sentence.',
        '--answer-tokens','16','--max-tokens','288','--wall-seconds','180']
for name,launcher,extra in (
    ('generic','scripts/run_video_demo.py',['--fps','1','--duration','8','--max-frames','8','--first-qa-frames','4','--qa-every','4','--playback','clock']),
    ('sparse_file','scripts/run_live_demo.py',['--input-kind','file','--fps','0.2','--duration','12','--max-frames','4','--first-qa-frames','2','--qa-every','2'])):
    cpu=ROOT/'runs'/f'{prefix}_{name}_cpu'
    gpu=ROOT/'runs'/f'{prefix}_{name}_gpu'
    args=[launcher,*common,*extra]
    if job(name+'_cpu',[*args,'--stage','preflight','--out',cpu],cpu,200):
        job(name+'_gpu',[*args,'--stage','gpu','--preflight',cpu,'--out',gpu],gpu,200)
write(out/'summary.json',{'all_passed':all(r['passed'] for r in rows),'jobs':rows,'gpu_model_loads_max':4,
    'scope':'0.8B finite closeout; no camera, 2B, training or formal benchmark claim'})
raise SystemExit(0 if all(r['passed'] for r in rows) else 1)
