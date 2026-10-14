"""600-second child watchdog for the finite quality verification."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--stage',choices=('preflight','gpu'),required=True)
p.add_argument('--arm',choices=('both','original','streaming'),default='both')
args,_=p.parse_known_args()
out=args.out.resolve()
if out.exists():p.error('Fresh output directory required')
out.parent.mkdir(parents=True,exist_ok=True)
log=out.with_name(out.name+'.console.log')
env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
sys.stdout.reconfigure(encoding='utf-8')
started=time.monotonic()
def child(arguments,console):
    with console.open('x',encoding='utf-8') as stream:
        try:
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate.py')),*arguments],
                env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=max(1.,600-(time.monotonic()-started)))
            code=result.returncode
        except subprocess.TimeoutExpired:code=124
    print(console.read_text(encoding='utf-8')[-9000:],flush=True)
    if code:raise SystemExit(code)

if args.stage=='gpu' and args.arm=='both':
    out.mkdir()
    reports=[];summaries=[]
    try:
        for arm in ('original','streaming'):
            arguments=list(sys.argv[1:])
            for flag in ('--out','--arm'):
                if flag in arguments:
                    i=arguments.index(flag);del arguments[i:i+2]
            arguments+=['--out',str(out/arm),'--arm',arm]
            child(arguments,out/(arm+'.console.log'))
            summary=json.loads((out/arm/'summary.json').read_text(encoding='utf-8'))
            summaries.append(summary);reports.extend(summary['reports'])
        if summaries[0]['source_hashes']!=summaries[1]['source_hashes'] or summaries[0]['input_identity']!=summaries[1]['input_identity']:
            raise RuntimeError('Independent arms have different sources or inputs')
        comparisons=[]
        for left,right in zip(reports[0]['answers'],reports[1]['answers']):
            if (left['file'],left['checkpoint'])!=(right['file'],right['checkpoint']):raise RuntimeError('Pairing drift')
            comparisons.append({'file':left['file'],'checkpoint':left['checkpoint'],
                'original':left['answer'],'streaming':right['answer'],
                'same_generated_ids':left['generated_ids']==right['generated_ids']})
        summary={**summaries[0],'reports':reports,'comparisons':comparisons,
                 'wall_seconds_both_arms':time.monotonic()-started,'independent_processes':True}
        (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        (out/'status.json').write_text(json.dumps({'status':'completed','stage':'gpu'}),encoding='utf-8')
    except BaseException:
        (out/'status.json').write_text(json.dumps({'status':'failed','stage':'gpu'}),encoding='utf-8')
        raise
else:
    child(sys.argv[1:],log)
