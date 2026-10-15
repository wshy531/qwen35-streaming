"""Bounded independent-process CPU suites for the 0.8B closeout."""
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[2]
MODEL = Path(os.environ.get('QWEN35_MODEL_DIR', 'D:/AI_project/models/Qwen3.5-0.8B'))
TARGETS = {'root': REPO, 'baseline': ROOT/'code/qwen35-streaming', 'streaming': ROOT/'code/qwen35-streaming-s4'}

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def suite(name, out):
    target = TARGETS[name]
    out.mkdir(parents=True, exist_ok=False)
    os.chdir(target)
    sys.path.insert(0, str(target))
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(42)
    tests = unittest.defaultTestLoader.discover(str(target/'tests'), top_level_dir=str(target))
    count = tests.countTestCases()
    sources = {p.relative_to(REPO).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
               for folder in ('tests','qwen35_local','scripts','streaming') if (target/folder).exists()
               for p in sorted((target/folder).rglob('*.py'))}
    write(out/'source_hashes.json',sources)
    started = time.monotonic()
    with (out/'unittest.log').open('w',encoding='utf-8',newline='\n') as log:
        result = unittest.TextTestRunner(stream=log,verbosity=2).run(tests)
    summary = {'suite':name,'discovered':count,'tests':result.testsRun,'passed':result.wasSuccessful(),
               'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
               'wall_seconds':time.monotonic()-started,'python':sys.executable,
               'packages':{n:importlib.metadata.version(n) for n in ('torch','transformers','numpy','av')}}
    write(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    return 0 if result.wasSuccessful() and not result.skipped else 1

if __name__=='__main__':
    if len(sys.argv)==4 and sys.argv[1]=='--suite':
        raise SystemExit(suite(sys.argv[2],Path(sys.argv[3]).resolve()))
    out=Path(sys.argv[1]).resolve()
    out.mkdir(parents=True,exist_ok=False)
    (out/'source_snapshot').mkdir()
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py','.md'):
            (out/'source_snapshot'/path.name).write_bytes(path.read_bytes())
    rows=[]
    for name in TARGETS:
        with (out/(name+'.console.log')).open('w',encoding='utf-8') as log:
            try:
                result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--suite',name,str(out/name)],
                                      stdout=log,stderr=subprocess.STDOUT,timeout=600,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
                code=result.returncode
            except subprocess.TimeoutExpired:
                code=124
        path=out/name/'summary.json'
        row=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'suite':name,'passed':False,'exit_code':code}
        rows.append(row)
        print(json.dumps(row),flush=True)
    python_files=subprocess.check_output(['git','ls-files','--cached','-z','--','*.py'],cwd=REPO).split(b'\0')
    syntax_errors=[]; parsed=0
    for rel in python_files:
        if not rel: continue
        path=REPO/rel.decode('utf-8')
        try: ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path));parsed+=1
        except (SyntaxError,UnicodeError) as exc: syntax_errors.append({'file':rel.decode('utf-8'),'error':str(exc)})
    summary={'suites':rows,'all_passed':all(r.get('passed') and not r.get('skipped') for r in rows),
             'tracked_python_ast_parsed':parsed,'syntax_errors':syntax_errors,
             'test_count':sum(r.get('tests',0) for r in rows),'random_weights_are_not_quality_evidence':True}
    write(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    raise SystemExit(0 if summary['all_passed'] and not syntax_errors else 1)
