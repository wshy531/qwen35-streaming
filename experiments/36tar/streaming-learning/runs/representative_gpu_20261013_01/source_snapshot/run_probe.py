"""Serial experiment child with a 360-second watchdog and persistent logs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--stage', choices=('preflight', 'gpu'), required=True)
args, _ = parser.parse_known_args()
out = args.out.resolve()
if out.exists():
    parser.error('Fresh output directory required')
out.parent.mkdir(parents=True, exist_ok=True)
out.mkdir()
snapshot = out / 'source_snapshot'
snapshot.mkdir()
source_hashes = {}
for path in Path(__file__).resolve().parent.iterdir():
    if path.suffix in ('.py', '.md', '.json'):
        shutil.copyfile(path, snapshot / path.name)
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
(out / 'bootstrap.json').write_text(json.dumps({'prepared_by': 'representative_v1_watchdog',
                                               'files': source_hashes}), encoding='utf-8')
log = out.with_name(out.name + '.console.log')
env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
with log.open('x', encoding='utf-8') as stream:
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('probe.py')), *sys.argv[1:]],
                                env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=360)
        code, timed_out = result.returncode, False
    except subprocess.TimeoutExpired:
        code, timed_out = 124, True
if out.exists():
    if not (out / 'status.json').exists():
        (out / 'status.json').write_text(json.dumps({'status': 'failed', 'stage': args.stage,
             'phase': 'bootstrap_or_watchdog', 'exit_code': code, 'timed_out': timed_out}), encoding='utf-8')
    (out / 'watchdog_status.json').write_text(json.dumps({'exit_code': code, 'timed_out': timed_out,
                                                       'limit_seconds': 360}), encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
print(log.read_text(encoding='utf-8')[-8500:])
print('Console log: ' + str(log))
sys.exit(code)
