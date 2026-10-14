"""Serial experiment child with 480-second hard watchdog."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--out', type=Path, required=True)
args, _ = parser.parse_known_args()
out = args.out.resolve()
if out.exists():
    parser.error('Fresh output directory required')
out.parent.mkdir(parents=True, exist_ok=True)
log = out.with_name(out.name + '.console.log')
env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1',
           HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
with log.open('x', encoding='utf-8') as stream:
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).with_name('probe.py')), *sys.argv[1:]],
                                env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=480)
        code, timed_out = result.returncode, False
    except subprocess.TimeoutExpired:
        code, timed_out = 124, True
if out.exists():
    (out / 'watchdog_status.json').write_text(json.dumps({'exit_code': code, 'timed_out': timed_out,
                                                       'limit_seconds': 480}), encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
print(log.read_text(encoding='utf-8')[-7500:])
print('Console log: ' + str(log))
sys.exit(code)
