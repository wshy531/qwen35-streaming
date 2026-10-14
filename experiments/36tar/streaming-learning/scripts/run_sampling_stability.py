"""Portable bounded child launcher for real-PTS sampling phase diagnostics."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--wall-seconds', type=int, default=240)
    args, _ = parser.parse_known_args()
    if not 1 <= args.wall_seconds <= 240 or args.out.exists():
        parser.error('Fresh output directory and wall budget 1..240 required')
    out = args.out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    log = out.with_name(out.name + '.console.log')
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    with log.open('x', encoding='utf-8') as stream:
        try:
            result = subprocess.run([sys.executable, '-m', 'scripts.sampling_stability', *sys.argv[1:]],
                cwd=root / 'code/qwen35-streaming-s4', env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=args.wall_seconds)
            code, timed_out = result.returncode, False
        except subprocess.TimeoutExpired:
            code, timed_out = 124, True
    if out.exists():
        (out / 'watchdog_status.json').write_bytes((json.dumps({'exit_code': code, 'timed_out': timed_out,
            'wall_budget_seconds_including_load': args.wall_seconds}, indent=2) + '\n').encode('utf-8'))
    sys.stdout.reconfigure(encoding='utf-8')
    print(log.read_text(encoding='utf-8')[-6000:])
    print(f'Console log: {log}')
    return code


if __name__ == '__main__':
    sys.exit(main())
