"""Bounded natural-video runner with full asset/code preflight matching."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    repo = root/"code/qwen35-streaming"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("preflight","gpu"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--assets",type=Path,default=root/"datasets/natural_commons_v1")
    parser.add_argument("--model-dir",type=Path,default=root/"models/Qwen3.5-0.8B")
    parser.add_argument("--preflight",type=Path)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error("New output ID required")
    if args.stage == "gpu":
        if args.preflight is None or read(args.preflight/"status.json") != {"status":"completed","stage":"preflight"}:
            parser.error("Successful matching interface preflight required; 15 questions/16 tokens/600 seconds")
        checked = read(args.preflight/"arguments.json")
        for key in ("assets","model_dir"):
            if Path(checked[key]).resolve() != getattr(args,key).resolve():
                parser.error(f"{key} changed")
        for relative,sha in read(args.preflight/"provenance.json")["code_hashes"].items():
            if digest(repo/relative) != sha:
                parser.error(f"Code changed: {relative}")
        proof = read(args.preflight/"preflight_summary.json")
        if proof["questions"] != 15:
            parser.error("Fixed workload changed")
        for name,sha in proof["asset_hashes"].items():
            path = (args.assets/name).resolve()
            if path.parent != args.assets.resolve() or digest(path) != sha:
                parser.error(f"Asset changed: {name}")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    log = args.out.with_name(args.out.name+".console.log")
    command = [sys.executable,"-m","scripts.natural_experiment","--stage",args.stage,"--out",str(args.out)]
    for key in ("assets","model_dir"):
        command += ["--"+key.replace("_","-"),str(getattr(args,key).resolve())]
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    timed_out = False
    with log.open("x",encoding="utf-8") as stream:
        try:
            result = subprocess.run(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code,timed_out = 124,True
    if args.out.exists():
        (args.out/"watchdog_status.json").write_text(json.dumps({"wall_time_limit_seconds":600,
            "timed_out":timed_out,"exit_code":code,"console_log":str(log)},indent=2),encoding="utf-8")
    print(f"Exit {code}; console: {log}")
    print(log.read_text(encoding="utf-8")[-5000:])
    return code


if __name__ == "__main__":
    sys.exit(main())
