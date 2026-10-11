"""S3 bounded runner: 100 questions + 4 controls, 600 seconds, checks required."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    project = Path(__file__).resolve().parents[1]
    repo = project/"code"/"qwen35-streaming"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("assets","cpu","preflight","gpu"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--assets",type=Path,default=project/"datasets"/"stability_synthetic")
    parser.add_argument("--model-dir",type=Path,default=project/"models"/"Qwen3.5-0.8B")
    parser.add_argument("--cpu-check",type=Path)
    parser.add_argument("--preflight",type=Path)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error("Choose a new output run ID")
    if args.stage == "gpu":
        if args.cpu_check is None or args.preflight is None:
            parser.error("GPU requires successful CPU and matching preflight")
        if read(args.cpu_check/"status.json") != {"status":"completed","stage":"cpu"} or not read(args.cpu_check/"cpu_summary.json")["passed"]:
            parser.error("CPU contracts failed")
        if read(args.preflight/"status.json") != {"status":"completed","stage":"preflight"}:
            parser.error("Preflight failed")
        checked = read(args.preflight/"arguments.json")
        for key in ("model_dir","assets"):
            if Path(checked[key]).resolve() != getattr(args,key).resolve():
                parser.error(f"{key} changed after preflight")
        for proof in (args.cpu_check,args.preflight):
            for relative,digest in read(proof/"provenance.json")["code_hashes"].items():
                if hashlib.sha256((repo/relative).read_bytes()).hexdigest() != digest:
                    parser.error(f"Code changed after checks: {relative}")
        report = read(args.preflight/"preflight_summary.json")
        if report["questions"] != 100 or report["sampled_frames"] != 200 or len(report["resets"]) != 20:
            parser.error("Preflight differs from registered workload")
        if hashlib.sha256((args.assets/"manifest.json").read_bytes()).hexdigest() != report["asset_manifest_sha256"]:
            parser.error("Asset manifest changed")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    log = args.out.with_name(args.out.name+".console.log")
    command = [sys.executable,"-m","scripts.stability_experiment","--stage",args.stage,"--out",str(args.out),
               "--model-dir",str(args.model_dir.resolve()),"--assets",str(args.assets.resolve())]
    environment = os.environ.copy()
    environment.update(PYTHONIOENCODING="utf-8",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    timed_out = False
    with log.open("x",encoding="utf-8") as stream:
        try:
            result = subprocess.run(command,cwd=repo,env=environment,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
            return_code = result.returncode
        except subprocess.TimeoutExpired:
            return_code,timed_out = 124,True
    if args.out.exists():
        (args.out/"watchdog_status.json").write_text(json.dumps({"wall_time_limit_seconds":600,
            "timed_out":timed_out,"exit_code":return_code,"console_log":str(log)},indent=2),encoding="utf-8")
    print(f"Exit {return_code}; console: {log}")
    print(log.read_text(encoding="utf-8")[-3500:])
    return return_code


if __name__ == "__main__":
    sys.exit(main())
