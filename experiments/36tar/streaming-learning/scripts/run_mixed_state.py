"""S4-A bounded runner pinned to the separate local Git branch checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    repo = root/"code/qwen35-streaming-s4"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("diagnostic","cpu","preflight","gpu"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--model-dir",type=Path,default=root/"models/Qwen3.5-0.8B")
    parser.add_argument("--video",type=Path,default=root/"datasets/natural_commons_v1/clean_hands.webm")
    parser.add_argument("--cpu-check",type=Path)
    parser.add_argument("--preflight",type=Path)
    parser.add_argument("--recurrent-reference",action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error("New run ID required")
    if args.stage == "gpu":
        if args.cpu_check is None or args.preflight is None:
            parser.error("GPU requires CPU tests and matching preflight; 10 groups/256 tokens/600 seconds")
        for proof,stage in ((args.cpu_check,"cpu"),(args.preflight,"preflight")):
            if read(proof/"status.json") != {"status":"completed","stage":stage}:
                parser.error(f"{stage} did not complete")
            for relative,digest in read(proof/"provenance.json")["code_hashes"].items():
                if sha(repo/relative) != digest:
                    parser.error(f"Code changed after checks: {relative}")
        if not read(args.cpu_check/"cpu_summary.json")["passed"]:
            parser.error("CPU tests failed")
        checked = read(args.preflight/"arguments.json")
        if bool(checked.get("recurrent_reference",False)) != args.recurrent_reference:
            parser.error("Precision rule changed after preflight")
        if args.recurrent_reference and not read(args.cpu_check/"cpu_summary.json").get("recurrent_reference",False):
            parser.error("Recurrent candidate CPU contracts required")
        for key in ("model_dir","video"):
            if Path(checked[key]).resolve() != getattr(args,key).resolve():
                parser.error(f"{key} changed after preflight")
        pre = read(args.preflight/"preflight_summary.json")
        if (pre["total_groups"] != 10 or sha(args.video) != pre["video_sha256"] or
                sha(args.model_dir/"DOWNLOAD_MANIFEST.json") != pre["model_manifest_sha256"]):
            parser.error("Budget or input assets changed")
        branch = subprocess.check_output(["git","branch","--show-current"],cwd=repo,text=True).strip()
        if branch != "36tar/s4-mixed-state":
            parser.error("GPU expects the registered personal S4 branch, never main")
    args.out.parent.mkdir(parents=True,exist_ok=True)
    log = args.out.with_name(args.out.name+".console.log")
    module = "scripts.chunk_experiment" if args.stage in ("diagnostic","cpu") else "scripts.chunk_weight_check"
    command = [sys.executable,"-m",module,"--stage",args.stage,"--out",str(args.out)]
    if args.recurrent_reference:
        command.append("--recurrent-reference")
    if args.stage in ("preflight","gpu"):
        command += ["--model-dir",str(args.model_dir.resolve()),"--video",str(args.video.resolve())]
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
    print(log.read_text(encoding="utf-8")[-4500:])
    return code


if __name__ == "__main__":
    sys.exit(main())
