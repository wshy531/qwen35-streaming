"""Bounded fixed FP32 state regression gated by exact CPU/input preflight."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from run_precision_layers import read,sha


def main():
    root = Path(__file__).resolve().parents[1]
    repo = root/"code/qwen35-streaming-s4"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("preflight","gpu"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--model-dir",type=Path,default=root/"models/Qwen3.5-0.8B")
    parser.add_argument("--video",type=Path,default=root/"datasets/natural_commons_v1/clean_hands.webm")
    parser.add_argument("--preflight",type=Path)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error("New run ID required")
    if args.stage == "gpu":
        if args.preflight is None or read(args.preflight/"status.json") != {"status":"completed","stage":"preflight"}:
            parser.error("Completed matching CPU preflight required")
        proof = read(args.preflight/"preflight_summary.json")
        budget = {"groups":10,"max_tokens":256,"generated_tokens":0,"wall_seconds":600,
                  "model_loads":1,"language_forwards":30,"vision_encodings":1}
        if not proof["cpu_controls"]["passed"] or proof["budget"] != budget:
            parser.error("CPU controls or budget mismatch")
        if sha(Path(__file__)) != proof["launcher_sha256"] or sha(root/"FP32混合状态回归方案.md") != proof["plan_sha256"]:
            parser.error("Launcher or registered plan changed")
        for name,digest in read(args.preflight/"provenance.json")["code_hashes"].items():
            if sha(repo/name) != digest:
                parser.error(f"Source changed: {name}")
        checked = read(args.preflight/"arguments.json")
        for key in ("model_dir","video"):
            if Path(checked[key]).resolve() != getattr(args,key).resolve():
                parser.error(f"Asset path changed: {key}")
        if sha(args.video) != proof["video_sha256"] or sha(args.model_dir/"DOWNLOAD_MANIFEST.json") != proof["model_manifest_sha256"]:
            parser.error("Asset content changed")
        if subprocess.check_output(["git","branch","--show-current"],cwd=repo,text=True).strip() != "36tar/s4-mixed-state":
            parser.error("Expected registered personal branch")
    log = args.out.with_name(args.out.name+".console.log")
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    command = [sys.executable,"-m","scripts.fp32_regression","--stage",args.stage,"--out",str(args.out),
        "--model-dir",str(args.model_dir.resolve()),"--video",str(args.video.resolve())]
    if args.preflight:
        command += ["--preflight",str(args.preflight.resolve())]
    with log.open("x",encoding="utf-8") as stream:
        try:
            result = subprocess.run(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
            code,timed_out = result.returncode,False
        except subprocess.TimeoutExpired:
            code,timed_out = 124,True
    if args.out.exists():
        (args.out/"watchdog_status.json").write_text(json.dumps({"wall_time_limit_seconds":600,"timed_out":timed_out,
            "exit_code":code,"console_log":str(log)},indent=2),encoding="utf-8")
    print(log.read_text(encoding="utf-8")[-7000:])
    return code


if __name__ == "__main__":
    sys.exit(main())
