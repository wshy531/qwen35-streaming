"""Run one isolated baseline stage with a 600-second wall-clock budget."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    project = Path(__file__).resolve().parents[1]
    repo = project / "code" / "qwen35-streaming"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu", "assets", "preflight", "gpu"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=project / "models" / "Qwen3.5-0.8B")
    parser.add_argument("--assets", type=Path, default=project / "datasets" / "baseline_synthetic")
    parser.add_argument("--repair-video-positions", action="store_true")
    parser.add_argument("--only-video", action="store_true")
    parser.add_argument("--preflight", type=Path, help="Required completed interface check before GPU")
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error("Output already exists; choose a new run ID")
    if args.stage == "gpu":
        if args.preflight is None:
            parser.error("GPU requires --preflight; budget is one GPU, <=16 output tokens/case, <=600 seconds")
        status = json.loads((args.preflight / "status.json").read_text(encoding="utf-8"))
        checked = json.loads((args.preflight / "arguments.json").read_text(encoding="utf-8"))
        if status != {"status": "completed", "stage": "preflight"}:
            parser.error("Interface preflight was not completed")
        if checked["repair_video_positions"] != args.repair_video_positions:
            parser.error("Position adapter differs from the preflight")
        for key in ("model_dir", "assets"):
            if Path(checked[key]).resolve() != getattr(args, key).resolve():
                parser.error(f"{key} differs from the preflight")
        import hashlib
        expected = json.loads((args.preflight / "provenance.json").read_text(encoding="utf-8"))["code_hashes"]
        for relative, digest in expected.items():
            if hashlib.sha256((repo / relative).read_bytes()).hexdigest() != digest:
                parser.error(f"Code changed after preflight: {relative}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log = args.out.with_name(args.out.name + ".console.log")
    command = [sys.executable, "-m", "scripts.baseline", "--stage", args.stage,
               "--out", str(args.out), "--model-dir", str(args.model_dir.resolve()),
               "--assets", str(args.assets.resolve())]
    if args.repair_video_positions:
        command.append("--repair-video-positions")
    if args.only_video:
        command.append("--only-video")
    environment = os.environ.copy()
    environment.update(PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    timed_out = False
    with log.open("x", encoding="utf-8") as stream:
        try:
            result = subprocess.run(command, cwd=repo, env=environment, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=600)
            return_code = result.returncode
        except subprocess.TimeoutExpired:
            timed_out = True  # subprocess.run kills and waits for the child.
            return_code = 124
    if args.out.exists():
        (args.out / "watchdog_status.json").write_text(json.dumps({
            "wall_time_limit_seconds": 600, "timed_out": timed_out,
            "exit_code": return_code, "console_log": str(log)}, indent=2), encoding="utf-8")
    print(f"Exit {return_code}; console: {log}")
    print(log.read_text(encoding="utf-8")[-3000:])
    return return_code


if __name__ == "__main__":
    sys.exit(main())
