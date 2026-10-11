"""CPU-gated four-call vision trace; frozen sources/assets and a 600-second watchdog."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from run_precision_layers import read


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


BUDGET = {"vision_encodings": 4, "language_forwards": 0, "generated_tokens": 0,
          "model_loads": 1, "wall_seconds": 600, "trace_buffer_bytes": 128 * 2**20}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    repo = root / "code/qwen35-streaming-s4"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "gpu"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=root / "models/Qwen3.5-0.8B")
    parser.add_argument("--video", type=Path, default=root / "datasets/natural_commons_v1/clean_hands.webm")
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error("New run ID required")
    if args.stage == "gpu":
        if args.preflight is None or read(args.preflight / "status.json") != {"status": "completed", "stage": "preflight"}:
            parser.error("Completed CPU preflight required")
        proof = read(args.preflight / "preflight_summary.json")
        if not proof.get("cpu_controls_passed") or proof["budget"] != BUDGET:
            parser.error("CPU controls or budget mismatch")
        for path, digest in ((Path(__file__), proof["launcher_sha256"]),
                             (root / "视觉首次差异定位方案.md", proof["plan_sha256"]),
                             (args.preflight / "cpu_summary.json", proof["cpu_report_sha256"]),
                             (root / "runs/s4b_shared_gpu_20261009_01/shared_features_summary.json", proof["historical_report_sha256"])):
            if sha(path) != digest:
                parser.error("Frozen plan, launcher or evidence changed")
        for rel, digest in read(args.preflight / "provenance.json")["code_hashes"].items():
            if sha(repo / rel) != digest:
                parser.error("Source changed")
        checked = read(args.preflight / "arguments.json")
        for key in ("model_dir", "video"):
            if Path(checked[key]).resolve() != getattr(args, key).resolve():
                parser.error("Asset path changed")
        if sha(args.video) != proof["video_sha256"] or sha(args.model_dir / "DOWNLOAD_MANIFEST.json") != proof["model_manifest_sha256"]:
            parser.error("Asset fingerprint changed")
        for item in proof["verified_model_assets"]:
            path = args.model_dir / item["name"]
            if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"].lower():
                parser.error("Model asset changed")
        if read(args.preflight / "cpu_summary.json")["input_identity"] != proof["input_identity"]:
            parser.error("Frozen input proof changed")
        if subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip() != "36tar/s4-mixed-state":
            parser.error("Expected personal branch")
    log = args.out.with_name(args.out.name + ".console.log")
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    command = [sys.executable, "-m", "scripts.vision_trace", "--stage", args.stage, "--out", str(args.out),
               "--model-dir", str(args.model_dir.resolve()), "--video", str(args.video.resolve())]
    if args.preflight:
        command += ["--preflight", str(args.preflight.resolve())]
    with log.open("x", encoding="utf-8") as stream:
        try:
            result = subprocess.run(command, cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT, timeout=600)
            code, timed_out = result.returncode, False
        except subprocess.TimeoutExpired:
            code, timed_out = 124, True
    if args.out.exists():
        (args.out / "watchdog_status.json").write_text(json.dumps({"wall_time_limit_seconds": 600,
            "timed_out": timed_out, "exit_code": code}, indent=2), encoding="utf-8")
    print(log.read_text(encoding="utf-8")[-6000:])
    return code


if __name__ == "__main__":
    sys.exit(main())
