"""CPU-gated fixed visual counterfactuals, with a 600s child watchdog."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from run_precision_layers import read, sha

BUDGET = {"sequences": 6, "input_frames": 28, "vision_encodings": 14, "answers": 24,
          "online_answers": 16, "reference_answers": 8, "max_new_tokens_per_answer": 8,
          "generated_tokens_max": 192, "language_forwards_max": 236,
          "main_tokens_max": 176, "max_tokens": 256, "model_loads": 1, "wall_seconds": 600}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    repo = root / "code/qwen35-streaming-s4"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "gpu"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=root / "models/Qwen3.5-0.8B")
    parser.add_argument("--dataset", type=Path, default=root / "datasets/natural_commons_v1")
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if args.out.exists():
        parser.error("New run ID required")
    if args.stage == "gpu":
        if args.preflight is None or read(args.preflight / "status.json") != {"status": "completed", "stage": "preflight"}:
            parser.error("Completed CPU/input preflight required")
        proof = read(args.preflight / "preflight_summary.json")
        if not proof.get("cpu_controls_passed") or proof.get("budget") != BUDGET or proof.get("numeric_controls") != 8:
            parser.error("CPU controls or registered budget mismatch")
        if sha(Path(__file__)) != proof["launcher_sha256"] or sha(root / "跨场景与早期画面对照方案.md") != proof["plan_sha256"]:
            parser.error("Launcher or registered plan changed")
        for name, digest in read(args.preflight / "provenance.json")["code_hashes"].items():
            if sha(repo / name) != digest:
                parser.error(f"Source changed: {name}")
        if sha(args.preflight / "scene_pair_summary.json") != proof["cpu_report_sha256"] or sha(args.preflight / "fixture_summary.json") != proof["fixture_summary_sha256"]:
            parser.error("CPU schedule/input evidence changed")
        cpu = read(args.preflight / "scene_pair_summary.json")
        if cpu["schedules"] != proof["expected_schedules"] or cpu["input_fingerprints"] != proof["input_fingerprints"]:
            parser.error("Frozen input/schedule metadata mismatch")
        fixture = root / "runs/s4b_scene_fixture_20261010_01"
        if sha(fixture / "fixture_sources.json") != proof["fixture_source_sha256"] or sha(fixture / "contact_sheet.png") != proof["contact_sheet_sha256"]:
            parser.error("Registered fixture evidence changed")
        checked = read(args.preflight / "arguments.json")
        for key in ("model_dir", "dataset"):
            if Path(checked[key]).resolve() != getattr(args, key).resolve():
                parser.error(f"Asset path changed: {key}")
        for name, digest in proof["video_hashes"].items():
            if sha(args.dataset / f"{name}.webm") != digest:
                parser.error("Source video changed")
        if sha(args.model_dir / "DOWNLOAD_MANIFEST.json") != proof["model_manifest_sha256"]:
            parser.error("Model manifest changed")
        for item in proof["verified_model_assets"]:
            path = args.model_dir / item["name"]
            if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"].lower():
                parser.error("Model asset changed")
        if subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip() != "36tar/s4-mixed-state":
            parser.error("Expected registered personal branch")
    log = args.out.with_name(args.out.name + ".console.log")
    env = os.environ.copy()
    env.update(PYTHONIOENCODING="utf-8", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    command = [sys.executable, "-m", "scripts.scene_pair_control", "--stage", args.stage, "--out", str(args.out),
               "--model-dir", str(args.model_dir.resolve()), "--dataset", str(args.dataset.resolve())]
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
            "timed_out": timed_out, "exit_code": code, "console_log": str(log)}, indent=2), encoding="utf-8")
    print(log.read_text(encoding="utf-8")[-4500:])
    return code


if __name__ == "__main__":
    sys.exit(main())
