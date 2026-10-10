"""CPU-gated four equal-length tail original-model controls; no launch before frozen evidence checks."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import json
from run_precision_layers import read, sha

BUDGET = {"cases": 4, "max_tokens": 256, "max_new_tokens_per_answer": 8, "generated_tokens_max": 32,
          "language_forwards_max": 36, "vision_encodings": 4, "model_loads": 1, "wall_seconds": 600}


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
        expected = [{"id": f"{mode}_{name}", "mode": mode, "input": name}
                    for mode in ("bf16_sdpa", "fp32_ieee_math") for name in ("hands_to_hands", "hands_to_cat")]
        if not proof.get("cpu_controls_passed") or proof.get("budget") != BUDGET or proof.get("cases") != expected:
            parser.error("CPU controls or registered budget mismatch")
        if sha(Path(__file__)) != proof["launcher_sha256"] or sha(root / "原模型六帧等长末尾对照方案.md") != proof["plan_sha256"]:
            parser.error("Launcher or registered plan changed")
        for name, digest in read(args.preflight / "provenance.json")["code_hashes"].items():
            if sha(repo / name) != digest:
                parser.error(f"Source changed: {name}")
        for path, digest in proof["original_reference_hashes"].items():
            if sha(Path(path)) != digest:
                parser.error("Original model or processor source changed")
        if sha(args.preflight / "cpu_summary.json") != proof["cpu_report_sha256"]:
            parser.error("CPU evidence changed")
        if read(args.preflight / "cpu_summary.json")["input_controls"] != proof["input_controls"]:
            parser.error("Frozen input evidence mismatch")
        if sha(root / "runs/s4b_scene_pairs_gpu_20261010_01/scene_pair_summary.json") != proof["historical_report_sha256"]:
            parser.error("Historical scene evidence changed")
        for path, digest in proof["linked_report_hashes"].items():
            if sha(Path(path)) != digest:
                parser.error("Linked original-model evidence changed")
        if read(args.preflight / "cpu_summary.json")["paired_input_control"] != proof["paired_input_control"]:
            parser.error("Frozen pair evidence mismatch")
        fixture = root / "runs/s4b_scene_fixture_20261010_01"
        meta = proof["fixture_metadata"]
        if sha(fixture / "fixture_sources.json") != meta["fixture_source_sha256"] or sha(fixture / "contact_sheet.png") != meta["contact_sheet_sha256"]:
            parser.error("Registered fixture evidence changed")
        checked = read(args.preflight / "arguments.json")
        for key in ("model_dir", "dataset"):
            if Path(checked[key]).resolve() != getattr(args, key).resolve():
                parser.error(f"Asset path changed: {key}")
        for source in meta["sources"]:
            if sha(args.dataset / f"{source['source']}.webm") != source["video_sha256"]:
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
    command = [sys.executable, "-m", "scripts.equal_tail_control", "--stage", args.stage, "--out", str(args.out),
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
