"""Read-only evidence audit; negative launch tests stop before any model load."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from run_precision_layers import read, sha
from run_equal_tail import BUDGET


def main():
    root = Path(__file__).resolve().parents[1]
    repo, runs = root / "code/qwen35-streaming-s4", root / "runs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--gpu", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    pre, gpu = args.preflight, args.gpu
    proof = read(pre / "preflight_summary.json")
    report = read(gpu / "equal_tail_summary.json")
    failed = runs / "s4b_original_first_gpu_20261010_01"
    assert read(failed / "status.json") == {"status": "failed", "stage": "gpu"}
    assert read(failed / "watchdog_status.json")["exit_code"] == 1
    assert "UnicodeDecodeError" in (failed / "traceback.txt").read_text(encoding="utf-8")
    assert not (failed / "original_first_summary.json").exists()
    assert sha(failed / "script_snapshot.py") == sha(runs / "s4b_original_first_preflight_20261010_01/script_snapshot.py")
    failed2 = runs / "s4b_original_first_gpu_20261010_02"
    assert read(failed2 / "status.json") == {"status": "failed", "stage": "gpu"}
    assert read(failed2 / "watchdog_status.json")["exit_code"] == 1
    assert "unsupported ScalarType BFloat16" in (failed2 / "traceback.txt").read_text(encoding="utf-8")
    assert not (failed2 / "original_first_summary.json").exists()
    assert read(failed2 / "cleanup.json")["hooks_restored"]
    assert sha(failed2 / "script_snapshot.py") == sha(runs / "s4b_original_first_preflight_20261010_02/script_snapshot.py")
    for directory, stage in ((pre, "preflight"), (gpu, "gpu")):
        assert read(directory / "status.json") == {"status": "completed", "stage": stage}
        watch = read(directory / "watchdog_status.json")
        assert watch["exit_code"] == 0 and not watch["timed_out"]
        assert sha(directory / "script_snapshot.py") == sha(repo / "scripts/equal_tail_control.py")
        for path, digest in read(directory / "provenance.json")["code_hashes"].items():
            assert sha(repo / path) == digest, path
    assert proof["budget"] == report["budget"] == BUDGET and proof["cpu_controls_passed"]
    assert sha(pre / "cpu_summary.json") == proof["cpu_report_sha256"]
    assert read(pre / "cpu_summary.json")["input_controls"] == proof["input_controls"]
    assert sha(root / "原模型六帧等长末尾对照方案.md") == proof["plan_sha256"]
    assert sha(root / "scripts/run_equal_tail.py") == proof["launcher_sha256"]
    assert sha(runs / "s4b_scene_pairs_gpu_20261010_01/scene_pair_summary.json") == proof["historical_report_sha256"]
    historic = read(runs / "s4b_scene_pairs_gpu_20261010_01/scene_pair_summary.json")
    for path, digest in proof["linked_report_hashes"].items():
        assert sha(Path(path)) == digest
    assert report["paired_input_control"] == proof["paired_input_control"] == read(pre / "cpu_summary.json")["paired_input_control"]
    pair = proof["paired_input_control"]
    assert pair["cases"] == ["hands_to_hands", "hands_to_cat"]
    assert pair["early_four_rgb_equal"] == [True] * 4 and pair["tail_two_rgb_different"] == [True] * 2
    assert all(pair["nonpixel_input_tensors_bitwise_equal"].values())
    assert pair["early_patch_rows_bitwise_equal"] and pair["tail_patch_rows_different"]
    assert pair["early_pixel_rows"] * 3 == pair["pixel_shape"][0] * 2
    assert pair["input_tokens"] == [205, 205]
    assert pair["experimental_frame_ids"] == [list(range(6))] * 2
    assert pair["experimental_pts"] == [[float(i) for i in range(6)]] * 2

    assert report["original_reference_hashes"] == proof["original_reference_hashes"]
    for path, digest in proof["original_reference_hashes"].items():
        assert sha(Path(path)) == digest
    assert Path(report["model_reference"]).resolve() == root / "code/qwen35-streaming/qwen35_local/modeling_qwen3_5.py"
    assert report["fixture_metadata"] == proof["fixture_metadata"]
    meta = proof["fixture_metadata"]
    assert sha(runs / "s4b_scene_fixture_20261010_01/fixture_sources.json") == meta["fixture_source_sha256"]
    assert sha(runs / "s4b_scene_fixture_20261010_01/contact_sheet.png") == meta["contact_sheet_sha256"]
    for item in meta["sources"]:
        assert sha(root / "datasets/natural_commons_v1" / f"{item['source']}.webm") == item["video_sha256"]
    for item in proof["verified_model_assets"]:
        path = root / "models/Qwen3.5-0.8B" / item["name"]
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"].lower()
    def generated(record, count, real_weights=True):
        ids, steps = record["generated_ids"], record["language_steps"]
        assert 1 <= len(ids) <= 8 and record["forward_counts"] == {"vision": 1, "language": len(ids)}
        assert len(record["complete_visual_feature_fingerprints"]) == 1
        assert record["incremental_decode_positions_verified"] and len(steps) == len(ids)
        assert steps[0]["cache_positions"] == list(range(count))
        for i, step in enumerate(steps):
            assert step["text_axis"] == step["cache_positions"]
            assert step["position_shape"] == [4, 1, count if i == 0 else 1]
            if i:
                assert step["cache_positions"] == [count + i - 1]
        if real_weights:
            assert record["registered_eos_ids"] == [248044, 248046]
        else:
            assert 248046 in record["registered_eos_ids"]
        assert record["stopped_on_registered_eos"] == (ids[-1] in record["registered_eos_ids"])
    for control in proof["input_controls"]:
        assert all(control["all_input_tensors_bitwise_equal"].values()) and control["hooks_restored"]
        assert control["tiny_prefill_logits"]["bitwise_equal"]
        assert all(s["all_tensors_bitwise_equal"] for s in control["tiny_prefill_states"].values())
        assert control["input_fingerprint"] == historic["input_fingerprints"][control["id"]]["first"]
        assert control["input_fingerprint"] == pair["input_fingerprints"][control["id"]]
        assert control["target"] == "HANDS"
        generated(control["tiny_generate"], control["input_tokens"], real_weights=False)
    assert [r["id"] for r in report["cases"]] == [r["id"] for r in proof["cases"]]
    for record in report["cases"]:
        control = next(c for c in proof["input_controls"] if c["id"] == record["input"])
        assert record["input_fingerprint"] == control["input_fingerprint"]
        assert record["input_tokens"] + 8 <= 256 and record["grid"] == [[3, 12, 16]]
        assert record["target"] == control["target"] and record["question"] == control["question"]
        assert record["correct"] == (record["answer"].strip().upper() == record["target"])
        assert record["matches_previous_online_ids"] == (record["generated_ids"] == control["previous_online_ids"])
        generated(record, record["input_tokens"])
    total = sum(len(r["generated_ids"]) for r in report["cases"])
    assert total == report["generated_tokens"] <= 32
    assert report["actual_forwards"] == {"vision": 4, "language": total} and total <= 36
    assert report["all_registered_cases_executed"] and report["all_input_fingerprints_verified"]
    assert report["wall_seconds_including_load"] <= 600
    assert report["peak_allocated_gib"] <= report["peak_reserved_gib"] < 8
    assert "never reset at conversion" in report["peak_scope"]
    assert read(gpu / "cleanup.json") == {"hooks_restored": True, "source_readers_closed_during_prepare": True}
    for name, digest in read(runs / "s4b_prefix_first_delivery_check_20261010_01/source_tree.json").items():
        if name != "S4_A.md":
            assert sha(repo / name) == digest, name
    original_count = 0
    for path in (root / "reference/qwen35-streaming").rglob("*"):
        if path.is_file():
            assert sha(path) == sha(root / "code/qwen35-streaming" / path.relative_to(root / "reference/qwen35-streaming"))
            original_count += 1
    wrong = next(r for r in historic["records"] if r["case"] == "hands_to_cat" and r["path"] == "online" and r["task"] == "first")
    assert wrong["answer"]["answer"] == "CAT" and not wrong["correct"]
    def rejected(name, directory, message):
        target = args.out / name
        result = subprocess.run([sys.executable, str(root / "scripts/run_equal_tail.py"), "--stage", "gpu", "--out", str(target),
                                 "--preflight", str(directory)], capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert result.returncode == 2 and message in result.stderr, result.stderr
        assert not target.exists() and not target.with_name(target.name + ".console.log").exists()
    rejected("wrong_budget", runs / "s4b_scene_pairs_preflight_20261010_01", "CPU controls or registered budget mismatch")
    def changed(name, field, value, error):
        directory = args.out / (name + "_proof")
        directory.mkdir()
        for file in ("status.json", "preflight_summary.json", "provenance.json", "arguments.json", "cpu_summary.json"):
            (directory / file).write_bytes((pre / file).read_bytes())
        proof_copy = read(directory / "preflight_summary.json")
        if field == "original_reference_hashes":
            proof_copy[field][next(iter(proof_copy[field]))] = value
        elif field == "input_controls":
            proof_copy[field][0]["input_fingerprint"] = value
        else:
            proof_copy[field] = value
        (directory / "preflight_summary.json").write_text(json.dumps(proof_copy, indent=2), encoding="utf-8")
        rejected(name, directory, error)
    changed("changed_original_source", "original_reference_hashes", "0" * 64, "Original model or processor source changed")
    changed("changed_input", "input_controls", "0" * 64, "Frozen input evidence mismatch")
    changed("changed_historical_evidence", "historical_report_sha256", "0" * 64, "Historical scene evidence changed")
    changed("changed_pair", "paired_input_control", {}, "Frozen pair evidence mismatch")
    git = lambda *a: subprocess.check_output(["git", *a], cwd=repo)
    assert git("branch", "--show-current").decode().strip() == "36tar/s4-mixed-state"
    git("diff", "--check")
    tree = {str(p.relative_to(repo)): sha(p) for p in sorted(repo.rglob("*")) if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts}
    (args.out / "source_tree.json").write_text(json.dumps(tree, indent=2), encoding="utf-8")
    (args.out / "tracked_changes.patch").write_bytes(git("diff", "--binary"))
    audit = {"four_equal_length_original_controls_verified": True, "same_early_pixels_and_layout_different_tail_verified": True, "all_input_and_source_hashes_verified": True,
             "budget_decode_positions_eos_and_cleanup_verified": True, "old_first_failure_preserved": True,
             "old_51_sources_unchanged_except_report": True, "original_baseline_files_unchanged": original_count,
             "five_negative_launch_gates_verified": True, "utf8_and_bf16_recording_failures_preserved": True,
             "source_files": len(tree), "no_inference_in_audit": True,
             "answers": [{"id": r["id"], "answer": r["answer"], "target": r["target"], "correct": r["correct"]} for r in report["cases"]],
             "branch": "36tar/s4-mixed-state", "base_commit": git("rev-parse", "HEAD").decode().strip(),
             "commit_created": False, "remote_push": False}
    (args.out / "verification.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(audit)


if __name__ == "__main__":
    main()
