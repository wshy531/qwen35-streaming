"""Read-only causal schedule, resource/budget, old-source and negative-gate audit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from run_precision_layers import read, sha


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    repo, runs = root / "code/qwen35-streaming-s4", root / "runs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    pre = runs / "s4b_pair_demo_preflight_20261010_01"
    gpu = runs / "s4b_pair_demo_gpu_20261010_01"
    proof, report = read(pre / "preflight_summary.json"), read(gpu / "pair_demo_summary.json")
    for directory, stage in ((pre, "preflight"), (gpu, "gpu")):
        assert read(directory / "status.json") == {"status": "completed", "stage": stage}
        watchdog = read(directory / "watchdog_status.json")
        assert watchdog["exit_code"] == 0 and not watchdog["timed_out"]
        assert sha(directory / "script_snapshot.py") == sha(repo / "scripts/pair_demo.py")
        for path, digest in read(directory / "provenance.json")["code_hashes"].items():
            assert sha(repo / path) == digest, path
        assert all(read(directory / "cleanup.json")[k] for k in ("reader_closed", "demo_closed", "pending_cleared", "hooks_restored"))
        assert read(directory / "cleanup.json")["main_cache_bytes_after_close"] == 0
        assert read(directory / "cleanup.json")["main_tokens_after_close"] == 0
    assert proof["cpu_controls_passed"] and proof["test_count"] == 9
    assert read(pre / "cpu_summary.json") == {"tests_run": 9, "failures": 0, "errors": 0, "passed": True,
                                             "scope": "tiny FP32 weights; policy, not semantics"}
    assert sha(pre / "pair_demo_summary.json") == proof["schedule_sha256"]
    assert report["schedule"] == proof["expected_schedule"] == read(pre / "pair_demo_summary.json")["schedule"]
    assert report["policy_checks_passed"] and report["budget"] == proof["budget"]
    assert sha(root / "同步成对帧演示验收方案.md") == proof["plan_sha256"]
    assert sha(root / "scripts/run_pair_demo.py") == proof["launcher_sha256"]
    events = [json.loads(line) for line in (gpu / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [e["index"] for e in events] == list(range(len(events)))
    received = [e for e in events if e["kind"] == "received"]
    encoded = [e for e in events if e["kind"] == "encoded"]
    answers = [e for e in events if e["kind"] == "answer"]
    resets = [e for e in events if e["kind"] == "capacity_reset"]
    assert len(received) == 23 and len(encoded) == 12 and len(answers) == 7 and len(resets) == 3
    assert [e["frame_id"] for e in received] == list(range(0, 661, 30))
    real_encoded_ids = sum([e["record"]["frame_ids"][:1 if e["record"]["synthetic_padding"] else 2] for e in encoded], [])
    assert real_encoded_ids == [e["frame_id"] for e in received]
    assert sum(e["record"]["synthetic_padding"] for e in encoded) == 1
    assert encoded[-1]["record"]["frame_ids"] == [660, 660]
    assert [e["answer"]["frame_id"] for e in answers] == [90, 150, 270, 330, 450, 510, 630]
    assert [e["epoch_real_frames"] for e in answers] == [4, 6, 4, 6, 4, 6, 4]
    assert [e["retained_frame_ids"] for e in resets] == [[180, 210], [360, 390], [540, 570]]
    for event in answers:
        prior = [e for e in events[:event["index"]] if e["kind"] == "received"][-1]
        assert event["answer"]["frame_id"] == prior["frame_id"] == event["status"]["session"]["last_encoded_id"]
        assert event["answer"]["timestamp"] == prior["timestamp"]
        assert event["main_unchanged_bitwise"] and event["main_counters_unchanged"]
        assert 1 <= len(event["answer"]["generated_ids"]) <= 8
    for event in resets:
        assert event["status"]["session"]["seen_tokens"] == event["main_cache_bytes"] == 0
        assert event["discarded_epoch"]["seen_tokens"] + event["new_block_tokens"] > 176
        next_pair = next(e for e in events[event["index"] + 1:] if e["kind"] == "encoded")
        assert next_pair["record"]["frame_ids"] == event["retained_frame_ids"]
        assert next_pair["status"]["session"]["encoded_real_frames"] == 2
    final = report["final_before_close"]
    assert final["received_real_frames"] == final["encoded_real_frames"] == 23
    assert final["session"]["encoded_real_frames"] == 5 and final["session"]["seen_tokens"] == 174
    assert final["finished"] and not final["invalid"] and final["pending_rgb_bytes"] == 0
    assert report["peak_main_tokens"] == 174 <= 176
    assert report["actual_forwards"]["vision"] == 12
    generated = sum(len(e["answer"]["generated_ids"]) for e in answers)
    assert generated == final["generated_tokens"] <= 64
    assert report["actual_forwards"]["language"] == 12 + 4 + 7 + generated <= 100
    assert report["wall_seconds_including_load"] < 600
    assert report["peak_allocated_gib"] <= report["peak_reserved_gib"] <= 8
    assert report["first_answer_matches_historical_hands"] == (answers[0]["answer"]["generated_ids"] == [72141, 50, 248046])
    assert events[-1]["kind"] == "closed" and events[-2]["kind"] == "eof"
    for name, digest in read(runs / "s4b_four_frame_delivery_check_20261010_02/source_tree.json").items():
        if name != "S4_A.md":
            assert sha(repo / name) == digest, name
    unchanged = 0
    for path in (root / "reference/qwen35-streaming").rglob("*"):
        if path.is_file():
            assert sha(path) == sha(root / "code/qwen35-streaming" / path.relative_to(root / "reference/qwen35-streaming"))
            unchanged += 1
    assert not read(runs / "s4b_online_gpu_20261009_01/online_summary.json")["all_interface_numeric_groups_passed"]
    manifest = read(root / "models/Qwen3.5-0.8B/DOWNLOAD_MANIFEST.json")
    assert manifest["files"] == proof["verified_model_assets"]
    for item in manifest["files"]:
        path = root / "models/Qwen3.5-0.8B" / item["name"]
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"].lower()
    def rejected(name, preflight, expected):
        target = args.out / name
        result = subprocess.run([sys.executable, str(root / "scripts/run_pair_demo.py"), "--stage", "gpu",
            "--out", str(target), "--preflight", str(preflight)], capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert result.returncode == 2 and expected in result.stderr, result.stderr
        assert not target.exists() and not target.with_name(target.name + ".console.log").exists()
    rejected("wrong_budget", runs / "s4b_four_frame_preflight_20261010_02", "CPU controls or registered budget mismatch")
    def modified_proof(name):
        folder = args.out / name
        folder.mkdir()
        for filename in ("status.json", "preflight_summary.json", "provenance.json", "arguments.json", "pair_demo_summary.json"):
            (folder / filename).write_bytes((pre / filename).read_bytes())
        return folder
    changed = modified_proof("changed_source_proof")
    provenance = read(changed / "provenance.json")
    provenance["code_hashes"]["streaming\\demo.py"] = "0" * 64
    (changed / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    rejected("changed_source", changed, "Source changed:")
    changed = modified_proof("changed_schedule_proof")
    altered = read(changed / "preflight_summary.json")
    altered["expected_schedule"][0]["timestamp"] += 1
    (changed / "preflight_summary.json").write_text(json.dumps(altered, indent=2), encoding="utf-8")
    rejected("changed_schedule", changed, "Frozen schedule metadata mismatch")
    git = lambda *a: subprocess.check_output(["git", *a], cwd=repo)
    assert git("branch", "--show-current").decode().strip() == "36tar/s4-mixed-state"
    git("diff", "--check")
    tree = {str(p.relative_to(repo)): sha(p) for p in sorted(repo.rglob("*"))
            if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts}
    (args.out / "source_tree.json").write_text(json.dumps(tree, indent=2), encoding="utf-8")
    (args.out / "tracked_changes.patch").write_bytes(git("diff", "--binary"))
    audit = {"cpu_nine_contracts_verified": True, "frozen_schedule_and_causal_cutoffs_verified": True,
             "all_23_real_frames_encoded_once_no_loss_across_three_resets": True,
             "one_explicit_padding_and_no_extra_eof_qa_verified": True, "all_seven_qa_main_states_bitwise_unchanged": True,
             "forward_token_time_resources_and_cleanup_verified": True, "old_45_source_files_unchanged_except_report": True,
             "original_baseline_files_unchanged": unchanged, "all_model_assets_sha_verified": len(manifest["files"]),
             "three_negative_launch_gates_verified": True, "no_inference_in_audit": True,
             "source_files": len(tree), "base_commit": git("rev-parse", "HEAD").decode().strip(),
             "branch": "36tar/s4-mixed-state", "commit_created": False, "remote_push": False}
    (args.out / "verification.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(audit)


if __name__ == "__main__":
    main()
