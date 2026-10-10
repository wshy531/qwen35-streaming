"""No-inference audit of paired input/semantic/numeric evidence and launch gates."""
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
    pre, gpu = runs / "s4b_scene_pairs_preflight_20261010_01", runs / "s4b_scene_pairs_gpu_20261010_01"
    proof, report = read(pre / "preflight_summary.json"), read(gpu / "scene_pair_summary.json")
    for directory, stage in ((pre, "preflight"), (gpu, "gpu")):
        assert read(directory / "status.json") == {"status": "completed", "stage": stage}
        assert read(directory / "watchdog_status.json")["exit_code"] == 0 and not read(directory / "watchdog_status.json")["timed_out"]
        assert sha(directory / "script_snapshot.py") == sha(repo / "scripts/scene_pair_control.py")
        for path, digest in read(directory / "provenance.json")["code_hashes"].items():
            assert sha(repo / path) == digest, path
        assert read(directory / "cleanup.json") == {"hooks_restored": True, "closed_sessions": [True] * 6, "all_six_sessions_closed": True}
        result = read(directory / "scene_pair_summary.json")
        assert result["budget"] == proof["budget"] and result["all_eight_numeric_controls_passed"]
        assert len(result["numerical_controls"]) == 8 and all(r["passed"] and r["generated_ids_equal"] for r in result["numerical_controls"])
    assert proof["cpu_controls_passed"] and proof["numeric_controls"] == 8
    assert sha(pre / "scene_pair_summary.json") == proof["cpu_report_sha256"]
    assert sha(pre / "fixture_summary.json") == proof["fixture_summary_sha256"]
    assert report["input_fingerprints"] == proof["input_fingerprints"]
    assert report["schedules"] == proof["expected_schedules"]
    assert read(pre / "fixture_summary.json") == read(gpu / "fixture_summary.json")
    fixture = read(gpu / "fixture_summary.json")
    assert all(fixture["layout_controls"].values())
    assert len(fixture["paired_pixel_controls"]) == 2 and all(x["last_two_rgb_bitwise_equal"] and x["early_four_rgb_different"] for x in fixture["paired_pixel_controls"])
    assert fixture["experimental_ids"] == list(range(6)) and fixture["experimental_pts"] == [float(i) for i in range(6)]
    assert sha(root / "跨场景与早期画面对照方案.md") == proof["plan_sha256"]
    assert sha(root / "scripts/run_scene_pairs.py") == proof["launcher_sha256"]
    fixed = runs / "s4b_scene_fixture_20261010_01"
    assert sha(fixed / "fixture_sources.json") == proof["fixture_source_sha256"]
    assert sha(fixed / "contact_sheet.png") == proof["contact_sheet_sha256"]
    for name, digest in proof["video_hashes"].items():
        assert sha(root / "datasets/natural_commons_v1" / f"{name}.webm") == digest
    events = [json.loads(line) for line in (gpu / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    case_names = ("cat_to_hands", "hands_to_hands", "hands_to_cat", "cat_to_cat")
    for name in case_names:
        subset = [e for e in events if e["case"] == name]
        assert [e["index"] for e in subset] == list(range(len(subset)))
        received = [e for e in subset if e["kind"] == "received"]
        encoded = [e for e in subset if e["kind"] == "encoded"]
        answers = [e for e in subset if e["kind"] == "answer"]
        assert [e["frame_id"] for e in received] == list(range(6))
        assert [e["record"]["frame_ids"] for e in encoded] == [[0, 1], [2, 3], [4, 5]]
        assert not any(e["record"]["synthetic_padding"] for e in encoded)
        assert [e["status"]["session"]["seen_tokens"] for e in encoded] == [59, 115, 171]
        assert [e["answer"]["frame_id"] for e in answers] == [3, 5]
        for e in answers:
            previous = [x for x in subset[:e["index"]] if x["kind"] == "received"][-1]
            assert e["answer"]["frame_id"] == previous["frame_id"] and e["answer"]["timestamp"] == previous["timestamp"]
            assert e["main_unchanged_bitwise"] and e["main_counters_unchanged"]
        assert not any(e["kind"] in ("capacity_reset", "error") for e in subset)
        assert subset[-1]["kind"] == "closed" and subset[-2]["kind"] == "eof"
    assert len(report["records"]) == 24
    assert all(all(identity.values()) for identity in report["exact_input_identities"].values())
    generated = sum(len(r["answer"]["generated_ids"]) for r in report["records"])
    assert generated == report["generated_tokens"] == 58 <= 192
    assert report["actual_forwards"] == {"vision": 14, "language": 20 + 24 + generated}
    assert report["actual_forwards"]["language"] <= 236 and report["wall_seconds_including_load"] <= 600
    assert report["peak_allocated_gib"] <= report["peak_reserved_gib"] < 8
    for r in report["records"]:
        assert r["correct"] == (r["answer"]["answer"].strip().upper() == r["target"])
        assert r["answer"]["generated_ids"][-1] == 248046
        if r["path"] in ("online", "tail_only"):
            assert r["answer"]["main_unchanged_bitwise_and_counters"]
    assert report["semantic_scores"] == {"online_last4": {"correct": 4, "total": 4}, "online_last6": {"correct": 4, "total": 4},
                                         "online_first6": {"correct": 3, "total": 4}, "tail_only": {"correct": 4, "total": 4}}
    wrong = [r for r in report["records"] if not r["correct"]]
    assert len(wrong) == 2 and all(r["case"] == "hands_to_cat" and r["task"] == "first" and r["target"] == "HANDS" and r["answer"]["answer"] == "CAT" for r in wrong)
    assert {r["path"] for r in wrong} == {"online", "shared_input_reference"}
    assert [p["early_visual_use_criterion_passed"] for p in report["paired_early_visual_controls"]] == [True, False]
    assert all(t["both_answers_correct"] and t["answers_changed"] for t in report["transition_controls"])
    for n in report["numerical_controls"]:
        assert n["logits"]["allclose_1e4"] and n["logits"]["fp32_reference_limits_passed"]
        for field, count in (("key_cache", 6), ("value_cache", 6), ("conv_states", 18), ("recurrent_states", 18)):
            assert n["branch_states"][field]["allclose_1e4"] and n["branch_states"][field]["populated_layers"] == count
    for name, digest in read(runs / "s4b_pair_demo_delivery_check_20261010_01/source_tree.json").items():
        if name != "S4_A.md":
            assert sha(repo / name) == digest, name
    original_count = 0
    for p in (root / "reference/qwen35-streaming").rglob("*"):
        if p.is_file():
            assert sha(p) == sha(root / "code/qwen35-streaming" / p.relative_to(root / "reference/qwen35-streaming"))
            original_count += 1
    assert not read(runs / "s4b_online_gpu_20261009_01/online_summary.json")["all_interface_numeric_groups_passed"]
    for item in proof["verified_model_assets"]:
        p = root / "models/Qwen3.5-0.8B" / item["name"]
        assert p.stat().st_size == item["bytes"] and sha(p) == item["sha256"].lower()
    def rejected(name, directory, expected):
        target = args.out / name
        result = subprocess.run([sys.executable, str(root / "scripts/run_scene_pairs.py"), "--stage", "gpu", "--out", str(target),
                                 "--preflight", str(directory)], capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert result.returncode == 2 and expected in result.stderr, result.stderr
        assert not target.exists() and not target.with_name(target.name + ".console.log").exists()
    rejected("wrong_budget", runs / "s4b_pair_demo_preflight_20261010_01", "CPU controls or registered budget mismatch")
    def copy_proof(name):
        directory = args.out / name
        directory.mkdir()
        for file in ("status.json", "preflight_summary.json", "provenance.json", "arguments.json", "scene_pair_summary.json", "fixture_summary.json"):
            (directory / file).write_bytes((pre / file).read_bytes())
        return directory
    altered = copy_proof("changed_source_proof")
    value = read(altered / "provenance.json")
    value["code_hashes"]["scripts\\scene_pair_control.py"] = "0" * 64
    (altered / "provenance.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    rejected("changed_source", altered, "Source changed:")
    altered = copy_proof("changed_input_proof")
    value = read(altered / "preflight_summary.json")
    value["input_fingerprints"]["cat_to_hands"]["prefix"] = "0" * 64
    (altered / "preflight_summary.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    rejected("changed_input", altered, "Frozen input/schedule metadata mismatch")
    git = lambda *a: subprocess.check_output(["git", *a], cwd=repo)
    assert git("branch", "--show-current").decode().strip() == "36tar/s4-mixed-state"
    git("diff", "--check")
    tree = {str(p.relative_to(repo)): sha(p) for p in sorted(repo.rglob("*")) if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts}
    (args.out / "source_tree.json").write_text(json.dumps(tree, indent=2), encoding="utf-8")
    (args.out / "tracked_changes.patch").write_bytes(git("diff", "--binary"))
    audit = {"frozen_sources_derived_ids_pts_and_input_controls_verified": True,
             "eight_strict_language_controls_and_main_isolation_verified": True,
             "current_subject_success_and_early_subject_failure_both_preserved": True,
             "one_of_two_early_visual_pairs_passes_only": True, "budget_forwards_generation_cleanup_verified": True,
             "old_48_sources_unchanged_except_report": True, "original_baseline_files_unchanged": original_count,
             "ten_model_assets_sha_verified": True, "three_negative_launch_gates_verified": True,
             "source_files": len(tree), "no_inference_in_audit": True,
             "branch": "36tar/s4-mixed-state", "base_commit": git("rev-parse", "HEAD").decode().strip(),
             "commit_created": False, "remote_push": False}
    (args.out / "verification.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(audit)


if __name__ == "__main__":
    main()
