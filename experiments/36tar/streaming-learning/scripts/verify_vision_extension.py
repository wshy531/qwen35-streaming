"""Audit finite extension evidence, frozen versions, failures and pre-GPU rejection gates."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(2**20), b""):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runs, repo = root / "runs", root / "code/qwen35-streaming-s4"
    args.out = args.out.resolve()
    assert args.out.is_relative_to(runs.resolve())
    args.out.mkdir(exist_ok=False)
    (args.out / "verification_script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    specs = [("vision_expand", "vision_expand.py", "run_vision_extension.py", ["01"], ["01"]),
             ("stage_timing", "stage_timing.py", "run_stage_timing.py", ["01", "02"], ["01", "02"]),
             ("chunk_probe", "chunk_size_probe.py", "run_chunk_probe.py", ["01"], ["01", "02"])]
    frozen, failures, changed_self = {}, [], []
    for kind, module, launcher, cpu_versions, gpu_versions in specs:
        for stage, versions in (("preflight", cpu_versions), ("gpu", gpu_versions)):
            for version in versions:
                folder = runs / f"s4b_{kind}_{stage}_20261010_{version}"
                failed = stage == "gpu" and version == "01" and kind in ("stage_timing", "chunk_probe")
                assert read(folder / "status.json") == {"status": "failed" if failed else "completed", "stage": stage}
                watchdog = read(folder / "watchdog_status.json")
                assert watchdog["exit_code"] == (1 if failed else 0) and not watchdog["timed_out"]
                proof_dir = folder if stage == "preflight" else Path(read(folder / "arguments.json")["preflight"])
                proof = read(proof_dir / "preflight_summary.json")
                assert sha(proof_dir / "cpu_summary.json") == proof["cpu_report_sha256"]
                for item, field in (("launcher_snapshot.py", "launcher_sha256"), ("plan_snapshot.md", "plan_sha256")):
                    assert sha(folder / item) == proof[field], (folder, field)
                source = read(folder / "provenance.json")["code_hashes"]
                assert sha(folder / "script_snapshot.py") == source[str(Path("scripts") / module)]
                for name, digest in source.items():
                    if sha(repo / name) != digest:
                        assert name == str(Path("scripts") / "stage_timing.py") and kind == "stage_timing" and version == "01"
                        changed_self.append(folder.name)
                if sha(root / "scripts" / launcher) != proof["launcher_sha256"]:
                    assert kind == "stage_timing" and version == "01"
                assert read(proof_dir / "cpu_summary.json")["input_identity"] == proof["input_identity"]
                frozen[folder.name] = {"module_snapshot_sha256": sha(folder / "script_snapshot.py"),
                    "launcher_snapshot_sha256": sha(folder / "launcher_snapshot.py"),
                    "plan_snapshot_sha256": sha(folder / "plan_snapshot.md"), "status": "failed" if failed else "completed"}
                if failed:
                    failures.append(folder.name)
    assert set(changed_self) == {"s4b_stage_timing_preflight_20261010_01", "s4b_stage_timing_gpu_20261010_01"}
    expansion_dir = runs / "s4b_vision_expand_gpu_20261010_01"
    expansion = read(expansion_dir / "expansion_summary.json")
    assert expansion["all_cases_bitwise_equal"] and len(expansion["cases"]) == 6
    assert expansion["actual_forwards"] == {"vision": 31, "language": 0}
    assert expansion["candidate_projection_operations"] == 1900 and expansion["methods_hooks_restored"]
    assert all(c["candidate_vs_pairs"]["bitwise_equal"] for c in expansion["cases"])
    assert all(not c["candidate_vs_original_full"]["bitwise_equal"] for c in expansion["cases"])
    assert expansion["comparison_buffer_peak_bound_bytes"] <= 128*2**20
    assert expansion["input_identity"] == read(runs / "s4b_vision_expand_preflight_20261010_01/preflight_summary.json")["input_identity"]
    timing_dir = runs / "s4b_stage_timing_gpu_20261010_02"
    timing = read(timing_dir / "timing_summary.json")
    assert timing["actual_forwards"] == {"vision": 25, "language": 36}
    assert timing["candidate_projection_operations"] == 1300 and len(timing["samples"]) == 5
    assert [r["iteration"] for r in timing["samples"]] == [1, 2, 3, 4, 5]
    numeric = timing["strict_shared_feature_correctness"]
    assert numeric["last_pair_logits"]["allclose_1e4"] and numeric["last_pair_logits"]["fp32_reference_limits_passed"]
    assert all(v["allclose_1e4"] for v in numeric["all_final_cache_fields"].values())
    assert timing["main_unchanged_all_six_branches_bitwise"] and timing["reset_empty_all_six"] and timing["fork_storage_independent"]
    assert timing["generated_tokens"] == 0 and timing["retained_feature_bytes"] <= 8*2**20
    for label, summary in timing["stage_summary_ms"].items():
        for unit, values in summary.items():
            assert values["min"] <= values["median"] <= values["max"]
            assert all(r[label][unit] > 0 for r in timing["samples"])
    failed_timing = runs / "s4b_stage_timing_gpu_20261010_01"
    assert "Temporal-unit projection budget exceeded" in (failed_timing / "traceback.txt").read_text()
    partial = read(failed_timing / "timing_partial.json")
    assert len(partial["samples"]) == 4 and partial["actual_forwards"] == {"vision": 21, "language": 30}
    assert not (failed_timing / "timing_summary.json").exists()
    blocked = runs / "s4b_chunk_probe_gpu_20261010_01"
    guard = read(blocked / "resource_precheck.json")
    assert guard["ram_available_bytes"] < 3*2**30 and guard["gpu_free_bytes"] >= 5.5*2**30
    assert not (blocked / "chunk_probe_partial.json").exists() and not (blocked / "chunk_probe_summary.json").exists()
    assert "Loading weights" not in (runs / (blocked.name + ".console.log")).read_text()
    probe_dir = runs / "s4b_chunk_probe_gpu_20261010_02"
    probe = read(probe_dir / "chunk_probe_summary.json")
    assert probe["actual_forwards"] == {"vision": 1, "language": 36}
    assert probe["chunk_rule_calls"] == 612 and probe["candidate_projection_operations"] == 100
    assert probe["all_strict_controls_passed"] and probe["hooks_methods_callables_restored"]
    assert [m["chunk_size"] for m in probe["modes"]] == [64, 32]
    for m in probe["modes"]:
        assert len(m["samples"]) == 3 and all(v["passed"] for v in m["numeric"].values())
        for control in m["numeric"].values():
            assert control["logits"]["allclose_1e4"] and control["logits"]["fp32_reference_limits_passed"]
            assert all(v["allclose_1e4"] for v in control.get("final_states", {}).values())
    assert probe["two_question_branches_main_unchanged_bitwise"]
    assert probe["feature_sha256"] == timing["shared_feature_sha256"] == "04386b4267395e5ce47ff540bf87a6e7213cb3e63817597d03d68fe2334b3356"
    assert probe["input_identity"] == timing["input_identity"]
    for kind, folder, summary in (("expand", expansion_dir, expansion), ("timing", timing_dir, timing), ("probe", probe_dir, probe)):
        proof = read(Path(read(folder / "arguments.json")["preflight"]) / "preflight_summary.json")
        assert summary["budget"] == proof["budget"] and summary["wall_seconds_including_load"] < 600
        assert (folder / "cleanup.json").exists()
    old = read(runs / "s4b_vision_exploration_delivery_check_20261010_02/source_tree.json")
    assert len(old) == 57 and all(sha(repo / name) == digest for name, digest in old.items())
    originals = list((root / "reference/qwen35-streaming").rglob("*"))
    originals = [p for p in originals if p.is_file()]
    assert len(originals) == 15
    assert all(sha(p) == sha(root / "code/qwen35-streaming" / p.relative_to(root / "reference/qwen35-streaming")) for p in originals)
    manifest = read(root / "models/Qwen3.5-0.8B/DOWNLOAD_MANIFEST.json")
    for item in manifest["files"]:
        p = root / "models/Qwen3.5-0.8B" / item["name"]
        assert p.stat().st_size == item["bytes"] and sha(p) == item["sha256"].lower()
    proof = read(runs / "s4b_chunk_probe_preflight_20261010_01/preflight_summary.json")
    assert all(sha(Path(p)) == digest for p, digest in proof["assets"].items())
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=repo, text=True).splitlines()
    tree = {name: sha(repo / name) for name in sorted(names)}
    assert len(tree) == 60
    write(args.out / "source_tree.json", tree)
    write(args.out / "source_snapshots.json", frozen)
    (args.out / "tracked_changes.patch").write_bytes(subprocess.check_output(["git", "diff", "--no-ext-diff", "HEAD"], cwd=repo))
    assert subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip() == "36tar/s4-mixed-state"
    gates = []
    good = runs / "s4b_chunk_probe_preflight_20261010_01"
    for label in ("budget", "source", "cpu", "input", "plan", "launcher", "asset", "model_manifest"):
        with tempfile.TemporaryDirectory(prefix="reject_", dir=args.out) as temporary:
            folder = Path(temporary).resolve()
            assert folder.is_relative_to(args.out) and args.out.is_relative_to(runs.resolve())
            for name in ("status.json", "preflight_summary.json", "provenance.json", "arguments.json", "cpu_summary.json"):
                shutil.copyfile(good / name, folder / name)
            proof = read(folder / "preflight_summary.json")
            expected_error = "Frozen plan, launcher or evidence changed"
            if label == "budget":
                proof["budget"]["chunk_rule_calls_max"] += 1
                expected_error = "CPU controls or budget mismatch"
            elif label == "source":
                source = read(folder / "provenance.json")
                source["code_hashes"]["scripts/vision_fixed_units.py"] = "0"*64
                write(folder / "provenance.json", source)
                expected_error = "Source changed"
            elif label == "cpu":
                (folder / "cpu_summary.json").write_text((folder / "cpu_summary.json").read_text() + "\n")
            elif label == "input":
                cpu = read(folder / "cpu_summary.json")
                cpu["input_identity"]["full_fingerprint"] = "0"*64
                write(folder / "cpu_summary.json", cpu)
                proof["cpu_report_sha256"] = sha(folder / "cpu_summary.json")
                expected_error = "Frozen input changed"
            elif label in ("plan", "launcher"):
                proof[label + "_sha256"] = "0"*64
            elif label == "asset":
                proof["assets"][next(iter(proof["assets"]))] = "0"*64
                expected_error = "Asset changed"
            else:
                proof["model_manifest_sha256"] = "0"*64
                expected_error = "Model manifest changed"
            write(folder / "preflight_summary.json", proof)
            target = args.out / ("should_not_run_" + label)
            result = subprocess.run([sys.executable, str(root / "scripts/run_chunk_probe.py"), "--experiment", "probe",
                "--stage", "gpu", "--preflight", str(folder), "--out", str(target)], capture_output=True,
                text=True, encoding="utf-8", timeout=30)
            assert result.returncode == 2 and expected_error in result.stderr, result.stderr
            assert not target.exists() and not target.with_name(target.name + ".console.log").exists()
            gates.append({"case": label, "rejected_before_log_or_gpu": True, "exit_code": 2})
    cleanup = []
    candidates = [(expansion_dir, "expansion_partial.json"), (timing_dir, "timing_partial.json"),
                  (probe_dir, "chunk_probe_partial.json")]
    for folder, name in candidates:
        path = (folder / name).resolve()
        if not path.exists():
            continue
        assert path.is_relative_to(folder.resolve()) and folder.resolve().is_relative_to(runs.resolve())
        partial = read(path)
        if folder == expansion_dir:
            assert partial["cases"] == expansion["cases"] and partial["actual_forwards"] == expansion["actual_forwards"]
        elif folder == timing_dir:
            final_samples = [{k: v for k, v in r.items() if k != "language_increment_total"} for r in timing["samples"]]
            assert partial["samples"] == final_samples and partial["warmup"] == timing["warmup_excluded"]
        else:
            assert partial["completed_modes"] == probe["modes"][:1]
            assert partial["current_size"] == 32 and partial["samples"] == probe["modes"][1]["samples"]
            assert partial["current_numeric"] == probe["modes"][1]["numeric"]
        cleanup.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path), "final_summary_covers_content": True})
        path.unlink()
    for module in ("vision_expand", "stage_timing", "chunk_size_probe"):
        cache_dir = (repo / "scripts/__pycache__").resolve()
        for path in cache_dir.glob(module + ".*.pyc"):
            path = path.resolve()
            assert path.is_relative_to(cache_dir) and cache_dir.is_relative_to(repo.resolve())
            cleanup.append({"path": str(path), "bytes": path.stat().st_size, "regenerable_bytecode": True})
            path.unlink()
    write(args.out / "cleanup_verification.json", {"removed": cleanup, "removed_bytes": sum(r["bytes"] for r in cleanup),
        "failed_timing_partial_preserved": (failed_timing / "timing_partial.json").exists(),
        "all_failures_and_snapshots_preserved": True})
    verification = {"passed": True, "cpu_runs": 4, "gpu_records": 5, "successful_gpu_runs": 3,
        "failed_budget_records": 1, "rejected_before_model_load": 1, "frozen_runs_checked": len(frozen),
        "old_source57_unchanged": True, "source_files": len(tree), "reference_files15_unchanged": True,
        "verified_model_assets": len(manifest["files"]), "verified_video_assets": len(proof["assets"]),
        "negative_gates": gates, "historical_versions_preserved": changed_self, "failures_preserved": failures,
        "candidate_original_full_bias_preserved": True, "timing_does_not_claim_real_time": True,
        "chunk32_three_sequential_samples_only": True}
    write(args.out / "verification.json", verification)
    print(verification)


if __name__ == "__main__":
    main()
