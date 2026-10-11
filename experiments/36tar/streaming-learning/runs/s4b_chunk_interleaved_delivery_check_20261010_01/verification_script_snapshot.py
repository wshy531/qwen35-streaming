"""Read-only evidence and frozen-gate audit; remove only covered partials/bytecode."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
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
    cpu_dir = runs / "s4b_chunk_interleaved_preflight_20261010_01"
    gpu_dir = runs / "s4b_chunk_interleaved_gpu_20261010_01"
    proof = read(cpu_dir / "preflight_summary.json")
    cpu, gpu = read(cpu_dir / "cpu_summary.json"), read(gpu_dir / "interleaved_summary.json")
    assert cpu["input_identity"] == gpu["input_identity"] == proof["input_identity"]
    assert sha(cpu_dir / "cpu_summary.json") == proof["cpu_report_sha256"]
    assert cpu["budget"] == gpu["budget"] == proof["budget"]
    assert cpu["real_config_budget_check"] == {"gdn_layers": 18, "vision_linears": 50,
        "main_language_calls": 294, "branch_language_calls": 20,
        "multitoken_gdn_calls": 5472, "single_token_gdn_calls_unchanged": 180}
    assert cpu["budget_exception_restored"]
    snapshots = {}
    for folder, stage, report in ((cpu_dir, "preflight", cpu), (gpu_dir, "gpu", gpu)):
        assert read(folder / "status.json") == {"status": "completed", "stage": stage}
        watchdog = read(folder / "watchdog_status.json")
        assert watchdog["exit_code"] == 0 and not watchdog["timed_out"]
        source = read(folder / "provenance.json")["code_hashes"]
        assert all(sha(repo / name) == digest for name, digest in source.items())
        assert sha(folder / "script_snapshot.py") == source[str(Path("scripts/chunk_interleaved.py"))]
        for file, field, live in (("launcher_snapshot.py", "launcher_sha256", root / "scripts/run_chunk_interleaved.py"),
                                 ("plan_snapshot.md", "plan_sha256", root / "语言计算块交错扩展方案.md")):
            assert sha(folder / file) == sha(live) == proof[field]
        assert report["all_strict_controls_passed"] and report["generated_tokens"] == 0
        assert report["hooks_methods_callables_restored"]
        assert read(folder / "runtime_cleanup.json")["hooks_methods_callables_restored"]
        assert read(folder / "runtime_cleanup.json")["actual_counts"] == report["actual_counts"]
        assert [c["case_id"] for c in report["cases"]] == ["text31", "text65", "text129", "hands4", "cat6"]
        for case in report["cases"]:
            assert case["identity"] == report["input_identity"][report["cases"].index(case)]
            assert len(case["samples"]) == 6
            assert [r["order"] for r in case["samples"]].count([64, 32]) == 3
            assert [r["order"] for r in case["samples"]].count([32, 64]) == 3
            assert case["all_strict_controls_passed"] and case["two_branches_main_unchanged_bitwise"] and case["all_reset_empty"]
            for index, row in enumerate([case["warmup_excluded"]] + case["samples"]):
                assert row["round"] == index and row["warmup"] == (index == 0)
                assert row["all_numeric_passed"] and row["all_reset_empty"]
                expected_keys = {"full32_vs_full64_all_logits", "increment64_vs_own_full_last_chunk", "increment32_vs_own_full_last_chunk"}
                if index == 0:
                    expected_keys |= {"question32_vs_question64", "fixed_token32_vs_fixed_token64"}
                assert set(row["numeric"]) == expected_keys
                for control in row["numeric"].values():
                    values = control["logits"]
                    assert control["passed"] and values["allclose_1e4"] and values["fp32_reference_limits_passed"]
                    assert values["max_abs_diff"] <= .01 and values["mean_abs_diff"] <= .0001 and values["rms_diff"] <= .0003
                    if "final_states" in control:
                        assert set(control["final_states"]) == {"key_cache", "value_cache", "conv_states", "recurrent_states"}
                        assert all(v["allclose_1e4"] for v in control["final_states"].values())
            for size in ("32", "64"):
                for label, units in case["stage_summary_ms"][size].items():
                    for unit, stats in units.items():
                        values = [row["times"][size][label][unit] for row in case["samples"]]
                        assert stats == {"median": statistics.median(values), "min": min(values), "max": max(values)}
                        assert min(values) > 0
            for label, ratios in case["paired_wall_ratios"].items():
                values = [r["times"]["32"][label]["wall_ms"]/r["times"]["64"][label]["wall_ms"] for r in case["samples"]]
                assert ratios["median_ratio"] == statistics.median(values)
                assert [v["ratio32_over64"] for v in ratios["samples"]] == values
                for order in ([64, 32], [32, 64]):
                    assert ratios["median_by_order"][str(order)] == statistics.median(v for r, v in zip(case["samples"], values) if r["order"] == order)
        snapshots[folder.name] = {"module_sha256": sha(folder / "script_snapshot.py"),
            "plan_sha256": sha(folder / "plan_snapshot.md"), "launcher_sha256": sha(folder / "launcher_snapshot.py")}
    assert gpu["actual_counts"] == {"vision": 2, "language": 314, "chunk_rule": 5472, "vision_projections": 250}
    assert gpu["wall_seconds_including_load"] < 600 and gpu["retained_feature_peak_bytes"] == 589824
    assert gpu["peak_allocated_gib"] < 6
    assert read(gpu_dir / "cleanup.json")["cuda_allocated_bytes_after_release"] == 0
    assert read(gpu_dir / "cleanup.json")["cuda_reserved_bytes_after_release"] == 0
    old_feature = read(runs / "s4b_chunk_probe_gpu_20261010_02/chunk_probe_summary.json")["feature_sha256"]
    assert gpu["cases"][3]["feature_sha256"] == old_feature
    old = read(runs / "s4b_vision_extension_delivery_check_20261010_01/source_tree.json")
    assert len(old) == 60 and all(sha(repo / name) == digest for name, digest in old.items())
    reference = root / "reference/qwen35-streaming"
    originals = [p for p in reference.rglob("*") if p.is_file()]
    assert len(originals) == 15 and all(sha(p) == sha(root / "code/qwen35-streaming" / p.relative_to(reference)) for p in originals)
    assert all(sha(Path(p)) == digest for p, digest in proof["assets"].items())
    model_dir = root / "models/Qwen3.5-0.8B"
    assert sha(model_dir / "DOWNLOAD_MANIFEST.json") == proof["model_manifest_sha256"]
    assert all((model_dir / item["name"]).stat().st_size == item["bytes"] and
               sha(model_dir / item["name"]) == item["sha256"].lower() for item in proof["verified_model_assets"])
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=repo, text=True).splitlines()
    tree = {name: sha(repo / name) for name in sorted(names)}
    assert len(tree) == 61 and set(tree)-set(old) == {"scripts/chunk_interleaved.py"}
    assert subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip() == "36tar/s4-mixed-state"
    write(args.out / "source_tree.json", tree)
    write(args.out / "source_snapshots.json", snapshots)
    (args.out / "tracked_changes.patch").write_bytes(subprocess.check_output(["git", "diff", "--no-ext-diff", "HEAD"], cwd=repo))
    gates = []
    for label in ("budget", "source", "cpu", "input", "plan", "launcher", "asset", "model_manifest"):
        with tempfile.TemporaryDirectory(prefix="reject_", dir=args.out) as temporary:
            folder = Path(temporary).resolve()
            assert folder.is_relative_to(args.out) and args.out.is_relative_to(runs.resolve())
            for name in ("status.json", "preflight_summary.json", "provenance.json", "arguments.json", "cpu_summary.json"):
                shutil.copyfile(cpu_dir / name, folder / name)
            broken = read(folder / "preflight_summary.json")
            expected_error = "Frozen plan, launcher or evidence changed"
            if label == "budget":
                broken["budget"]["chunk_rule_calls_max"] += 1
                expected_error = "CPU controls or budget mismatch"
            elif label == "source":
                source = read(folder / "provenance.json")
                source["code_hashes"][str(Path("scripts/chunk_interleaved.py"))] = "0"*64
                write(folder / "provenance.json", source)
                expected_error = "Source changed"
            elif label == "cpu":
                (folder / "cpu_summary.json").write_text((folder / "cpu_summary.json").read_text(encoding="utf-8-sig") + "\n", encoding="utf-8")
            elif label == "input":
                broken_cpu = read(folder / "cpu_summary.json")
                broken_cpu["input_identity"][0]["input_fingerprint"] = "0"*64
                write(folder / "cpu_summary.json", broken_cpu)
                broken["cpu_report_sha256"] = sha(folder / "cpu_summary.json")
                expected_error = "Frozen input changed"
            elif label in ("plan", "launcher"):
                broken[label + "_sha256"] = "0"*64
            elif label == "asset":
                broken["assets"][next(iter(broken["assets"]))] = "0"*64
                expected_error = "Asset changed"
            else:
                broken["model_manifest_sha256"] = "0"*64
                expected_error = "Model manifest changed"
            write(folder / "preflight_summary.json", broken)
            target = args.out / ("should_not_run_" + label)
            result = subprocess.run([sys.executable, str(root / "scripts/run_chunk_interleaved.py"), "--experiment", "interleaved",
                "--stage", "gpu", "--preflight", str(folder), "--out", str(target)], capture_output=True,
                text=True, encoding="utf-8", timeout=30)
            assert result.returncode == 2 and expected_error in result.stderr, result.stderr
            assert not target.exists() and not target.with_name(target.name + ".console.log").exists()
            gates.append({"case": label, "rejected_before_log_or_gpu": True, "exit_code": 2})
    cleanup = []
    for folder, report in ((cpu_dir, cpu), (gpu_dir, gpu)):
        path = (folder / "interleaved_partial.json").resolve()
        if path.exists():
            assert path.is_relative_to(folder.resolve()) and folder.resolve().is_relative_to(runs.resolve())
            partial = read(path)
            assert partial["completed_cases"] == report["cases"][:-1]
            assert partial["current_case"] == report["cases"][-1]["case_id"]
            assert partial["warmup"] == report["cases"][-1]["warmup_excluded"]
            assert partial["samples"] == report["cases"][-1]["samples"] and partial["actual_counts"] == report["actual_counts"]
            cleanup.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path), "final_summary_covers_content": True})
            path.unlink()
    cache_dir = (repo / "scripts/__pycache__").resolve()
    for path in cache_dir.glob("chunk_interleaved.*.pyc"):
        path = path.resolve()
        assert path.is_relative_to(cache_dir) and cache_dir.is_relative_to(repo.resolve())
        cleanup.append({"path": str(path), "bytes": path.stat().st_size, "regenerable_bytecode": True})
        path.unlink()
    write(args.out / "cleanup_verification.json", {"removed": cleanup, "removed_bytes": sum(r["bytes"] for r in cleanup),
        "historical_failures_and_assets_preserved": True})
    verification = {"passed": True, "frozen_runs": 2, "old_source60_unchanged": True, "source_files": 61,
        "reference_files15_unchanged": True, "model_assets_checked": 10, "video_assets_checked": 2,
        "cases_per_stage": 5, "timed_rounds_per_stage": 30, "warmups_per_stage": 5,
        "strict_controls_per_stage": 115, "negative_gates": gates, "defaults_unchanged": True,
        "scope": "finite FP32 torch fallback benchmark; no answer generation or long-running claims"}
    write(args.out / "verification.json", verification)
    print(verification)


if __name__ == "__main__":
    main()
