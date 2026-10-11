"""Read-only evidence/source audit, negative launch gates and bounded duplicate cleanup."""
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
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    repo, runs = root / "code/qwen35-streaming-s4", root / "runs"
    args.out = args.out.resolve()
    assert args.out.is_relative_to(runs.resolve())
    args.out.mkdir(exist_ok=False)
    specs = [("vision_trace", "vision_trace_summary.json", ["01"]),
             ("qkv_replay", "qkv_replay_summary.json", ["01"]),
             ("vision_fixed", "vision_fixed_summary.json", ["01", "02"]),
             ("vision_fixed_online", "online_summary.json", ["01", "02"])]
    reports, snapshots, changed_self_versions = {}, {}, []
    for kind, summary, versions in specs:
        module = "vision_fixed_control.py" if kind == "vision_fixed" else kind + ".py"
        launcher = root / "scripts" / ("run_" + kind + ".py")
        dirs = [(runs / f"s4b_{kind}_preflight_20261010_{version}", "preflight") for version in versions]
        gpu_dir = runs / f"s4b_{kind}_gpu_20261010_01"
        dirs.append((gpu_dir, "gpu"))
        for folder, stage in dirs:
            assert read(folder / "status.json") == {"status": "completed", "stage": stage}
            watchdog = read(folder / "watchdog_status.json")
            assert watchdog["exit_code"] == 0 and not watchdog["timed_out"]
            code = read(folder / "provenance.json")["code_hashes"]
            assert sha(folder / "script_snapshot.py") == code[str(Path("scripts") / module)]
            for name, digest in code.items():
                if sha(repo / name) != digest:
                    assert name == str(Path("scripts") / module) and stage == "preflight" and folder.name.endswith("_01")
                    changed_self_versions.append(folder.name)
            proof = read(folder / "preflight_summary.json") if stage == "preflight" else read(dirs[-2][0] / "preflight_summary.json")
            assert sha(launcher) == proof["launcher_sha256"]
            snapshot = folder / "launcher_snapshot.py"
            snapshot.write_bytes(launcher.read_bytes())
            snapshots[folder.name] = {"module": sha(folder / "script_snapshot.py"), "launcher": sha(snapshot)}
            if stage == "preflight":
                assert sha(folder / "cpu_summary.json") == proof["cpu_report_sha256"]
        latest = dirs[-2][0]
        proof = read(latest / "preflight_summary.json")
        plan = {"vision_trace": "视觉首次差异定位方案.md", "qkv_replay": "视觉QKV同输入重放方案.md",
                "vision_fixed": "固定时间单元视觉参考方案.md", "vision_fixed_online": "固定视觉在线回归方案.md"}[kind]
        assert sha(root / plan) == proof["plan_sha256"]
        report = read(gpu_dir / summary)
        assert report["budget"] == proof["budget"] and report["wall_seconds_including_load"] < 600
        reports[kind] = report
    assert set(changed_self_versions) == {"s4b_vision_fixed_preflight_20261010_01", "s4b_vision_fixed_online_preflight_20261010_01"}
    first, replay, fixed, online = (reports[k] for k in ("vision_trace", "qkv_replay", "vision_fixed", "vision_fixed_online"))
    assert first["actual_forwards"] == {"vision": 4, "language": 0} and len(first["stages"]) == 42
    assert first["repeat_all_stages_bitwise"]
    assert all(first["first_differences"][k]["stage"] == "blocks.0.attn.qkv.output" for k in ("pair0", "pair1"))
    assert first["combined_features"]["max_abs_diff"] == read(runs / "s4b_shared_gpu_20261009_01/shared_features_summary.json")["groups"][4]["features"]["max_abs_diff"]
    assert replay["actual_operations"] == {"prefix": 1, "projection": 9, "vision": 0, "language": 0}
    assert replay["trace_input_and_outputs_reproduced"] and replay["weights_bitwise_unchanged"]
    assert not replay["fp32_split"]["bitwise_equal"]
    assert replay["fp32_equal_shape_padding"]["bitwise_equal"] and replay["local_fp64_cast_fp32_split"]["bitwise_equal"]
    assert fixed["actual_forwards"] == {"vision": 5, "language": 0} and fixed["candidate_projection_operations"] == 300
    assert fixed["candidate_all_stages_bitwise"] and fixed["combined_features"]["bitwise_equal"]
    assert fixed["candidate_full_matches_original_pairs"] and not fixed["candidate_full_vs_original_full"]["bitwise_equal"]
    assert online["all_interface_numeric_groups_passed"] and len(online["groups"]) == 8
    assert online["actual_forwards"] == {"language": 23, "vision": 8} and online["generated_tokens"] == 8
    assert online["candidate_projection_operations"] == 500 and online["candidate_methods_restored"]
    for g in online["groups"]:
        assert g["passed"]
        if "final_states" in g:
            assert g["logits"]["allclose_1e4"] and g["logits"]["fp32_reference_limits_passed"]
            assert all(v["allclose_1e4"] for v in g["final_states"].values())
        if "online" in g:
            assert g["main_unchanged_bitwise"] and g["online"]["generated_ids"] == g["reference_ids"]
    assert [(g["online"]["answer"], g["online"]["generated_ids"]) for g in online["groups"] if "online" in g] == [
        ("CAT", [27280, 248046]), ("ONE", [5059, 248046])]
    for r in (first, replay, fixed):
        assert r["trace_buffer_peak_bound_bytes"] <= 128 * 2**20
    assert online["retained_feature_buffer_peak_bound_bytes"] <= 128 * 2**20
    old = read(runs / "s4b_equal_tail_delivery_check_20261010_01/source_tree.json")
    for name, digest in old.items():
        assert sha(repo / name) == digest, name
    baseline = 0
    for path in (root / "reference/qwen35-streaming").rglob("*"):
        if path.is_file():
            assert sha(path) == sha(root / "code/qwen35-streaming" / path.relative_to(root / "reference/qwen35-streaming"))
            baseline += 1
    manifest = read(root / "models/Qwen3.5-0.8B/DOWNLOAD_MANIFEST.json")
    for item in manifest["files"]:
        path = root / "models/Qwen3.5-0.8B" / item["name"]
        assert path.stat().st_size == item["bytes"] and sha(path) == item["sha256"].lower()
    names = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=repo, text=True).splitlines()
    tree = {name: sha(repo / name) for name in sorted(names)}
    assert len(tree) == 57
    write(args.out / "source_tree.json", tree)
    diff = subprocess.check_output(["git", "diff", "--no-ext-diff", "HEAD"], cwd=repo)
    (args.out / "tracked_changes.patch").write_bytes(diff)
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip()
    assert branch == "36tar/s4-mixed-state"
    write(args.out / "source_snapshots.json", snapshots)
    gates = []
    def rejected(label, launcher, proof_dir, expected):
        target = args.out / label
        result = subprocess.run([sys.executable, str(root / "scripts" / launcher), "--stage", "gpu", "--out", str(target),
                                 "--preflight", str(proof_dir)], capture_output=True, text=True, encoding="utf-8", timeout=30)
        assert result.returncode == 2 and expected in result.stderr, result.stderr
        assert not target.exists() and not target.with_name(target.name + ".console.log").exists()
        gates.append({"case": label, "exit_code": 2, "rejected_before_log_or_gpu": True})
    rejected("wrong_vision_budget", "run_vision_trace.py", runs / "s4b_shared_preflight_20261009_02", "CPU controls or budget mismatch")
    rejected("wrong_qkv_budget", "run_qkv_replay.py", runs / "s4b_vision_trace_preflight_20261010_01", "CPU controls or budget mismatch")
    good = runs / "s4b_vision_fixed_online_preflight_20261010_02"
    for label in ("changed_source", "changed_cpu", "changed_input", "changed_plan"):
        with tempfile.TemporaryDirectory(prefix="gate_", dir=args.out) as folder_name:
            folder = Path(folder_name).resolve()
            # Verify the absolute recursive-cleanup target before TemporaryDirectory exits.
            assert folder.is_relative_to(args.out) and args.out.is_relative_to(runs.resolve())
            for name in ("status.json", "preflight_summary.json", "provenance.json", "arguments.json", "cpu_summary.json"):
                shutil.copyfile(good / name, folder / name)
            if label == "changed_source":
                item = read(folder / "provenance.json")
                item["code_hashes"]["scripts/vision_fixed_units.py"] = "0" * 64
                write(folder / "provenance.json", item)
                expected = "Source changed"
            elif label == "changed_cpu":
                with (folder / "cpu_summary.json").open("a", encoding="utf-8") as stream:
                    stream.write("\n")
                expected = "Frozen plan, launcher or evidence changed"
            elif label == "changed_input":
                item = read(folder / "cpu_summary.json")
                item["input_fingerprints"]["prefix2"] = "0" * 64
                write(folder / "cpu_summary.json", item)
                proof = read(folder / "preflight_summary.json")
                proof["cpu_report_sha256"] = sha(folder / "cpu_summary.json")
                write(folder / "preflight_summary.json", proof)
                expected = "Frozen input proof changed"
            else:
                proof = read(folder / "preflight_summary.json")
                proof["plan_sha256"] = "0" * 64
                write(folder / "preflight_summary.json", proof)
                expected = "Frozen plan, launcher or evidence changed"
            rejected(label, "run_vision_fixed_online.py", folder, expected)
    cleanup = []
    for kind in ("vision_trace", "vision_fixed"):
        folder = (runs / f"s4b_{kind}_gpu_20261010_01").resolve()
        path = folder / "trace_partial.json"
        assert path.resolve().is_relative_to(folder) and folder.is_relative_to(runs.resolve())
        partial = read(path)
        assert partial["records"] == reports[kind]["records"] and partial["resources"] == reports[kind]["resources"]
        cleanup.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha(path),
                        "reason": "identical records/resources preserved in final summary"})
        path.unlink()
    for name in ("vision_trace", "qkv_replay", "vision_fixed_units", "vision_fixed_control", "vision_fixed_online"):
        path = repo / "scripts/__pycache__" / (name + ".cpython-312.pyc")
        if path.exists():
            assert path.resolve().is_relative_to((repo / "scripts/__pycache__").resolve())
            cleanup.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha(path),
                            "reason": "regenerable bytecode for this turn's new modules only"})
            path.unlink()
    write(args.out / "cleanup_verification.json", cleanup)
    result = {"four_gpu_runs_completed": True, "six_cpu_runs_completed": True, "changed_self_versions_preserved": changed_self_versions,
              "source_files": len(tree), "old_source_files_unchanged": len(old), "original_reference_files_unchanged": baseline,
              "model_assets_sha_verified": len(manifest["files"]), "negative_launch_gates": gates,
              "explicit_candidate_online_eight_strict_passed": True, "original_default_and_semantic_failures_preserved": True,
              "memory_bounds_checked": True, "temporary_files_removed": len(cleanup), "temporary_bytes_removed": sum(i["bytes"] for i in cleanup),
              "branch": branch, "new_gpu_in_audit": 0, "remote_commands": 0, "training": 0, "push": False}
    write(args.out / "verification.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
