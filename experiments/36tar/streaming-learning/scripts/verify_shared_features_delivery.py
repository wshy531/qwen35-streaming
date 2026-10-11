"""Audit exact-input control, reproduced failures, historical sources and launch gates; no inference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from run_precision_layers import read,sha


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    repo,runs = root/"code/qwen35-streaming-s4",root/"runs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    names = ("s4b_shared_preflight_20261009_01","s4b_shared_preflight_20261009_02","s4b_shared_gpu_20261009_01")
    cpu1,cpu2,gpu_dir = [runs/name for name in names]
    for path,stage in ((cpu1,"preflight"),(cpu2,"preflight"),(gpu_dir,"gpu")):
        assert read(path/"status.json") == {"status":"completed","stage":stage}
        assert read(path/"watchdog_status.json")["exit_code"] == 0
        assert not read(path/"watchdog_status.json")["timed_out"]
        assert sha(path/"script_snapshot.py") == sha(repo/"scripts/shared_features_control.py")
        for file,digest in read(path/"provenance.json")["code_hashes"].items():
            assert sha(repo/file) == digest,(path,file)
    proof = read(cpu2/"preflight_summary.json")
    assert sha(root/"共享视觉特征控制方案.md") == proof["plan_sha256"]
    assert sha(root/"scripts/run_shared_features.py") == proof["launcher_sha256"]
    assert sha(cpu1/"launcher_snapshot.py") == read(cpu1/"preflight_summary.json")["launcher_sha256"]
    assert read(cpu1/"preflight_summary.json")["launcher_sha256"] != proof["launcher_sha256"]
    assert read(cpu1/"cpu_summary.json")["all_shared_groups_passed"]
    assert read(cpu2/"cpu_summary.json")["all_shared_groups_passed"] and proof["hooks_cleanup_verified"]
    gpu = read(gpu_dir/"shared_features_summary.json")
    old = read(runs/"s4b_online_gpu_20261009_01/online_summary.json")
    assert gpu["budget"] == proof["budget"]
    assert gpu["input_fingerprints"] == proof["input_fingerprints"] == old["input_fingerprints"]
    assert gpu["actual_forwards"] == {"language":23,"vision":6} and gpu["generated_tokens"] == 8
    assert gpu["wall_seconds_including_load"] < 600
    groups = {g["id"]:g for g in gpu["groups"]}
    assert len(groups) == 7 and gpu["all_shared_groups_passed"]
    for name in ("prefix2_shared","qa2_shared","prefix4_shared","qa4_shared","eof3_shared"):
        group = groups[name]
        assert group["passed"] and group["logits"]["allclose_1e4"] and group["logits"]["fp32_reference_limits_passed"]
        for field,count in (("key_cache",6),("value_cache",6),("conv_states",18),("recurrent_states",18)):
            assert group["final_states"][field]["allclose_1e4"]
            assert group["final_states"][field]["populated_layers"] == count
        if "input_identity" in group:
            assert all(group["input_identity"][key] for key in ("tokens_equal","four_axis_positions_equal",
                "complete_embeddings_bitwise_equal","visual_features_bitwise_equal"))
            assert group["input_identity"]["feature_bytes"] <= 393216
    for name,old_index in (("prefix4_vision_bias",4),("eof3_vision_bias",7)):
        group = groups[name]
        assert not group["features"]["bitwise_equal"]
        assert group["features"]["max_abs_diff"] <= 1e-4 and group["features"]["allclose_1e4"]
        assert not group["within_old_strict_language_gate"] and not group["online_vs_whole"]["passed"]
        assert group["online_vs_whole"]["logits"]["max_abs_diff"] == old["groups"][old_index]["logits"]["max_abs_diff"]
    for name,expected in (("qa2_shared",[27280,248046]),("qa4_shared",[5059,248046])):
        group = groups[name]
        assert group["online"]["generated_ids"] == group["reference_ids"] == expected
        assert group["main_unchanged_bitwise"] and group["main_counters_unchanged"]
    assert groups["qa2_shared"]["online"]["answer"] == "CAT"
    assert groups["qa4_shared"]["online"]["answer"] == "ONE"
    assert groups["eof3_shared"]["input_sealed"] and groups["eof3_shared"]["padding_record"]["synthetic_padding"]
    assert not old["all_interface_numeric_groups_passed"] and sum(g["passed"] for g in old["groups"]) == 5
    old_tree = read(runs/"s4b_online_delivery_check_20261009_01/source_tree.json")
    for name,digest in old_tree.items():
        if name != "S4_A.md":
            assert sha(repo/name) == digest,name
    baseline_files = 0
    for path in (root/"reference/qwen35-streaming").rglob("*"):
        if path.is_file():
            assert sha(path) == sha(root/"code/qwen35-streaming"/path.relative_to(root/"reference/qwen35-streaming"))
            baseline_files += 1
    model_dir = root/"models/Qwen3.5-0.8B"
    manifest = read(model_dir/"DOWNLOAD_MANIFEST.json")
    for file in manifest["files"]:
        path = model_dir/file["name"]
        assert path.stat().st_size == file["bytes"] and sha(path) == file["sha256"].lower()
    launcher = root/"scripts/run_shared_features.py"
    def rejected(name,preflight,message):
        target = args.out/name
        result = subprocess.run([sys.executable,str(launcher),"--stage","gpu","--out",str(target),
            "--preflight",str(preflight)],capture_output=True,text=True,encoding="utf-8",timeout=30)
        assert result.returncode == 2 and message in result.stderr,result.stderr
        assert not target.exists() and not target.with_name(target.name+".console.log").exists()
    rejected("wrong_budget",runs/"s4b_online_preflight_20261009_02","CPU controls or registered budget mismatch")
    changed = args.out/"changed_source_proof"
    changed.mkdir()
    for name in ("status.json","preflight_summary.json","provenance.json","arguments.json"):
        (changed/name).write_bytes((cpu2/name).read_bytes())
    modified = read(changed/"provenance.json")
    modified["code_hashes"]["scripts/shared_features_control.py"] = "0"*64
    (changed/"provenance.json").write_text(json.dumps(modified,indent=2),encoding="utf-8")
    rejected("changed_source",changed,"Source changed: scripts/shared_features_control.py")
    git = lambda *args:subprocess.check_output(["git",*args],cwd=repo)
    assert git("branch","--show-current").decode().strip() == "36tar/s4-mixed-state"
    git("diff","--check")
    tree = {str(p.relative_to(repo)):sha(p) for p in sorted(repo.rglob("*")) if p.is_file() and
        ".git" not in p.parts and "__pycache__" not in p.parts}
    (args.out/"source_tree.json").write_text(json.dumps(tree,indent=2),encoding="utf-8")
    (args.out/"tracked_changes.patch").write_bytes(git("diff","--binary"))
    report = {"shared_strict_five_of_five_verified":True,"different_vision_two_failures_preserved":True,
        "old_online_error_reproduced_exactly":True,"old_five_of_eight_failure_preserved":True,
        "eos_real_two_answers_stop_verified":True,"cat_one_semantic_failures_preserved":True,
        "original_42_executable_sources_unchanged":True,"baseline_files_unchanged":baseline_files,
        "all_model_assets_sha_verified":len(manifest["files"]),"provenance_inputs_budget_verified":True,
        "negative_budget_and_source_gates_verified_before_launch":True,"no_inference_in_audit":True,
        "first_cpu_child_success_outer_display_error_separated":True,"source_files":len(tree),
        "branch":"36tar/s4-mixed-state","base_commit":git("rev-parse","HEAD").decode().strip(),
        "commit_created":False,"remote_push":False}
    (args.out/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
