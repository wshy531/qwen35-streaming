"""Audit initial device error, repaired four-frame proof and old-source continuity; no inference."""
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
    names = ("s4b_four_frame_preflight_20261010_01","s4b_four_frame_gpu_20261010_01",
             "s4b_four_frame_preflight_20261010_02","s4b_four_frame_gpu_20261010_02")
    snapshots = {sha(runs/name/"script_snapshot.py"):name for name in names}
    historical_sources = {}
    for name in names:
        differences = []
        for path,digest in read(runs/name/"provenance.json")["code_hashes"].items():
            if sha(repo/path) != digest:
                assert path.replace("\\", "/") == "scripts/four_frame_online.py" and digest in snapshots
                differences.append(path)
        historical_sources[name] = differences
    old_pre,failed,pre,gpu_dir = [runs/name for name in names]
    for run,stage in ((old_pre,"preflight"),(pre,"preflight"),(gpu_dir,"gpu")):
        assert read(run/"status.json") == {"status":"completed","stage":stage}
        assert read(run/"watchdog_status.json")["exit_code"] == 0 and not read(run/"watchdog_status.json")["timed_out"]
    assert read(failed/"status.json") == {"status":"failed","stage":"gpu"}
    assert read(failed/"watchdog_status.json")["exit_code"] == 1 and not read(failed/"watchdog_status.json")["timed_out"]
    assert "Expected all tensors to be on the same device" in (failed/"traceback.txt").read_text()
    assert not (failed/"four_frame_online_summary.json").exists()
    assert sha(old_pre/"script_snapshot.py") == sha(failed/"script_snapshot.py")
    assert sha(pre/"script_snapshot.py") == sha(gpu_dir/"script_snapshot.py") == sha(repo/"scripts/four_frame_online.py")
    assert sha(old_pre/"script_snapshot.py") != sha(pre/"script_snapshot.py")
    proof = read(pre/"preflight_summary.json")
    first_proof = read(old_pre/"preflight_summary.json")
    assert proof["budget"] == first_proof["budget"] and proof["input_fingerprints"] == first_proof["input_fingerprints"]
    assert proof["expected_original_ids"] == first_proof["expected_original_ids"] == [72141,50,248046]
    assert proof["cpu_controls_passed"] and proof["hooks_cleanup_verified"] and proof["all_model_assets_verified"] == 10
    assert read(pre/"cpu_summary.json")["actual_forwards"] == {"language":43,"vision":4}
    assert read(pre/"cpu_summary.json")["generated_tokens"] == 32
    assert sha(root/"在线四帧主体公平对照方案.md") == proof["plan_sha256"] == first_proof["plan_sha256"]
    assert sha(root/"scripts/run_four_frame_online.py") == proof["launcher_sha256"]
    original_path = runs/"s4b_protocol_capability_gpu_20261009_01/protocol_capability_summary.json"
    assert sha(original_path) == proof["original_gpu_summary_sha256"]
    gpu = read(gpu_dir/"four_frame_online_summary.json")
    assert gpu["all_numeric_groups_passed"] and gpu["semantic_expected_ids_match"]
    assert gpu["actual_forwards"] == {"language":23,"vision":4} and gpu["generated_tokens"] == 12
    assert gpu["input_fingerprints"] == proof["input_fingerprints"] and gpu["budget"] == proof["budget"]
    assert gpu["frame_ids"] == [0,30,60,90] and gpu["timestamps"] == [0.,1.001,2.002,3.003]
    assert gpu["question_cutoff"] == {"frame_id":90,"timestamp":3.003}
    assert len(gpu["groups"]) == 4 and all(g["passed"] for g in gpu["groups"])
    for g in gpu["groups"][:2]:
        assert g["logits"]["allclose_1e4"] and g["logits"]["fp32_reference_limits_passed"]
        for field,count in (("key_cache",6),("value_cache",6),("conv_states",18),("recurrent_states",18)):
            assert g["final_states"][field]["allclose_1e4"] and g["final_states"][field]["populated_layers"] == count
    for g in gpu["groups"][1:]:
        assert g["online"]["answer"] == "HANDS" and g["online"]["generated_ids"] == [72141,50,248046]
        assert g["main_unchanged_bitwise_and_counters"] and g["online"]["frame_id"] == 90
    for g in gpu["groups"][2:]:
        assert g["initial_logits_bitwise_equal"] and g["branch_states_bitwise_equal"] and g["ids_match_first"]
    reset = gpu["groups"][3]
    assert all(reset[k] for k in ("reset_empty_contract","actual_features_bitwise_equal","complete_embeddings_bitwise_equal",
        "main_after_replay_bitwise_equal","main_status_after_replay_equal","input_fingerprints_equal"))
    for name,digest in read(runs/"s4b_protocol_capability_delivery_check_20261009_02/source_tree.json").items():
        if name != "S4_A.md":
            assert sha(repo/name) == digest,name
    unchanged_baseline = 0
    for p in (root/"reference/qwen35-streaming").rglob("*"):
        if p.is_file():
            assert sha(p) == sha(root/"code/qwen35-streaming"/p.relative_to(root/"reference/qwen35-streaming"))
            unchanged_baseline += 1
    assert not read(runs/"s4b_online_gpu_20261009_01/online_summary.json")["all_interface_numeric_groups_passed"]
    original = read(original_path)
    assert next(c for c in original["cases"] if c["id"] == "fp32_ieee_math_frames2_subject")["answer"] == "CAT"
    assert next(c for c in original["cases"] if c["id"] == "fp32_ieee_math_frames4_hands_count")["answer"] == "ONE"
    model_dir = root/"models/Qwen3.5-0.8B"
    manifest = read(model_dir/"DOWNLOAD_MANIFEST.json")
    for file in manifest["files"]:
        p = model_dir/file["name"]
        assert p.stat().st_size == file["bytes"] and sha(p) == file["sha256"].lower()
    def rejected(name,preflight,message):
        target = args.out/name
        result = subprocess.run([sys.executable,str(root/"scripts/run_four_frame_online.py"),"--stage","gpu","--out",str(target),
            "--preflight",str(preflight)],capture_output=True,text=True,encoding="utf-8",timeout=30)
        assert result.returncode == 2 and message in result.stderr,result.stderr
        assert not target.exists() and not target.with_name(target.name+".console.log").exists()
    rejected("wrong_budget",runs/"s4b_shared_preflight_20261009_02","CPU controls or registered budget mismatch")
    rejected("old_source",old_pre,"Source changed:")
    changed = args.out/"changed_answer_proof"
    changed.mkdir()
    for name in ("status.json","preflight_summary.json","provenance.json","arguments.json"):
        (changed/name).write_bytes((pre/name).read_bytes())
    changed_proof = read(changed/"preflight_summary.json")
    changed_proof["original_gpu_summary_sha256"] = "0"*64
    (changed/"preflight_summary.json").write_text(json.dumps(changed_proof,indent=2),encoding="utf-8")
    rejected("changed_answer",changed,"Original answer evidence changed")
    git = lambda *a:subprocess.check_output(["git",*a],cwd=repo)
    assert git("branch","--show-current").decode().strip() == "36tar/s4-mixed-state"
    git("diff","--check")
    tree = {str(p.relative_to(repo)):sha(p) for p in sorted(repo.rglob("*")) if p.is_file() and
        ".git" not in p.parts and "__pycache__" not in p.parts}
    (args.out/"source_tree.json").write_text(json.dumps(tree,indent=2),encoding="utf-8")
    (args.out/"tracked_changes.patch").write_bytes(git("diff","--binary"))
    report = {"initial_device_error_and_exact_source_preserved":True,"historical_source_matches":historical_sources,
        "repaired_four_groups_and_hands_ids_verified":True,"repeated_qa_reset_bitwise_and_main_isolation_verified":True,
        "input_budget_provenance_verified":True,"old_44_executable_sources_unchanged":True,
        "baseline_files_unchanged":unchanged_baseline,"old_cat_one_and_numeric_failures_preserved":True,
        "three_negative_launch_gates_verified":True,"all_model_assets_sha_verified":len(manifest["files"]),
        "no_inference_in_audit":True,"source_files":len(tree),"branch":"36tar/s4-mixed-state",
        "base_commit":git("rev-parse","HEAD").decode().strip(),"commit_created":False,"remote_push":False}
    (args.out/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()

