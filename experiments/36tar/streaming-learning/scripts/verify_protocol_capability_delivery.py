"""Audit original-model same-input controls and preserved history; never runs inference."""
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
    pre = runs/"s4b_protocol_capability_preflight_20261009_01"
    gpu_dir = runs/"s4b_protocol_capability_gpu_20261009_01"
    proof = read(pre/"preflight_summary.json")
    gpu = read(gpu_dir/"protocol_capability_summary.json")
    for run,stage in ((pre,"preflight"),(gpu_dir,"gpu")):
        assert read(run/"status.json") == {"status":"completed","stage":stage}
        assert read(run/"watchdog_status.json")["exit_code"] == 0 and not read(run/"watchdog_status.json")["timed_out"]
        assert sha(run/"script_snapshot.py") == sha(repo/"scripts/protocol_capability_control.py")
        for name,digest in read(run/"provenance.json")["code_hashes"].items():
            assert sha(repo/name) == digest,name
    assert proof["cpu_controls_passed"] and len(proof["input_comparisons"]) == 4
    assert sha(root/"同画面输入与能力控制方案.md") == proof["plan_sha256"]
    assert sha(root/"scripts/run_protocol_capability.py") == proof["launcher_sha256"]
    for name,digest in proof["original_reference_hashes"].items():
        assert sha(Path(name)) == digest,name
    for item in proof["input_comparisons"]:
        assert item["all_input_tensors_bitwise_equal"] and all(item["tensor_equalities"].values())
        assert item["tiny_original_vs_modified_prefill"]["bitwise_equal"]
    assert gpu["budget"] == proof["budget"]
    assert gpu["actual_forwards"] == {"language":18,"vision":8} and gpu["generated_tokens"] == 18
    assert gpu["all_registered_cases_executed"] and gpu["all_input_fingerprints_verified"]
    assert [x["id"] for x in gpu["cases"]] == [x["id"] for x in proof["cases"]]
    assert Path(gpu["model_reference"]).resolve() == (root/"code/qwen35-streaming/qwen35_local/modeling_qwen3_5.py").resolve()
    specifications = {x["id"]:x for x in proof["input_comparisons"]}
    expected = {"frames2_subject":("CAT",[27280,248046]),"frames2_hands_count":("ONE",[5059,248046]),
        "frames4_subject":("HANDS",[72141,50,248046]),"frames4_hands_count":("ONE",[5059,248046])}
    for mode in ("bf16_sdpa","fp32_ieee_math"):
        for input_id,(answer,ids) in expected.items():
            case = next(x for x in gpu["cases"] if x["id"] == mode+"_"+input_id)
            assert case["answer"] == answer and case["generated_ids"] == ids and case["stopped_on_registered_eos"]
            assert case["input_fingerprint"] == specifications[input_id]["input_fingerprint"]
            assert case["forward_counts"]["vision"] == 1
    natural = read(runs/"natural_gpu_20261009_01/gpu_summary.json")
    old = next(x for x in natural["questions"] if x["video"] == "clean_hands" and x["sample_index"] == 3)
    assert old["input_fingerprint"] == specifications["frames4_subject"]["input_fingerprint"] and old["answer"].strip() == "HANDS"
    shared = read(runs/"s4b_shared_gpu_20261009_01/shared_features_summary.json")
    assert shared["all_shared_groups_passed"] and shared["groups"][1]["online"]["answer"] == "CAT"
    assert shared["groups"][3]["online"]["answer"] == "ONE"
    assert not read(runs/"s4b_online_gpu_20261009_01/online_summary.json")["all_interface_numeric_groups_passed"]
    for name,digest in read(runs/"s4b_shared_delivery_check_20261009_01/source_tree.json").items():
        if name != "S4_A.md":
            assert sha(repo/name) == digest,name
    for name in ("streaming/inputs.py","scripts/baseline.py"):
        assert sha(repo/name) == sha(root/"code/qwen35-streaming"/name),name
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
    def reject(name,preflight,message):
        target = args.out/name
        result = subprocess.run([sys.executable,str(root/"scripts/run_protocol_capability.py"),"--stage","gpu",
            "--out",str(target),"--preflight",str(preflight)],capture_output=True,text=True,encoding="utf-8",timeout=30)
        assert result.returncode == 2 and message in result.stderr,result.stderr
        assert not target.exists() and not target.with_name(target.name+".console.log").exists()
    reject("wrong_budget",runs/"s4b_shared_preflight_20261009_02","CPU controls or registered budget mismatch")
    changed = args.out/"changed_original_proof"
    changed.mkdir()
    for name in ("status.json","preflight_summary.json","provenance.json","arguments.json"):
        (changed/name).write_bytes((pre/name).read_bytes())
    tampered = read(changed/"preflight_summary.json")
    first = next(iter(tampered["original_reference_hashes"]))
    tampered["original_reference_hashes"][first] = "0"*64
    (changed/"preflight_summary.json").write_text(json.dumps(tampered,indent=2),encoding="utf-8")
    reject("changed_original",changed,"Original model or processor source changed")
    git = lambda *a:subprocess.check_output(["git",*a],cwd=repo)
    assert git("branch","--show-current").decode().strip() == "36tar/s4-mixed-state"
    git("diff","--check")
    tree = {str(p.relative_to(repo)):sha(p) for p in sorted(repo.rglob("*")) if p.is_file() and
        ".git" not in p.parts and "__pycache__" not in p.parts}
    (args.out/"source_tree.json").write_text(json.dumps(tree,indent=2),encoding="utf-8")
    (args.out/"tracked_changes.patch").write_bytes(git("diff","--binary"))
    report = {"four_input_combinations_bitwise_equal":True,"tiny_fresh_prefill_bitwise_equal":4,
        "original_model_source_path_and_sha_verified":True,"eight_fixed_cases_answers_and_ids_verified":True,
        "two_modes_same_answers_verified":True,"two_frame_cat_four_frame_hands_one_preserved":True,
        "historical_positive_input_rebuilt":True,"old_failures_preserved":True,"budget_and_provenance_verified":True,
        "all_model_assets_sha_verified":len(manifest["files"]),"old_43_executable_sources_unchanged":True,
        "baseline_files_unchanged":baseline_files,"negative_budget_and_original_source_gates_verified":True,
        "no_inference_in_audit":True,"source_files":len(tree),"branch":"36tar/s4-mixed-state",
        "base_commit":git("rev-parse","HEAD").decode().strip(),"commit_created":False,"remote_push":False}
    (args.out/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
