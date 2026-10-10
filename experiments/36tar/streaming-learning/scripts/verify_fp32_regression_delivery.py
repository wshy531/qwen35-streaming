"""Verify fixed FP32 regression/CPU protocol evidence without inference."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from run_precision_layers import read,sha


def main():
    root = Path(__file__).resolve().parents[1]
    repo,runs = root/"code/qwen35-streaming-s4",root/"runs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    pre_dir = runs/"s4_fp32_regression_preflight_20261009_01"
    gpu_dir = runs/"s4_fp32_regression_gpu_20261009_01"
    probe_dir = runs/"s4b_protocol_cpu_20261009_01"
    for directory,stage in ((pre_dir,"preflight"),(gpu_dir,"gpu"),(probe_dir,"cpu_protocol_probe")):
        assert read(directory/"status.json") == {"status":"completed","stage":stage}
        for name,digest in read(directory/"provenance.json")["code_hashes"].items():
            assert sha(repo/name) == digest,(directory,name)
        assert sha(directory/"script_snapshot.py") == read(directory/"script_snapshot.py.json")["sha256"]
    pre,gpu,probe = read(pre_dir/"preflight_summary.json"),read(gpu_dir/"fp32_regression_summary.json"),read(probe_dir/"protocol_summary.json")
    assert pre["cpu_controls"]["passed"]
    assert sha(root/"FP32混合状态回归方案.md") == pre["plan_sha256"]
    assert sha(root/"scripts/run_fp32_regression.py") == pre["launcher_sha256"]
    assert gpu["input_fingerprints"] == pre["input_fingerprints"] == read(runs/"s4_preflight_20261009_01/preflight_summary.json")["input_fingerprints"]
    assert gpu["budget"] == pre["budget"] and gpu["budget"]["generated_tokens"] == 0
    assert len(gpu["groups"]) == 10 and gpu["numeric_groups_passed"] and gpu["state_isolation_reset_capacity_passed"]
    assert gpu["vision_dtype"] == gpu["language_dtype"] == "torch.float32"
    active = gpu["active_backend_settings"]
    assert active["matmul_fp32"] == active["conv_fp32"] == "ieee"
    assert active["math_sdp"] and not active["flash_sdp"] and not active["mem_efficient_sdp"]
    assert gpu["backend_and_metrics_restored"] and not read(gpu_dir/"watchdog_status.json")["timed_out"]
    assert sum(1+len(g["chunks"]) for g in gpu["groups"][:7])+7 == gpu["budget"]["language_forwards"] == 30
    numerical = [g for g in gpu["groups"] if "logits" in g]+[gpu["groups"][7]["main_continuation"]]
    for group in numerical:
        assert group["logits"]["fp32_reference_limits_passed"] and group["logits"]["allclose_1e4"]
        assert all(v["allclose_1e4"] for v in group["final_states"].values())
        assert [group["final_states"][name]["populated_layers"] for name in ("key_cache","value_cache","conv_states","recurrent_states")] == [6,6,18,18]
    for group in gpu["groups"][:7]:
        assert group["cache_lengths"][-1] == sum(group["chunks"])
    fork = gpu["groups"][7]
    assert fork["cache_storage_independent"] and fork["main_unchanged_before_continuation_bitwise"]
    assert gpu["groups"][8]["logits"]["bitwise_equal"] and gpu["groups"][8]["all_state_and_positions_cleared"]
    assert gpu["groups"][9]["rejected_before_forward"] and gpu["groups"][9]["all_state_unchanged_bitwise"]
    assert probe["checked_short_protocol_interfaces_passed"]
    assert probe["whole_input_fingerprint_with_positions"] == gpu["input_fingerprints"]["video"]
    assert probe["component_token_counts"] == [3,56,56,34]
    assert sha(root/"scripts/probe_online_protocol.py") == read(probe_dir/"script_snapshot.py.json")["sha256"]
    for name,digest in read(runs/"s4_full_precision_delivery_check_20261009_01/source_tree.json").items():
        if name != "S4_A.md":
            assert sha(repo/name) == digest,name
    negative = args.out/"wrong_preflight"
    result = subprocess.run([sys.executable,str(root/"scripts/run_fp32_regression.py"),"--stage","gpu","--out",str(negative),
        "--preflight",str(runs/"s4_full_precision_preflight_20261009_01")],capture_output=True,text=True,timeout=30)
    assert result.returncode == 2 and "CPU controls or budget mismatch" in result.stderr
    assert not negative.exists() and not negative.with_name(negative.name+".console.log").exists()
    assert not read(runs/"s4_gpu_20261009_01/gpu_summary.json")["registered_numeric_limits_passed"]
    assert not read(runs/"s4_recurrent_gpu_20261009_01/gpu_summary.json")["registered_numeric_limits_passed"]
    git = lambda *a:subprocess.check_output(["git",*a],cwd=repo)
    assert git("branch","--show-current").decode().strip() == "36tar/s4-mixed-state"
    git("diff","--check")
    tree = {str(p.relative_to(repo)):sha(p) for p in sorted(repo.rglob("*")) if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts}
    (args.out/"source_tree.json").write_text(json.dumps(tree,indent=2),encoding="utf-8")
    (args.out/"tracked_changes.patch").write_bytes(git("diff","--binary"))
    report = {"fp32_fixed_ten_checks_passed":True,"fork_main_continuation_verified":True,
        "input_and_source_provenance_matched":True,"all_old_executable_sources_unchanged":True,
        "short_cpu_protocol_probe_verified":True,"historical_bf16_failures_preserved":True,
        "wrong_preflight_rejected_before_launch":True,"source_files":len(tree),"no_inference_in_audit":True,
        "branch":"36tar/s4-mixed-state","base_commit":git("rev-parse","HEAD").decode().strip(),"commit_created":False,"remote_push":False}
    (args.out/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
