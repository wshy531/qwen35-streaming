"""Verify fixed FP32 text evidence and preserve BF16 failures without inference."""
import argparse
from pathlib import Path
import subprocess
import sys
from run_precision_layers import read, sha
import json


def main():
    root = Path(__file__).resolve().parents[1]
    repo,runs = root/"code/qwen35-streaming-s4",root/"runs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    pre_dir = runs/"s4_full_precision_preflight_20261009_01"
    gpu_dir = runs/"s4_full_precision_gpu_20261009_01"
    pre,gpu = read(pre_dir/"preflight_summary.json"),read(gpu_dir/"full_precision_summary.json")
    for folder,stage in ((pre_dir,"preflight"),(gpu_dir,"gpu")):
        assert read(folder/"status.json") == {"status":"completed","stage":stage}
        assert not read(folder/"watchdog_status.json")["timed_out"]
        for name,digest in read(folder/"provenance.json")["code_hashes"].items():
            assert sha(repo/name) == digest,(folder,name)
        assert sha(folder/"script_snapshot.py") == read(folder/"script_snapshot.py.json")["sha256"]
    assert sha(root/"高精度整模型诊断方案.md") == pre["plan_sha256"]
    assert sha(root/"scripts/run_precision_fullmodel.py") == pre["launcher_sha256"]
    assert pre["cpu_controls"]["passed"] and gpu["backend_restored"]
    assert pre["input_fingerprints"] == gpu["input_fingerprints"] == read(runs/"s4_layers_preflight_20261009_02/preflight_summary.json")["input_fingerprints"]
    assert gpu["budget"] == pre["budget"]
    assert len(gpu["groups"]) == 4 and gpu["language_layers"] == 24 and gpu["parameter_count"] == 852985920
    assert [{k:g[k] for k in ("mode","input","chunks")} for g in gpu["groups"]] == pre["groups"]
    original = read(runs/"s4_gpu_20261009_01/gpu_summary.json")["groups"][0]["logits"]
    assert all(gpu["groups"][0]["logits"][k] == v for k,v in original.items())
    assert not gpu["bf16_reduction_control_passed"] and gpu["fp32_two_groups_passed"]
    assert not gpu["groups"][0]["logits"]["registered_numeric_limits_passed"]
    assert gpu["groups"][0]["active_backend_settings"]["bf16_reduction"]
    assert not gpu["groups"][1]["active_backend_settings"]["bf16_reduction"]
    replay_calls = 0
    for group in gpu["groups"]:
        assert group["cache_length"] == sum(group["chunks"])
        assert [group["final_states"][name]["populated_layers"] for name in ("key_cache","value_cache","conv_states","recurrent_states")] == [6,6,18,18]
        assert len([k for k in group["traced_tensors"] if k.startswith("layer.")]) == 24
        replay = group["same_input_component_replay"]
        assert replay["identical_input_gated_norm"]["bitwise_equal"]
        replay_calls += replay["calls"]
    assert replay_calls == gpu["budget"]["component_replay_calls"] == 26
    assert sum(1+len(g["chunks"]) for g in gpu["groups"]) == 13
    for group in gpu["groups"][2:]:
        assert group["logits"]["fp32_reference_limits_passed"] and group["logits"]["allclose_1e4"]
        assert all(x["allclose_1e4"] for x in group["final_states"].values())
        assert set(group["parameter_dtype_counts"]) == {"torch.float32"}
        active = group["active_backend_settings"]
        assert active["matmul_fp32"] == active["conv_fp32"] == "ieee"
        assert active["math_sdp"] and not active["flash_sdp"] and not active["mem_efficient_sdp"]
    for name,digest in read(runs/"s4_precision_delivery_check_20261009_01/source_tree.json").items():
        if name == "S4_A.md":
            continue  # Human-facing report updated; all earlier executable source retained exactly.
        assert sha(repo/name) == digest,name
    # Wrong experiment preflight must reject before creating a run/launching GPU.
    rejected = args.out/"wrong_preflight"
    result = subprocess.run([sys.executable,str(root/"scripts/run_precision_fullmodel.py"),"--stage","gpu","--out",str(rejected),
        "--preflight",str(runs/"s4_layers_preflight_20261009_02")],capture_output=True,text=True,timeout=30)
    assert result.returncode == 2 and "CPU controls or budget mismatch" in result.stderr
    assert not rejected.exists() and not rejected.with_name(rejected.name+".console.log").exists()
    git = lambda *a:subprocess.check_output(["git",*a],cwd=repo)
    assert git("branch","--show-current").decode().strip() == "36tar/s4-mixed-state"
    git("diff","--check")
    tree = {str(p.relative_to(repo)):sha(p) for p in sorted(repo.rglob("*")) if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts}
    (args.out/"source_tree.json").write_text(json.dumps(tree,indent=2),encoding="utf-8")
    (args.out/"tracked_changes.patch").write_bytes(git("diff","--binary"))
    report = {"four_groups_verified":True,"original_bf16_reproduced_and_failure_preserved":True,
        "bf16_reduction_control_failure_preserved":True,"fp32_text_two_groups_and_four_states_passed":True,
        "same_input_norm_and_projection_replay_checked":True,"wrong_preflight_rejected_before_launch":True,
        "old_executable_sources_unchanged":True,"source_files":len(tree),"no_inference_in_audit":True,
        "branch":"36tar/s4-mixed-state","head":git("rev-parse","HEAD").decode().strip(),
        "commit_created":False,"remote_push":False}
    (args.out/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
