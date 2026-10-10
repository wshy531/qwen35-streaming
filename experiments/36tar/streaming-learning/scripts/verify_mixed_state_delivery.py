"""Read-only S4 evidence audit and lightweight review artifacts; no model forward."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT/"code/qwen35-streaming-s4"
RUNS = ROOT/"runs"
OUT = RUNS/"s4_delivery_check_20261009_01"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def git(*args):
    return subprocess.check_output(["git",*args],cwd=REPO)


def main():
    OUT.mkdir(exist_ok=False)
    originals = [p for p in (ROOT/"reference/qwen35-streaming").rglob("*") if p.is_file()]
    for p in originals:
        assert p.read_bytes() == (ROOT/"code/qwen35-streaming"/p.relative_to(ROOT/"reference/qwen35-streaming")).read_bytes()
    modified = []
    for p in originals:
        relative = p.relative_to(ROOT/"reference/qwen35-streaming")
        a,b = p.read_bytes().replace(b"\r\n",b"\n"),(REPO/relative).read_bytes().replace(b"\r\n",b"\n")
        if a != b:
            modified.append(relative.as_posix())
    assert sorted(modified) == ["THIRD_PARTY_NOTICES.md","qwen35_local/modeling_qwen3_5.py"]
    assert git("branch","--show-current").decode().strip() == "36tar/s4-mixed-state"
    assert git("rev-parse","HEAD").decode().strip() == "61e8de673fe9abb7281f803c1d1c94dc65f6a064"
    git("diff","--check")
    matched = {}
    for name in ("s4_cpu_20261009_01","s4_preflight_20261009_01","s4_gpu_20261009_01","s4_after_20261009_01"):
        proof = read(RUNS/name/"provenance.json")
        for relative,digest in proof["code_hashes"].items():
            assert sha(REPO/relative) == digest,relative
        matched[name] = len(proof["code_hashes"])
    assert read(RUNS/"s4_cpu_20261009_01/cpu_summary.json")["passed"]
    for name in ("s4_before_20261009_01","s4_after_20261009_01"):
        cases = read(RUNS/name/"diagnostic_summary.json")["cases"]
        assert len(cases) == 4
        assert sum(c["equivalent"] for c in cases) == (2 if "before" in name else 4)
    gpu = read(RUNS/"s4_gpu_20261009_01/gpu_summary.json")
    pre = read(RUNS/"s4_preflight_20261009_01/preflight_summary.json")
    assert gpu["input_fingerprints"] == pre["input_fingerprints"]
    assert gpu["generated_tokens"] == 0 and len(gpu["groups"]) == 10
    assert read(RUNS/"s4_gpu_20261009_01/status.json") == {"status":"failed","stage":"gpu"}
    watchdog = read(RUNS/"s4_gpu_20261009_01/watchdog_status.json")
    assert watchdog["exit_code"] == 1 and not watchdog["timed_out"]
    assert not gpu["registered_numeric_limits_passed"] and gpu["state_isolation_reset_capacity_passed"]
    for group in gpu["groups"][:7]:
        assert group["cache_lengths"][-1] == sum(group["chunks"])
    counts = sum(g["logits"]["registered_numeric_limits_passed"] for g in gpu["groups"][:7])
    assert counts == 1
    fork,reset,cap = gpu["groups"][7:]
    assert fork["main_unchanged_bitwise"] and not fork["logits"]["registered_numeric_limits_passed"]
    assert reset["all_state_and_positions_cleared"] and reset["logits"]["bitwise_equal"]
    assert all(x["all_tensors_bitwise_equal"] for x in reset["final_states"].values())
    assert cap["rejected_before_forward"] and cap["all_state_unchanged_bitwise"]
    script = REPO/"scripts/precision_diagnostic_cpu.py"
    current = script.read_bytes()
    # Preserve the exact first diagnostic revision, reconstructed and hash checked.
    v1 = current.decode("utf-8").replace("Qwen3_5RMSNorm,l2norm,torch_chunk_gated_delta_rule)",
                                        "Qwen3_5RMSNorm,torch_chunk_gated_delta_rule)")
    start = v1.index('    report = {"output":difference(torch.cat((first,last),1),full),"state":difference(s,state)}')
    end = v1.index("\n\n\n@torch.inference_mode()",start)
    v1 = v1[:start]+'    return {"output":difference(torch.cat((first,last),1),full),"state":difference(s,state)}'+v1[end:]
    for name,data in (("s4_precision_cpu_20261009_01",v1.encode("utf-8")),("s4_precision_cpu_20261009_02",current)):
        proof = read(RUNS/name/"provenance.json")
        assert hashlib.sha256(data).hexdigest() == proof["diagnostic_script_sha256"]
        target = RUNS/name/"script_snapshot.py"
        assert not target.exists()
        target.write_bytes(data)
    precision = read(RUNS/"s4_precision_cpu_20261009_02/precision_summary.json")
    assert precision["random_kernel"]["torch.bfloat16"]["normalization_shape_probe"]["query"]["max_abs_diff"] > 0
    assert precision["random_kernel"]["torch.float32"]["normalization_shape_probe"]["query"]["bitwise_equal"]
    negative_out = OUT/"rejected_asset_run"
    negative = subprocess.run([sys.executable,str(ROOT/"scripts/run_mixed_state.py"),"--stage","gpu",
        "--out",str(negative_out),"--cpu-check",str(RUNS/"s4_cpu_20261009_01"),
        "--preflight",str(RUNS/"s4_preflight_20261009_01"),"--video",str(ROOT/"datasets/natural_commons_v1/hands_vfr.mp4")],
        cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert negative.returncode == 2 and "video changed after preflight" in negative.stderr
    assert not negative_out.exists() and not negative_out.with_name(negative_out.name+".console.log").exists()
    write(OUT/"source_tree.json",{str(p.relative_to(REPO)):sha(p) for p in sorted(REPO.rglob("*"))
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts})
    (OUT/"tracked_changes.patch").write_bytes(git("diff","--binary"))
    write(OUT/"git_review.json",{"base_commit":git("rev-parse","HEAD").decode().strip(),
        "branch":git("branch","--show-current").decode().strip(),"status":git("status","--short").decode(),
        "remote_push":False,"commit_created":False,"write_permission_confirmed":False})
    write(OUT/"verification.json",{"original_baseline_files_unchanged":len(originals),
        "normalized_original_files_modified_in_s4":modified,"matched_run_code_hash_counts":matched,
        "gpu_input_fingerprints_match_preflight":True,"gpu_numerical_failure_preserved":True,
        "seven_comparison_groups_passed":counts,"fork_reset_capacity_management_passed":True,
        "two_precision_script_revisions_hash_verified":True,"changed_video_rejected_before_run":True,
        "no_model_forward_no_extra_gpu":True,"source_tree_sha256":sha(OUT/"source_tree.json"),
        "tracked_patch_sha256":sha(OUT/"tracked_changes.patch")})
    print((OUT/"verification.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
