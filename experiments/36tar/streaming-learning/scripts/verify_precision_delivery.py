"""Audit precision runs against current or exact retained historical sources."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[1]
    repo,runs = root/"code/qwen35-streaming-s4",root/"runs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    names = ["s4_cpu_20261009_01","s4_preflight_20261009_01","s4_gpu_20261009_01",
        "s4_layers_preflight_20261009_01","s4_layers_preflight_20261009_02","s4_layers_gpu_20261009_01",
        "s4_recurrence_cpu_20261009_01","s4_recurrent_cpu_20261009_01","s4_recurrent_cpu_20261009_02",
        "s4_recurrent_preflight_20261009_01","s4_recurrent_gpu_20261009_01"]
    historical = list((runs/"s4_before_recurrent_20261009_01").glob("*.py"))
    historical += [runs/"s4_recurrent_cpu_20261009_01/chunk_weight_check_snapshot.py"]
    historical += [runs/name/"script_snapshot.py" for name in names if (runs/name/"script_snapshot.py").exists()]
    versions = {sha(p):str(p.relative_to(root)) for p in historical}
    matched = {}
    for name in names:
        proof = read(runs/name/"provenance.json")
        current_count,snapshots = 0,{}
        for relative,digest in proof["code_hashes"].items():
            if sha(repo/relative) == digest:
                current_count += 1
            else:
                assert digest in versions,(name,relative,digest)
                snapshots[relative] = versions[digest]
        matched[name] = {"current_source_count":current_count,"exact_historical_sources":snapshots}
    originals = [p for p in (root/"reference/qwen35-streaming").rglob("*") if p.is_file()]
    for path in originals:
        assert path.read_bytes() == (root/"code/qwen35-streaming"/path.relative_to(root/"reference/qwen35-streaming")).read_bytes()
    layers = read(runs/"s4_layers_gpu_20261009_01/precision_layers_summary.json")
    pre = read(runs/"s4_layers_preflight_20261009_02/preflight_summary.json")
    assert layers["input_fingerprints"] == pre["input_fingerprints"]
    assert len(layers["groups"]) == 4 and layers["norm_override_restored"]
    assert not layers["candidate_two_groups_passed"]
    for group in layers["groups"][:2]:
        for field in ("in_proj_qkv","in_proj_a","in_proj_b","query","key","value","g","beta","query_norm","key_norm"):
            assert group["first_linear_layer"]["linear.0."+field]["bitwise_equal"]
        assert not group["first_linear_layer"]["linear.0.core"]["bitwise_equal"]
    old = read(runs/"s4_gpu_20261009_01/gpu_summary.json")
    for a,b in zip(layers["groups"][:2],(old["groups"][0],old["groups"][2])):
        assert a["logits"] == b["logits"]
    cpu = read(runs/"s4_recurrent_cpu_20261009_02/cpu_summary.json")
    assert cpu["passed"] and cpu["tests"] == 29 and cpu["recurrent_reference"]
    candidate = read(runs/"s4_recurrent_gpu_20261009_01/gpu_summary.json")
    pre = read(runs/"s4_recurrent_preflight_20261009_01/preflight_summary.json")
    assert candidate["input_fingerprints"] == pre["input_fingerprints"] == old["input_fingerprints"]
    assert candidate["recurrent_reference"] and len(candidate["groups"]) == 10
    assert candidate["generated_tokens"] == 0 and not candidate["registered_numeric_limits_passed"]
    assert candidate["state_isolation_reset_capacity_passed"]
    assert len(candidate["original_prefill_controls"]) == 3
    assert sum(g["logits"]["registered_numeric_limits_passed"] for g in candidate["groups"][:7]) == 1
    assert read(runs/"s4_recurrent_gpu_20261009_01/status.json") == {"status":"failed","stage":"gpu"}
    assert not read(runs/"s4_recurrent_gpu_20261009_01/watchdog_status.json")["timed_out"]
    recurrence = read(runs/"s4_recurrence_cpu_20261009_01/recurrence_summary.json")
    assert all(v["bitwise_equal"] for v in recurrence["groups"][3]["comparisons"].values())
    negative_out = args.out/"mode_mismatch_run"
    result = subprocess.run([sys.executable,str(root/"scripts/run_mixed_state.py"),"--stage","gpu","--out",str(negative_out),
        "--cpu-check",str(runs/"s4_recurrent_cpu_20261009_02"),"--preflight",str(runs/"s4_recurrent_preflight_20261009_01")],
        cwd=root,capture_output=True,text=True,timeout=30)
    assert result.returncode == 2 and "Precision rule changed after preflight" in result.stderr
    assert not negative_out.exists() and not negative_out.with_name(negative_out.name+".console.log").exists()
    git = lambda *a:subprocess.check_output(["git",*a],cwd=repo)
    assert git("branch","--show-current").decode().strip() == "36tar/s4-mixed-state"
    assert git("rev-parse","HEAD").decode().strip() == "61e8de673fe9abb7281f803c1d1c94dc65f6a064"
    git("diff","--check")
    tree = {str(p.relative_to(repo)):sha(p) for p in sorted(repo.rglob("*"))
        if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts}
    write(args.out/"source_tree.json",tree)
    (args.out/"tracked_changes.patch").write_bytes(git("diff","--binary"))
    write(args.out/"git_review.json",{"branch":"36tar/s4-mixed-state","base_commit":git("rev-parse","HEAD").decode().strip(),
        "status":git("status","--short").decode(),"commit_created":False,"remote_push":False})
    write(args.out/"verification.json",{"original_baseline_files_unchanged":len(originals),
        "matched_provenance":matched,"original_gpu_two_groups_exactly_reproduced":True,
        "first_original_gpu_difference_at_core":True,"norm_fp32_rejected":True,
        "recurrent_cpu29_pass_gpu_failure_preserved":True,"all_gpu_input_fingerprints_match":True,
        "recurrent_original_prefill_shifts_recorded":True,"mode_mismatch_rejected_before_run":True,
        "source_files":len(tree),"no_forward_or_extra_gpu_in_audit":True})
    print({"source_files":len(tree),"original_files":len(originals),"matched_runs":len(names),
           "mode_mismatch_rejected":True,"gpu_failures_retained":True})


if __name__ == "__main__":
    main()
