"""Audit failed online GPU and CPU-only EOS candidate, with exact source snapshots."""
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
    names = ("s4b_online_preflight_20261009_01","s4b_online_gpu_20261009_01","s4b_online_preflight_20261009_02")
    snapshots = {sha(p):str(p.relative_to(root)) for name in names for p in (runs/name).glob("*_snapshot.py")}
    matched = {}
    for name in names:
        originals = read(runs/name/"provenance.json")["code_hashes"]
        historical = {}
        for path,digest in originals.items():
            if sha(repo/path) != digest:
                assert digest in snapshots,(name,path)
                historical[path] = snapshots[digest]
        matched[name] = historical
    gpu_dir = runs/names[1]
    gpu = read(gpu_dir/"online_summary.json")
    pre = read(runs/names[0]/"preflight_summary.json")
    candidate = read(runs/names[2]/"preflight_summary.json")
    assert read(gpu_dir/"status.json") == {"status":"failed","stage":"gpu"}
    assert read(gpu_dir/"watchdog_status.json")["exit_code"] == 1 and not read(gpu_dir/"watchdog_status.json")["timed_out"]
    assert gpu["actual_forwards"] == {"language":31,"vision":8} and gpu["generated_tokens"] == 16
    assert gpu["budget"] == pre["budget"] == candidate["budget"]
    assert gpu["input_fingerprints"] == pre["input_fingerprints"] == candidate["input_fingerprints"]
    assert pre["cpu_tests"] == 12 and candidate["cpu_tests"] == 13
    assert pre["cpu_controls_passed"] and candidate["cpu_controls_passed"]
    assert sha(root/"在线成对帧问答验收方案.md") == pre["plan_sha256"] == candidate["plan_sha256"]
    assert sha(root/"scripts/run_online_pairs.py") == pre["launcher_sha256"] == candidate["launcher_sha256"]
    assert not gpu["all_interface_numeric_groups_passed"]
    assert [g["id"] for g in gpu["groups"] if g["passed"]] == ["vision_pair1","prefix2","qa2","vision_pair2","reset_then_pair"]
    assert [g["id"] for g in gpu["groups"] if not g["passed"]] == ["prefix4","qa4","eof_odd_padding"]
    for group in (gpu["groups"][2],gpu["groups"][5]):
        assert group["main_unchanged_bitwise"] and group["online"]["generated_ids"] == group["reference_ids"]
        assert group["online"]["generated_ids"][-3:] == [248046,198,248044]
    assert gpu["groups"][2]["online"]["answer"].strip() == "CAT"
    assert gpu["groups"][5]["online"]["answer"].strip() == "ONE"
    assert gpu["groups"][-1]["input_sealed"] and gpu["groups"][-1]["padding_record"]["synthetic_padding"]
    old = read(runs/"s4_fp32_regression_delivery_check_20261009_02/source_tree.json")
    for name,digest in old.items():
        if name != "S4_A.md":
            assert sha(repo/name) == digest,name
    refs = list((root/"reference/qwen35-streaming").rglob("*"))
    for p in refs:
        if p.is_file():
            assert p.read_bytes() == (root/"code/qwen35-streaming"/p.relative_to(root/"reference/qwen35-streaming")).read_bytes()
    negative = args.out/"wrong_preflight"
    result = subprocess.run([sys.executable,str(root/"scripts/run_online_pairs.py"),"--stage","gpu","--out",str(negative),
        "--preflight",str(runs/"s4_fp32_regression_preflight_20261009_01")],capture_output=True,text=True,timeout=30)
    assert result.returncode == 2 and "CPU controls or registered budget mismatch" in result.stderr
    assert not negative.exists() and not negative.with_name(negative.name+".console.log").exists()
    git = lambda *a:subprocess.check_output(["git",*a],cwd=repo)
    assert git("branch","--show-current").decode().strip() == "36tar/s4-mixed-state"
    git("diff","--check")
    tree = {str(p.relative_to(repo)):sha(p) for p in sorted(repo.rglob("*")) if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts}
    (args.out/"source_tree.json").write_text(json.dumps(tree,indent=2),encoding="utf-8")
    (args.out/"tracked_changes.patch").write_bytes(git("diff","--binary"))
    report = {"matched_provenance":matched,"failed_gpu_five_of_eight_preserved":True,"gpu_actual_budget_verified":True,
        "cpu12_then_eos_candidate_cpu13_verified":True,"candidate_has_no_new_gpu_claim":True,"wrong_preflight_rejected_before_launch":True,
        "cat_one_counterexamples_preserved":True,"input_fingerprints_match":True,"old_executable_sources_unchanged":True,
        "source_files":len(tree),"no_inference_in_audit":True,"branch":"36tar/s4-mixed-state",
        "base_commit":git("rev-parse","HEAD").decode().strip(),"commit_created":False,"remote_push":False}
    (args.out/"verification.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
