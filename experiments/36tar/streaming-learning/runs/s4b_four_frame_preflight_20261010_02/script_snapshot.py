"""Fixed four-frame subject QA, shared exact inputs, repeated QA and real reset replay."""
import argparse
import itertools
import json
from pathlib import Path
import tempfile
import time
import traceback

import psutil
import torch
from transformers import AutoProcessor

from scripts.baseline import PROJECT,REPO,REVISION,provenance,write_json,file_hash
from scripts.fp32_regression import strict_logits,strict_states
from scripts.online_experiment import reference_inputs,reference_answer,pair_record,numerical_pass,Q1
from scripts.shared_features_control import Capture,shared_input
from streaming.inputs import tensor_fingerprint
from streaming.online import OnlinePairSession,fp32_reference
from streaming.window import read_packets

PLAN = PROJECT/"在线四帧主体公平对照方案.md"
LAUNCHER = PROJECT/"scripts/run_four_frame_online.py"
BUDGET = {"groups":4,"max_tokens":256,"max_new_tokens_per_answer":8,"generated_tokens_max":32,
          "language_forwards_max":43,"vision_encodings":4,"model_loads":1,"wall_seconds":600}


def exact_states(left,right):
    return all(x["all_tensors_bitwise_equal"] for x in strict_states(left,right).values())


@torch.inference_mode()
def controls(model,processor,reader,canvas,out=None,proof=None):
    device = model.get_input_embeddings().weight.device
    capture = Capture(model)
    session = OnlinePairSession(model,processor,canvas=canvas)
    frames,groups = [],[]
    generated = 0
    report = {"groups":groups,"budget":BUDGET,"question":Q1,"canvas":list(canvas),
        "scope":"one fixed four-frame causal replay, repeated isolated QA and reset; not a long-run/live benchmark",
        "experiment_query_trigger":"after four delivered frames; underlying two-frame ask contract unchanged"}
    def append_actual(packets):
        before = session.main.seen_tokens
        capture.active = True
        try:
            results = [session.append(p) for p in packets]
        finally:
            capture.active = False
        return results[-1]["logits"],before
    def identity_inputs():
        inputs = reference_inputs(model,processor,frames,canvas)
        qinputs = reference_inputs(model,processor,frames,canvas,Q1)
        fingerprints = {"prefix4":tensor_fingerprint(inputs),"question4_subject":tensor_fingerprint(qinputs)}
        if proof:
            assert fingerprints == proof["input_fingerprints"]
        # BatchFeature.to mutates its own mapping. Keep the independent CPU
        # template untouched; it is later compared against CPU suffix IDs.
        device_inputs = type(inputs)({key:value.to(device) for key,value in inputs.items()})
        actual,identity = shared_input(model,processor,session,device_inputs,capture.features,capture.embeddings,capture.positions)
        assert all(value.device.type == "cpu" for value in inputs.values())
        return inputs,qinputs,actual,identity,fingerprints
    def ask_isolated():
        snapshot = session.main.cache.clone()
        count = session.main.seen_tokens,session.main.next_rope_position
        answer = session.ask(Q1,8,return_evidence=True)
        evidence = answer.pop("evidence")
        unchanged = exact_states(session.main.cache,snapshot) and count == (session.main.seen_tokens,session.main.next_rope_position)
        return answer,evidence,unchanged
    try:
        with fp32_reference():
            # The first question is issued before requesting any fifth sample.
            frames.extend((next(reader),next(reader)))
            append_actual(frames)
            assert session.main.seen_tokens > 0
            frames.extend((next(reader),next(reader)))
            logits,start = append_actual(frames[2:])
            inputs,qinputs,actual,identity,fingerprints = identity_inputs()
            full = model(inputs_embeds=actual,position_ids=inputs.position_ids.to(device),use_cache=True)
            record = pair_record(logits,session,full,start)
            groups.append({"id":"four_frame_prefix_shared",**record,"input_identity":identity,"passed":numerical_pass(record)})
            if out:
                write_json(out/"four_frame_online_summary.json",{**report,"partial":True,
                    "actual_forwards":capture.counts.copy(),"generated_tokens":generated})
            main_snapshot = session.main.cache.clone()
            original_status = session.status()
            answer,first_evidence,unchanged = ask_isolated()
            assert torch.equal(qinputs.input_ids,torch.cat((inputs.input_ids,first_evidence["suffix_ids"]),1))
            reference = reference_answer(model,processor,actual,inputs.position_ids.to(device),Q1,8)
            generated += len(answer["generated_ids"])+len(reference["generated_ids"])
            record = {"id":"four_frame_subject_shared","online":answer,"reference_answer":reference["answer"],
                "reference_ids":reference["generated_ids"],"main_unchanged_bitwise_and_counters":unchanged,
                "logits":strict_logits(first_evidence["initial_logits"],reference["initial_logits"]),
                "final_states":strict_states(first_evidence["branch_cache"],reference["cache"])}
            record["passed"] = numerical_pass(record) and unchanged and answer["generated_ids"] == reference["generated_ids"]
            groups.append(record)
            if out:
                write_json(out/"four_frame_online_summary.json",{**report,"partial":True,
                    "actual_forwards":capture.counts.copy(),"generated_tokens":generated})
            repeat,repeated_evidence,unchanged = ask_isolated()
            generated += len(repeat["generated_ids"])
            record = {"id":"same_prefix_repeated_subject","online":repeat,"main_unchanged_bitwise_and_counters":unchanged,
                "ids_match_first":repeat["generated_ids"] == answer["generated_ids"],
                "initial_logits_bitwise_equal":torch.equal(first_evidence["initial_logits"],repeated_evidence["initial_logits"]),
                "branch_states_bitwise_equal":exact_states(first_evidence["branch_cache"],repeated_evidence["branch_cache"])}
            record["passed"] = all(record[key] for key in ("main_unchanged_bitwise_and_counters","ids_match_first",
                "initial_logits_bitwise_equal","branch_states_bitwise_equal"))
            groups.append(record)
            if out:
                write_json(out/"four_frame_online_summary.json",{**report,"partial":True,
                    "actual_forwards":capture.counts.copy(),"generated_tokens":generated})
            saved_features = [x.clone() for x in capture.features]
            del full,reference,repeated_evidence
            session.reset()
            reset_status = session.status()
            reset_cleared = (session.main.seen_tokens == session.main.next_rope_position == session.accepted_frames == session.encoded_frames == 0
                and session.pending is None and not session.ledger and session.vision_encodings == 0 and
                model.model.rope_deltas is None and not session.invalid and not session.finished and not session.closed and
                all(x is None for field in ("key_cache","value_cache","conv_states","recurrent_states") for x in getattr(session.main.cache,field)))
            assert reset_cleared
            capture.features.clear(); capture.embeddings.clear(); capture.positions.clear()
            # Explicit replay of already known frames after reset, not new future observations.
            append_actual(frames[:2])
            append_actual(frames[2:])
            second_inputs,second_qinputs,second_actual,second_identity,second_fingerprints = identity_inputs()
            reset_answer,reset_evidence,unchanged = ask_isolated()
            generated += len(reset_answer["generated_ids"])
            record = {"id":"reset_replay_four_frame_subject","online":reset_answer,"reset_empty_contract":reset_cleared,
                "reset_status":reset_status,"input_identity":second_identity,"input_fingerprints_equal":fingerprints == second_fingerprints,
                "actual_features_bitwise_equal":len(saved_features) == len(capture.features) and all(torch.equal(a,b) for a,b in zip(saved_features,capture.features)),
                "complete_embeddings_bitwise_equal":torch.equal(actual,second_actual),"main_after_replay_bitwise_equal":exact_states(main_snapshot,session.main.cache),
                "main_status_after_replay_equal":session.status() == original_status,"main_unchanged_bitwise_and_counters":unchanged,
                "ids_match_first":reset_answer["generated_ids"] == answer["generated_ids"],
                "initial_logits_bitwise_equal":torch.equal(first_evidence["initial_logits"],reset_evidence["initial_logits"]),
                "branch_states_bitwise_equal":exact_states(first_evidence["branch_cache"],reset_evidence["branch_cache"])}
            record["passed"] = all(record[key] for key in ("reset_empty_contract","input_fingerprints_equal","actual_features_bitwise_equal",
                "complete_embeddings_bitwise_equal","main_after_replay_bitwise_equal","main_status_after_replay_equal",
                "main_unchanged_bitwise_and_counters","ids_match_first","initial_logits_bitwise_equal","branch_states_bitwise_equal"))
            groups.append(record)
        assert len(groups) == 4 and capture.counts["vision"] == 4 and capture.counts["language"] <= 43 and generated <= 32
        report.update(all_numeric_groups_passed=all(g["passed"] for g in groups),actual_forwards=capture.counts,generated_tokens=generated,
            input_fingerprints=fingerprints,frame_ids=[p.frame_id for p in frames],timestamps=[p.timestamp for p in frames],
            question_cutoff={"frame_id":frames[-1].frame_id,"timestamp":frames[-1].timestamp},
            semantic_expected_ids_match=None if proof is None else all(groups[i]["online"]["generated_ids"] == proof["expected_original_ids"] for i in (1,2,3)))
        if out:
            write_json(out/"four_frame_online_summary.json",report)
        print({"all_numeric_groups_passed":report["all_numeric_groups_passed"],"answer":answer["answer"],
            "actual_forwards":capture.counts,"generated_tokens":generated},flush=True)
        return report
    finally:
        session.close()
        capture.close()


def preflight(out,args):
    from tests.test_online import small_model
    torch.set_num_threads(1)
    manifest = json.loads((args.model_dir/"DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8-sig"))
    assert manifest["revision"] == REVISION
    for file in manifest["files"]:
        path = args.model_dir/file["name"]
        assert path.stat().st_size == file["bytes"] and file_hash(path) == file["sha256"].lower()
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    model = small_model(processor)
    hooks_before = len(model.model.visual._forward_hooks),len(model.model.language_model._forward_pre_hooks)
    reader = read_packets(args.video,1)
    try:
        cpu = controls(model,processor,reader,(32,32))
    finally:
        reader.close()
    assert cpu["all_numeric_groups_passed"]
    assert hooks_before == (len(model.model.visual._forward_hooks),len(model.model.language_model._forward_pre_hooks))
    write_json(out/"cpu_summary.json",cpu)
    reader = read_packets(args.video,1)
    try:
        frames = tuple(itertools.islice(reader,4))
    finally:
        reader.close()
    fingerprints = {"prefix4":tensor_fingerprint(reference_inputs(model,processor,frames,(192,256))),
        "question4_subject":tensor_fingerprint(reference_inputs(model,processor,frames,(192,256),Q1))}
    historical = json.loads((PROJECT/"runs/s4b_protocol_capability_preflight_20261009_01/preflight_summary.json").read_text())
    spec = next(x for x in historical["input_comparisons"] if x["id"] == "frames4_subject")
    assert fingerprints["question4_subject"] == spec["input_fingerprint"]
    original = json.loads((PROJECT/"runs/s4b_protocol_capability_gpu_20261009_01/protocol_capability_summary.json").read_text())
    expected = next(x for x in original["cases"] if x["id"] == "fp32_ieee_math_frames4_subject")
    assert expected["answer"] == "HANDS" and expected["generated_ids"] == [72141,50,248046]
    write_json(out/"preflight_summary.json",{"cpu_controls_passed":True,"cpu_groups":4,"hooks_cleanup_verified":True,"budget":BUDGET,
        "input_fingerprints":fingerprints,"expected_original_ids":expected["generated_ids"],"expected_original_answer":expected["answer"],
        "original_gpu_summary_sha256":file_hash(PROJECT/"runs/s4b_protocol_capability_gpu_20261009_01/protocol_capability_summary.json"),
        "video_sha256":file_hash(args.video),"model_manifest_sha256":file_hash(args.model_dir/"DOWNLOAD_MANIFEST.json"),
        "all_model_assets_verified":len(manifest["files"]),"plan_sha256":file_hash(PLAN),"launcher_sha256":file_hash(LAUNCHER)})
    print({"cpu_groups_passed":4,"real_subject_input_matches_original":True,"budget":BUDGET},flush=True)


@torch.inference_mode()
def gpu(out,args):
    from qwen35_local import Qwen3_5ForConditionalGeneration
    begin = time.perf_counter()
    torch.manual_seed(42)
    free,total = torch.cuda.mem_get_info()
    guard = {"gpu_free_bytes":free,"gpu_total_bytes":total,"ram_available_bytes":psutil.virtual_memory().available}
    write_json(out/"resource_precheck.json",guard)
    assert free >= 5.5*2**30 and guard["ram_available_bytes"] >= 3*2**30,"Resource guard; no load"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir,local_files_only=True,dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    assert torch.cuda.mem_get_info()[0] >= 2*2**30
    model.float()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(not x.linear_attn.experimental_recurrent_reference for x in model.model.language_model.layers if x.layer_type == "linear_attention")
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    proof = json.loads((args.preflight/"preflight_summary.json").read_text())
    reader = read_packets(args.video,1)
    try:
        report = controls(model,processor,reader,(192,256),out=out,proof=proof)
    finally:
        reader.close()
    report.update(wall_seconds_including_load=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,resource_precheck=guard)
    write_json(out/"four_frame_online_summary.json",report)
    if not report["all_numeric_groups_passed"] or not report["semantic_expected_ids_match"]:
        raise RuntimeError("Fixed four-frame numeric/expected answer check failed; preserve evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("preflight","gpu"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--model-dir",type=Path,default=PROJECT/"models/Qwen3.5-0.8B")
    parser.add_argument("--video",type=Path,default=PROJECT/"datasets/natural_commons_v1/clean_hands.webm")
    parser.add_argument("--preflight",type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"tmp").mkdir()
    tempfile.tempdir = str((args.out/"tmp").resolve())
    write_json(args.out/"provenance.json",provenance())
    (args.out/"script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    write_json(args.out/"arguments.json",{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    try:
        (preflight if args.stage == "preflight" else gpu)(args.out,args)
        write_json(args.out/"status.json",{"status":"completed","stage":args.stage})
    except Exception:
        (args.out/"traceback.txt").write_text(traceback.format_exc(),encoding="utf-8")
        write_json(args.out/"status.json",{"status":"failed","stage":args.stage})
        raise


if __name__ == "__main__":
    main()
