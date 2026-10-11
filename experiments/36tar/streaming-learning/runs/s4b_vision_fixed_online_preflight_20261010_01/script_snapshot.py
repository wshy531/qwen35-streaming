"""Explicit fixed-unit candidate regression; original online source remains untouched."""
import argparse
import gc
import itertools
import json
from pathlib import Path
import time
import traceback

import psutil
import torch
from transformers import AutoProcessor

from scripts.baseline import PROJECT, provenance, write_json, file_hash
from scripts.chunk_weight_check import encoded
from scripts.fp32_regression import strict_logits, strict_states
from scripts.online_experiment import Q1, Q2, reference_inputs, reference_answer, pair_record, numerical_pass
from scripts.pair_demo import verify_assets
from scripts.shared_features_control import controls as shared_controls
from scripts.vision_trace import hook_state
from scripts.vision_fixed_control import forward_attributes
from scripts.vision_fixed_units import FixedTemporalLinears
from streaming.inputs import tensor_fingerprint
from streaming.online import OnlinePairSession, fp32_reference
from streaming.window import read_packets

PLAN = PROJECT / "固定视觉在线回归方案.md"
LAUNCHER = PROJECT / "scripts/run_vision_fixed_online.py"
HISTORY = PROJECT / "runs/s4b_vision_fixed_gpu_20261010_01/vision_fixed_summary.json"
BUDGET = {"groups": 8, "max_tokens": 256, "max_new_tokens_per_answer": 8,
          "generated_tokens_max": 32, "language_forwards_max": 47, "vision_encodings": 8,
          "candidate_projection_operations_max": 500, "model_loads": 1, "wall_seconds": 600,
          "trace_buffer_bytes": 128 * 2**20}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


@torch.inference_mode()
def preflight(out, args):
    from tests.test_online import small_model
    torch.set_num_threads(1)
    manifest = verify_assets(args.model_dir)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = small_model(processor)
    before, attrs = hook_state(model), forward_attributes(model)
    reader = read_packets(args.video, 1)
    try:
        with FixedTemporalLinears(model.model.visual, max_operations=500) as fixed:
            cpu = shared_controls(model, processor, reader, (32, 32))
    finally:
        reader.close()
    assert cpu["all_shared_groups_passed"]
    biases = [r for r in cpu["groups"] if r["kind"] == "diagnostic_different_inputs"]
    assert len(biases) == 2 and all(r["within_old_strict_language_gate"] and r["online_vs_whole"]["passed"] for r in biases)
    assert all(r["features"]["bitwise_equal"] for r in biases)
    assert before == hook_state(model) and attrs == forward_attributes(model)
    reader = read_packets(args.video, 1)
    try:
        frames = tuple(itertools.islice(reader, 4))
    finally:
        reader.close()
    fingerprints = {name: tensor_fingerprint(reference_inputs(model, processor, subset, (192, 256), question))
                    for name, subset, question in (("prefix2", frames[:2], None), ("prefix4", frames, None),
                        ("question2", frames[:2], Q1), ("question4", frames, Q2), ("eof3", frames[:3], None))}
    assert fingerprints == read(PROJECT / "runs/s4b_shared_gpu_20261009_01/shared_features_summary.json")["input_fingerprints"]
    assert read(HISTORY)["candidate_all_stages_bitwise"]
    cpu.update(hooks_and_methods_restored=True, input_fingerprints=fingerprints,
               independent_vision_controls_passed=True, candidate_projection_operations=fixed.operations,
               scope="tiny random candidate, CPU interface only")
    write_json(out / "cpu_summary.json", cpu)
    write_json(out / "preflight_summary.json", {"cpu_controls_passed": True, "budget": BUDGET,
        "input_fingerprints": fingerprints, "cpu_report_sha256": file_hash(out / "cpu_summary.json"),
        "video_sha256": file_hash(args.video), "model_manifest_sha256": file_hash(args.model_dir / "DOWNLOAD_MANIFEST.json"),
        "verified_model_assets": manifest["files"], "historical_report_sha256": file_hash(HISTORY),
        "plan_sha256": file_hash(PLAN), "launcher_sha256": file_hash(LAUNCHER),
        "original_protocol_source_sha256": file_hash(Path(__file__).with_name("online_experiment.py"))})
    print({"cpu_shared_strict_groups": 5, "cpu_independent_vision_groups": 2, "restoration": True}, flush=True)


@torch.inference_mode()
def gpu(out,args):
    from transformers import AutoProcessor
    from qwen35_local import Qwen3_5ForConditionalGeneration
    torch.manual_seed(42)
    begin = time.perf_counter()
    free,total = torch.cuda.mem_get_info()
    guard = {"gpu_free_bytes":free,"gpu_total_bytes":total,"ram_available_bytes":psutil.virtual_memory().available}
    write_json(out/"resource_precheck.json",guard)
    assert free >= 5.5*2**30 and guard["ram_available_bytes"] >= 3*2**30,"Memory guard; no loading"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir,local_files_only=True,dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    assert torch.cuda.mem_get_info()[0] >= 2*2**30 and psutil.virtual_memory().available >= 2*2**30
    model.float()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(not layer.linear_attn.experimental_recurrent_reference for layer in model.model.language_model.layers
               if layer.layer_type == "linear_attention")
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    proof = json.loads((args.preflight/"preflight_summary.json").read_text(encoding="utf-8-sig"))
    before_hooks, before_methods = hook_state(model), forward_attributes(model)
    counters = {"language":0,"vision":0}
    visual_parts = []
    def count_language(_m,_a,_o):
        counters["language"] += 1
    def capture_vision(_m,_a,output):
        counters["vision"] += 1
        visual_parts.append(output.pooler_output.detach().cpu())
    def guard_language(_module,_args):
        if counters["language"] >= BUDGET["language_forwards_max"]:
            raise RuntimeError("Language budget exceeded before forward")
    def guard_vision(_module,_args):
        if counters["vision"] >= BUDGET["vision_encodings"]:
            raise RuntimeError("Vision budget exceeded before forward")
    hooks = [model.model.language_model.register_forward_hook(count_language),model.model.visual.register_forward_hook(capture_vision),
        model.model.language_model.register_forward_pre_hook(guard_language),model.model.visual.register_forward_pre_hook(guard_vision)]
    reader = read_packets(args.video,1)
    session = OnlinePairSession(model,processor)
    frames,groups,fingerprints = [],[],{}
    generated_count = 0
    report = {"groups":groups,"budget":BUDGET,"resource_precheck":guard,"questions":[Q1,Q2],
        "scope":"causal sequential file replay with new-pair encoding and real isolated QA; not live camera/long-running service"}
    try:
        with fp32_reference(), FixedTemporalLinears(model.model.visual, max_operations=500) as fixed:
            for pair_index,question in enumerate((Q1,Q2)):
                before = session.main.seen_tokens
                first = next(reader)
                second = next(reader)
                frames.extend((first,second))
                waiting = session.append(first)
                assert waiting["logits"] is None
                blocked = False
                try:
                    session.ask(question,8)
                except ValueError:
                    blocked = True
                assert blocked
                new = session.append(second)
                online_features = visual_parts.pop()
                inputs = reference_inputs(model,processor,frames,(192,256))
                key = f"prefix{len(frames)}"
                fingerprints[key] = tensor_fingerprint(inputs)
                assert fingerprints[key] == proof["input_fingerprints"][key]
                inputs = inputs.to("cuda")
                embeddings = encoded(model,inputs)
                whole_features = visual_parts.pop()
                full = model(inputs_embeds=embeddings,position_ids=inputs.position_ids,use_cache=True)
                count = online_features.shape[0]
                feature_diff = strict_logits(online_features[None],whole_features[-count:][None])
                feature_pass = feature_diff["max_abs_diff"] <= 1e-4 and feature_diff["allclose_1e4"]
                if pair_index == 1:
                    past_diff = strict_logits(first_features[None],whole_features[:first_features.shape[0]][None])
                    feature_pass = feature_pass and past_diff["max_abs_diff"] <= 1e-4 and past_diff["allclose_1e4"]
                else:
                    first_features = whole_features
                    first_full = full
                groups.append({"id":f"vision_pair{pair_index+1}","features":feature_diff,"passed":feature_pass,
                    **({"past_features_vs_later_full_video":past_diff} if pair_index else {})})
                start = before if before else session.builder.prefix_ids.shape[1]
                record = pair_record(new["logits"],session,full,start)
                groups.append({"id":f"prefix{len(frames)}",**record,"passed":numerical_pass(record)})
                snapshot = session.main.cache.clone()
                answer = session.ask(question,8,return_evidence=True)
                reference = reference_answer(model,processor,embeddings,inputs.position_ids,question,8)
                evidence = answer.pop("evidence")
                q_record = {"id":f"qa{len(frames)}","online":answer,"reference_answer":reference["answer"],
                    "reference_ids":reference["generated_ids"],"logits":strict_logits(evidence["initial_logits"],reference["initial_logits"]),
                    "final_states":strict_states(evidence["branch_cache"],reference["cache"]),
                    "main_unchanged_bitwise":all(v["all_tensors_bitwise_equal"] for v in strict_states(session.main.cache,snapshot).values())}
                q_record["passed"] = numerical_pass(q_record) and q_record["main_unchanged_bitwise"] and answer["generated_ids"] == reference["generated_ids"]
                groups.append(q_record)
                generated_count += len(answer["generated_ids"])+len(reference["generated_ids"])
                q_inputs = reference_inputs(model,processor,frames,(192,256),question)
                assert torch.equal(q_inputs.input_ids,torch.cat((inputs.input_ids.cpu(),evidence["suffix_ids"]),1))
                fingerprints[f"question{len(frames)}"] = tensor_fingerprint(q_inputs)
                assert fingerprints[f"question{len(frames)}"] == proof["input_fingerprints"][f"question{len(frames)}"]
                del evidence,reference,snapshot,q_inputs
                write_json(out/"online_summary.json",report)
                print({"prefix_frames":len(frames),"math_passed":q_record["passed"],"answer":answer["answer"]},flush=True)
                del full,inputs,embeddings,new,online_features,whole_features
            session.reset()
            assert session.main.seen_tokens == 0 and session.pending is None and not session.ledger and model.model.rope_deltas is None
            session.append(frames[0])
            reset = session.append(frames[1])
            visual_parts.clear()
            record = pair_record(reset["logits"],session,first_full,session.builder.prefix_ids.shape[1])
            groups.append({"id":"reset_then_pair","all_counters_cleared":True,**record,"passed":numerical_pass(record)})
            session.close()
            eof = OnlinePairSession(model,processor)
            eof.append(frames[0]); eof.append(frames[1]); eof.append(frames[2])
            old = eof.main.seen_tokens
            flushed = eof.finish()
            visual_parts.clear()
            inputs = reference_inputs(model,processor,frames[:3],(192,256))
            fingerprints["eof3"] = tensor_fingerprint(inputs)
            assert fingerprints["eof3"] == proof["input_fingerprints"]["eof3"]
            inputs = inputs.to("cuda")
            embeddings = encoded(model,inputs)
            visual_parts.clear()
            full = model(inputs_embeds=embeddings,position_ids=inputs.position_ids,use_cache=True)
            record = pair_record(flushed["logits"],eof,full,old)
            rejected = False
            try:
                eof.append(frames[3])
            except ValueError:
                rejected = True
            groups.append({"id":"eof_odd_padding","padding_record":eof.ledger[-1],"input_sealed":rejected,
                **record,"passed":numerical_pass(record) and rejected and eof.ledger[-1]["synthetic_padding"]})
            eof.close()
        assert len(groups) == 8
        assert counters["vision"] == BUDGET["vision_encodings"] and counters["language"] <= BUDGET["language_forwards_max"]
        assert generated_count <= BUDGET["generated_tokens_max"]
        assert fixed.operations <= 500
        assert before_methods == forward_attributes(model)
        report.update(candidate_projection_operations=fixed.operations, candidate_methods_restored=True,
            reference_protocol="explicit fixed temporal-unit linears; original default full vision preserved",
            input_fingerprints=fingerprints,actual_forwards=counters,generated_tokens=generated_count,
            all_interface_numeric_groups_passed=all(g["passed"] for g in groups),
            wall_seconds_including_load=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        write_json(out/"online_summary.json",report)
        print({k:report[k] for k in ("all_interface_numeric_groups_passed","actual_forwards","generated_tokens","wall_seconds_including_load","peak_allocated_gib","peak_reserved_gib")},flush=True)
        if not report["all_interface_numeric_groups_passed"]:
            raise RuntimeError("Online registered interfaces/numerics failed; retain evidence")
    finally:
        reader.close()
        for hook in hooks:
            hook.remove()
        session.close()
        if "eof" in locals():
            eof.close()
        assert before_hooks == hook_state(model) and before_methods == forward_attributes(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "gpu"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=PROJECT / "models/Qwen3.5-0.8B")
    parser.add_argument("--video", type=Path, default=PROJECT / "datasets/natural_commons_v1/clean_hands.webm")
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "provenance.json", provenance())
    (args.out / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    write_json(args.out / "arguments.json", {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    try:
        (preflight if args.stage == "preflight" else gpu)(args.out,args)
        write_json(args.out / "status.json", {"status":"completed", "stage":args.stage})
    except Exception:
        (args.out / "traceback.txt").write_text(traceback.format_exc(),encoding="utf-8")
        write_json(args.out / "status.json", {"status":"failed", "stage":args.stage})
        raise
    finally:
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            write_json(args.out / "cleanup.json", {"cuda_allocated_bytes_after_return":torch.cuda.memory_allocated(),
                "cuda_reserved_bytes_after_return":torch.cuda.memory_reserved(),"control_frame_returned":True})


if __name__ == "__main__":
    main()
