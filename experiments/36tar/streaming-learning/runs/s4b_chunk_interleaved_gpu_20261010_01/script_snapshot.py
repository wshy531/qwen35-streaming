"""Five bounded cases, balanced 64/32 orders, strict checks on every timed round."""
import argparse
import gc
from contextlib import contextmanager
from pathlib import Path
import time
import traceback

import psutil
import torch
from transformers import AutoProcessor

from scripts.baseline import PROJECT, provenance, write_json, file_hash, cache_measure
from scripts.chunk_size_probe import ChunkRule, callables, compare, FIELDS
from scripts.online_experiment import Q1
from scripts.pair_demo import verify_assets
from scripts.stage_timing import measured, distribution, read
from scripts.vision_expand import general_inputs, frames_for, assets
from scripts.vision_fixed_units import FixedTemporalLinears
from scripts.vision_fixed_control import forward_attributes
from scripts.vision_trace import hook_state, tensor_hash
from streaming.inputs import tensor_fingerprint
from streaming.hybrid import HybridSequence
from streaming.online import PairInputBuilder, fp32_reference

PLAN = PROJECT / "语言计算块交错扩展方案.md"
LAUNCHER = PROJECT / "scripts/run_chunk_interleaved.py"
BUDGET = {"vision_encodings": 2, "candidate_projection_operations_max": 250,
    "language_forwards": 314, "chunk_rule_calls_max": 5472, "generated_tokens": 0,
    "model_loads": 1, "wall_seconds": 600, "retained_features_bytes": 8*2**20}
CASES = [("text31", "text", 31, [7, 24]), ("text65", "text", 65, [7, 31, 27]),
         ("text129", "text", 129, [7, 31, 32, 59]),
         ("hands4", "clean_hands", 4, None), ("cat6", "cat_play", 6, None)]


def prepare(model, processor, spec):
    name, source, length, layout = spec
    if source == "text":
        text = "A video consists of consecutive images captured over time. Each new frame supplies visual information. "*32
        ids = processor.tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[:, :length]
        assert ids.shape[1] == length and sum(layout) == length
        positions = torch.arange(length)[None, None].expand(4, 1, -1).clone()
        batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "position_ids": positions}
        identity = {"case_id": name, "kind": "synthetic_exact_length_text_probe", "tokens": length,
            "layout": layout, "input_fingerprint": tensor_fingerprint(batch), "token_ids": ids[0].tolist()}
    else:
        batch, pairs, identity = general_inputs(model, processor, frames_for(source, length), (192, 256))
        prefix = PairInputBuilder(processor, (192, 256)).prefix_ids.shape[1]
        layout = [prefix] + [p.input_ids.shape[1] for p in pairs]
        assert sum(layout) == batch.input_ids.shape[1]
        identity.update(case_id=name, kind="fixed_actual_video_features", layout=layout, tokens=batch.input_ids.shape[1])
    assert min(layout) >= 2
    return batch, layout, identity


def compact_control(left, right, cache, reference):
    result = compare(left, right, cache, reference)
    return {"passed": result["passed"], "logits": result["logits"],
        "final_states": {field: {key: values[key] for key in ("max_abs_diff", "allclose_1e4", "all_tensors_bitwise_equal")}
                         for field, values in result["final_states"].items()}}


def reset_owned(sequence):
    sequence.reset()
    assert sequence.seen_tokens == 0 and sequence.cache.get_seq_length() == 0
    assert all(t is None for field in FIELDS for t in getattr(sequence.cache, field))


@torch.inference_mode()
def run_mode(model, embeddings, positions, layout, question, fixed_token, warmup, cuda):
    whole = HybridSequence(model, max_tokens=256)
    whole_logits, full_time = measured(lambda: whole.append(inputs_embeds=embeddings, position_ids=positions), cuda)
    main = HybridSequence(model, max_tokens=256)
    times, offset = {"full_prefix": full_time}, 0
    for index, length in enumerate(layout):
        stop = offset + length
        last, times[f"increment_{index}"] = measured(lambda: main.append(
            inputs_embeds=embeddings[:, offset:stop], position_ids=positions[:, :, offset:stop]), cuda)
        if index < len(layout)-1:
            del last
        offset = stop
    assert whole.seen_tokens == main.seen_tokens == embeddings.shape[1]
    assert whole.next_rope_position == main.next_rope_position
    times["increment_total"] = {unit: sum(times[f"increment_{i}"][unit] for i in range(len(layout)))
                                for unit in full_time}
    result = {"whole": whole, "whole_logits": whole_logits, "main": main, "last": last, "times": times}
    if warmup:
        snapshot = main.cache.clone()
        counters = main.seen_tokens, main.next_rope_position
        branch = main.fork()
        for field in FIELDS:
            for a, b in zip(getattr(main.cache, field), getattr(branch.cache, field)):
                if a is not None:
                    assert a.data_ptr() != b.data_ptr()
        q = branch.append(input_ids=question)
        token = branch.append(input_ids=fixed_token)
        assert counters == (main.seen_tokens, main.next_rope_position)
        assert all(torch.equal(a, b) for field in FIELDS for a, b in
                   zip(getattr(main.cache, field), getattr(snapshot, field)) if a is not None)
        result.update(question_logits=q, token_logits=token, branch=branch,
            branch_main_unchanged_bitwise=True, fork_storage_independent=True, cache=cache_measure(main.cache))
        del snapshot
    return result


@torch.inference_mode()
def experiment(model, processor, out, expected=None, cuda=False):
    hooks_before, attrs_before, functions_before = hook_state(model), forward_attributes(model), callables(model)
    counts = {"vision": 0, "language": 0, "chunk_rule": 0, "vision_projections": 0}
    def guard(name, maximum):
        def hook(_module, _args):
            if counts[name] >= maximum:
                raise RuntimeError(f"{name} budget exceeded")
            counts[name] += 1
        return hook
    handles = [model.model.visual.register_forward_pre_hook(guard("vision", 2)),
               model.model.language_model.register_forward_pre_hook(guard("language", 314))]
    @contextmanager
    def rules(size):
        with ChunkRule(model, size, maximum=5472-counts["chunk_rule"]) as chunk:
            try:
                yield
            finally:
                counts["chunk_rule"] += chunk.calls
    device = next(model.parameters()).device
    builder = PairInputBuilder(processor, (192, 256))
    question = builder.question_ids(Q1).to(device)
    fixed_token = builder.tokenize("HANDS")[:, :1].to(device)
    records, identities, feature_peak = [], [], 0
    try:
        with fp32_reference():
            for spec in CASES:
                cpu, layout, identity = prepare(model, processor, spec)
                assert identity["tokens"] + question.shape[1] + 1 <= 256
                if expected is not None:
                    assert identity == expected[len(identities)], "Frozen input mismatch"
                identities.append(identity)
                positions = cpu["position_ids"].to(device)
                ids = cpu["input_ids"].to(device)
                embeddings = model.get_input_embeddings()(ids)
                feature_hash = None
                if "pixel_values_videos" in cpu:
                    with FixedTemporalLinears(model.model.visual, max_operations=250-counts["vision_projections"]) as vision:
                        features = model.model.visual(cpu["pixel_values_videos"].to(device),
                            grid_thw=cpu["video_grid_thw"].to(device)).pooler_output
                    counts["vision_projections"] += vision.operations
                    feature_peak = max(feature_peak, features.numel()*features.element_size())
                    assert feature_peak <= BUDGET["retained_features_bytes"]
                    feature_hash = tensor_hash(features)
                    embeddings = embeddings.masked_scatter((ids == processor.video_token_id)[..., None].expand_as(embeddings), features)
                    del features, vision
                del cpu, ids
                rows, warmup_row, cache = [], None, None
                for round_id in range(7):
                    warmup = round_id == 0
                    formal_id = round_id - 1
                    order = [64, 32] if warmup or formal_id % 2 == 0 else [32, 64]
                    outputs = {}
                    for size in order:
                        with rules(size):
                            outputs[size] = run_mode(model, embeddings, positions, layout, question, fixed_token, warmup, cuda)
                    left, right = outputs[32], outputs[64]
                    numeric = {"full32_vs_full64_all_logits": compact_control(left["whole_logits"], right["whole_logits"],
                        left["whole"].cache, right["whole"].cache)}
                    for size in (64, 32):
                        value = outputs[size]
                        numeric[f"increment{size}_vs_own_full_last_chunk"] = compact_control(value["last"],
                            value["whole_logits"][:, -value["last"].shape[1]:], value["main"].cache, value["whole"].cache)
                    if warmup:
                        from scripts.fp32_regression import strict_logits
                        q = strict_logits(left["question_logits"], right["question_logits"])
                        numeric["question32_vs_question64"] = {"logits": q,
                            "passed": bool(q["fp32_reference_limits_passed"] and q["allclose_1e4"])}
                        numeric["fixed_token32_vs_fixed_token64"] = compact_control(left["token_logits"], right["token_logits"],
                            left["branch"].cache, right["branch"].cache)
                        assert all(v["branch_main_unchanged_bitwise"] and v["fork_storage_independent"] for v in outputs.values())
                        cache = right["cache"]
                    row = {"round": round_id, "warmup": warmup, "order": order,
                        "times": {str(size): outputs[size]["times"] for size in (64, 32)}, "numeric": numeric,
                        "all_numeric_passed": all(v["passed"] for v in numeric.values()),
                        "all_reset_empty": True}
                    for value in outputs.values():
                        reset_owned(value["main"])
                        reset_owned(value["whole"])
                        if warmup:
                            reset_owned(value["branch"])
                    del outputs, left, right, value
                    if warmup:
                        warmup_row = row
                    else:
                        rows.append(row)
                    write_json(out / "interleaved_partial.json", {"completed_cases": records, "current_case": spec[0],
                        "warmup": warmup_row, "samples": rows, "actual_counts": counts})
                assert [r["order"] for r in rows].count([64, 32]) == [r["order"] for r in rows].count([32, 64]) == 3
                labels = rows[0]["times"]["64"].keys()
                summary = {str(size): {label: distribution([r["times"][str(size)][label] for r in rows]) for label in labels}
                           for size in (64, 32)}
                ratios = {label: {"samples": [{"ratio32_over64": r["times"]["32"][label]["wall_ms"]/r["times"]["64"][label]["wall_ms"],
                    "order": r["order"]} for r in rows]} for label in ("full_prefix", "increment_total")}
                import statistics
                for label in ratios:
                    values = ratios[label]["samples"]
                    ratios[label]["median_ratio"] = statistics.median(v["ratio32_over64"] for v in values)
                    ratios[label]["median_by_order"] = {str(order): statistics.median(v["ratio32_over64"] for v in values if v["order"] == order)
                                                         for order in ([64, 32], [32, 64])}
                record = {"case_id": spec[0], "identity": identity, "feature_sha256": feature_hash, "layout": layout,
                    "warmup_excluded": warmup_row, "samples": rows, "stage_summary_ms": summary,
                    "paired_wall_ratios": ratios, "cache": cache,
                    "all_strict_controls_passed": warmup_row["all_numeric_passed"] and all(r["all_numeric_passed"] for r in rows),
                    "two_branches_main_unchanged_bitwise": True, "all_reset_empty": True}
                records.append(record)
                del embeddings, positions, row
            assert counts["vision"] == 2 and counts["language"] == 314
            if cuda:
                assert counts["vision_projections"] == 250 and counts["chunk_rule"] == 5472
            report = {"cases": records, "input_identity": identities, "budget": BUDGET, "actual_counts": counts,
                "all_strict_controls_passed": all(c["all_strict_controls_passed"] for c in records),
                "retained_feature_peak_bytes": feature_peak, "generated_tokens": 0,
                "scope": "finite shared features FP32 paired microbenchmark; defaults/BF16/semantics/real time not claimed"}
    finally:
        for handle in handles:
            handle.remove()
        assert hooks_before == hook_state(model) and attrs_before == forward_attributes(model) and functions_before == callables(model)
        write_json(out / "runtime_cleanup.json", {"actual_counts": counts, "hooks_methods_callables_restored": True})
    report["hooks_methods_callables_restored"] = True
    return report


@torch.inference_mode()
def preflight(out, args):
    from tests.test_online import small_model
    torch.set_num_threads(1)
    manifest = verify_assets(args.model_dir)
    config = read(args.model_dir / "config.json")
    gdn = sum(kind == "linear_attention" for kind in config["text_config"]["layer_types"])
    linears = config["vision_config"]["depth"]*4+2
    layouts = [2, 3, 4, 3, 4]
    main_calls = 7*2*sum(n+1 for n in layouts)
    branch_calls = len(CASES)*2*2
    assert gdn == 18 and linears == 50
    assert main_calls + branch_calls == BUDGET["language_forwards"] == 314
    assert (main_calls+len(CASES)*2)*gdn == BUDGET["chunk_rule_calls_max"] == 5472
    assert linears*(2+3) == BUDGET["candidate_projection_operations_max"] == 250
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = small_model(processor)
    report = experiment(model, processor, out)
    assert [len(i["layout"]) for i in report["input_identity"]] == layouts
    assert report["all_strict_controls_passed"]
    before = callables(model)
    try:
        with fp32_reference(), ChunkRule(model, 32, maximum=0):
            sequence = HybridSequence(model)
            sequence.append(input_ids=processor.tokenizer("budget exception probe", return_tensors="pt").input_ids)
        raise AssertionError("No budget rejection")
    except RuntimeError as error:
        assert "Chunk-rule call budget" in str(error)
    assert before == callables(model)
    report.update(scope="tiny CPU contract only, not real capability or GPU timing",
        budget_exception_restored=True, real_config_budget_check={"gdn_layers": gdn, "vision_linears": linears,
        "main_language_calls": main_calls, "branch_language_calls": branch_calls,
        "multitoken_gdn_calls": 5472, "single_token_gdn_calls_unchanged": 180})
    write_json(out / "cpu_summary.json", report)
    write_json(out / "preflight_summary.json", {"cpu_controls_passed": True, "budget": BUDGET,
        "input_identity": report["input_identity"], "cpu_report_sha256": file_hash(out / "cpu_summary.json"),
        "assets": assets(), "model_manifest_sha256": file_hash(args.model_dir / "DOWNLOAD_MANIFEST.json"),
        "verified_model_assets": manifest["files"], "plan_sha256": file_hash(PLAN), "launcher_sha256": file_hash(LAUNCHER)})
    print({"cpu_controls_passed": True, "cases": len(report["cases"]), "actual_counts": report["actual_counts"]}, flush=True)


@torch.inference_mode()
def gpu(out, args):
    from qwen35_local import Qwen3_5ForConditionalGeneration
    begin = time.perf_counter()
    free, total = torch.cuda.mem_get_info()
    guard = {"gpu_free_bytes": free, "gpu_total_bytes": total, "ram_available_bytes": psutil.virtual_memory().available}
    write_json(out / "resource_precheck.json", guard)
    assert free >= 5.5*2**30 and guard["ram_available_bytes"] >= 3*2**30, "Preload resource guard"
    model = None
    try:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        torch.cuda.reset_peak_memory_stats()
        assert torch.cuda.mem_get_info()[0] >= 2*2**30 and psutil.virtual_memory().available >= 2*2**30
        model.float()
        processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
        report = experiment(model, processor, out, read(args.preflight / "preflight_summary.json")["input_identity"], cuda=True)
        report.update(wall_seconds_including_load=time.perf_counter()-begin,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        write_json(out / "interleaved_summary.json", report)
        print({"all_strict_controls_passed": report["all_strict_controls_passed"], "actual_counts": report["actual_counts"],
            "peak_allocated_gib": report["peak_allocated_gib"], "cases": [{"id": c["case_id"],
                "passed": c["all_strict_controls_passed"], "paired_wall_ratios": c["paired_wall_ratios"]} for c in report["cases"]]}, flush=True)
        assert report["all_strict_controls_passed"], "Strict numerical failures preserved in final summary"
        del report, processor
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        write_json(out / "cleanup.json", {"model_variable_deleted": True,
            "scope": "exception frames may retain tensors until exit", "cuda_allocated_bytes_after_release": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes_after_release": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "gpu"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "provenance.json", provenance())
    (args.out / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    (args.out / "launcher_snapshot.py").write_bytes(LAUNCHER.read_bytes())
    (args.out / "plan_snapshot.md").write_bytes(PLAN.read_bytes())
    write_json(args.out / "arguments.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    try:
        (preflight if args.stage == "preflight" else gpu)(args.out, args)
        write_json(args.out / "status.json", {"status": "completed", "stage": args.stage})
    except Exception:
        (args.out / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(args.out / "status.json", {"status": "failed", "stage": args.stage})
        raise


if __name__ == "__main__":
    main()
