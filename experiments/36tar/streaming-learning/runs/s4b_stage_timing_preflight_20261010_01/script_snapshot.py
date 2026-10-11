"""Synchronized stage timing with shared features and frozen finite work counts."""
import argparse
import gc
import json
from pathlib import Path
import statistics
import time
import traceback

import psutil
import torch
from transformers import AutoProcessor

from scripts.baseline import PROJECT, provenance, write_json, file_hash, cache_measure
from scripts.fp32_regression import strict_logits, strict_states
from scripts.online_experiment import Q1
from scripts.pair_demo import verify_assets
from scripts.vision_expand import general_inputs, frames_for, assets
from scripts.vision_fixed_control import forward_attributes
from scripts.vision_fixed_units import FixedTemporalLinears
from scripts.vision_trace import hook_state, tensor_hash
from streaming.hybrid import HybridSequence
from streaming.online import PairInputBuilder, fp32_reference

PLAN = PROJECT / "分阶段耗时验收方案.md"
LAUNCHER = PROJECT / "scripts/run_stage_timing.py"
BUDGET = {"vision_encodings": 25, "candidate_projection_operations_max": 1250,
          "language_forwards": 36, "generated_tokens": 0, "model_loads": 1,
          "wall_seconds": 600, "retained_features_bytes": 8*2**20}
FIELDS = ("key_cache", "value_cache", "conv_states", "recurrent_states")


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def measured(function, cuda=False):
    if cuda:
        torch.cuda.synchronize()
        start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start = time.perf_counter()
    if cuda:
        start_event.record()
    value = function()
    if cuda:
        end_event.record()
        end_event.synchronize()
    result = {"wall_ms": (time.perf_counter()-start)*1000}
    if cuda:
        result["cuda_event_ms"] = start_event.elapsed_time(end_event)
    return value, result


def distribution(values):
    return {key: {"median": statistics.median([r[key] for r in values]),
        "min": min(r[key] for r in values), "max": max(r[key] for r in values)}
        for key in values[0]}


def cpu_stages(model, processor):
    decode, prepare = [], []
    for _ in range(5):
        frames, timing = measured(lambda: frames_for("clean_hands", 4))
        decode.append(timing)
        batch, timing = measured(lambda: general_inputs(model, processor, frames, (192, 256)))
        prepare.append(timing)
        identity = batch[2]
        del batch, frames
    return {"decode_first4": {"samples": decode, "summary": distribution(decode)},
            "prepare_full_and_pairs": {"samples": prepare, "summary": distribution(prepare)}}, identity


@torch.inference_mode()
def experiment(model, processor, out, expected=None, cuda=False):
    before, attrs = hook_state(model), forward_attributes(model)
    counts = {"vision": 0, "language": 0}
    def guard(name, maximum):
        def hook(_module, _args):
            if counts[name] >= maximum:
                raise RuntimeError(f"{name} budget exceeded")
            counts[name] += 1
        return hook
    handles = [model.model.visual.register_forward_pre_hook(guard("vision", 25)),
               model.model.language_model.register_forward_pre_hook(guard("language", 36))]
    device = next(model.parameters()).device
    samples, warmup, total_operations = [], {}, 0
    try:
        with fp32_reference():
            cpu, identity = cpu_stages(model, processor)
            if expected is not None:
                assert identity == expected, "Frozen inputs changed"
            full_cpu, pairs_cpu, same = general_inputs(model, processor, frames_for("clean_hands", 4), (192, 256))
            assert same == identity
            full = {k: v.to(device) for k, v in full_cpu.items()}
            pairs = [{k: v.to(device) for k, v in b.items()} for b in pairs_cpu]
            builder = PairInputBuilder(processor, (192, 256))
            question_ids = builder.question_ids(Q1).to(device)
            fixed_token = builder.tokenize("HANDS")[:, :1].to(device)
            prefix_length = builder.prefix_ids.shape[1]
            cuts = [0, prefix_length, prefix_length + pairs[0]["input_ids"].shape[1], full["input_ids"].shape[1]]
            assert cuts[-1] + question_ids.shape[1] + 1 <= 256
            def visual(batch):
                return model.model.visual(batch["pixel_values_videos"], grid_thw=batch["video_grid_thw"]).pooler_output
            for iteration in range(6):
                row = {"iteration": iteration, "warmup": iteration == 0}
                # Alternate placement of original relative to candidate to reduce one-sided ordering effects.
                if iteration % 2 == 0:
                    value, row["vision_original_full4"] = measured(lambda: visual(full), cuda)
                    del value
                fixed = FixedTemporalLinears(model.model.visual, max_operations=1250-total_operations)
                _, row["candidate_install"] = measured(fixed.__enter__, cuda)
                try:
                    def paired():
                        return [visual(b) for b in pairs]
                    if iteration % 2 == 0:
                        candidate, row["vision_candidate_full4"] = measured(lambda: visual(full), cuda)
                        parts, row["vision_candidate_pairs2plus2"] = measured(paired, cuda)
                    else:
                        parts, row["vision_candidate_pairs2plus2"] = measured(paired, cuda)
                        candidate, row["vision_candidate_full4"] = measured(lambda: visual(full), cuda)
                    if iteration == 0:
                        assert torch.equal(candidate, torch.cat(parts)), "Timing input candidate mismatch"
                finally:
                    _, row["candidate_restore"] = measured(lambda: fixed.__exit__(None, None, None), cuda)
                    total_operations += fixed.operations
                del candidate, parts, fixed
                if iteration % 2:
                    value, row["vision_original_full4"] = measured(lambda: visual(full), cuda)
                    del value
                if iteration == 0:
                    with FixedTemporalLinears(model.model.visual, max_operations=1250-total_operations) as setup:
                        features = visual(full)
                    total_operations += setup.operations
                    retained = features.numel()*features.element_size()
                    assert retained <= BUDGET["retained_features_bytes"]
                    feature_sha = tensor_hash(features)
                    embeddings = model.get_input_embeddings()(full["input_ids"])
                    mask = (full["input_ids"] == processor.video_token_id)[..., None].expand_as(embeddings)
                    embeddings = embeddings.masked_scatter(mask, features)
                    del features, setup, mask
                whole = HybridSequence(model, max_tokens=256)
                whole_logits, row["language_full_prefix"] = measured(lambda: whole.append(
                    inputs_embeds=embeddings, position_ids=full["position_ids"]), cuda)
                main = HybridSequence(model, max_tokens=256)
                for index, label in enumerate(("language_prompt_init", "language_first_pair", "language_second_pair")):
                    left, right = cuts[index:index+2]
                    last_logits, row[label] = measured(lambda: main.append(inputs_embeds=embeddings[:, left:right],
                        position_ids=full["position_ids"][:, :, left:right]), cuda)
                    if index != 2:
                        del last_logits
                if iteration == 0:
                    logits = strict_logits(last_logits, whole_logits[:, -last_logits.shape[1]:])
                    states = strict_states(main.cache, whole.cache)
                    assert logits["fp32_reference_limits_passed"] and logits["allclose_1e4"]
                    assert all(r["allclose_1e4"] for r in states.values())
                    correctness = {"last_pair_logits": logits, "all_final_cache_fields": states,
                        "scope": "shared actual features; last new pair logits and full final cache"}
                    cache = cache_measure(main.cache)
                del whole_logits, last_logits, whole
                snapshot = main.cache.clone()
                positions_before = main.seen_tokens, main.next_rope_position
                branch, row["fork"] = measured(main.fork, cuda)
                for field in FIELDS:
                    for a, b in zip(getattr(main.cache, field), getattr(branch.cache, field)):
                        if a is not None:
                            assert a.data_ptr() != b.data_ptr()
                value, row["question_prefill"] = measured(lambda: branch.append(input_ids=question_ids), cuda)
                del value
                value, row["fixed_single_token_continue"] = measured(lambda: branch.append(input_ids=fixed_token), cuda)
                del value, branch
                assert positions_before == (main.seen_tokens, main.next_rope_position)
                for field in FIELDS:
                    assert all(torch.equal(a, b) for a, b in zip(getattr(main.cache, field), getattr(snapshot, field)) if a is not None)
                del snapshot
                _, row["reset"] = measured(main.reset, cuda)
                assert main.seen_tokens == 0 and main.cache.get_seq_length() == 0
                del main
                if iteration == 0:
                    warmup = row
                else:
                    samples.append(row)
                write_json(out / "timing_partial.json", {"samples": samples, "warmup": warmup, "actual_forwards": counts})
            assert counts == {"vision": 25, "language": 36}
            if cuda:
                assert total_operations == 1250
            labels = [key for key in samples[0] if isinstance(samples[0][key], dict)]
            summaries = {key: distribution([row[key] for row in samples]) for key in labels}
            for row in samples:
                row["language_increment_total"] = {key: sum(row[label][key] for label in
                    ("language_prompt_init", "language_first_pair", "language_second_pair")) for key in row["language_prompt_init"]}
            summaries["language_increment_total"] = distribution([row["language_increment_total"] for row in samples])
            report = {"budget": BUDGET, "actual_forwards": counts, "candidate_projection_operations": total_operations,
                "input_identity": identity, "cpu_stages": cpu, "samples": samples, "warmup_excluded": warmup,
                "stage_summary_ms": summaries, "shared_feature_sha256": feature_sha,
                "retained_feature_bytes": retained, "strict_shared_feature_correctness": correctness,
                "main_cache": cache, "main_unchanged_all_six_branches_bitwise": True,
                "fork_storage_independent": True, "reset_empty_all_six": True,
                "question_tokens": question_ids.shape[1], "fixed_token_id": int(fixed_token[0, 0]),
                "generated_tokens": 0, "scope": "finite synchronized FP32 microbenchmark; no answer/real-time/BF16 claim"}
            del full, pairs, embeddings, question_ids, fixed_token
    finally:
        for handle in handles:
            handle.remove()
    assert before == hook_state(model) and attrs == forward_attributes(model)
    report["methods_hooks_restored"] = True
    return report


@torch.inference_mode()
def preflight(out, args):
    from tests.test_online import small_model
    torch.set_num_threads(1)
    manifest = verify_assets(args.model_dir)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = small_model(processor)
    report = experiment(model, processor, out)
    report["scope"] = "tiny random CPU contract and timing-shape checks, no GPU speed or semantic evidence"
    write_json(out / "cpu_summary.json", report)
    write_json(out / "preflight_summary.json", {"cpu_controls_passed": True, "budget": BUDGET,
        "input_identity": report["input_identity"], "cpu_report_sha256": file_hash(out / "cpu_summary.json"),
        "assets": assets(), "model_manifest_sha256": file_hash(args.model_dir / "DOWNLOAD_MANIFEST.json"),
        "verified_model_assets": manifest["files"], "plan_sha256": file_hash(PLAN), "launcher_sha256": file_hash(LAUNCHER)})
    print({"cpu_controls_passed": True, "actual_forwards": report["actual_forwards"], "branches_reset_passed": True}, flush=True)


@torch.inference_mode()
def gpu(out, args):
    from qwen35_local import Qwen3_5ForConditionalGeneration
    begin = time.perf_counter()
    free, total = torch.cuda.mem_get_info()
    guard = {"gpu_free_bytes": free, "gpu_total_bytes": total, "ram_available_bytes": psutil.virtual_memory().available}
    write_json(out / "resource_precheck.json", guard)
    assert free >= 5.5*2**30 and guard["ram_available_bytes"] >= 3*2**30, "Resource guard, no loading"
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
        write_json(out / "timing_summary.json", report)
        print({"actual_forwards": report["actual_forwards"], "peak_allocated_gib": report["peak_allocated_gib"],
               "stage_summary_ms": report["stage_summary_ms"]}, flush=True)
        del report, processor
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        write_json(out / "cleanup.json", {"model_released": True,
            "cuda_allocated_bytes_after_release": torch.cuda.memory_allocated(),
            "cuda_reserved_bytes_after_release": torch.cuda.memory_reserved()})


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
