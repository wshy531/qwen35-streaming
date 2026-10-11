"""Explicit reversible 32-token torch GDN execution candidate; defaults preserved."""
import argparse
import gc
from pathlib import Path
import time
import traceback

import psutil
import torch
from transformers import AutoProcessor

from qwen35_local.modeling_qwen3_5 import torch_chunk_gated_delta_rule
from scripts.baseline import PROJECT, provenance, write_json, file_hash, cache_measure
from scripts.fp32_regression import strict_logits, strict_states
from scripts.online_experiment import Q1
from scripts.pair_demo import verify_assets
from scripts.stage_timing import measured, distribution, read
from scripts.vision_expand import general_inputs, frames_for, assets
from scripts.vision_fixed_units import FixedTemporalLinears
from scripts.vision_fixed_control import forward_attributes
from scripts.vision_trace import hook_state, tensor_hash
from streaming.hybrid import HybridSequence
from streaming.online import PairInputBuilder, fp32_reference

PLAN = PROJECT / "语言计算块对照方案.md"
LAUNCHER = PROJECT / "scripts/run_chunk_probe.py"
BUDGET = {"vision_encodings": 1, "candidate_projection_operations_max": 100,
    "language_forwards": 36, "chunk_rule_calls_max": 612, "generated_tokens": 0,
    "model_loads": 1, "wall_seconds": 600, "retained_features_bytes": 8*2**20}
FIELDS = ("key_cache", "value_cache", "conv_states", "recurrent_states")


def callables(model):
    return [(name, module.chunk_gated_delta_rule) for name, module in model.named_modules()
            if hasattr(module, "chunk_gated_delta_rule")]


class ChunkRule:
    def __init__(self, model, size, maximum=612):
        self.model, self.size, self.maximum = model, size, maximum
        self.saved, self.calls = [], 0

    def __enter__(self):
        modules = [m for m in self.model.modules() if hasattr(m, "chunk_gated_delta_rule")]
        assert modules and all(m.chunk_gated_delta_rule is torch_chunk_gated_delta_rule for m in modules), "Only frozen torch fallback allowed"
        for module in modules:
            original = module.chunk_gated_delta_rule
            self.saved.append((module, original))
            def rule(*args, _original=original, **kwargs):
                if self.calls >= self.maximum:
                    raise RuntimeError("Chunk-rule call budget exceeded")
                self.calls += 1
                kwargs["chunk_size"] = self.size
                return _original(*args, **kwargs)
            module.chunk_gated_delta_rule = rule
        return self

    def __exit__(self, *_exception):
        for module, original in self.saved:
            module.chunk_gated_delta_rule = original
        self.saved.clear()


def compare(actual, expected, cache, reference_cache):
    logits, states = strict_logits(actual, expected), strict_states(cache, reference_cache)
    passed = logits["fp32_reference_limits_passed"] and logits["allclose_1e4"] and all(r["allclose_1e4"] for r in states.values())
    return {"logits": logits, "final_states": states, "passed": bool(passed)}


@torch.inference_mode()
def experiment(model, processor, out, expected=None, cuda=False):
    before, attrs, functions = hook_state(model), forward_attributes(model), callables(model)
    counts = {"vision": 0, "language": 0}
    def guard(name, maximum):
        def hook(_module, _args):
            if counts[name] >= maximum:
                raise RuntimeError(f"{name} budget exceeded")
            counts[name] += 1
        return hook
    handles = [model.model.visual.register_forward_pre_hook(guard("vision", 1)),
               model.model.language_model.register_forward_pre_hook(guard("language", 36))]
    device = next(model.parameters()).device
    reports, total_calls = [], 0
    try:
        with fp32_reference():
            full_cpu, pairs_cpu, identity = general_inputs(model, processor, frames_for("clean_hands", 4), (192, 256))
            if expected is not None:
                assert identity == expected
            batch = {k: v.to(device) for k, v in full_cpu.items()}
            builder = PairInputBuilder(processor, (192, 256))
            prefix = builder.prefix_ids.shape[1]
            cuts = [0, prefix, prefix + pairs_cpu[0].input_ids.shape[1], batch["input_ids"].shape[1]]
            question = builder.question_ids(Q1).to(device)
            fixed_token = builder.tokenize("HANDS")[:, :1].to(device)
            assert cuts[-1] + question.shape[1] + 1 <= 256
            with FixedTemporalLinears(model.model.visual, max_operations=100) as vision:
                features = model.model.visual(batch["pixel_values_videos"], grid_thw=batch["video_grid_thw"]).pooler_output
            feature_bytes = features.numel()*features.element_size()
            assert feature_bytes <= BUDGET["retained_features_bytes"]
            feature_hash = tensor_hash(features)
            embeddings = model.get_input_embeddings()(batch["input_ids"])
            embeddings = embeddings.masked_scatter((batch["input_ids"] == processor.video_token_id)[..., None].expand_as(embeddings), features)
            projection_ops = vision.operations
            del features, vision
            for size in (64, 32):
                samples, numeric = [], {}
                with ChunkRule(model, size, maximum=612-total_calls) as chunk:
                    for iteration in range(4):
                        row = {"iteration": iteration, "warmup": iteration == 0}
                        whole = HybridSequence(model)
                        all_logits, row["full_prefix"] = measured(lambda: whole.append(
                            inputs_embeds=embeddings, position_ids=batch["position_ids"]), cuda)
                        main = HybridSequence(model)
                        for index, label in enumerate(("prompt", "first_pair", "second_pair")):
                            left, right = cuts[index:index+2]
                            last, row[label] = measured(lambda: main.append(inputs_embeds=embeddings[:, left:right],
                                position_ids=batch["position_ids"][:, :, left:right]), cuda)
                            if index < 2:
                                del last
                        if iteration == 0:
                            numeric["increment_vs_own_full"] = compare(last, all_logits[:, -last.shape[1]:], main.cache, whole.cache)
                            if size == 64:
                                reference_logits, reference_whole = all_logits, whole
                            else:
                                numeric["full32_vs_full64_all_logits"] = compare(all_logits, reference_logits, whole.cache, reference_whole.cache)
                            snapshot = main.cache.clone()
                            positions_before = main.seen_tokens, main.next_rope_position
                            branch = main.fork()
                            for field in FIELDS:
                                for a, b in zip(getattr(main.cache, field), getattr(branch.cache, field)):
                                    if a is not None:
                                        assert a.data_ptr() != b.data_ptr()
                            q_logits, row["warmup_question"] = measured(lambda: branch.append(input_ids=question), cuda)
                            fixed_logits, row["warmup_fixed_token"] = measured(lambda: branch.append(input_ids=fixed_token), cuda)
                            if size == 64:
                                reference_q, reference_fixed, reference_branch = q_logits, fixed_logits, branch
                            else:
                                numeric["question32_vs_question64"] = {"logits": strict_logits(q_logits, reference_q)}
                                q = numeric["question32_vs_question64"]["logits"]
                                numeric["question32_vs_question64"]["passed"] = bool(q["allclose_1e4"] and q["fp32_reference_limits_passed"])
                                numeric["fixed_token32_vs_fixed_token64"] = compare(fixed_logits, reference_fixed, branch.cache, reference_branch.cache)
                                del q_logits, fixed_logits, branch
                            assert positions_before == (main.seen_tokens, main.next_rope_position)
                            for field in FIELDS:
                                assert all(torch.equal(a, b) for a, b in zip(getattr(main.cache, field), getattr(snapshot, field)) if a is not None)
                            del snapshot
                            cache_info = cache_measure(main.cache)
                        if not (size == 64 and iteration == 0):
                            del all_logits, whole
                        del last
                        main.reset()
                        assert main.seen_tokens == 0 and main.cache.get_seq_length() == 0
                        del main
                        if iteration == 0:
                            warmup = row
                        else:
                            row["increment_total"] = {key: sum(row[label][key] for label in ("prompt", "first_pair", "second_pair")) for key in row["prompt"]}
                            samples.append(row)
                        write_json(out / "chunk_probe_partial.json", {"completed_modes": reports, "current_size": size,
                            "current_numeric": numeric, "samples": samples, "actual_forwards": counts,
                            "chunk_calls": total_calls + chunk.calls})
                        assert all(r["passed"] for r in numeric.values()), f"Strict 32 probe failed at size {size}"
                    mode = {"chunk_size": size, "numeric": numeric, "samples": samples, "warmup_excluded": warmup,
                        "stage_summary_ms": {label: distribution([r[label] for r in samples]) for label in
                                             ("full_prefix", "prompt", "first_pair", "second_pair", "increment_total")},
                        "cache": cache_info, "chunk_rule_calls": chunk.calls}
                total_calls += chunk.calls
                reports.append(mode)
            assert counts == {"vision": 1, "language": 36}
            if cuda:
                assert total_calls == 612 and projection_ops == 100
            del reference_logits, reference_whole, reference_q, reference_fixed, reference_branch
            report = {"budget": BUDGET, "actual_forwards": counts, "chunk_rule_calls": total_calls,
                "candidate_projection_operations": projection_ops, "input_identity": identity, "modes": reports,
                "retained_feature_bytes": feature_bytes, "feature_sha256": feature_hash,
                "all_strict_controls_passed": True, "two_question_branches_main_unchanged_bitwise": True,
                "scope": "explicit 32 execution candidate, finite FP32 input; not default/BF16/semantics"}
    finally:
        for handle in handles:
            handle.remove()
        assert before == hook_state(model) and attrs == forward_attributes(model) and functions == callables(model)
        write_json(out / "runtime_cleanup.json", {"actual_forwards": counts, "chunk_rule_calls_completed_modes": total_calls,
            "hooks_methods_callables_restored": True})
    report["hooks_methods_callables_restored"] = True
    return report


@torch.inference_mode()
def preflight(out, args):
    from tests.test_online import small_model
    torch.set_num_threads(1)
    manifest = verify_assets(args.model_dir)
    config = read(args.model_dir / "config.json")["text_config"]
    layers = sum(kind == "linear_attention" for kind in config["layer_types"])
    assert layers == 18 and layers * 34 == BUDGET["chunk_rule_calls_max"]
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = small_model(processor)
    report = experiment(model, processor, out)
    original = callables(model)
    batch, _, _ = general_inputs(model, processor, frames_for("clean_hands", 4), (192, 256))
    try:
        with fp32_reference(), ChunkRule(model, 32, maximum=0):
            main = HybridSequence(model)
            main.append(input_ids=batch.input_ids[:, :3])
        raise AssertionError("No call budget rejection")
    except RuntimeError as error:
        assert "Chunk-rule call budget" in str(error)
    assert original == callables(model)
    report.update(scope="tiny CPU contract, no official capability/speed evidence", budget_exception_restored=True,
                  real_config_budget_check={"gdn_layers": layers, "multitoken_calls": 34, "expected_core_calls": 612})
    write_json(out / "cpu_summary.json", report)
    write_json(out / "preflight_summary.json", {"cpu_controls_passed": True, "budget": BUDGET,
        "input_identity": report["input_identity"], "cpu_report_sha256": file_hash(out / "cpu_summary.json"),
        "assets": assets(), "model_manifest_sha256": file_hash(args.model_dir / "DOWNLOAD_MANIFEST.json"),
        "verified_model_assets": manifest["files"], "plan_sha256": file_hash(PLAN), "launcher_sha256": file_hash(LAUNCHER)})
    print({"cpu_controls_passed": True, "numeric_passed": True, "real_budget_passed": True}, flush=True)


@torch.inference_mode()
def gpu(out, args):
    from qwen35_local import Qwen3_5ForConditionalGeneration
    begin = time.perf_counter()
    free, total = torch.cuda.mem_get_info()
    guard = {"gpu_free_bytes": free, "gpu_total_bytes": total, "ram_available_bytes": psutil.virtual_memory().available}
    write_json(out / "resource_precheck.json", guard)
    assert free >= 5.5*2**30 and guard["ram_available_bytes"] >= 3*2**30
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
        write_json(out / "chunk_probe_summary.json", report)
        print({"all_strict_controls_passed": True, "peak_allocated_gib": report["peak_allocated_gib"],
            "modes": [{"size": m["chunk_size"], "times": m["stage_summary_ms"]} for m in report["modes"]]}, flush=True)
        del report, processor
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        write_json(out / "cleanup.json", {"model_variable_deleted": True,
            "scope": "exception frames may retain tensors until exit", "cuda_allocated_bytes_after_release": torch.cuda.memory_allocated(),
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
