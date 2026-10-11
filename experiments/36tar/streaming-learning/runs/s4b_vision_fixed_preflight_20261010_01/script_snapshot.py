"""Four traced candidate visions plus one original reference, with bounded memory."""
import argparse
import gc
import json
from pathlib import Path
import time
import traceback

import psutil
import torch
from transformers import AutoProcessor

from scripts.baseline import PROJECT, provenance, write_json, file_hash
from scripts.pair_demo import verify_assets
from scripts.vision_trace import inputs, packets, tensor_hash, difference, controls, hook_state
from scripts.vision_fixed_units import FixedTemporalLinears
from streaming.online import fp32_reference

PLAN = PROJECT / "固定时间单元视觉参考方案.md"
LAUNCHER = PROJECT / "scripts/run_vision_fixed.py"
HISTORY = PROJECT / "runs/s4b_vision_trace_gpu_20261010_01/vision_trace_summary.json"
BUDGET = {"vision_encodings": 5, "candidate_projection_operations_max": 300,
          "language_forwards": 0, "generated_tokens": 0, "model_loads": 1,
          "wall_seconds": 600, "trace_buffer_bytes": 128 * 2**20}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def forward_attributes(model):
    return [(name, "forward" in m.__dict__, m.__dict__.get("forward")) for name, m in model.named_modules()]


@torch.inference_mode()
def fixed_controls(model, batch, out=None, history=None):
    before, attrs = hook_state(model), forward_attributes(model)
    device = next(model.parameters()).device
    counts = {"vision": 0, "language": 0}
    captured = []
    def vision_start(_module, _args):
        if counts["vision"] >= 5:
            raise RuntimeError("Five-vision budget exceeded")
        counts["vision"] += 1
    def language_start(_module, _args):
        counts["language"] += 1
        raise RuntimeError("Language forbidden")
    def candidate_output(_module, _args, result):
        if counts["vision"] == 2:
            captured.append(result.pooler_output.detach().cpu().clone())
    hooks = [model.model.visual.register_forward_pre_hook(vision_start),
             model.model.language_model.register_forward_pre_hook(language_start),
             model.model.visual.register_forward_hook(candidate_output)]
    try:
        with fp32_reference():
            visual = {k: batch[0][k].to(device) for k in ("pixel_values_videos", "video_grid_thw")}
            baseline = model.model.visual(visual["pixel_values_videos"], grid_thw=visual["video_grid_thw"]).pooler_output.cpu()
            del visual
            if history:
                assert tensor_hash(baseline) == history["records"]["full"][-1]["sha256"]
            with FixedTemporalLinears(model.model.visual) as fixed:
                report = controls(model, batch, out)
                operations = fixed.operations
                assert fixed.calls == 4 and operations <= 300
            assert len(captured) == 1 and counts == {"vision": 5, "language": 0}
            additional = baseline.numel() * baseline.element_size() + captured[0].numel() * captured[0].element_size()
            report["trace_buffer_peak_bound_bytes"] += additional
            assert report["trace_buffer_peak_bound_bytes"] <= BUDGET["trace_buffer_bytes"]
            report.update(budget=BUDGET, actual_forwards=counts, candidate_forwards=4,
                          candidate_projection_operations=operations,
                          candidate_full_vs_original_full=difference(captured[0], baseline),
                          candidate_all_stages_bitwise=all(r["bitwise_equal"] for mode in ("repeat", "pair0", "pair1")
                                                          for r in report["records"][mode]),
                          original_full_sha256=tensor_hash(baseline),
                          candidate_full_matches_original_pairs=(tensor_hash(captured[0]) == history["combined_features"]["actual_sha256"])
                                                                  if history else None,
                          scope="experimental consistent vision protocol; original full computation differs and is preserved")
    finally:
        for h in hooks:
            h.remove()
        captured.clear()
    assert before == hook_state(model) and attrs == forward_attributes(model)
    report.update(fixed_methods_restored=True, hooks_restored=True)
    return report


@torch.inference_mode()
def preflight(out, args):
    from tests.test_online import small_model
    torch.set_num_threads(1)
    manifest = verify_assets(args.model_dir)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = small_model(processor)
    batch, identity = inputs(model, processor, packets(args.video), (192, 256))
    assert identity == read(HISTORY)["input_identity"]
    attrs, before = forward_attributes(model), hook_state(model)
    weights = {k: v.clone() for k, v in model.model.visual.state_dict().items()}
    cpu = fixed_controls(model, batch)
    assert cpu["candidate_all_stages_bitwise"]
    assert all(torch.equal(v, weights[k]) for k, v in model.model.visual.state_dict().items())
    try:
        with FixedTemporalLinears(model.model.visual, max_operations=0):
            model.model.visual(batch[1].pixel_values_videos, grid_thw=batch[1].video_grid_thw)
        raise AssertionError("Budget exception not triggered")
    except RuntimeError as error:
        assert "projection budget" in str(error)
    assert attrs == forward_attributes(model) and before == hook_state(model)
    cpu.update(exception_cleanup_verified=True, weights_bitwise_unchanged=True, input_identity=identity,
               scope="tiny random visual weights; contract only")
    write_json(out / "cpu_summary.json", cpu)
    write_json(out / "preflight_summary.json", {"cpu_controls_passed": True, "budget": BUDGET,
        "input_identity": identity, "cpu_report_sha256": file_hash(out / "cpu_summary.json"),
        "video_sha256": file_hash(args.video), "model_manifest_sha256": file_hash(args.model_dir / "DOWNLOAD_MANIFEST.json"),
        "verified_model_assets": manifest["files"], "historical_report_sha256": file_hash(HISTORY),
        "plan_sha256": file_hash(PLAN), "launcher_sha256": file_hash(LAUNCHER)})
    print({"cpu_controls_passed": True, "candidate_all_stages_bitwise": True, "exception_cleanup": True}, flush=True)


@torch.inference_mode()
def gpu(out, args):
    from qwen35_local import Qwen3_5ForConditionalGeneration
    begin = time.perf_counter()
    free, total = torch.cuda.mem_get_info()
    guard = {"gpu_free_bytes": free, "gpu_total_bytes": total, "ram_available_bytes": psutil.virtual_memory().available}
    write_json(out / "resource_precheck.json", guard)
    assert free >= 5.5 * 2**30 and guard["ram_available_bytes"] >= 3 * 2**30, "Resource guard; no loading"
    model = None
    try:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        torch.cuda.reset_peak_memory_stats()
        assert torch.cuda.mem_get_info()[0] >= 2 * 2**30 and psutil.virtual_memory().available >= 2 * 2**30
        model.float()
        processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
        batch, identity = inputs(model, processor, packets(args.video), (192, 256))
        assert identity == read(args.preflight / "preflight_summary.json")["input_identity"]
        report = fixed_controls(model, batch, out, read(HISTORY))
        report.update(input_identity=identity, wall_seconds_including_load=time.perf_counter() - begin,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        write_json(out / "vision_fixed_summary.json", report)
        print({k: report[k] for k in ("candidate_all_stages_bitwise", "combined_features",
                                    "candidate_full_vs_original_full", "candidate_full_matches_original_pairs",
                                    "candidate_projection_operations", "peak_allocated_gib")}, flush=True)
        del batch, processor, report
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
    parser.add_argument("--model-dir", type=Path, default=PROJECT / "models/Qwen3.5-0.8B")
    parser.add_argument("--video", type=Path, default=PROJECT / "datasets/natural_commons_v1/clean_hands.webm")
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "provenance.json", provenance())
    (args.out / "script_snapshot.py").write_bytes(Path(__file__).read_bytes())
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
