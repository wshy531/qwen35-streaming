"""Same-input projection shape/precision controls following the fixed vision trace."""
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
from scripts.vision_trace import inputs, packets, tensor_hash, difference
from streaming.online import fp32_reference

PLAN = PROJECT / "视觉QKV同输入重放方案.md"
LAUNCHER = PROJECT / "scripts/run_qkv_replay.py"
HISTORY = PROJECT / "runs/s4b_vision_trace_gpu_20261010_01/vision_trace_summary.json"
BUDGET = {"projection_operations": 9, "prefix_builds": 1, "vision_encodings": 0,
          "language_forwards": 0, "generated_tokens": 0, "model_loads": 1,
          "wall_seconds": 600, "trace_buffer_bytes": 128 * 2**20}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


@torch.inference_mode()
def replay(model, batch, history=None):
    visual = model.model.visual
    device = next(model.parameters()).device
    full = batch[0]
    pixels, grid = full.pixel_values_videos.to(device), full.video_grid_thw.to(device)
    counts = {"prefix": 0, "projection": 0, "vision": 0, "language": 0}
    hooks = []
    def forbidden_vision(_module, _args):
        counts["vision"] += 1
        raise RuntimeError("Full vision forward forbidden")
    def forbidden_language(_module, _args):
        counts["language"] += 1
        raise RuntimeError("Language forward forbidden")
    hooks.append(visual.register_forward_pre_hook(forbidden_vision))
    hooks.append(model.model.language_model.register_forward_pre_hook(forbidden_language))
    try:
        with fp32_reference():
            x = visual.blocks[0].norm1(visual.patch_embed(pixels) + visual.fast_pos_embed_interpolate(grid))
            counts["prefix"] += 1
            assert x.shape[0] % 2 == 0
            half = x.shape[0] // 2
            layer = visual.blocks[0].attn.qkv
            fingerprints = {"input": tensor_hash(x), "weight": tensor_hash(layer.weight), "bias": tensor_hash(layer.bias)}
            if history:
                expected = {r["stage"]: r for r in history["records"]["full"]}
                assert fingerprints["input"] == expected["blocks.0.norm1.output"]["sha256"], "Actual norm1 input changed"
            held = []
            peak_bound = 0
            def project(value, weight=None, bias=None):
                nonlocal peak_bound
                if counts["projection"] >= 9:
                    raise RuntimeError("Projection budget exceeded before operation")
                # Conservative live diagnostic tensors plus CPU FP64 metric/hash workspaces.
                size = value.shape[0] * layer.out_features * value.element_size()
                fixed = sum(t.numel() * t.element_size() for t in held) + 32 * 2**20
                bound = fixed + 12 * size
                assert bound <= BUDGET["trace_buffer_bytes"], "Projection buffer bound exceeded"
                peak_bound = max(peak_bound, bound)
                counts["projection"] += 1
                return layer(value) if weight is None else torch.nn.functional.linear(value, weight, bias)
            whole = project(x).cpu()
            held.append(whole)
            repeat = project(x).cpu()
            repeated = difference(repeat, whole)
            del repeat
            parts = [project(x[:half]).cpu(), project(x[half:]).cpu()]
            split = torch.cat(parts)
            del parts
            held.append(split)
            split_comparison = difference(split, whole)
            zeros = torch.zeros_like(x[:half])
            padded0 = project(torch.cat((x[:half], zeros)))[:half].cpu()
            padded1 = project(torch.cat((zeros, x[half:])))[half:].cpu()
            padded = torch.cat((padded0, padded1))
            del padded0, padded1, zeros
            padding_comparison = difference(padded, whole)
            del padded
            if history:
                assert tensor_hash(whole) == expected["blocks.0.attn.qkv.output"]["sha256"]
                for i in (0, 1):
                    actual = tensor_hash(split[i * half:(i + 1) * half])
                    record = next(r for r in history["records"][f"pair{i}"] if r["stage"] == "blocks.0.attn.qkv.output")
                    assert actual == record["actual_sha256"], "Same-input split did not reproduce trace"
            double_x, double_w, double_b = x.double(), layer.weight.double(), layer.bias.double()
            double_whole = project(double_x, double_w, double_b).float().cpu()
            double_parts = [project(double_x[:half], double_w, double_b).float().cpu(),
                            project(double_x[half:], double_w, double_b).float().cpu()]
            double_split = torch.cat(double_parts)
            del double_parts, double_x, double_w, double_b
            precision_comparison = difference(double_split, double_whole)
            accuracy = {"fp32_whole_vs_local_fp64": difference(whole, double_whole),
                        "fp32_split_vs_local_fp64": difference(split, double_whole)}
            assert fingerprints["weight"] == tensor_hash(layer.weight) and fingerprints["bias"] == tensor_hash(layer.bias)
            assert counts == {"prefix": 1, "projection": 9, "vision": 0, "language": 0}
            report = {"budget": BUDGET, "actual_operations": counts, "fingerprints": fingerprints,
                      "fp32_repeat": repeated, "fp32_split": split_comparison, "fp32_equal_shape_padding": padding_comparison,
                      "local_fp64_cast_fp32_split": precision_comparison, **accuracy,
                      "weights_bitwise_unchanged": True, "trace_input_and_outputs_reproduced": history is not None,
                      "input_shape": list(x.shape), "weight_shape": list(layer.weight.shape),
                      "trace_buffer_peak_bound_bytes": peak_bound,
                      "rss_bytes": psutil.Process().memory_info().rss,
                      "ram_available_bytes": psutil.virtual_memory().available,
                      "scope": "one projection on fixed actual input; no complete-vision repair claim"}
            held.clear()
            return report
    finally:
        for hook in hooks:
            hook.remove()


@torch.inference_mode()
def preflight(out, args):
    from tests.test_online import small_model
    from scripts.vision_trace import hook_state
    torch.set_num_threads(1)
    manifest = verify_assets(args.model_dir)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = small_model(processor)
    batch, identity = inputs(model, processor, packets(args.video), (192, 256))
    assert identity == read(HISTORY)["input_identity"]
    before = hook_state(model)
    cpu = replay(model, batch)
    assert cpu["fp32_repeat"]["bitwise_equal"] and cpu["weights_bitwise_unchanged"]
    assert cpu["local_fp64_cast_fp32_split"]["allclose_1e4"] and before == hook_state(model)
    cpu.update(input_identity=identity, hooks_restored=True, scope="tiny random projection; CPU interface only")
    write_json(out / "cpu_summary.json", cpu)
    write_json(out / "preflight_summary.json", {"cpu_controls_passed": True, "budget": BUDGET,
        "input_identity": identity, "cpu_report_sha256": file_hash(out / "cpu_summary.json"),
        "video_sha256": file_hash(args.video), "model_manifest_sha256": file_hash(args.model_dir / "DOWNLOAD_MANIFEST.json"),
        "verified_model_assets": manifest["files"], "historical_report_sha256": file_hash(HISTORY),
        "plan_sha256": file_hash(PLAN), "launcher_sha256": file_hash(LAUNCHER)})
    print({"cpu_controls_passed": True, "actual_operations": cpu["actual_operations"]}, flush=True)


@torch.inference_mode()
def gpu(out, args):
    from qwen35_local import Qwen3_5ForConditionalGeneration
    from scripts.vision_trace import hook_state
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
        before = hook_state(model)
        report = replay(model, batch, read(HISTORY))
        assert before == hook_state(model)
        report.update(hooks_restored=True, input_identity=identity,
                      wall_seconds_including_load=time.perf_counter() - begin,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        write_json(out / "qkv_replay_summary.json", report)
        print({k: report[k] for k in ("fp32_split", "fp32_equal_shape_padding", "local_fp64_cast_fp32_split",
                                    "peak_allocated_gib", "trace_buffer_peak_bound_bytes")}, flush=True)
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
