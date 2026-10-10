"""Bounded four-frame FP32 vision trace; no language evaluation or tensor dumps."""
import argparse
import gc
import hashlib
import itertools
import json
from pathlib import Path
import time
import traceback

import psutil
import torch
from transformers import AutoProcessor

from scripts.baseline import PROJECT, REPO, provenance, write_json, file_hash
from scripts.online_experiment import reference_inputs
from scripts.pair_demo import verify_assets
from streaming.inputs import tensor_fingerprint
from streaming.online import PairInputBuilder, fp32_reference
from streaming.window import read_packets

PLAN = PROJECT / "视觉首次差异定位方案.md"
LAUNCHER = PROJECT / "scripts/run_vision_trace.py"
HISTORY = PROJECT / "runs/s4b_shared_gpu_20261009_01/shared_features_summary.json"
BUDGET = {"vision_encodings": 4, "language_forwards": 0, "generated_tokens": 0,
          "model_loads": 1, "wall_seconds": 600, "trace_buffer_bytes": 128 * 2**20}


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def tensor_hash(value):
    value = value.detach().cpu().contiguous()
    h = hashlib.sha256(str((value.dtype, tuple(value.shape))).encode())
    h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def difference(left, right):
    assert left.shape == right.shape and left.dtype == right.dtype
    equal = torch.equal(left, right)
    delta = left.double()
    delta.sub_(right)
    return {"shape": list(left.shape), "dtype": str(left.dtype), "bitwise_equal": equal,
            "different_elements": int(torch.count_nonzero(left != right)),
            "max_abs_diff": float(delta.abs().max()), "mean_abs_diff": float(delta.abs().mean()),
            "rms_diff": float(delta.square().mean().sqrt()),
            "allclose_1e4": bool(torch.allclose(left, right, atol=1e-4, rtol=1e-4)),
            "actual_sha256": tensor_hash(left), "reference_sha256": tensor_hash(right)}


def packets(video):
    reader = read_packets(video, 1)
    try:
        frames = tuple(itertools.islice(reader, 4))
    finally:
        reader.close()
    assert len(frames) == 4
    return frames


def inputs(model, processor, frames, canvas):
    builder = PairInputBuilder(processor, canvas)
    full = reference_inputs(model, processor, frames, canvas)
    first, first_meta = builder.pair(*frames[:2], builder.prefix_ids.shape[1], builder.prefix_ids.shape[1])
    next_rope = int(first.position_ids[1:].max()) + 1
    second, second_meta = builder.pair(*frames[2:], builder.prefix_ids.shape[1] + first.input_ids.shape[1], next_rope)
    assert torch.equal(torch.cat((first.pixel_values_videos, second.pixel_values_videos)), full.pixel_values_videos)
    grids = torch.cat((first.video_grid_thw, second.video_grid_thw))
    assert grids[:, 0].tolist() == [1, 1] and int(full.video_grid_thw[0, 0]) == 2
    assert torch.equal(grids[:, 1:], full.video_grid_thw[:, 1:].expand(2, -1))
    assert torch.equal(torch.cat((builder.prefix_ids, first.input_ids, second.input_ids), 1), full.input_ids)
    prefix = torch.arange(builder.prefix_ids.shape[1])[None, None].expand(4, -1, -1)
    assert torch.equal(torch.cat((prefix, first.position_ids, second.position_ids), 2), full.position_ids)
    report = {"pixels_bitwise_equal": True, "grids_aligned_by_temporal_unit": True,
              "tokens_bitwise_equal": True, "four_axis_positions_bitwise_equal": True,
              "full_grid": full.video_grid_thw.tolist(), "pair_grids": grids.tolist(),
              "full_pixels_shape": list(full.pixel_values_videos.shape),
              "frame_ids": [f.frame_id for f in frames], "timestamps": [f.timestamp for f in frames],
              "frame_sha256": [hashlib.sha256(f.rgb.tobytes()).hexdigest() for f in frames],
              "pairs": [first_meta, second_meta],
              "input_fingerprints": {"prefix4": tensor_fingerprint(full),
                                     "prefix2": tensor_fingerprint(reference_inputs(model, processor, frames[:2], canvas))}}
    return (full, first, second), report


class Trace:
    """CPU references only, aligned per time unit; temporary edits restored on all exits."""
    def __init__(self, model, buffer_limit=BUDGET["trace_buffer_bytes"], detailed_block=0):
        self.model = model
        self.visual = model.model.visual
        self.limit = buffer_limit
        self.block = detailed_block
        self.references, self.stages, self.hooks, self.methods = {}, [], [], {}
        self.reference_bytes = self.buffer_peak_bound = self.rss_peak = 0
        self.calls, self.language_calls = 0, 0
        self.records, self.resources, self.boundaries = {}, [], {}
        self.mode = "full"
        self.offset = None
        self.order = []

    def emit(self, name, value):
        assert isinstance(value, torch.Tensor) and value.ndim >= 1
        size = value.numel() * value.element_size()
        # Covers the temporary CPU copy, float64 difference/statistic/hash workspaces.
        bound = self.reference_bytes + 12 * size
        if bound > self.limit:
            raise MemoryError(f"Trace buffer hard bound exceeded at {name}: {bound} > {self.limit}")
        self.buffer_peak_bound = max(self.buffer_peak_bound, bound)
        self.rss_peak = max(self.rss_peak, psutil.Process().memory_info().rss)
        cpu = value.detach().to("cpu", copy=True).contiguous()
        self.order.append(name)
        if self.mode == "full":
            assert name not in self.references
            self.references[name] = cpu
            self.reference_bytes += size
            self.stages.append(name)
            self.records[self.mode].append({"stage": name, "shape": list(cpu.shape), "dtype": str(cpu.dtype),
                                            "sha256": tensor_hash(cpu)})
        else:
            reference = self.references[name]
            if self.offset is not None:
                assert reference.shape[0] == cpu.shape[0] * 2, (name, reference.shape, cpu.shape)
                reference = reference[self.offset * cpu.shape[0]:(self.offset + 1) * cpu.shape[0]]
            self.records[self.mode].append({"stage": name, **difference(cpu, reference)})

    def module(self, module, name, pre=False):
        if pre:
            def capture(_module, args):
                self.emit(name, args[0])
            self.hooks.append(module.register_forward_pre_hook(capture))
        else:
            def capture(_module, _args, value):
                self.emit(name, value)
            self.hooks.append(module.register_forward_hook(capture))

    def __enter__(self):
        self.hooks.append(self.visual.register_forward_pre_hook(self.vision_start))
        self.hooks.append(self.model.model.language_model.register_forward_pre_hook(self.language_start))
        self.module(self.visual.patch_embed, "patch.output")
        for attr, name in (("fast_pos_embed_interpolate", "position.interpolate"), ("rot_pos_emb", "position.rotary")):
            saved = (attr in self.visual.__dict__, self.visual.__dict__.get(attr))
            original = getattr(self.visual, attr)
            self.methods[attr] = saved
            def wrapped(*args, _original=original, _name=name, **kwargs):
                value = _original(*args, **kwargs)
                self.emit(_name, value)
                return value
            setattr(self.visual, attr, wrapped)
        for i, block in enumerate(self.visual.blocks):
            self.module(block, f"blocks.{i}.input", pre=True)
            self.module(block, f"blocks.{i}.output")
            if i == self.block:
                for attr in ("norm1", "attn.qkv", "attn.proj", "norm2", "mlp.linear_fc1", "mlp.linear_fc2"):
                    module = block.get_submodule(attr)
                    self.module(module, f"blocks.{i}.{attr}.output")
                    if attr in ("attn.proj", "mlp.linear_fc2"):
                        self.module(module, f"blocks.{i}.{attr}.input", pre=True)
        self.hooks.append(self.visual.blocks[0].register_forward_pre_hook(self.positions, with_kwargs=True))
        for name, module in self.visual.merger.named_modules():
            if name and (isinstance(module, (torch.nn.LayerNorm, torch.nn.Linear, torch.nn.GELU))):
                self.module(module, "merger." + name + ".output")
        return self

    def positions(self, _module, _args, kwargs):
        self.boundaries[self.mode] = kwargs["cu_seqlens"].detach().cpu().tolist()
        for name, value in zip(("cos", "sin"), kwargs["position_embeddings"]):
            self.emit("position." + name, value)

    def vision_start(self, _module, _args):
        if self.calls >= BUDGET["vision_encodings"]:
            raise RuntimeError("Vision budget exceeded before forward")
        self.calls += 1

    def language_start(self, _module, _args):
        self.language_calls += 1
        raise RuntimeError("Language forward forbidden in vision-only diagnostic")

    def start(self, mode, offset=None):
        self.mode, self.offset, self.order = mode, offset, []
        self.records[mode] = []

    def finish(self):
        assert self.order == self.stages, "Capture order differs"
        info = {"mode": self.mode, "rss_bytes": psutil.Process().memory_info().rss,
                "ram_available_bytes": psutil.virtual_memory().available, "reference_bytes": self.reference_bytes}
        if torch.cuda.is_initialized():
            info.update(cuda_allocated_bytes=torch.cuda.memory_allocated(), cuda_reserved_bytes=torch.cuda.memory_reserved())
        self.resources.append(info)

    def clear(self):
        self.references.clear()
        self.reference_bytes = 0

    def __exit__(self, *_exception):
        for handle in self.hooks:
            handle.remove()
        for attr, (existed, value) in self.methods.items():
            if existed:
                setattr(self.visual, attr, value)
            else:
                delattr(self.visual, attr)
        self.hooks.clear()
        self.methods.clear()
        self.clear()


def hook_state(model):
    return [(name, len(module._forward_pre_hooks), len(module._forward_hooks))
            for name, module in model.named_modules()]


@torch.inference_mode()
def controls(model, batch, out=None):
    device = next(model.parameters()).device
    before = hook_state(model)
    pooled = []
    with Trace(model) as trace, fp32_reference():
        for mode, which, offset in (("full", 0, None), ("repeat", 0, None), ("pair0", 1, 0), ("pair1", 2, 1)):
            assert psutil.virtual_memory().available >= 1 * 2**30, "RAM guard before vision forward"
            trace.start(mode, offset)
            visual = {name: batch[which][name].to(device) for name in ("pixel_values_videos", "video_grid_thw")}
            result = model.model.visual(visual["pixel_values_videos"], grid_thw=visual["video_grid_thw"])
            trace.emit("visual.pooler_output", result.pooler_output)
            trace.finish()
            pooled.append(result.pooler_output.detach().cpu().clone())
            del visual, result
            if device.type == "cuda":
                torch.cuda.synchronize()
            if out:
                write_json(out / "trace_partial.json", {"records": trace.records, "resources": trace.resources})
        full_boundaries = trace.boundaries["full"]
        assert full_boundaries == trace.boundaries["repeat"]
        size = full_boundaries[1]
        assert full_boundaries == [0, size, size * 2]
        assert trace.boundaries["pair0"] == trace.boundaries["pair1"] == [0, size]
        combined = torch.cat(pooled[2:])
        report = {"budget": BUDGET, "stages": trace.stages, "records": trace.records,
                  "attention_boundaries": trace.boundaries, "boundaries_aligned": True,
                  "repeat_all_stages_bitwise": all(x["bitwise_equal"] for x in trace.records["repeat"]),
                  "first_differences": {mode: next((r for r in trace.records[mode] if not r["bitwise_equal"]), None)
                                        for mode in ("repeat", "pair0", "pair1")},
                  "combined_features": difference(combined, pooled[0]),
                  "actual_forwards": {"vision": trace.calls, "language": trace.language_calls},
                  "generated_tokens": 0, "reference_tensor_bytes": trace.reference_bytes,
                  "trace_buffer_peak_bound_bytes": trace.buffer_peak_bound, "trace_rss_peak_bytes": trace.rss_peak,
                  "resources": trace.resources, "scope": "vision-only first observed boundary; not a unique-root-cause proof"}
    assert before == hook_state(model) and trace.reference_bytes == 0
    report.update(hooks_and_methods_restored=True, reference_tensors_released=True)
    return report


@torch.inference_mode()
def preflight(out, args):
    from tests.test_online import small_model
    torch.set_num_threads(1)
    manifest = verify_assets(args.model_dir)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = small_model(processor)
    frames = packets(args.video)
    batch, identity = inputs(model, processor, frames, (192, 256))
    assert identity["input_fingerprints"] == {k: read(HISTORY)["input_fingerprints"][k] for k in ("prefix4", "prefix2")}
    before = hook_state(model)
    with fp32_reference():
        native = model.model.visual(batch[0].pixel_values_videos, grid_thw=batch[0].video_grid_thw).pooler_output
    cpu = controls(model, batch)
    assert tensor_hash(native) == cpu["records"]["full"][-1]["sha256"]
    assert cpu["repeat_all_stages_bitwise"] and before == hook_state(model)
    try:
        with Trace(model, buffer_limit=1) as trace:
            trace.start("full")
            model.model.visual(batch[0].pixel_values_videos, grid_thw=batch[0].video_grid_thw)
        raise AssertionError("Buffer-limit failure was not triggered")
    except MemoryError:
        pass
    assert before == hook_state(model) and not trace.references
    cpu.update(capture_does_not_change_output=True, exception_cleanup_verified=True,
               input_identity=identity, scope="real processor, tiny random weights; interface evidence only")
    write_json(out / "cpu_summary.json", cpu)
    write_json(out / "preflight_summary.json", {"cpu_controls_passed": True, "budget": BUDGET,
        "input_identity": identity, "cpu_report_sha256": file_hash(out / "cpu_summary.json"),
        "video_sha256": file_hash(args.video), "model_manifest_sha256": file_hash(args.model_dir / "DOWNLOAD_MANIFEST.json"),
        "verified_model_assets": manifest["files"], "historical_report_sha256": file_hash(HISTORY),
        "plan_sha256": file_hash(PLAN), "launcher_sha256": file_hash(LAUNCHER)})
    print({"cpu_controls_passed": True, "capture_stages": len(cpu["stages"]), "input_identity": True}, flush=True)


@torch.inference_mode()
def gpu(out, args):
    from qwen35_local import Qwen3_5ForConditionalGeneration
    begin = time.perf_counter()
    torch.manual_seed(42)
    free, total = torch.cuda.mem_get_info()
    guard = {"gpu_free_bytes": free, "gpu_total_bytes": total, "ram_available_bytes": psutil.virtual_memory().available}
    write_json(out / "resource_precheck.json", guard)
    assert free >= 5.5 * 2**30 and guard["ram_available_bytes"] >= 3 * 2**30, "Resource guard; no model loading"
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
        report = controls(model, batch, out)
        report.update(input_identity=identity, wall_seconds_including_load=time.perf_counter() - begin,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30, resource_precheck=guard)
        write_json(out / "vision_trace_summary.json", report)
        print({key: report[key] for key in ("first_differences", "combined_features", "peak_allocated_gib",
                                          "peak_reserved_gib", "trace_buffer_peak_bound_bytes")}, flush=True)
        del batch, processor, report
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()
        write_json(out / "cleanup.json", {"model_released": True, "cuda_allocated_bytes_after_release": torch.cuda.memory_allocated(),
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
