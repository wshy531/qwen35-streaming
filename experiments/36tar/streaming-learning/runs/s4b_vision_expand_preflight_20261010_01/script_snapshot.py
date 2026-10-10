"""Six bounded independent vision cases; candidate vs pair and original reference."""
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

from scripts.baseline import PROJECT, provenance, write_json, file_hash
from scripts.online_experiment import reference_inputs
from scripts.pair_demo import verify_assets
from scripts.vision_trace import tensor_hash, difference, hook_state
from scripts.vision_fixed_control import forward_attributes
from scripts.vision_fixed_units import FixedTemporalLinears
from streaming.inputs import tensor_fingerprint
from streaming.online import PairInputBuilder, fp32_reference
from streaming.window import read_packets

PLAN = PROJECT / "扩展视觉一致性验收方案.md"
LAUNCHER = PROJECT / "scripts/run_vision_extension.py"
BUDGET = {"vision_encodings": 31, "candidate_projection_operations_max": 1900,
          "language_forwards": 0, "generated_tokens": 0, "model_loads": 1,
          "wall_seconds": 600, "comparison_buffer_bytes": 128 * 2**20}
CASES = [("hands6_standard", "clean_hands", 6, (192, 256)),
         ("cat6_standard", "cat_play", 6, (192, 256)),
         ("hands8_small", "clean_hands", 8, (128, 192)),
         ("cat8_portrait", "cat_play", 8, (256, 128)),
         ("cat4_square", "cat_play", 4, (256, 256)),
         ("hands5_eof", "clean_hands", 5, (192, 256))]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def frames_for(source, count):
    reader = read_packets(PROJECT / f"datasets/natural_commons_v1/{source}.webm", 1)
    try:
        frames = tuple(itertools.islice(reader, count))
    finally:
        reader.close()
    assert len(frames) == count
    return frames


def general_inputs(model, processor, frames, canvas):
    full = reference_inputs(model, processor, frames, canvas)
    builder = PairInputBuilder(processor, canvas)
    padded = frames if len(frames) % 2 == 0 else frames + (frames[-1],)
    seen = rope = builder.prefix_ids.shape[1]
    pairs, metadata = [], []
    for index in range(0, len(padded), 2):
        batch, meta = builder.pair(*padded[index:index+2], seen, rope,
            synthetic_padding=index + 1 >= len(frames))
        seen += batch.input_ids.shape[1]
        rope = int(batch.position_ids[1:].max()) + 1
        pairs.append(batch)
        metadata.append(meta)
    assert torch.equal(torch.cat([b.pixel_values_videos for b in pairs]), full.pixel_values_videos)
    assert torch.equal(torch.cat([builder.prefix_ids] + [b.input_ids for b in pairs], 1), full.input_ids)
    prefix = torch.arange(builder.prefix_ids.shape[1])[None, None].expand(4, -1, -1)
    assert torch.equal(torch.cat([prefix] + [b.position_ids for b in pairs], 2), full.position_ids)
    assert torch.equal(torch.cat([b.mm_token_type_ids for b in pairs], 1),
                       full.mm_token_type_ids[:, builder.prefix_ids.shape[1]:])
    assert int(full.video_grid_thw[0, 0]) == len(pairs)
    assert all(b.video_grid_thw.tolist() == [[1, *full.video_grid_thw[0, 1:].tolist()]] for b in pairs)
    identity = {"pixels_tokens_positions_types_equal": True, "full_grid": full.video_grid_thw.tolist(),
        "canvas": list(canvas), "real_frames": len(frames), "padded_frames": len(padded),
        "frame_ids": [f.frame_id for f in frames], "timestamps": [f.timestamp for f in frames],
        "frame_sha256": [hashlib.sha256(f.rgb.tobytes()).hexdigest() for f in frames],
        "pairs": metadata, "full_fingerprint": tensor_fingerprint(full)}
    return full, pairs, identity


def assets():
    return {str(PROJECT / f"datasets/natural_commons_v1/{source}.webm"):
            file_hash(PROJECT / f"datasets/natural_commons_v1/{source}.webm")
            for source in ("clean_hands", "cat_play")}


@torch.inference_mode()
def controls(model, processor, out, expected=None, real=False):
    before, attrs = hook_state(model), forward_attributes(model)
    counts = {"vision": 0, "language": 0}
    def vision_start(_module, _args):
        if counts["vision"] >= BUDGET["vision_encodings"]:
            raise RuntimeError("Visual budget exceeded")
        counts["vision"] += 1
    def language_start(_module, _args):
        counts["language"] += 1
        raise RuntimeError("No language in visual expansion")
    handles = [model.model.visual.register_forward_pre_hook(vision_start),
               model.model.language_model.register_forward_pre_hook(language_start)]
    device = next(model.parameters()).device
    reports, identities, total_ops, buffer_peak = [], [], 0, 0
    try:
        with fp32_reference():
            for case_id, source, count, canvas in CASES:
                frames = frames_for(source, count)
                full, pairs, identity = general_inputs(model, processor, frames, canvas)
                identity["case_id"] = case_id
                if expected is not None:
                    assert identity == expected[len(identities)], "Frozen input mismatch"
                identities.append(identity)
                def visual(batch):
                    value = model.model.visual(batch.pixel_values_videos.to(device),
                        grid_thw=batch.video_grid_thw.to(device)).pooler_output
                    return value.cpu().clone()
                original = visual(full)
                remaining = BUDGET["candidate_projection_operations_max"] - total_ops
                with FixedTemporalLinears(model.model.visual, max_operations=remaining) as fixed:
                    candidate = visual(full)
                    parts = [visual(pair) for pair in pairs]
                    operations, calls = fixed.operations, fixed.calls
                total_ops += operations
                combined = torch.cat(parts)
                input_bytes = sum(v.numel()*v.element_size() for b in [full, *pairs] for v in b.values())
                bound = sum(f.rgb.nbytes for f in frames) + input_bytes + 20*candidate.numel()*candidate.element_size()
                buffer_peak = max(buffer_peak, bound)
                assert bound <= BUDGET["comparison_buffer_bytes"]
                reference = difference(candidate, combined)
                bias = difference(candidate, original)
                reports.append({"case_id": case_id, "identity": identity, "candidate_vs_pairs": reference,
                    "candidate_vs_original_full": bias, "candidate_projection_operations": operations,
                    "candidate_vision_calls": calls, "buffer_bound_bytes": bound,
                    "cpu_random_weights_only": not real})
                # Preserve evidence even if a later correctness assertion fails.
                write_json(out / "expansion_partial.json", {"cases": reports, "actual_forwards": counts,
                    "candidate_projection_operations": total_ops})
                assert reference["bitwise_equal"], f"Candidate differs for {case_id}"
                del frames, full, pairs, original, candidate, parts, combined
    finally:
        for handle in handles:
            handle.remove()
    assert before == hook_state(model) and attrs == forward_attributes(model)
    assert counts == {"vision": 31, "language": 0}
    if real:
        assert total_ops == 1900
    return {"cases": reports, "input_identity": identities, "budget": BUDGET,
        "actual_forwards": counts, "candidate_projection_operations": total_ops,
        "comparison_buffer_peak_bound_bytes": buffer_peak, "all_cases_bitwise_equal": True,
        "methods_hooks_restored": True, "scope": "fixed candidate FP32 numerical evidence, not original-full/BF16/semantics"}


@torch.inference_mode()
def preflight(out, args):
    from tests.test_online import small_model
    torch.set_num_threads(1)
    manifest = verify_assets(args.model_dir)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = small_model(processor)
    weights = {k: v.clone() for k, v in model.model.visual.state_dict().items()}
    report = controls(model, processor, out)
    assert all(torch.equal(v, weights[k]) for k, v in model.model.visual.state_dict().items())
    attrs, hooks = forward_attributes(model), hook_state(model)
    full, pairs, _ = general_inputs(model, processor, frames_for("clean_hands", 5), (192, 256))
    for kind in ("budget", "grid"):
        try:
            with FixedTemporalLinears(model.model.visual, max_operations=0 if kind == "budget" else 300):
                grid = pairs[0].video_grid_thw if kind == "budget" else pairs[0].video_grid_thw.repeat(2, 1)
                model.model.visual(pairs[0].pixel_values_videos, grid_thw=grid)
            raise AssertionError("Missing rejection")
        except (RuntimeError, ValueError) as error:
            assert ("projection budget" if kind == "budget" else "Single-video grid") in str(error)
        assert attrs == forward_attributes(model) and hooks == hook_state(model)
    report.update(weights_bitwise_unchanged=True, budget_grid_exception_cleanup=True)
    write_json(out / "cpu_summary.json", report)
    write_json(out / "preflight_summary.json", {"cpu_controls_passed": True, "budget": BUDGET,
        "input_identity": report["input_identity"], "cpu_report_sha256": file_hash(out / "cpu_summary.json"),
        "assets": assets(), "model_manifest_sha256": file_hash(args.model_dir / "DOWNLOAD_MANIFEST.json"),
        "verified_model_assets": manifest["files"], "plan_sha256": file_hash(PLAN), "launcher_sha256": file_hash(LAUNCHER)})
    print({"cpu_controls_passed": True, "cases": len(report["cases"]), "exception_cleanup": True}, flush=True)


@torch.inference_mode()
def gpu(out, args):
    from qwen35_local import Qwen3_5ForConditionalGeneration
    start = time.perf_counter()
    free, total = torch.cuda.mem_get_info()
    guard = {"gpu_free_bytes": free, "gpu_total_bytes": total, "ram_available_bytes": psutil.virtual_memory().available}
    write_json(out / "resource_precheck.json", guard)
    assert free >= 5.5*2**30 and guard["ram_available_bytes"] >= 3*2**30, "No load: resource guard"
    model = None
    try:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        torch.cuda.reset_peak_memory_stats()
        assert torch.cuda.mem_get_info()[0] >= 2*2**30 and psutil.virtual_memory().available >= 2*2**30
        model.float()
        processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
        report = controls(model, processor, out, read(args.preflight / "preflight_summary.json")["input_identity"], real=True)
        report.update(wall_seconds_including_load=time.perf_counter()-start,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        write_json(out / "expansion_summary.json", report)
        print({"cases_bitwise_equal": len(report["cases"]), "actual_forwards": report["actual_forwards"],
               "peak_allocated_gib": report["peak_allocated_gib"]}, flush=True)
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
