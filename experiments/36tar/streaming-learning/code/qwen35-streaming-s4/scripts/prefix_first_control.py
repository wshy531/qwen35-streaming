"""Original-model FIRST on only the frozen early four frames; old experiments stay read-only."""
import argparse
from contextlib import nullcontext
import inspect
import json
from pathlib import Path
import tempfile
import time
import traceback
import psutil
import torch
from transformers import AutoProcessor
from scripts.baseline import PROJECT, provenance, write_json, file_hash
from scripts.fp32_regression import strict_logits, strict_states
from scripts.pair_demo import verify_assets
from scripts.protocol_capability_control import original_package, source_assets, ORIGINAL
from scripts.scene_pair_control import FIRST, prepare
from scripts.original_first_control import build, Counter, generate, feature_fingerprint
from streaming.online import PairInputBuilder, fp32_reference

NAMES = {"early_hands": "hands_to_cat", "early_cat": "cat_to_cat"}
TARGETS = {"early_hands": "HANDS", "early_cat": "CAT"}
PLAN = PROJECT / "原模型早期四帧对照方案.md"
LAUNCHER = PROJECT / "scripts/run_prefix_first.py"
OLD_SCENE = PROJECT / "runs/s4b_scene_pairs_gpu_20261010_01/scene_pair_summary.json"
OLD_SIX = PROJECT / "runs/s4b_original_first_gpu_20261010_03/original_first_summary.json"
BUDGET = {"cases": 4, "max_tokens": 256, "max_new_tokens_per_answer": 8, "generated_tokens_max": 32,
          "language_forwards_max": 36, "vision_encodings": 4, "model_loads": 1, "wall_seconds": 600}


def independent_inputs(processor, frames):
    builder = PairInputBuilder(processor)
    prefix = builder.prefix_ids
    size = prefix.shape[1]
    seen = rope = size
    parts = []
    for i in range(0, len(frames), 2):
        inputs, _ = builder.pair(frames[i], frames[i + 1], seen, rope)
        parts.append(inputs)
        seen += inputs.input_ids.shape[1]
        rope = int(inputs.position_ids[1:].max()) + 1
    suffix = builder.question_ids(FIRST)
    positions = torch.stack((torch.arange(seen, seen + suffix.shape[1]),
                             *[torch.arange(rope, rope + suffix.shape[1]) for _ in range(3)]))[:, None, :]
    ids = torch.cat((prefix, *[p.input_ids for p in parts], suffix), 1)
    grid = parts[0].video_grid_thw.clone()
    grid[0, 0] = len(parts)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            "mm_token_type_ids": (ids == processor.video_token_id).long() * 2,
            "video_grid_thw": grid, "pixel_values_videos": torch.cat([p.pixel_values_videos for p in parts]),
            "position_ids": torch.cat((torch.arange(size)[None, None, :].expand(4, 1, -1),
                                        *[p.position_ids for p in parts], positions), 2)}


def preflight(out, args):
    from tests.test_online import small_model
    torch.set_num_threads(1)
    for dtype in (torch.bfloat16, torch.float32):
        probe = torch.arange(12, dtype=dtype).reshape(3, 4)
        assert feature_fingerprint(probe) == feature_fingerprint(probe.clone())
        changed = probe.clone()
        changed[0, 0] = 1
        assert feature_fingerprint(probe) != feature_fingerprint(changed)
    manifest = verify_assets(args.model_dir)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    current = small_model(processor)
    package = original_package()
    original = package.Qwen3_5ForConditionalGeneration(package.Qwen3_5Config(**current.config.to_dict())).eval()
    original.set_attn_implementation("sdpa")
    original.load_state_dict(current.state_dict(), strict=True)
    assert Path(inspect.getfile(type(original))).resolve() == ORIGINAL / "modeling_qwen3_5.py"
    cases, metadata = prepare(args.dataset)
    historic = json.loads(OLD_SCENE.read_text(encoding="utf-8"))
    six_report = json.loads(OLD_SIX.read_text(encoding="utf-8"))
    entries = []
    with torch.inference_mode(), fp32_reference():
        for name, old_name in NAMES.items():
            full, full_spec = build(original, processor, cases[old_name])
            assert full_spec["input_fingerprint"] == historic["input_fingerprints"][old_name]["first"]
            frames = cases[old_name][:4]
            inputs, spec = build(original, processor, frames)
            paired = independent_inputs(processor, frames)
            assert set(inputs) == set(paired)
            equality = {key: torch.equal(inputs[key], paired[key]) for key in inputs}
            assert all(equality.values())
            suffix = PairInputBuilder(processor).question_ids(FIRST)
            boundary = inputs.input_ids.shape[1] - suffix.shape[1]
            prefix = {key: torch.equal(inputs[key][..., :boundary], full[key][..., :boundary])
                      for key in ("input_ids", "attention_mask", "mm_token_type_ids", "position_ids")}
            pixel_count = inputs.pixel_values_videos.shape[0]
            prefix["pixels_equal_first_two_temporal_groups"] = torch.equal(inputs.pixel_values_videos, full.pixel_values_videos[:pixel_count])
            prefix["pixel_count_is_two_thirds"] = pixel_count * 3 == full.pixel_values_videos.shape[0] * 2
            prefix["same_question_suffix_ids"] = torch.equal(inputs.input_ids[:, -suffix.shape[1]:], full.input_ids[:, -suffix.shape[1]:])
            assert all(prefix.values()) and inputs.video_grid_thw.tolist() == [[2, 12, 16]]
            old = original(**inputs, use_cache=True)
            new = current(**inputs, use_cache=True)
            metrics, states = strict_logits(new.logits, old.logits), strict_states(new.past_key_values, old.past_key_values)
            assert metrics["bitwise_equal"] and all(s["all_tensors_bitwise_equal"] for s in states.values())
            del old, new
            counter = Counter(original)
            try:
                inputs, spec = build(original, processor, frames)
                tiny = generate(original, processor, inputs, counter)
            finally:
                cleaned = counter.close()
            assert cleaned
            previous = {r["mode"]: {"answer": r["answer"], "generated_ids": r["generated_ids"], "correct": r["correct"]}
                        for r in six_report["cases"] if r["input"] == old_name}
            assert len(previous) == 2
            entries.append({"id": name, "original_six_input": old_name, "target": TARGETS[name], "question": FIRST, **spec,
                            "experimental_frame_ids": [p.frame_id for p in frames], "experimental_pts": [p.timestamp for p in frames],
                            "full_six_fingerprint": full_spec["input_fingerprint"], "prefix_boundary_tokens": boundary,
                            "all_input_tensors_bitwise_equal": equality, "prefix_only_controls": prefix,
                            "tiny_prefill_logits": metrics, "tiny_prefill_states": states, "tiny_generate": tiny,
                            "hooks_restored": cleaned, "previous_six_by_mode": previous})
    write_json(out / "cpu_summary.json", {"input_controls": entries, "scope": "two tiny CPU interface controls; random answers are not capability evidence"})
    proof = {"cpu_controls_passed": True, "input_controls": entries, "budget": BUDGET,
             "cases": [{"id": f"{mode}_{name}", "mode": mode, "input": name} for mode in ("bf16_sdpa", "fp32_ieee_math") for name in NAMES],
             "cpu_report_sha256": file_hash(out / "cpu_summary.json"), "original_reference_hashes": source_assets(),
             "verified_model_assets": manifest["files"], "model_manifest_sha256": file_hash(args.model_dir / "DOWNLOAD_MANIFEST.json"),
             "fixture_metadata": metadata, "historical_report_sha256": file_hash(OLD_SCENE), "original_six_report_sha256": file_hash(OLD_SIX),
             "plan_sha256": file_hash(PLAN), "launcher_sha256": file_hash(LAUNCHER), "bf16_and_fp32_feature_hash_probes_passed": True}
    write_json(out / "preflight_summary.json", proof)
    print({"two_four_frame_inputs_and_original_tiny_controls_passed": True, "six_to_four_prefix_identity_verified": True, "budget": BUDGET}, flush=True)


@torch.inference_mode()
def gpu(out, args):
    begin = time.perf_counter()
    proof = json.loads((args.preflight / "preflight_summary.json").read_text(encoding="utf-8"))
    free, total = torch.cuda.mem_get_info()
    guard = {"gpu_free_bytes": free, "gpu_total_bytes": total, "ram_available_bytes": psutil.virtual_memory().available}
    write_json(out / "resource_precheck.json", guard)
    assert free >= 5.5 * 2**30 and guard["ram_available_bytes"] >= 3 * 2**30, "Resource guard; no load"
    package = original_package()
    torch.manual_seed(42)
    model = package.Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir, local_files_only=True, dtype=torch.bfloat16,
                        attn_implementation="sdpa").to("cuda").eval()
    assert Path(inspect.getfile(type(model))).resolve() == ORIGINAL / "modeling_qwen3_5.py"
    torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    cases, metadata = prepare(args.dataset)
    assert metadata == proof["fixture_metadata"] and source_assets() == proof["original_reference_hashes"]
    counter = Counter(model)
    report = {"budget": BUDGET, "cases": [], "model_reference": str(ORIGINAL / "modeling_qwen3_5.py"),
              "original_reference_hashes": source_assets(), "fixture_metadata": metadata, "resource_precheck": guard,
              "scope": "original unmodified model, independent complete early-four-frame encoding; fixed FIRST"}
    generated = 0
    try:
        for mode in ("bf16_sdpa", "fp32_ieee_math"):
            if mode == "fp32_ieee_math":
                assert torch.cuda.mem_get_info()[0] >= 2 * 2**30
                model.float()
                assert all(p.dtype == torch.float32 for p in model.parameters())
            with (fp32_reference() if mode == "fp32_ieee_math" else nullcontext()):
                for spec in proof["input_controls"]:
                    frames = cases[spec["original_six_input"]][:4]
                    inputs, identity = build(model, processor, frames)
                    assert identity["input_fingerprint"] == spec["input_fingerprint"]
                    answer = generate(model, processor, inputs, counter)
                    generated += len(answer["generated_ids"])
                    previous = spec["previous_six_by_mode"][mode]
                    record = {"id": f"{mode}_{spec['id']}", "input": spec["id"], "mode": mode, "target": spec["target"],
                              "question": FIRST, **identity, **answer, "correct": answer["answer"].strip().upper() == spec["target"],
                              "experimental_frame_ids": [p.frame_id for p in frames], "experimental_pts": [p.timestamp for p in frames],
                              "previous_six": previous, "answer_changed_from_six": answer["generated_ids"] != previous["generated_ids"]}
                    report["cases"].append(record)
                    write_json(out / "prefix_first_summary.json", report)
                    print({"id": record["id"], "answer": answer["answer"], "target": spec["target"], "correct": record["correct"]}, flush=True)
            if mode == "bf16_sdpa":
                report["bf16_stage_peak_memory"] = {"allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                                                     "reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        assert [r["id"] for r in report["cases"]] == [r["id"] for r in proof["cases"]]
        assert counter.counts["vision"] == 4 and counter.counts["language"] <= 36 and generated <= 32
        report.update(actual_forwards=counter.counts, generated_tokens=generated, all_registered_cases_executed=True,
                      all_input_fingerprints_verified=True, wall_seconds_including_load=time.perf_counter() - begin,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                      peak_scope="after BF16 load through BF16/FP32 conversion and all inference; never reset at conversion")
        write_json(out / "prefix_first_summary.json", report)
    finally:
        restored = counter.close()
        write_json(out / "cleanup.json", {"hooks_restored": restored, "source_readers_closed_during_prepare": True})
        assert restored


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "gpu"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=PROJECT / "models/Qwen3.5-0.8B")
    parser.add_argument("--dataset", type=Path, default=PROJECT / "datasets/natural_commons_v1")
    parser.add_argument("--preflight", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "tmp").mkdir()
    tempfile.tempdir = str((args.out / "tmp").resolve())
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
