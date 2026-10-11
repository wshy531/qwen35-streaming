"""Original-model equal-length FIRST controls, same early hands and different final pair."""
import argparse
import hashlib
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

from scripts.baseline import PROJECT, provenance, write_json, file_hash, add_video_positions, check_position_oracle
from scripts.fp32_regression import strict_logits, strict_states
from scripts.online_experiment import reference_inputs
from scripts.pair_demo import verify_assets
from scripts.protocol_capability_control import original_package, source_assets, ORIGINAL
from scripts.scene_pair_control import FIRST, prepare, FIXTURE
from streaming.inputs import tensor_fingerprint
from streaming.online import PairInputBuilder, fp32_reference

NAMES = ("hands_to_hands", "hands_to_cat")
TARGETS = {name: "HANDS" for name in NAMES}
PLAN = PROJECT / "原模型六帧等长末尾对照方案.md"
LAUNCHER = PROJECT / "scripts/run_equal_tail.py"
OLD_REPORT = PROJECT / "runs/s4b_scene_pairs_gpu_20261010_01/scene_pair_summary.json"
LINKED_REPORTS = (PROJECT / "runs/s4b_original_first_gpu_20261010_03/original_first_summary.json",
                  PROJECT / "runs/s4b_prefix_first_gpu_20261010_01/prefix_first_summary.json")
BUDGET = {"cases": 4, "max_tokens": 256, "max_new_tokens_per_answer": 8, "generated_tokens_max": 32,
          "language_forwards_max": 36, "vision_encodings": 4, "model_loads": 1, "wall_seconds": 600}


from scripts.original_first_control import build, Counter, generate, feature_fingerprint, independent_pair_inputs


def equal_tail_inputs(model, processor, cases):
    left, right = (cases[name] for name in NAMES)
    early_equal = [bool((left[i].rgb == right[i].rgb).all()) for i in range(4)]
    tail_different = [not bool((left[i].rgb == right[i].rgb).all()) for i in (4, 5)]
    a, a_spec = build(model, processor, left)
    b, b_spec = build(model, processor, right)
    layout = {key: torch.equal(a[key], b[key]) for key in a if key != "pixel_values_videos"}
    pixels_a, pixels_b = a.pixel_values_videos, b.pixel_values_videos
    assert pixels_a.shape == pixels_b.shape and pixels_a.shape[0] % 3 == 0
    boundary = pixels_a.shape[0] // 3 * 2
    result = {"cases": list(NAMES), "early_four_rgb_equal": early_equal, "tail_two_rgb_different": tail_different,
              "nonpixel_input_tensors_bitwise_equal": layout, "pixel_shape": list(pixels_a.shape),
              "early_pixel_rows": boundary, "early_patch_rows_bitwise_equal": torch.equal(pixels_a[:boundary], pixels_b[:boundary]),
              "tail_patch_rows_different": not torch.equal(pixels_a[boundary:], pixels_b[boundary:]),
              "experimental_frame_ids": [[p.frame_id for p in frames] for frames in (left, right)],
              "experimental_pts": [[p.timestamp for p in frames] for frames in (left, right)],
              "input_tokens": [a_spec["input_tokens"], b_spec["input_tokens"]],
              "input_fingerprints": {NAMES[0]: a_spec["input_fingerprint"], NAMES[1]: b_spec["input_fingerprint"]}}
    assert all(early_equal) and all(tail_different) and all(layout.values())
    assert result["early_patch_rows_bitwise_equal"] and result["tail_patch_rows_different"]
    assert result["input_tokens"] == [205, 205] and a.video_grid_thw.tolist() == [[3, 12, 16]]
    assert result["experimental_frame_ids"] == [list(range(6))] * 2 and result["experimental_pts"] == [[float(i) for i in range(6)]] * 2
    return result


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
    historical = json.loads(OLD_REPORT.read_text(encoding="utf-8"))
    pair_control = equal_tail_inputs(original, processor, cases)
    entries = []
    with torch.inference_mode(), fp32_reference():
        for name in NAMES:
            inputs, spec = build(original, processor, cases[name])
            paired = independent_pair_inputs(processor, cases[name])
            assert set(inputs) == set(paired)
            equality = {key: torch.equal(inputs[key], paired[key]) for key in inputs}
            assert all(equality.values())
            assert spec["input_fingerprint"] == historical["input_fingerprints"][name]["first"]
            old = original(**inputs, use_cache=True)
            new = current(**inputs, use_cache=True)
            metrics = strict_logits(new.logits, old.logits)
            states = strict_states(new.past_key_values, old.past_key_values)
            assert metrics["bitwise_equal"] and all(s["all_tensors_bitwise_equal"] for s in states.values())
            del old, new
            counter = Counter(original)
            try:
                # Restore the explicitly computed delta after any prior forward.
                inputs, spec = build(original, processor, cases[name])
                tiny = generate(original, processor, inputs, counter)
            finally:
                cleaned = counter.close()
            assert cleaned
            previous = next(r for r in historical["records"] if r["case"] == name and r["path"] == "online" and r["task"] == "first")
            entries.append({"id": name, "target": TARGETS[name], "question": FIRST, **spec,
                            "all_input_tensors_bitwise_equal": equality, "tiny_prefill_logits": metrics,
                            "tiny_prefill_states": states, "tiny_generate": tiny, "hooks_restored": cleaned,
                            "previous_online_answer": previous["answer"]["answer"], "previous_online_ids": previous["answer"]["generated_ids"]})
    write_json(out / "cpu_summary.json", {"paired_input_control": pair_control, "input_controls": entries, "scope": "two tiny original-model interface controls; random answers are not capability evidence"})
    proof = {"cpu_controls_passed": True, "paired_input_control": pair_control, "input_controls": entries, "budget": BUDGET,
             "cases": [{"id": f"{mode}_{name}", "mode": mode, "input": name} for mode in ("bf16_sdpa", "fp32_ieee_math") for name in NAMES],
             "cpu_report_sha256": file_hash(out / "cpu_summary.json"), "original_reference_hashes": source_assets(),
             "verified_model_assets": manifest["files"], "model_manifest_sha256": file_hash(args.model_dir / "DOWNLOAD_MANIFEST.json"),
             "fixture_metadata": metadata, "historical_report_sha256": file_hash(OLD_REPORT),
             "linked_report_hashes": {str(p): file_hash(p) for p in LINKED_REPORTS},
             "plan_sha256": file_hash(PLAN), "launcher_sha256": file_hash(LAUNCHER)}
    write_json(out / "preflight_summary.json", proof)
    print({"equal_length_same_early_pixels_and_layout_different_tail_verified": True, "two_inputs_and_tiny_fresh_prefills_bitwise_equal": True, "tiny_original_generate_and_positions_verified": True,
           "original_source": str(ORIGINAL), "budget": BUDGET}, flush=True)


@torch.inference_mode()
def gpu(out, args):
    begin = time.perf_counter()
    proof = json.loads((args.preflight / "preflight_summary.json").read_text(encoding="utf-8"))
    free, total = torch.cuda.mem_get_info()
    guard = {"gpu_free_bytes": free, "gpu_total_bytes": total, "ram_available_bytes": psutil.virtual_memory().available}
    write_json(out / "resource_precheck.json", guard)
    assert free >= 5.5 * 2**30 and guard["ram_available_bytes"] >= 3 * 2**30, "Resource guard; no load"
    torch.manual_seed(42)
    package = original_package()
    model = package.Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir, local_files_only=True, dtype=torch.bfloat16,
                        attn_implementation="sdpa").to("cuda").eval()
    assert Path(inspect.getfile(type(model))).resolve() == ORIGINAL / "modeling_qwen3_5.py"
    torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    cases, metadata = prepare(args.dataset)
    assert metadata == proof["fixture_metadata"] and source_assets() == proof["original_reference_hashes"]
    pair_control = equal_tail_inputs(model, processor, cases)
    assert pair_control == proof["paired_input_control"]
    counter = Counter(model)
    report = {"budget": BUDGET, "paired_input_control": pair_control, "cases": [], "model_reference": str(ORIGINAL / "modeling_qwen3_5.py"),
              "original_reference_hashes": source_assets(), "fixture_metadata": metadata, "resource_precheck": guard,
              "scope": "original unmodified model, independent complete six-frame visual encoding; fixed FIRST with equal-length/different-tail control"}
    generated = 0
    try:
        for mode in ("bf16_sdpa", "fp32_ieee_math"):
            if mode == "fp32_ieee_math":
                assert torch.cuda.mem_get_info()[0] >= 2 * 2**30
                model.float()
                assert all(p.dtype == torch.float32 for p in model.parameters())
            with (fp32_reference() if mode == "fp32_ieee_math" else nullcontext()):
                for spec in proof["input_controls"]:
                    inputs, identity = build(model, processor, cases[spec["id"]])
                    assert identity["input_fingerprint"] == spec["input_fingerprint"]
                    answer = generate(model, processor, inputs, counter)
                    generated += len(answer["generated_ids"])
                    record = {"id": f"{mode}_{spec['id']}", "input": spec["id"], "mode": mode,
                              "target": spec["target"], "question": FIRST, **identity, **answer,
                              "correct": answer["answer"].strip().upper() == spec["target"],
                              "matches_previous_online_ids": answer["generated_ids"] == spec["previous_online_ids"]}
                    report["cases"].append(record)
                    write_json(out / "equal_tail_summary.json", report)
                    print({"id": record["id"], "answer": answer["answer"], "target": spec["target"], "correct": record["correct"]}, flush=True)
            if mode == "bf16_sdpa":
                report["bf16_stage_peak_memory"] = {"allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                                                     "reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        assert [r["id"] for r in report["cases"]] == [r["id"] for r in proof["cases"]]
        assert counter.counts["vision"] == 4 and counter.counts["language"] <= 36 and generated <= 32
        report.update(actual_forwards=counter.counts, generated_tokens=generated, all_registered_cases_executed=True,
                      all_input_fingerprints_verified=True, wall_seconds_including_load=time.perf_counter() - begin,
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                      peak_scope="after BF16 load through BF16/FP32 conversion and all inference; never reset at conversion")
        write_json(out / "equal_tail_summary.json", report)
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
