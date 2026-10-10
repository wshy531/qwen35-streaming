"""Four original-model FIRST controls on the frozen six-frame failed/control pair."""
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

NAMES = ("hands_to_cat", "cat_to_cat")
TARGETS = {"hands_to_cat": "HANDS", "cat_to_cat": "CAT"}
PLAN = PROJECT / "原模型早期主体对照方案.md"
LAUNCHER = PROJECT / "scripts/run_original_first.py"
OLD_REPORT = PROJECT / "runs/s4b_scene_pairs_gpu_20261010_01/scene_pair_summary.json"
BUDGET = {"cases": 4, "max_tokens": 256, "max_new_tokens_per_answer": 8, "generated_tokens_max": 32,
          "language_forwards_max": 36, "vision_encodings": 4, "model_loads": 1, "wall_seconds": 600}


def feature_fingerprint(features):
    # NumPy cannot represent BF16. Preserve dtype/shape and hash the exact bytes.
    value = features.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build(model, processor, frames):
    model.model.rope_deltas = None
    inputs = reference_inputs(model, processor, frames, (192, 256), FIRST)
    previous_positions = inputs.position_ids.clone()
    adapted = add_video_positions(model, inputs)
    assert torch.equal(previous_positions, inputs.position_ids)
    oracle = check_position_oracle(model, inputs)
    assert inputs.input_ids.shape[1] + 8 <= 256
    return inputs, {"input_fingerprint": tensor_fingerprint(inputs), "input_tokens": inputs.input_ids.shape[1],
                    "grid": inputs.video_grid_thw.tolist(), "position_adapter": adapted, "position_oracle": oracle}


def independent_pair_inputs(processor, frames):
    builder = PairInputBuilder(processor)
    prefix = builder.prefix_ids
    prefix_size = prefix.shape[1]
    seen = rope = prefix_size
    parts = []
    for i in (0, 2, 4):
        inputs, record = builder.pair(frames[i], frames[i + 1], seen, rope)
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
            "video_grid_thw": grid,
            "pixel_values_videos": torch.cat([p.pixel_values_videos for p in parts]),
            "position_ids": torch.cat((torch.arange(prefix_size)[None, None, :].expand(4, 1, -1),
                                        *[p.position_ids for p in parts], positions), 2)}


class Counter:
    def __init__(self, model):
        self.model = model
        self.counts = {"vision": 0, "language": 0}
        self.features = []
        self.language_steps = []
        self.before = len(model.model.visual._forward_hooks), len(model.model.language_model._forward_pre_hooks)
        self.hooks = [model.model.visual.register_forward_hook(self.vision),
                      model.model.language_model.register_forward_pre_hook(self.language, with_kwargs=True)]

    def vision(self, _module, _args, output):
        self.counts["vision"] += 1
        self.features.append(feature_fingerprint(output.pooler_output))

    def language(self, _module, _args, kwargs):
        self.counts["language"] += 1
        point = kwargs["cache_position"].detach().cpu()
        position = kwargs["position_ids"].detach().cpu()
        self.language_steps.append({"length": len(point), "cache_positions": point.tolist(),
                                    "position_shape": list(position.shape), "text_axis": position[0, 0].tolist()})
        assert torch.equal(position[0, 0], point)

    def close(self):
        for hook in self.hooks:
            hook.remove()
        return self.before == (len(self.model.model.visual._forward_hooks), len(self.model.model.language_model._forward_pre_hooks))


@torch.inference_mode()
def generate(model, processor, inputs, counter):
    device = model.get_input_embeddings().weight.device
    device_inputs = type(inputs)({key: value.to(device) for key, value in inputs.items()})
    assert all(value.device.type == "cpu" for value in inputs.values())
    model.model.rope_deltas = model.model.rope_deltas.to(device)
    eos = model.generation_config.eos_token_id
    eos_ids = set([] if eos is None else eos if isinstance(eos, list) else [eos])
    if processor.tokenizer.eos_token_id is not None:
        eos_ids.add(processor.tokenizer.eos_token_id)
    before = counter.counts.copy()
    feature_start = len(counter.features)
    step_start = len(counter.language_steps)
    if device.type == "cuda":
        torch.cuda.synchronize()
    begin = time.perf_counter()
    output = model.generate(**device_inputs, max_new_tokens=8, do_sample=False, eos_token_id=sorted(eos_ids))
    if device.type == "cuda":
        torch.cuda.synchronize()
    ids = output[0, inputs.input_ids.shape[1]:].tolist()
    steps = counter.language_steps[step_start:]
    assert steps[0]["cache_positions"] == list(range(inputs.input_ids.shape[1]))
    for i, step in enumerate(steps[1:]):
        assert step["cache_positions"] == [inputs.input_ids.shape[1] + i]
    counts = {key: counter.counts[key] - before[key] for key in before}
    assert counts["vision"] == 1 and counts["language"] == len(ids) and 1 <= len(ids) <= 8
    return {"answer": processor.tokenizer.decode(ids, skip_special_tokens=True), "generated_ids": ids,
            "stopped_on_registered_eos": ids[-1] in eos_ids, "registered_eos_ids": sorted(eos_ids),
            "forward_counts": counts, "generation_seconds": time.perf_counter() - begin,
            "complete_visual_feature_fingerprints": counter.features[feature_start:],
            "language_steps": steps, "incremental_decode_positions_verified": True}


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
    write_json(out / "cpu_summary.json", {"input_controls": entries, "scope": "two tiny original-model interface controls; random answers are not capability evidence"})
    proof = {"cpu_controls_passed": True, "input_controls": entries, "budget": BUDGET,
             "cases": [{"id": f"{mode}_{name}", "mode": mode, "input": name} for mode in ("bf16_sdpa", "fp32_ieee_math") for name in NAMES],
             "cpu_report_sha256": file_hash(out / "cpu_summary.json"), "original_reference_hashes": source_assets(),
             "verified_model_assets": manifest["files"], "model_manifest_sha256": file_hash(args.model_dir / "DOWNLOAD_MANIFEST.json"),
             "fixture_metadata": metadata, "historical_report_sha256": file_hash(OLD_REPORT),
             "plan_sha256": file_hash(PLAN), "launcher_sha256": file_hash(LAUNCHER)}
    write_json(out / "preflight_summary.json", proof)
    print({"two_inputs_and_tiny_fresh_prefills_bitwise_equal": True, "tiny_original_generate_and_positions_verified": True,
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
    counter = Counter(model)
    report = {"budget": BUDGET, "cases": [], "model_reference": str(ORIGINAL / "modeling_qwen3_5.py"),
              "original_reference_hashes": source_assets(), "fixture_metadata": metadata, "resource_precheck": guard,
              "scope": "original unmodified model, independent complete six-frame visual encoding; fixed FIRST semantic control"}
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
                    write_json(out / "original_first_summary.json", report)
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
        write_json(out / "original_first_summary.json", report)
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
