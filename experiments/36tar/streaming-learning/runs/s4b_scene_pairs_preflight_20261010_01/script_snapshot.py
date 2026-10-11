"""Frozen 2x2 early/late visual counterfactuals and shared-input language controls."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np
from PIL import Image
import psutil
import torch
from transformers import AutoProcessor

from scripts.baseline import PROJECT, provenance, write_json, file_hash, cache_measure
from scripts.fp32_regression import strict_logits, strict_states
from scripts.online_experiment import Q1, reference_inputs, reference_answer
from scripts.pair_demo import verify_assets
from scripts.shared_features_control import Capture, shared_input
from streaming.demo import PairDemo
from streaming.inputs import tensor_fingerprint
from streaming.online import OnlinePairSession, fp32_reference
from streaming.window import FramePacket, read_packets

FIRST = Q1.replace("LAST", "FIRST")
PLAN = PROJECT / "跨场景与早期画面对照方案.md"
LAUNCHER = PROJECT / "scripts/run_scene_pairs.py"
FIXTURE = PROJECT / "runs/s4b_scene_fixture_20261010_01"
SPECS = (("cat_to_hands", "cat_play", "clean_hands", "CAT", "HANDS"),
         ("hands_to_hands", "clean_hands", "clean_hands", "HANDS", "HANDS"),
         ("hands_to_cat", "clean_hands", "cat_play", "HANDS", "CAT"),
         ("cat_to_cat", "cat_play", "cat_play", "CAT", "CAT"))
BUDGET = {"sequences": 6, "input_frames": 28, "vision_encodings": 14, "answers": 24,
          "online_answers": 16, "reference_answers": 8, "max_new_tokens_per_answer": 8,
          "generated_tokens_max": 192, "language_forwards_max": 236,
          "main_tokens_max": 176, "max_tokens": 256, "model_loads": 1, "wall_seconds": 600}


def rgb_hash(rgb):
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def prepare(dataset):
    registered = json.loads((FIXTURE / "fixture_sources.json").read_text(encoding="utf-8"))
    sources, metadata = {}, []
    for spec in registered["sources"]:
        name = spec["source"]
        path = dataset / f"{name}.webm"
        assert file_hash(path) == spec["video_sha256"]
        reader = read_packets(path, 1)
        try:
            packets = tuple(itertools.islice(reader, 6))
        finally:
            reader.close()
        assert reader.gi_frame is None and len(packets) == 6
        actual = [{"frame_id": p.frame_id, "source_pts": p.timestamp, "decoded_count": p.decoded_count,
                   "shape": list(p.rgb.shape), "rgb_sha256": rgb_hash(p.rgb)} for p in packets]
        assert actual == spec["frames"]
        sources[name] = tuple(np.asarray(Image.fromarray(p.rgb).resize((256, 192), Image.Resampling.BICUBIC)).copy() for p in packets)
        metadata.append({"source": name, "video_sha256": file_hash(path), "frames": actual,
                         "derived_rgb_sha256": [rgb_hash(rgb) for rgb in sources[name]]})
    cases = {}
    for name, early, late, first, last in SPECS:
        rgb = sources[early][:4] + sources[late][4:]
        cases[name] = tuple(FramePacket(i, float(i), pixels, i + 1) for i, pixels in enumerate(rgb))
    for name in sources:
        cases[f"tail_{name}"] = tuple(FramePacket(i, float(i), sources[name][i], i + 1) for i in (4, 5))
    paired = []
    for a, b in (("cat_to_hands", "hands_to_hands"), ("hands_to_cat", "cat_to_cat")):
        assert all(np.array_equal(cases[a][i].rgb, cases[b][i].rgb) for i in (4, 5))
        assert all(not np.array_equal(cases[a][i].rgb, cases[b][i].rgb) for i in range(4))
        paired.append({"cases": [a, b], "last_two_rgb_bitwise_equal": True, "early_four_rgb_different": True})
    return cases, {"sources": metadata, "paired_pixel_controls": paired,
                   "experimental_pts": [0., 1., 2., 3., 4., 5.], "experimental_ids": list(range(6)),
                   "scope": "controlled in-memory splice, original frame IDs/PTS above; not continuous natural footage",
                   "fixture_source_sha256": file_hash(FIXTURE / "fixture_sources.json"),
                   "contact_sheet_sha256": file_hash(FIXTURE / "contact_sheet.png")}


class MainCapture(Capture):
    """Capture only the owned main cache, excluding QA/ref branches even in demo.append."""
    def __init__(self, model):
        self.main = None
        super().__init__(model)

    def language(self, module, args, kwargs):
        self.active = self.main is not None and kwargs.get("past_key_values") is self.main.cache
        super().language(module, args, kwargs)

    def clear(self, main):
        self.features.clear()
        self.embeddings.clear()
        self.positions.clear()
        self.main = main


def exact(left, right):
    return all(item["all_tensors_bitwise_equal"] for item in strict_states(left, right).values())


def normalize(answer):
    return answer.strip().upper()


def policy_schedule(events):
    result = []
    for event in events:
        kind = event["kind"]
        if kind == "received":
            result.append({"kind": kind, "id": event["frame_id"], "pts": event["timestamp"]})
        elif kind == "encoded":
            result.append({"kind": kind, "record": event["record"], "main_tokens": event["status"]["session"]["seen_tokens"]})
        elif kind == "answer":
            result.append({"kind": kind, "id": event["answer"]["frame_id"], "pts": event["answer"]["timestamp"],
                           "epoch_real_frames": event["epoch_real_frames"]})
        elif kind in ("capacity_reset", "error"):
            raise AssertionError(f"Unexpected event for this bounded fixture: {kind}")
    return result


@torch.inference_mode()
def controls(model, processor, cases, out, proof=None):
    capture = MainCapture(model)
    hooks_before = len(model.model.visual._forward_hooks) - 1, len(model.model.language_model._forward_pre_hooks) - 1
    records, fingerprints, schedules, identities, numerical = [], {}, {}, {}, []
    generated = 0
    closed_sessions = []
    report = {"budget": BUDGET, "records": records, "numerical_controls": numerical,
              "questions": {"last": Q1, "first": FIRST}, "scope": "four controlled 6-frame splices plus two tail-only controls"}
    def save_partial():
        write_json(out / "scene_pair_summary.json", report)
    def install_ask(session):
        original = session.ask
        evidence = {}
        def audited(question, max_new_tokens=8):
            before = session.status()
            snapshot = session.main.cache.clone()
            value = original(question, max_new_tokens, return_evidence=True)
            details = value.pop("evidence")
            unchanged = before == session.status() and exact(snapshot, session.main.cache)
            assert unchanged
            evidence[question] = details
            value["main_unchanged_bitwise_and_counters"] = unchanged
            return value
        audited.original = original
        session.ask = audited
        return evidence
    try:
        with (out / "events.jsonl").open("x", encoding="utf-8") as stream, fp32_reference():
            for name, early, late, first, last in SPECS:
                packets = cases[name]
                session = OnlinePairSession(model, processor)
                capture.clear(session.main)
                evidence = install_ask(session)
                events = []
                def sink(event):
                    events.append(event)
                    stream.write(json.dumps({"case": name, **event}, ensure_ascii=False) + "\n")
                    stream.flush()
                demo = PairDemo(session, Q1, 8, sink, audit=True)
                try:
                    for packet in packets:
                        demo.append(packet)
                    assert demo.questions == 2 and demo.resets == 0 and session.main.seen_tokens == 171
                    auto = [e["answer"] for e in events if e["kind"] == "answer"]
                    assert [a["frame_id"] for a in auto] == [3, 5]
                    for count, answer, target in ((4, auto[0], first), (6, auto[1], last)):
                        records.append({"case": name, "path": "online", "frames": count, "task": "last", "target": target,
                                        "answer": answer, "correct": normalize(answer["answer"]) == target})
                        generated += len(answer["generated_ids"])
                    answer_first = session.ask(FIRST, 8)
                    records.append({"case": name, "path": "online", "frames": 6, "task": "first", "target": first,
                                    "answer": answer_first, "correct": normalize(answer_first["answer"]) == first})
                    generated += len(answer_first["generated_ids"])
                    inputs = reference_inputs(model, processor, packets, (192, 256))
                    fingerprints[name] = {"prefix": tensor_fingerprint(inputs), **{task: tensor_fingerprint(
                        reference_inputs(model, processor, packets, (192, 256), q)) for task, q in (("last", Q1), ("first", FIRST))}}
                    assert len(capture.embeddings) == 4 and len(capture.features) == 3
                    device = model.get_input_embeddings().weight.device
                    device_inputs = type(inputs)({key: value.to(device) for key, value in inputs.items()})
                    actual, identity = shared_input(model, processor, session, device_inputs, capture.features, capture.embeddings, capture.positions)
                    assert all(value.device.type == "cpu" for value in inputs.values())
                    identities[name] = identity
                    snapshot, before = session.main.cache.clone(), session.status()
                    for task, q, online, target in (("last", Q1, auto[1], last), ("first", FIRST, answer_first, first)):
                        reference = reference_answer(model, processor, actual, inputs.position_ids.to(device), q, 8)
                        metric = strict_logits(evidence[q]["initial_logits"], reference["initial_logits"])
                        states = strict_states(evidence[q]["branch_cache"], reference["cache"])
                        passed = (metric["fp32_reference_limits_passed"] and metric["allclose_1e4"] and
                                  all(v["allclose_1e4"] for v in states.values()) and online["generated_ids"] == reference["generated_ids"])
                        numerical.append({"case": name, "task": task, "logits": metric, "branch_states": states,
                                          "generated_ids_equal": online["generated_ids"] == reference["generated_ids"], "passed": passed})
                        public = {"answer": reference["answer"], "generated_ids": reference["generated_ids"], "frame_id": 5, "timestamp": 5.}
                        records.append({"case": name, "path": "shared_input_reference", "frames": 6, "task": task, "target": target,
                                        "answer": public, "correct": normalize(public["answer"]) == target})
                        generated += len(public["generated_ids"])
                        del reference
                    assert session.status() == before and exact(snapshot, session.main.cache)
                    del snapshot, actual, device_inputs, evidence
                    demo.finish("configured_six_frame_boundary")
                    schedules[name] = policy_schedule(events)
                    save_partial()
                finally:
                    session.ask = session.ask.original
                    demo.close()
                    cleared = session.closed and session.main.seen_tokens == 0 and session.pending is None and cache_measure(session.main.cache)["total_tensor_bytes"] == 0
                    closed_sessions.append(cleared)
                    assert cleared
            for source, target in (("cat_play", "CAT"), ("clean_hands", "HANDS")):
                name = f"tail_{source}"
                session = OnlinePairSession(model, processor)
                capture.clear(session.main)
                evidence = install_ask(session)
                try:
                    for packet in cases[name]:
                        session.append(packet)
                    assert session.main.seen_tokens == 59
                    for task, q in (("last", Q1), ("first", FIRST)):
                        answer = session.ask(q, 8)
                        records.append({"case": name, "path": "tail_only", "frames": 2, "task": task, "target": target,
                                        "answer": answer, "correct": normalize(answer["answer"]) == target})
                        generated += len(answer["generated_ids"])
                    fingerprints[name] = {"prefix": tensor_fingerprint(reference_inputs(model, processor, cases[name], (192, 256)))}
                    del evidence
                    save_partial()
                finally:
                    session.ask = session.ask.original
                    session.close()
                    cleared = session.closed and session.main.seen_tokens == 0 and session.pending is None and cache_measure(session.main.cache)["total_tensor_bytes"] == 0
                    closed_sessions.append(cleared)
                    assert cleared
            assert len(records) == 24 and len(numerical) == 8
            assert capture.counts["vision"] == 14 and capture.counts["language"] == 20 + 24 + generated <= 236
            assert generated <= 192
            if proof:
                assert fingerprints == proof["input_fingerprints"] and schedules == proof["expected_schedules"]
            paired = []
            for a, b in (("cat_to_hands", "hands_to_hands"), ("hands_to_cat", "cat_to_cat")):
                rows = [next(r for r in records if r["case"] == case and r["path"] == "online" and r["task"] == "first") for case in (a, b)]
                correct = all(r["correct"] for r in rows)
                changed = normalize(rows[0]["answer"]["answer"]) != normalize(rows[1]["answer"]["answer"])
                paired.append({"cases": [a, b], "both_first_answers_correct": correct, "first_answers_changed": changed,
                               "early_visual_use_criterion_passed": correct and changed})
            scores = {}
            for name, filt in (("online_last4", lambda r: r["path"] == "online" and r["task"] == "last" and r["frames"] == 4),
                               ("online_last6", lambda r: r["path"] == "online" and r["task"] == "last" and r["frames"] == 6),
                               ("online_first6", lambda r: r["path"] == "online" and r["task"] == "first"),
                               ("tail_only", lambda r: r["path"] == "tail_only")):
                rows = [r for r in records if filt(r)]
                scores[name] = {"correct": sum(r["correct"] for r in rows), "total": len(rows)}
            transitions = []
            for name in ("cat_to_hands", "hands_to_cat"):
                rows = [r for r in records if r["case"] == name and r["path"] == "online" and r["task"] == "last"]
                transitions.append({"case": name, "both_answers_correct": all(r["correct"] for r in rows),
                                    "answers_changed": normalize(rows[0]["answer"]["answer"]) != normalize(rows[1]["answer"]["answer"])})
            report.update(all_eight_numeric_controls_passed=all(r["passed"] for r in numerical),
                          actual_forwards=capture.counts, generated_tokens=generated, input_fingerprints=fingerprints,
                          schedules=schedules, exact_input_identities=identities, semantic_scores=scores,
                          paired_early_visual_controls=paired, transition_controls=transitions,
                          online_main_isolation_verified=True)
            save_partial()
            print({"numeric_controls": report["all_eight_numeric_controls_passed"], "scores": scores,
                   "paired_early": paired, "forwards": capture.counts}, flush=True)
            return report
    finally:
        capture.close()
        cleanup = {"hooks_restored": hooks_before == (len(model.model.visual._forward_hooks), len(model.model.language_model._forward_pre_hooks)),
                   "closed_sessions": closed_sessions, "all_six_sessions_closed": len(closed_sessions) == 6 and all(closed_sessions)}
        capture.clear(None)
        write_json(out / "cleanup.json", cleanup)
        assert cleanup["hooks_restored"]


def layout_controls(model, processor, cases):
    full = {name: reference_inputs(model, processor, cases[name], (192, 256)) for name, *_ in SPECS}
    anchor = full["cat_to_hands"]
    for item in full.values():
        for key in ("input_ids", "position_ids", "mm_token_type_ids", "attention_mask", "video_grid_thw"):
            assert torch.equal(anchor[key], item[key])
    pixels_per_pair = int(anchor.video_grid_thw[0, 1:].prod())
    for a, b in (("cat_to_hands", "hands_to_hands"), ("hands_to_cat", "cat_to_cat")):
        assert torch.equal(full[a].pixel_values_videos[-pixels_per_pair:], full[b].pixel_values_videos[-pixels_per_pair:])
        assert not torch.equal(full[a].pixel_values_videos[:-pixels_per_pair], full[b].pixel_values_videos[:-pixels_per_pair])
    return {"all_four_token_position_grid_layouts_bitwise_equal": True,
            "paired_terminal_processor_pixels_bitwise_equal": True, "paired_early_processor_pixels_different": True}


def preflight(out, args):
    from tests.test_online import small_model
    torch.set_num_threads(1)
    manifest = verify_assets(args.model_dir)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = small_model(processor)
    cases, meta = prepare(args.dataset)
    layout = layout_controls(model, processor, cases)
    assert FIRST != Q1
    session = OnlinePairSession(model, processor)
    question_lengths = [session.builder.question_ids(q).shape[1] for q in (Q1, FIRST)]
    assert max(question_lengths) <= 64 and 171 + max(question_lengths) + 8 <= 256
    session.close()
    report = controls(model, processor, cases, out)
    assert report["all_eight_numeric_controls_passed"]
    write_json(out / "fixture_summary.json", {**meta, "layout_controls": layout})
    proof = {"cpu_controls_passed": True, "numeric_controls": 8, "budget": BUDGET,
             "input_fingerprints": report["input_fingerprints"], "expected_schedules": report["schedules"],
             "cpu_report_sha256": file_hash(out / "scene_pair_summary.json"), "fixture_summary_sha256": file_hash(out / "fixture_summary.json"),
             "fixture_source_sha256": meta["fixture_source_sha256"], "contact_sheet_sha256": meta["contact_sheet_sha256"],
             "video_hashes": {s["source"]: s["video_sha256"] for s in meta["sources"]},
             "model_manifest_sha256": file_hash(args.model_dir / "DOWNLOAD_MANIFEST.json"), "verified_model_assets": manifest["files"],
             "plan_sha256": file_hash(PLAN), "launcher_sha256": file_hash(LAUNCHER), "question_lengths": question_lengths,
             "scope": "tiny weights CPU evidence, all semantic outputs are random"}
    write_json(out / "preflight_summary.json", proof)


@torch.inference_mode()
def gpu(out, args):
    from qwen35_local import Qwen3_5ForConditionalGeneration
    begin = time.perf_counter()
    free, total = torch.cuda.mem_get_info()
    guard = {"gpu_free_bytes": free, "gpu_total_bytes": total, "ram_available_bytes": psutil.virtual_memory().available}
    write_json(out / "resource_precheck.json", guard)
    assert free >= 5.5 * 2**30 and guard["ram_available_bytes"] >= 3 * 2**30, "Resource guard; no load"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir, local_files_only=True, dtype=torch.bfloat16,
                    attn_implementation="sdpa").to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    assert torch.cuda.mem_get_info()[0] >= 2 * 2**30
    model.float()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(not l.linear_attn.experimental_recurrent_reference for l in model.model.language_model.layers if l.layer_type == "linear_attention")
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    cases, meta = prepare(args.dataset)
    layout = layout_controls(model, processor, cases)
    write_json(out / "fixture_summary.json", {**meta, "layout_controls": layout})
    proof = json.loads((args.preflight / "preflight_summary.json").read_text())
    assert json.loads((args.preflight / "fixture_summary.json").read_text()) == {**meta, "layout_controls": layout}
    report = controls(model, processor, cases, out, proof)
    report.update(wall_seconds_including_load=time.perf_counter() - begin,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30, resource_precheck=guard)
    write_json(out / "scene_pair_summary.json", report)
    if not report["all_eight_numeric_controls_passed"]:
        raise RuntimeError("Strict shared-input language control failed; preserve all semantic/numeric evidence")


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
