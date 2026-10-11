"""Finite causal pair-demo replay with CPU-frozen event schedule and guarded GPU."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import traceback
import unittest

import psutil
import torch
from transformers import AutoProcessor

from scripts.baseline import PROJECT, REPO, REVISION, provenance, write_json, file_hash, cache_measure
from scripts.online_experiment import Q1
from streaming.demo import PairDemo
from streaming.online import OnlinePairSession
from streaming.window import read_packets

PLAN = PROJECT / "同步成对帧演示验收方案.md"
LAUNCHER = PROJECT / "scripts/run_pair_demo.py"
BUDGET = {"samples_max": 23, "vision_encodings_max": 12, "questions_max": 8, "resets_max": 6,
          "language_forwards_max": 100, "generated_tokens_max": 64, "max_new_tokens_per_answer": 8,
          "main_tokens_max": 176, "max_tokens": 256, "model_loads": 1, "wall_seconds": 600}


def schedule(events):
    """Only deterministic input/policy metadata; never random/model answer content."""
    result = []
    for event in events:
        kind = event["kind"]
        item = {"kind": kind, "epoch": event["status"]["epoch"]}
        if kind == "received":
            item.update(frame_id=event["frame_id"], timestamp=event["timestamp"], decoded_count=event["decoded_count"])
        elif kind == "encoded":
            item.update(record=event["record"], block_tokens=event["block_tokens"], main_tokens=event["status"]["session"]["seen_tokens"])
        elif kind == "capacity_reset":
            item.update(retained_frame_ids=event["retained_frame_ids"], synthetic_padding=event["synthetic_padding"],
                        discarded_epoch=event["discarded_epoch"], new_block_tokens=event["new_block_tokens"])
        elif kind == "answer":
            item.update(frame_id=event["answer"]["frame_id"], timestamp=event["answer"]["timestamp"],
                        epoch_real_frames=event["epoch_real_frames"], main_tokens=event["status"]["session"]["seen_tokens"])
        elif kind == "eof":
            item.update(reason=event["reason"])
        else:
            continue
        result.append(item)
    return result


@torch.inference_mode()
def replay(model, processor, video, out, proof=None):
    counts = {"vision": 0, "language": 0}
    events = []  # The experiment is explicitly finite; no RGB/logits retained here.
    hooks_before = len(model.model.visual._forward_pre_hooks), len(model.model.language_model._forward_pre_hooks)
    def count_vision(_m, _args):
        counts["vision"] += 1
        assert counts["vision"] <= BUDGET["vision_encodings_max"]
    def count_language(_m, _args):
        counts["language"] += 1
        assert counts["language"] <= BUDGET["language_forwards_max"]
    hooks = [model.model.visual.register_forward_pre_hook(count_vision),
             model.model.language_model.register_forward_pre_hook(count_language)]
    reader = read_packets(video, 1)
    session = OnlinePairSession(model, processor, canvas=(192, 256))
    peak_main_bytes = peak_main_tokens = 0
    with (out / "events.jsonl").open("x", encoding="utf-8") as stream:
        def sink(event):
            nonlocal peak_main_bytes, peak_main_tokens
            if model.get_input_embeddings().weight.device.type == "cuda":
                torch.cuda.synchronize()
            event["process_rss_bytes"] = psutil.Process().memory_info().rss
            event["main_cache_bytes"] = cache_measure(session.main.cache)["total_tensor_bytes"]
            peak_main_bytes = max(peak_main_bytes, event["main_cache_bytes"])
            peak_main_tokens = max(peak_main_tokens, session.main.seen_tokens)
            events.append(event)
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
            if event["kind"] in ("answer", "capacity_reset", "eof", "error"):
                print({"event": event["kind"], "epoch": event["status"]["epoch"],
                       "received": event["status"]["received_real_frames"],
                       "answer": event.get("answer"), "main_tokens": session.main.seen_tokens}, flush=True)
        demo = PairDemo(session, Q1, 8, sink, audit=True)
        try:
            for packet in reader:
                assert demo.received < BUDGET["samples_max"], "Unexpected extra sampled frame; no model submission"
                demo.append(packet)  # Completes QA before asking the reader for its next packet.
                assert demo.questions <= BUDGET["questions_max"] and demo.resets <= BUDGET["resets_max"]
                assert demo.generated <= BUDGET["generated_tokens_max"]
            demo.finish("source_eof")
            final = demo.status()
            assert demo.received == demo.encoded == 23 and demo.pairs == counts["vision"] == 12
            assert demo.generated <= 64 and demo.questions <= 8 and demo.resets <= 6
            assert session.pending is None and session.finished
            actual_schedule = schedule(events)
            if proof:
                assert actual_schedule == proof["expected_schedule"], "Runtime schedule differs from frozen CPU proof"
            answers = [e for e in events if e["kind"] == "answer"]
            assert all(e["main_unchanged_bitwise"] and e["main_counters_unchanged"] for e in answers)
            report = {"budget": BUDGET, "question": Q1, "canvas": [192, 256], "final_before_close": final,
                      "actual_forwards": counts, "schedule": actual_schedule, "answers": answers,
                      "peak_main_cache_bytes": peak_main_bytes, "peak_main_tokens": peak_main_tokens,
                      "peak_process_rss_bytes": max(e["process_rss_bytes"] for e in events),
                      "event_count_before_close": len(events), "policy_checks_passed": True,
                      "first_answer_matches_historical_hands": None if proof is None else answers[0]["answer"]["generated_ids"] == [72141, 50, 248046],
                      "scope": "finite sequential file replay; no wall-clock pacing, no live camera or long-run evidence"}
            write_json(out / "pair_demo_summary.json", report)
        finally:
            reader.close()
            demo.close()
            for hook in hooks:
                hook.remove()
            cleanup = {"reader_closed": reader.gi_frame is None, "demo_closed": demo.closed,
                       "main_tokens_after_close": session.main.seen_tokens,
                       "main_cache_bytes_after_close": cache_measure(session.main.cache)["total_tensor_bytes"],
                       "pending_cleared": demo.pending is None and session.pending is None,
                       "hooks_restored": hooks_before == (len(model.model.visual._forward_pre_hooks), len(model.model.language_model._forward_pre_hooks))}
            write_json(out / "cleanup.json", cleanup)
    assert cleanup == {"reader_closed": True, "demo_closed": True, "main_tokens_after_close": 0,
                       "main_cache_bytes_after_close": 0, "pending_cleared": True, "hooks_restored": True}
    report["cleanup"] = cleanup
    write_json(out / "pair_demo_summary.json", report)
    return report


def verify_assets(model_dir):
    manifest = json.loads((model_dir / "DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8-sig"))
    assert manifest["revision"] == REVISION and len(manifest["files"]) == 10
    for item in manifest["files"]:
        path = model_dir / item["name"]
        assert path.stat().st_size == item["bytes"] and file_hash(path) == item["sha256"].lower()
    return manifest


def preflight(out, args):
    from tests.test_online import small_model
    from tests.test_pair_demo import DemoTests
    torch.set_num_threads(1)
    with (out / "unittest.log").open("w", encoding="utf-8") as stream:
        tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DemoTests))
    write_json(out / "cpu_summary.json", {"tests_run": tests.testsRun, "failures": len(tests.failures),
                                         "errors": len(tests.errors), "passed": tests.wasSuccessful(), "scope": "tiny FP32 weights; policy, not semantics"})
    assert tests.wasSuccessful() and tests.testsRun == 9
    manifest = verify_assets(args.model_dir)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    report = replay(small_model(processor), processor, args.video, out)
    proof = {"cpu_controls_passed": True, "budget": BUDGET, "expected_schedule": report["schedule"],
             "schedule_sha256": file_hash(out / "pair_demo_summary.json"),
             "video_sha256": file_hash(args.video), "model_manifest_sha256": file_hash(args.model_dir / "DOWNLOAD_MANIFEST.json"),
             "verified_model_assets": manifest["files"], "all_model_assets_verified": len(manifest["files"]),
             "plan_sha256": file_hash(PLAN), "launcher_sha256": file_hash(LAUNCHER),
             "test_count": tests.testsRun}
    write_json(out / "preflight_summary.json", proof)
    print({"cpu_tests_passed": 9, "samples": 23, "questions": report["final_before_close"]["questions"],
           "resets": report["final_before_close"]["resets"], "forwards": report["actual_forwards"]}, flush=True)


@torch.inference_mode()
def gpu(out, args):
    from qwen35_local import Qwen3_5ForConditionalGeneration
    begin = time.perf_counter()
    torch.manual_seed(42)
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
    assert all(not layer.linear_attn.experimental_recurrent_reference for layer in model.model.language_model.layers if layer.layer_type == "linear_attention")
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    proof = json.loads((args.preflight / "preflight_summary.json").read_text())
    report = replay(model, processor, args.video, out, proof)
    report.update(wall_seconds_including_load=time.perf_counter() - begin,
                  peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30, resource_precheck=guard)
    write_json(out / "pair_demo_summary.json", report)
    print({"policy_checks_passed": True, "first_answer_matches_historical_hands": report["first_answer_matches_historical_hands"],
           "actual_forwards": report["actual_forwards"], "wall_seconds_including_load": report["wall_seconds_including_load"]}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "gpu"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=PROJECT / "models/Qwen3.5-0.8B")
    parser.add_argument("--video", type=Path, default=PROJECT / "datasets/natural_commons_v1/clean_hands.webm")
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
