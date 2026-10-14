"""Generic bounded file-video demo. CPU preflight must precede real GPU run."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import traceback
import unittest

import psutil
import torch
from transformers import AutoProcessor

from scripts.baseline import REPO, file_hash, write_json
from scripts.pair_demo import verify_assets
from streaming.demo import PairDemo
from streaming.online import OnlinePairSession
from streaming.viewer import Viewer
from streaming.window import read_packets
from streaming.playback import ClockPlayback,SourceDigest
from streaming.repeat_source import repeat_packets, provenance


def source_hashes():
    hashes = {str(p.relative_to(REPO)).replace("\\", "/"):file_hash(p) for p in sorted(REPO.rglob("*.py"))}
    hashes["@launcher"] = file_hash(REPO.parents[1]/"scripts/run_video_demo.py")
    return hashes


def branch_allowed(branch, allowed):
    return bool(branch) and branch not in ("main","master","HEAD") and (not allowed or branch in allowed)


def request(args):
    return {k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items()
            if k not in ("stage","out","preflight","ui_port")}


def budget(args):
    pairs = math.ceil(args.max_frames / 2)
    history_queries = len(getattr(args,"history_time",[]))
    return {"model_loads":1,"frames_max":args.max_frames,"vision_forwards_max":pairs+history_queries,
            "questions_max":pairs+history_queries,"language_forwards_max":pairs * (2 + args.answer_tokens + 1)+history_queries*(args.answer_tokens+4),
            "generated_tokens_max":(pairs+history_queries) * args.answer_tokens,"history_tokens_max":args.max_tokens-96-args.answer_tokens,
            "wall_seconds":args.wall_seconds,"free_gpu_gib_min":5.5,"available_ram_gib_min":3,
            "precision":"fp32_ieee_math","resize":"letterbox","video_logits":"none","qa_logits":"last",
            "playback":getattr(args,"playback","fast"),"queue_capacity":1,
            "consumer_delay_seconds":getattr(args,"consumer_delay_seconds",0.),
            "repeat_count":getattr(args,"repeat_count",1),"repeat_period_seconds":getattr(args,"repeat_period",None),
            "first_qa_frames":getattr(args,"first_qa_frames",4),
            "external_history_enabled":getattr(args,"remember_history",False),
            "external_history_queries_max":history_queries,
            "external_history_rgb_max_bytes":884736 if getattr(args,"remember_history",False) else 0,
            "external_history_max_pairs":3 if getattr(args,"remember_history",False) else 0,
            "external_history_every_pairs":getattr(args,"memory_every_pairs",4),
            "external_history_query_after":getattr(args,"history_query_after",None)}


@torch.inference_mode()
def replay(model, processor, args, expected_schedule=None, expected_source=None, source_factory=None):
    # Timing depends on real inference latency. CPU proof checks every selected
    # source sample in fast mode; clock GPU runs verify the same complete source.
    effective_playback = "live" if source_factory else args.playback if args.stage == "gpu" else "fast"
    scope = "真实权重 GPU · "+("按PTS墙钟播放 / 最新帧替换" if effective_playback == "clock" else "顺序文件重放")
    if args.stage != "gpu":
        scope = "CPU 微型随机权重：全输入接口预检，回答没有语义意义；不模拟GPU耗时"
    if source_factory:
        scope += (" · 独立文件读帧进程 / 原视频PTS墙钟播放" if getattr(args,"input_kind",None)=="file"
                  else " · 实时接收接口 / 接收时刻计时，非传感器曝光延迟")
    if args.consumer_delay_seconds:
        scope += f" · 人工消费者延迟{args.consumer_delay_seconds}秒（调度压力）"
    if getattr(args,"repeat_count",1) > 1:
        scope += f" · 明确循环{args.repeat_count}次 / 合成会话时间；不是自然长视频"
    if getattr(args,"remember_history",False):
        scope += " · 可选有限真实帧记忆 / 明确时间检索，非无限或语义记忆"
    viewer = Viewer(scope)
    session = OnlinePairSession(model,processor,max_tokens=args.max_tokens,max_question_tokens=96,
                                max_answer_tokens=args.answer_tokens,resize_mode="letterbox",video_logits="none")
    counts = {"vision":0,"language":0,"lm_head_rows":0}
    limits = budget(args)
    def guard(name, maximum):
        def hook(_module, inputs):
            counts[name] += inputs[0].shape[1] if name == "lm_head_rows" else 1
            if counts[name] > maximum:
                raise RuntimeError(f"Registered {name} budget exceeded")
        return hook
    hooks = [model.model.visual.register_forward_pre_hook(guard("vision",limits["vision_forwards_max"])),
             model.model.language_model.register_forward_pre_hook(guard("language",limits["language_forwards_max"])),
             model.lm_head.register_forward_pre_hook(guard("lm_head_rows",limits["language_forwards_max"]))]
    repeated = getattr(args,"repeat_count",1) > 1
    reader = None if source_factory else (repeat_packets(args.video,args.fps,args.repeat_period,args.repeat_count)
              if repeated else read_packets(args.video,args.fps))
    fingerprint = SourceDigest()
    playback = None
    event_lock = threading.RLock()
    schedule = []  # Only bounded scalar metadata. No RGB, logits or cache retained.
    begin = time.monotonic()
    latency = {"count":0,"sum_seconds":0.0,"max_seconds":0.0}
    peak_rss = 0
    answer_lags = []  # Scalars only, bounded by registered question budget.
    max_pair_gap = 0.
    resources = []  # Bounded scalar observations after GPU synchronization.
    current_provenance = None
    with (args.out/"events.jsonl").open("x",encoding="utf-8") as stream:
        def sink(event):
            nonlocal peak_rss,max_pair_gap
            if model.device.type == "cuda":
                torch.cuda.synchronize()
            peak_rss = max(peak_rss,psutil.Process().memory_info().rss)
            event["available_wall_seconds"] = time.monotonic()-begin
            if repeated:
                event["source_provenance"] = current_provenance
            if event["kind"] in ("encoded","answer","capacity_reset"):
                state = event["status"]
                sample = {"kind":event["kind"],"wall_seconds":event["available_wall_seconds"],
                    "epoch":state["epoch"],"seen_tokens":state["session"]["seen_tokens"],
                    "session_pts":state["last_received_pts"],"memory_from_pts":state["memory_from_pts"],
                    "rss_bytes":psutil.Process().memory_info().rss,
                    "ram_available_bytes":psutil.virtual_memory().available}
                if model.device.type == "cuda":
                    sample.update(allocated_bytes=torch.cuda.memory_allocated(),reserved_bytes=torch.cuda.memory_reserved())
                resources.append(sample)
            if event["kind"] == "answer":
                lag = max(0.,event["available_wall_seconds"]-event["answer"]["timestamp"])
                event["answer"]["answer_lag_seconds"] = lag if effective_playback in ("clock","live") else None
                event["answer"]["available_wall_seconds"] = event["available_wall_seconds"]
                if effective_playback in ("clock","live"):
                    answer_lags.append(lag)
            if event["kind"] == "history_answer":
                event["answer"]["trigger_encoded_pts"] = event["status"]["session"]["last_encoded_pts"]
                event["answer"]["trigger_received_pts"] = event["status"]["last_received_pts"]
                event["answer"]["trigger_finished"] = event["status"]["finished"]
                event["answer"]["available_wall_seconds"] = event["available_wall_seconds"]
            if event["kind"] == "encoded":
                pts = event["record"]["timestamps"]
                max_pair_gap = max(max_pair_gap,pts[1]-pts[0])
            with event_lock:
                viewer.event(event)
                stream.write(json.dumps(event,ensure_ascii=False)+"\n")
                stream.flush()
            if event["kind"] in ("encoded","capacity_reset","answer","history_answer","eof"):
                s = event["status"]
                schedule.append({"kind":event["kind"],"epoch":s["epoch"],"memory_from_pts":s["memory_from_pts"],
                                 "encoded_frames":s["encoded_real_frames"],"seen_tokens":s["session"]["seen_tokens"]})
                print(json.dumps({"event":event["kind"],"timestamp":s["last_received_pts"],"memory_from":s["memory_from_pts"],
                                  "answer":event.get("answer",{}).get("answer")},ensure_ascii=False),flush=True)
        demo_type = PairDemo
        memory_options = {}
        if getattr(args,"remember_history",False):
            from streaming.remembering import RememberingPairDemo
            demo_type = RememberingPairDemo
            memory_options["memory_every_pairs"] = args.memory_every_pairs
        demo = demo_type(session,args.question,args.answer_tokens,sink,audit=True,
                         first_qa_frames=args.first_qa_frames,qa_every_frames=args.qa_every,**memory_options)
        historical_answers = []
        history_triggered = False
        def maybe_query_history(at_finish=False):
            nonlocal history_triggered
            targets = getattr(args,"history_time",[])
            if history_triggered or not targets:
                return
            threshold = getattr(args,"history_query_after",None)
            if threshold is None:
                if not at_finish:
                    return
            elif session.last_encoded_pts is None or session.last_encoded_pts < threshold:
                return
            for target in targets:
                if time.monotonic()-begin >= args.wall_seconds:
                    raise TimeoutError("Registered history query wall budget exceeded")
                historical_answers.append(demo.ask_at(target,args.history_question,args.answer_tokens))
            history_triggered = True
        try:
            if args.ui_port >= 0:
                print("VIEWER "+viewer.start(args.ui_port),flush=True)
            reason = "source_eof"
            if effective_playback in ("clock","live"):
                def source_frame(packet, timing):
                    if repeated:
                        timing = {**timing,"source_provenance":provenance(packet)}
                    with event_lock:
                        viewer.frame(packet.rgb,packet.timestamp,timing)
                        stream.write(json.dumps({"kind":"source_frame","frame_id":packet.frame_id,
                            "timestamp":packet.timestamp,"source_provenance":provenance(packet),
                            "playback":timing},ensure_ascii=False)+"\n")
                        stream.flush()
                playback = (source_factory(source_frame) if source_factory else ClockPlayback(reader,args.duration,args.max_frames,source_frame)).start()
                # All timestamps/lag are measured from the source playback origin.
                begin = playback.origin
            source = playback if playback is not None else reader
            for packet in source:
                current_provenance = provenance(packet)
                if playback is None:
                    if packet.timestamp >= args.duration:
                        reason = "video_duration_limit"
                        break
                    fingerprint.add(packet)
                    viewer.frame(packet.rgb,packet.timestamp,{"source_provenance":current_provenance} if repeated else None)
                if time.monotonic()-begin >= args.wall_seconds:
                    raise TimeoutError("Registered inference wall budget exceeded")
                start = time.perf_counter()
                # Explicit overload injection; never used silently in normal runs.
                if effective_playback in ("clock","live") and args.consumer_delay_seconds:
                    time.sleep(args.consumer_delay_seconds)
                demo.append(packet)
                # Query only after actual completed encoding, never a pending
                # received frame or a producer timestamp. Same model consumer.
                maybe_query_history()
                if model.device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
                latency["count"] += 1
                latency["sum_seconds"] += elapsed + packet.decode_seconds
                latency["max_seconds"] = max(latency["max_seconds"],elapsed+packet.decode_seconds)
                if playback is None and demo.received >= args.max_frames:
                    reason = "frame_budget_limit"
                    break
            if playback is not None:
                playback.close()
                reason = playback.reason
            demo.finish(reason)
            if demo.received == 0:
                raise ValueError("No video frames delivered in the requested duration")
            maybe_query_history(at_finish=True)
            source_fingerprint = playback.digest.snapshot() if playback is not None else fingerprint.snapshot()
            if expected_source is not None and source_fingerprint != expected_source:
                raise AssertionError("Complete CPU/GPU selected source fingerprints differ")
            if effective_playback == "fast" and expected_schedule is not None and schedule != expected_schedule:
                raise AssertionError("CPU/GPU input, capacity or trigger schedule differs")
            report = {"scope":viewer.scope,"budget":limits,"counts":counts,"final":demo.status(),"schedule":schedule,
                      "append_including_qa_and_decode_latency":latency,"replay_wall_seconds":time.monotonic()-begin,
                      "peak_process_rss_bytes":peak_rss,"cross_reset_memory":False,
                      "end_to_end_speedup_proven":False,"realtime":False,
                      "requested_playback":args.playback,"effective_playback":effective_playback,
                      "source_fingerprint":source_fingerprint,"max_encoded_pair_gap_seconds":max_pair_gap,
                      "answer_lags_seconds":answer_lags,
                      "resource_samples":resources,"synthetic_repeat":repeated,
                      "repeat_count":getattr(args,"repeat_count",1),
                      "repeat_period_seconds":getattr(args,"repeat_period",None),
                      "playback":None if playback is None else playback.status()}
            report.update(external_history_enabled=getattr(args,"remember_history",False),
                          historical_answers=historical_answers,
                          history_query_schedule="after source finish" if getattr(args,"history_query_after",None) is None else "once after actual encoded PTS reaches threshold",
                          history_query_triggered=history_triggered,
                          pending_history_times=[] if history_triggered else list(getattr(args,"history_time",[])),
                          history_query_after=getattr(args,"history_query_after",None),
                          main_history_across_reset=False)
            if model.device.type == "cuda":
                report.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                              peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            write_json(args.out/"summary.json",report)
            return report
        finally:
            if playback is not None:
                playback.close()
            elif reader is not None:
                reader.close()
            # Save final state before close resets session counters.
            viewer.save(args.out/"viewer.html")
            demo.close()
            for hook in hooks:
                hook.remove()
            viewer.close()


def preflight(args):
    from tests.test_online import small_model
    from tests.test_video_demo import VideoDemoTests
    from tests.test_playback import PlaybackTests
    from tests.test_repeat_source import RepeatSourceTests
    from tests.test_frame_memory import FrameMemoryTests,RememberingTests
    from tests.test_history_demo import HistoryDemoTests
    from tests.test_history_trigger import HistoryTriggerTests
    RememberingTests.model_dir = args.model_dir
    with (args.out/"unittest.log").open("w",encoding="utf-8") as stream:
        suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (VideoDemoTests,PlaybackTests,RepeatSourceTests,FrameMemoryTests,RememberingTests,HistoryDemoTests,HistoryTriggerTests)])
        result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    write_json(args.out/"cpu_tests.json",{"tests":result.testsRun,"passed":result.wasSuccessful(),
                                        "errors":len(result.errors),"failures":len(result.failures)})
    if not result.wasSuccessful():
        raise AssertionError("CPU interface controls failed")
    assets = verify_assets(args.model_dir)
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    report = replay(small_model(processor),processor,args)
    write_json(args.out/"proof.json",{"request":request(args),"budget":budget(args),"source_hashes":source_hashes(),
        "video_sha256":file_hash(args.video),"model_manifest_sha256":file_hash(args.model_dir/"DOWNLOAD_MANIFEST.json"),
        "assets":assets["files"],"schedule":report["schedule"],"source_fingerprint":report["source_fingerprint"],
        "clock_schedule_comparison":"complete_source_fingerprint; delivered schedule depends on latency",
        "summary_sha256":file_hash(args.out/"summary.json"),
        "cpu_tests_sha256":file_hash(args.out/"cpu_tests.json"),"cpu_tests_passed":True})


def check_proof(args):
    if args.preflight is None:
        raise ValueError("Fresh --preflight directory is required")
    read = lambda name:json.loads((args.preflight/name).read_text(encoding="utf-8"))
    proof = read("proof.json")
    if read("status.json") != {"status":"completed","stage":"preflight"}:
        raise ValueError("Preflight did not complete")
    if proof["request"] != request(args) or proof["budget"] != budget(args) or proof["source_hashes"] != source_hashes():
        raise ValueError("CPU proof request, budget or source differs")
    if (proof["summary_sha256"] != file_hash(args.preflight/"summary.json") or
        proof["cpu_tests_sha256"] != file_hash(args.preflight/"cpu_tests.json") or
        not read("cpu_tests.json")["passed"] or not proof["cpu_tests_passed"]):
        raise ValueError("CPU control evidence changed or failed")
    if proof["video_sha256"] != file_hash(args.video) or proof["model_manifest_sha256"] != file_hash(args.model_dir/"DOWNLOAD_MANIFEST.json"):
        raise ValueError("Input/model manifest differs")
    if verify_assets(args.model_dir)["files"] != proof["assets"]:
        raise ValueError("Model assets differ")
    branch = subprocess.check_output(["git","branch","--show-current"],cwd=REPO,text=True).strip()
    if not branch_allowed(branch,args.allow_branch):
        raise ValueError("GPU requires a non-main branch satisfying --allow-branch (if specified)")
    write_json(args.out/"checked_branch.json",{"branch":branch,"allow_branch":args.allow_branch})
    return proof


def gpu(args):
    proof = check_proof(args)
    from qwen35_local import Qwen3_5ForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device unavailable")
    free,total = torch.cuda.mem_get_info()
    guard = {"gpu_free_bytes":free,"gpu_total_bytes":total,"ram_available_bytes":psutil.virtual_memory().available}
    write_json(args.out/"resource_precheck.json",guard)
    if free < 5.5*2**30 or guard["ram_available_bytes"] < 3*2**30:
        raise RuntimeError("Resource guard refused model load")
    start = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation="sdpa").to("cuda").eval()
    if torch.cuda.mem_get_info()[0] < 2*2**30:
        raise RuntimeError("Insufficient room for FP32 conversion")
    model.float()
    torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    report = replay(model,processor,args,proof["schedule"],proof["source_fingerprint"])
    report["wall_seconds_including_load"] = time.perf_counter()-start
    write_json(args.out/"summary.json",report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("preflight","gpu"),required=True)
    parser.add_argument("--video",type=Path,required=True)
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--question",default="Describe what is happening in the video.")
    parser.add_argument("--fps",type=float,default=1)
    parser.add_argument("--duration",type=float,default=60)
    parser.add_argument("--max-frames",type=int,default=60)
    parser.add_argument("--answer-tokens",type=int,default=24)
    parser.add_argument("--qa-every",type=int,default=4)
    parser.add_argument("--first-qa-frames",type=int,default=4,help="Real encoded frames per new memory epoch before first QA")
    parser.add_argument("--playback",choices=("fast","clock"),default="fast")
    parser.add_argument("--repeat-count",type=int,default=1,help="Explicit bounded file-repeat stress input, not natural long video")
    parser.add_argument("--repeat-period",type=float,help="Declared loop period strictly longer than source PTS; session time is synthetic")
    parser.add_argument("--consumer-delay-seconds",type=float,default=0.,help="Explicit clock-mode overload injection, not a performance setting")
    parser.add_argument("--max-tokens",type=int,default=320)
    parser.add_argument("--wall-seconds",type=int,default=600)
    parser.add_argument("--allow-branch",action="append",default=[])
    parser.add_argument("--ui-port",type=int,default=-1,help="-1 disables live view, 0 uses a free loopback port")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--preflight",type=Path)
    parser.add_argument("--remember-history",action="store_true",help="Optional bounded CPU real-frame bank; main reset still forgets")
    parser.add_argument("--memory-every-pairs",type=int,default=4,help="Store every Nth completed real pair, max 3 pairs / 864KiB persistent RGB")
    parser.add_argument("--history-time",type=float,action="append",default=[],help="Explicit session timestamp queried after file finish; at most 8")
    parser.add_argument("--history-question",default="Describe the visible subject and action in these retained historical frames.")
    parser.add_argument("--history-query-after",type=float,help="Run all preset history queries once after encoded PTS reaches this time; default is after EOF")
    args = parser.parse_args(argv)
    if (not 1 <= args.memory_every_pairs <= 64 or len(args.history_time) > 8 or
        any(not math.isfinite(t) or t < 0 for t in args.history_time) or
        (args.history_time and not args.remember_history)):
        parser.error("History queries require --remember-history, <=8 finite nonnegative times, sampling interval 1..64")
    if args.history_query_after is not None and (
        not math.isfinite(args.history_query_after) or not 0 <= args.history_query_after <= 600 or
        not args.remember_history or not args.history_time):
        parser.error("--history-query-after requires history targets and enabled memory, finite 0..600 seconds")
    if (not math.isfinite(args.fps) or not 0 < args.fps <= 10 or not math.isfinite(args.duration) or
        not 0 < args.duration <= 600 or not 1 <= args.max_frames <= 600 or not 1 <= args.answer_tokens <= 48 or
        not 1 <= args.qa_every <= 600 or not 1 <= args.first_qa_frames <= 600 or
        not math.isfinite(args.consumer_delay_seconds) or not 0 <= args.consumer_delay_seconds <= 5 or
        (args.consumer_delay_seconds > 0 and args.playback != "clock") or not 200 <= args.max_tokens <= 512 or
        not 1 <= args.wall_seconds <= 600 or not -1 <= args.ui_port <= 65535):
        parser.error("Arguments exceed bounded demo limits; FPS<=10, duration/frames/wall<=600, answer<=48, tokens200..512")
    if (not 1 <= args.repeat_count <= 30 or
        (args.repeat_count > 1 and (args.repeat_period is None or not math.isfinite(args.repeat_period) or not 0 < args.repeat_period <= 600)) or
        (args.repeat_count == 1 and args.repeat_period is not None)):
        parser.error("Repeat count 2..30 requires finite positive --repeat-period <=600; one-shot mode has no repeat period")
    args.video,args.model_dir,args.out = args.video.resolve(),args.model_dir.resolve(),args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"tmp").mkdir()
    tempfile.tempdir = str(args.out/"tmp")
    torch.set_num_threads(1)
    torch.manual_seed(42)
    write_json(args.out/"arguments.json",request(args))
    write_json(args.out/"budget.json",budget(args))
    write_json(args.out/"source_hashes.json",source_hashes())
    try:
        (preflight if args.stage == "preflight" else gpu)(args)
        write_json(args.out/"status.json",{"status":"completed","stage":args.stage})
    except Exception:
        (args.out/"traceback.txt").write_text(traceback.format_exc(),encoding="utf-8")
        write_json(args.out/"status.json",{"status":"failed","stage":args.stage})
        raise


if __name__ == "__main__":
    main()
