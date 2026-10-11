"""S3 fixed finite stress/reset experiment; no model changes or persistent cache."""
import argparse
import inspect
import json
from pathlib import Path
import statistics
import tempfile
import time
import traceback
import unittest
import weakref

import psutil
import torch

from scripts.baseline import (PROJECT, REPO, REVISION, add_video_positions,
    check_position_oracle, check_tiny_video_adapter, file_hash, provenance, write_json)
from scripts.window_experiment import PROMPT, ModelTrace, asset_cases
from streaming.inputs import make_window_inputs, tensor_fingerprint
from streaming.session import WindowSession
from streaming.window import read_packets

BUDGET = {"sampled_frames":200,"questions":100,"resets":20,"control_prefills":4,
          "fps":1,"window":4,"every":2,"max_new_tokens":16,"wall_seconds":600}


def create_assets(root):
    import av
    import numpy as np
    from PIL import Image, ImageDraw
    root.mkdir(parents=True,exist_ok=False)
    sequences = {"continuous":(["red"]*2+["blue"]*2+["green"]*2+["yellow"]*2)*15,
                 "reset_A":["red","red","blue","blue"],
                 "reset_B":["green","green","blue","blue"]}
    manifest = {"source":"self-generated synthetic, not a benchmark","fps":1,"videos":[]}
    for name,colors in sequences.items():
        target = root/(name+".mp4")
        with av.open(str(target),"w") as container:
            stream = container.add_stream("libx264",rate=1,
                options={"bf":"0","g":"1","crf":"0","rc-lookahead":"0","sync-lookahead":"0"})
            stream.width = stream.height = 256
            stream.pix_fmt = "yuv420p"
            for color in colors:
                image = Image.new("RGB",(256,256),"white")
                ImageDraw.Draw(image).rectangle((48,48,208,208),fill=color)
                for encoded in stream.encode(av.VideoFrame.from_ndarray(np.asarray(image),format="rgb24")):
                    container.mux(encoded)
            for encoded in stream.encode():
                container.mux(encoded)
        manifest["videos"].append({"id":name,"file":target.name,"colors":colors,
                                 "sha256":file_hash(target),"frames":len(colors)})
    write_json(root/"manifest.json",manifest)
    return manifest


def cases(root):
    result = {x["id"]:x for x in asset_cases(root)}
    if set(result) != {"continuous","reset_A","reset_B"}:
        raise ValueError("Unexpected fixed workload assets")
    assert len(result["continuous"]["colors"]) == 120
    assert result["reset_A"]["colors"] == ["red","red","blue","blue"]
    assert result["reset_B"]["colors"] == ["green","green","blue","blue"]
    return result


def fresh_controls(root):
    for key,case in cases(root).items():
        if key == "continuous":
            continue
        fresh = WindowSession(session_id=key)
        for packet in read_packets(root/case["file"],1):
            fresh.append(packet)
            if fresh.window.evidence()["accepted_samples"] % 2 == 0:
                yield key+":"+str(len(fresh.window.snapshot())),fresh.window.snapshot()


def workload(session,root,resets):
    assets = cases(root)
    schedule = [("continuous",assets["continuous"])] + [
        (f"reset_{i+1}",assets["reset_A" if i%2 == 0 else "reset_B"]) for i in range(20)]
    for index,(identity,case) in enumerate(schedule):
        if index:
            before = session.evidence()
            rope_before = None if session.model is None else session.model.model.rope_deltas
            rope_before_value = None if rope_before is None else rope_before.detach().cpu().tolist()
            session.reset(identity)
            after = session.evidence()
            passed = (after["accepted_samples"] == 0 and after["window_pixel_bytes"] == 0 and
                      after["questions"] == 0 and after["frame_ids"] == [] and
                      (session.model is None or session.model.model.rope_deltas is None))
            resets.append({"identity":identity,"before":before,"after":after,
                           "rope_before":rope_before_value,"rope_cleared":True,"passed":passed})
            assert passed
            del rope_before
        for packet in read_packets(root/case["file"],1):
            session.append(packet)
            if session.window.evidence()["accepted_samples"] % 2 == 0:
                session.questions += 1
                frames = session.window.snapshot()
                yield case,frames,session.evidence()
        assert session.window.evidence()["accepted_samples"] == len(case["colors"])


def cpu(out):
    from tests.test_stream_window import WindowTests
    from tests.test_stream_session import SessionTests
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(x)
                              for x in (WindowTests,SessionTests)])
    with (out/"unittest.log").open("w",encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report = {"tests":result.testsRun,"errors":len(result.errors),"failures":len(result.failures),
              "passed":result.wasSuccessful(),"scope":"CPU window/session contracts"}
    write_json(out/"cpu_summary.json",report)
    assert result.wasSuccessful()
    print(json.dumps(report),flush=True)


def preflight(out,args):
    from transformers import AutoProcessor
    torch.set_num_threads(1)
    assert json.loads((args.model_dir/"DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8-sig"))["revision"] == REVISION
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    tiny,model = check_tiny_video_adapter()
    controls = {}
    for key,frames in fresh_controls(args.assets):
        inputs,_ = make_window_inputs(processor,frames,PROMPT)
        add_video_positions(model,inputs)
        check_position_oracle(model,inputs)
        controls[key] = tensor_fingerprint(inputs)
    resets,windows = [],[]
    session = WindowSession()
    for case,frames,evidence in workload(session,args.assets,resets):
        inputs,info = make_window_inputs(processor,frames,PROMPT)
        add_video_positions(model,inputs)
        check_position_oracle(model,inputs)
        fingerprint = tensor_fingerprint(inputs)
        key = case["id"]+":"+str(len(frames))
        assert len(frames) <= 4 and inputs.input_ids.shape[1] <= 1024
        if case["id"] != "continuous":
            assert fingerprint == controls[key]
        windows.append({"case":case["id"],**evidence,"input_fingerprint":fingerprint,
                        "input_tokens":inputs.input_ids.shape[1],"reset_control_match":
                        None if case["id"] == "continuous" else True})
    assert len(windows) == 100 and len(resets) == 20
    report = {"budget":BUDGET,"tiny_adapter_test":tiny,"controls":controls,"windows":windows,
              "resets":resets,"questions":len(windows),"sampled_frames":200,
              "asset_manifest_sha256":file_hash(args.assets/"manifest.json"),"scope":"CPU interface, no model capability"}
    write_json(out/"preflight_summary.json",report)
    print(json.dumps({"questions":100,"sampled_frames":200,"resets":20,"control_matches":40}),flush=True)


def infer(model,processor,frames,generate,reference=None):
    # References remain CPU-only; no temporary cache/input/output escapes this function.
    model.model.rope_deltas = None
    inputs,info = make_window_inputs(processor,frames,PROMPT)
    add_video_positions(model,inputs)
    fingerprint = tensor_fingerprint(inputs)
    length = inputs.input_ids.shape[1]
    inputs = inputs.to("cuda")
    model.model.rope_deltas = model.model.rope_deltas.to("cuda")
    refs,first_logits = [],[]
    def capture(module,call_args,kwargs,output):
        cache = output.past_key_values
        if cache is not None and not any(x() is cache for x in refs):
            refs.append(weakref.ref(cache))
        if not first_logits:
            first_logits.append(output.logits[:,-1,:].detach().float().cpu())
    handle = model.register_forward_hook(capture,with_kwargs=True)
    trace = ModelTrace(model)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        with torch.inference_mode():
            if generate:
                output = model.generate(**inputs,max_new_tokens=16,do_sample=False)
            else:
                output = model(**inputs,use_cache=True,logits_to_keep=1)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        measured = trace.summary()
        answer = processor.batch_decode(output[:,length:],skip_special_tokens=True)[0] if generate else None
        record = {"input_fingerprint":fingerprint,"input_tokens":length,"answer":answer,
                  "generated_tokens":output.shape[1]-length if generate else 0,
                  "generation_seconds":elapsed,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes":torch.cuda.max_memory_reserved(),
                  "cuda_intervals_ms":measured["cuda_intervals_ms"],
                  "calls":[{k:v for k,v in x.items() if k != "cache"} |
                           ({"cache_total_bytes":x["cache"]["total_tensor_bytes"],"cache_seq_length":x["cache"]["seq_length"]}
                            if "cache" in x else {}) for x in measured["module_calls"]],
                  "temporal_patch_means":info["temporal_patch_means"]}
        if reference is not None:
            ref_logits,ref_fingerprint = reference
            delta = (first_logits[0]-ref_logits).abs()
            record["control_comparison"] = {"input_match":fingerprint == ref_fingerprint,
                "bitwise_logits_match":torch.equal(first_logits[0],ref_logits),
                "max_abs_logit_diff":delta.max().item(),"atol":5e-4,"rtol":5e-4,
                "logits_close":torch.allclose(first_logits[0],ref_logits,atol=5e-4,rtol=5e-4)}
        # Control logits are the only tensor deliberately returned, and always CPU.
        control = (first_logits[0],fingerprint) if not generate else None
        return record,refs,control
    finally:
        handle.remove()
        trace.close()


def resource_snapshot():
    torch.cuda.synchronize()
    memory = psutil.Process().memory_info()
    return {"allocated_bytes":torch.cuda.memory_allocated(),"reserved_bytes":torch.cuda.memory_reserved(),
            "rss_bytes":memory.rss,"private_bytes":memory.private}


def resource_summary(records):
    steady = records[10:]
    summaries = {}
    for key in ("allocated_bytes","reserved_bytes","rss_bytes","private_bytes"):
        values = [x["resources_after_return"][key] for x in steady]
        first,last = statistics.median(values[:10]),statistics.median(values[-10:])
        center = (len(values)-1)/2
        slope = sum((i-center)*value for i,value in enumerate(values))/sum((i-center)**2 for i in range(len(values)))
        summaries[key] = {"min":min(values),"max":max(values),"range":max(values)-min(values),
                          "first_10_median":first,"last_10_median":last,"median_growth":last-first,
                          "linear_slope_bytes_per_question":slope}
    allocated = summaries["allocated_bytes"]
    passed = (allocated["range"] <= 2*2**20 and allocated["median_growth"] <= 2*2**20 and
              all(summaries[x]["median_growth"] <= 64*2**20 for x in ("rss_bytes","private_bytes")))
    return {"steady_questions":[11,100],"metrics":summaries,"passed_registered_resource_limits":passed}


def gpu(out,args):
    from transformers import AutoProcessor
    from qwen35_local import Qwen3_5ForConditionalGeneration
    assert torch.cuda.is_available()
    torch.manual_seed(42)
    start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir,local_files_only=True,
              dtype=torch.bfloat16,attn_implementation="sdpa").to("cuda").eval()
    assert Path(inspect.getfile(type(model))).resolve().parent == REPO/"qwen35_local"
    controls,control_records = {},[]
    for key,frames in fresh_controls(args.assets):
        record,refs,control = infer(model,processor,frames,False)
        assert all(ref() is None for ref in refs)
        controls[key] = control
        control_records.append({"key":key,**record,"cache_weakrefs_alive":0})
    del control,frames
    report = {"budget":BUDGET,"model_revision":REVISION,"gpu":torch.cuda.get_device_name(),
              "control_prefills":control_records,"resets":[],"questions":[],
              "resource_measurement":"after function return/synchronize; no gc.collect or empty_cache",
              "scope":"finite synthetic independent-window stress, not long-memory or wall-clock real-time"}
    session = WindowSession(model)
    for case,frames,evidence in workload(session,args.assets,report["resets"]):
        assert len(report["questions"]) < 100
        key = case["id"]+":"+str(len(frames))
        record,refs,_ = infer(model,processor,frames,True,controls.get(key))
        alive = sum(ref() is not None for ref in refs)
        resources = resource_snapshot()
        expected = case["colors"][frames[-1].frame_id]
        entry = {"question":len(report["questions"])+1,"case":case["id"],**evidence,**record,
                 "expected":expected,"exact_after_strip":record["answer"].strip().lower() == expected,
                 "cache_weakrefs_alive":alive,"resources_after_return":resources}
        report["questions"].append(entry)
        write_json(out/"gpu_summary.json",report)
        if entry["question"] % 10 == 0:
            print(json.dumps({"question":entry["question"],"resets":session.reset_count,
                  "cache_alive":alive,"allocated_mib":resources["allocated_bytes"]/2**20,
                  "rss_mib":resources["rss_bytes"]/2**20},ensure_ascii=False),flush=True)
    assert len(report["questions"]) == 100 and session.reset_count == 20
    comparisons = [x["control_comparison"] for x in report["questions"] if "control_comparison" in x]
    assert len(comparisons) == 40
    report["resource_summary"] = resource_summary(report["questions"])
    report["exact_answers"] = sum(x["exact_after_strip"] for x in report["questions"])
    state_pass = (all(x["cache_weakrefs_alive"] == 0 and len(x["frame_ids"]) <= 4 and
                     next(c for c in x["calls"] if c["kind"] == "prefill")["past_tokens_before"] == 0
                     for x in report["questions"]) and all(x["passed"] for x in report["resets"]) and
                  all(x["input_match"] and x["logits_close"] for x in comparisons))
    report["state_reset_passed"] = state_pass
    report["registered_acceptance_passed"] = state_pass and report["resource_summary"]["passed_registered_resource_limits"]
    report["wall_seconds_including_load"] = time.perf_counter()-start
    write_json(out/"gpu_summary.json",report)
    print(json.dumps({"answers_correct":report["exact_answers"],"state_reset_passed":state_pass,
                      "registered_acceptance_passed":report["registered_acceptance_passed"],
                      "seconds":report["wall_seconds_including_load"]}),flush=True)
    if not report["registered_acceptance_passed"]:
        raise RuntimeError("Registered acceptance failed; inspect measured evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("assets","cpu","preflight","gpu"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--assets",type=Path,default=PROJECT/"datasets"/"stability_synthetic")
    parser.add_argument("--model-dir",type=Path,default=PROJECT/"models"/"Qwen3.5-0.8B")
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"tmp").mkdir()
    tempfile.tempdir = str((args.out/"tmp").resolve())
    prov = provenance()
    prov["psutil"] = psutil.__version__
    write_json(args.out/"provenance.json",prov)
    write_json(args.out/"arguments.json",{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    try:
        if args.stage == "assets":
            write_json(args.out/"assets_summary.json",create_assets(args.assets))
        elif args.stage == "cpu":
            cpu(args.out)
        elif args.stage == "preflight":
            preflight(args.out,args)
        else:
            gpu(args.out,args)
        write_json(args.out/"status.json",{"status":"completed","stage":args.stage})
    except Exception:
        (args.out/"traceback.txt").write_text(traceback.format_exc(),encoding="utf-8")
        write_json(args.out/"status.json",{"status":"failed","stage":args.stage})
        raise


if __name__ == "__main__":
    main()
