"""S2 sequential video replay and independent bounded-window questions."""
import argparse
import inspect
import itertools
import json
from pathlib import Path
import tempfile
import time
import traceback
import unittest

import torch

from scripts.baseline import (PROJECT, REPO, REVISION, add_video_positions, cache_measure,
                             check_position_oracle, check_tiny_video_adapter,
                             file_hash, provenance, write_json)
from streaming.window import FramePacket, iter_windows, read_packets
from streaming.inputs import make_window_inputs, tensor_fingerprint

PROMPT = "What color is the square in the last frame of the provided video? Answer with one English color word."


def create_assets(root):
    import av
    import numpy as np
    from PIL import Image, ImageDraw
    root.mkdir(parents=True, exist_ok=False)
    sequences = {"same_prefix_future_blue": ["red"]*2+["blue"]*2+["green"]*2+["yellow"]*2+["red"]*2+["blue"]*2,
                 "same_prefix_future_green": ["red"]*2+["green"]*2+["blue"]*2+["yellow"]*2+["red"]*2+["green"]*2}
    manifest = {"source":"self-generated synthetic, not a benchmark", "fps":1, "videos":[]}
    for name, colors in sequences.items():
        target = root / (name+".mp4")
        with av.open(str(target), "w") as container:
            stream = container.add_stream("libx264", rate=1,
                options={"bf":"0", "g":"1", "crf":"0", "rc-lookahead":"0", "sync-lookahead":"0"})
            stream.width = stream.height = 256
            stream.pix_fmt = "yuv420p"
            for color in colors:
                image = Image.new("RGB", (256,256), "white")
                ImageDraw.Draw(image).rectangle((48,48,208,208), fill=color)
                for encoded in stream.encode(av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")):
                    container.mux(encoded)
            for encoded in stream.encode():
                container.mux(encoded)
        manifest["videos"].append({"id":name, "file":target.name, "colors":colors,
                                    "sha256":file_hash(target), "frames":len(colors)})
    write_json(root/"manifest.json",manifest)
    return manifest


def asset_cases(root):
    manifest = json.loads((root/"manifest.json").read_text(encoding="utf-8"))
    for case in manifest["videos"]:
        path = (root/case["file"]).resolve()
        if path.parent != root.resolve() or file_hash(path) != case["sha256"]:
            raise ValueError("Unexpected asset path or hash")
    return manifest["videos"]


def cpu_tests(out):
    from tests.test_stream_window import WindowTests
    with (out/"unittest.log").open("w",encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream,verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(WindowTests))
    report = {"tests":result.testsRun,"errors":len(result.errors),"failures":len(result.failures),
              "passed":result.wasSuccessful(), "scope":"window and sampling contracts, no model capability"}
    write_json(out/"cpu_summary.json",report)
    if not result.wasSuccessful():
        raise RuntimeError("Window contracts failed")
    print(json.dumps(report))


def preflight(out, args):
    import numpy as np
    from transformers import AutoProcessor
    torch.set_num_threads(1)
    model_manifest = json.loads((args.model_dir/"DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8-sig"))
    if model_manifest["revision"] != REVISION:
        raise ValueError("Unexpected model revision")
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    tiny_test, position_model = check_tiny_video_adapter()
    cases = asset_cases(args.assets)
    summaries = []
    first_hashes = []
    for case in cases:
        snapshots = []
        for frames,evidence in iter_windows(read_packets(args.assets/case["file"],args.fps),args.window,args.every):
            inputs, info = make_window_inputs(processor,frames,args.prompt)
            # Independent full processor reference, CFR only, using ORIGINAL
            # frame numbers rather than window-relative numbers.
            messages = [{"role":"user","content":[{"type":"video"},{"type":"text","text":args.prompt}]}]
            text = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            reference = processor(text=[text], videos=[np.stack([p.rgb for p in frames])], do_sample_frames=False,
                video_metadata=[{"fps":1.0,"total_num_frames":frames[-1].decoded_count,
                                 "frames_indices":[p.frame_id for p in frames]}],return_tensors="pt")
            assert set(reference) == set(inputs)
            for key in reference:
                torch.testing.assert_close(inputs[key],reference[key],rtol=0,atol=0)
            info["position_adapter"] = add_video_positions(position_model,inputs)
            info.update(check_position_oracle(position_model,inputs))
            info.update(evidence)
            info["input_fingerprint"] = tensor_fingerprint(inputs)
            info["input_tokens"] = inputs.input_ids.shape[1]
            info["full_processor_matches"] = True
            snapshots.append(info)
        if not snapshots:
            raise ValueError("No window triggered")
        first_hashes.append(snapshots[0]["input_fingerprint"])
        summaries.append({"id":case["id"],"windows":snapshots})
    assert len(set(first_hashes)) == 1
    # Explicit known-frame odd padding and irregular PTS; never read a new frame.
    reader = read_packets(args.assets/cases[0]["file"],args.fps)
    try:
        original_frames = list(itertools.islice(reader,4))
    finally:
        reader.close()
    odd = []
    for length in (1,2,3,4):
        inputs,info = make_window_inputs(processor,original_frames[:length],args.prompt)
        add_video_positions(position_model,inputs)
        check_position_oracle(position_model,inputs)
        assert all(i <= original_frames[length-1].frame_id for i in info["temporal_padding_frame_ids"])
        odd.append({"frames":length,**info,"position_shape":list(inputs.position_ids.shape)})
    irregular = [FramePacket(p.frame_id,t,p.rgb,p.decoded_count) for p,t in zip(original_frames,[0.2,1.4,3.9,5.7])]
    _,irregular_info = make_window_inputs(processor,irregular[:3],args.prompt)
    assert irregular_info["rendered_timestamps"] == ["0.8","3.9"]
    report = {"tiny_adapter_test":tiny_test,"same_prefix_model_inputs_match":True,"cases":summaries,
              "odd_known_frame_padding":odd,"irregular_pts":irregular_info,
              "questions":sum(len(c["windows"]) for c in summaries),"capability_scope":"CPU interface only"}
    if report["questions"] > 12:
        raise ValueError("Declared GPU question budget exceeded")
    write_json(out/"preflight_summary.json",report)
    print(json.dumps({"windows":report["questions"],"prefix_match":True,"lengths_checked":[1,2,3,4]}))


class ModelTrace:
    def __init__(self, model):
        self.records = []
        self.handles = []
        for kind,module in (("vision",model.model.visual),("language",model.model.language_model)):
            self.handles.append(module.register_forward_pre_hook(self.before(kind),with_kwargs=True))
            self.handles.append(module.register_forward_hook(self.after(kind),with_kwargs=True))

    def before(self,kind):
        def hook(module, args, kwargs):
            entry = {"kind":kind}
            if kind == "language":
                embeddings = kwargs.get("inputs_embeds")
                ids = kwargs.get("input_ids")
                length = (embeddings if embeddings is not None else ids).shape[1]
                cache = kwargs.get("past_key_values")
                entry.update(query_tokens=length, past_tokens_before=0 if cache is None else cache.get_seq_length())
                entry["kind"] = "prefill" if not any(x["kind"] == "prefill" for x in self.records) else "decode"
            entry["start"] = torch.cuda.Event(enable_timing=True)
            entry["end"] = torch.cuda.Event(enable_timing=True)
            entry["start"].record()
            self.records.append(entry)
        return hook

    def after(self,kind):
        def hook(module,args,kwargs,output):
            entry = self.records[-1]
            entry["end"].record()
            if kind == "language" and output.past_key_values is not None:
                entry["cache"] = cache_measure(output.past_key_values)
        return hook

    def summary(self):
        entries = [{k:v for k,v in record.items() if k not in ("start","end")} |
                   {"cuda_interval_ms":record["start"].elapsed_time(record["end"])} for record in self.records]
        prefills = [x for x in entries if x["kind"] == "prefill"]
        assert len(prefills) == 1 and prefills[0]["past_tokens_before"] == 0
        return {"module_calls":entries, "cuda_intervals_ms":
                {kind:sum(x["cuda_interval_ms"] for x in entries if x["kind"] == kind)
                 for kind in ("vision","prefill","decode")}}

    def close(self):
        for handle in self.handles:
            handle.remove()


def gpu(out,args):
    from transformers import AutoProcessor
    from qwen35_local import Qwen3_5ForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.manual_seed(42)
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir,local_files_only=True,
                    dtype=torch.bfloat16,attn_implementation="sdpa").to("cuda").eval()
    assert Path(inspect.getfile(type(model))).resolve().parent == REPO/"qwen35_local"
    report = {"mode":"synchronous PTS replay, independent original-model windows",
              "model_revision":REVISION,"gpu":torch.cuda.get_device_name(),"cases":[],
              "capability_scope":"synthetic last-frame color, not long-memory or real-time evidence"}
    first_hashes = []
    for case in asset_cases(args.assets):
        pending_decode = 0.0
        def counted():
            nonlocal pending_decode
            for packet in read_packets(args.assets/case["file"],args.fps):
                pending_decode += packet.decode_seconds
                yield packet
        trace = ModelTrace(model)
        try:
            for frames,evidence in iter_windows(counted(),args.window,args.every):
                if len(report["cases"]) >= 12:
                    raise RuntimeError("Question budget exceeded")
                trace.records.clear()
                model.model.rope_deltas = None
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                inputs,info = make_window_inputs(processor,frames,args.prompt)
                info["position_adapter"] = add_video_positions(model,inputs)
                fingerprint = tensor_fingerprint(inputs)
                processor_seconds = time.perf_counter()-start
                transfer_start = time.perf_counter()
                inputs = inputs.to("cuda")
                model.model.rope_deltas = model.model.rope_deltas.to("cuda")
                torch.cuda.synchronize()
                transfer_seconds = time.perf_counter()-transfer_start
                generation_start = time.perf_counter()
                with torch.inference_mode():
                    generated = model.generate(**inputs,max_new_tokens=16,do_sample=False)
                torch.cuda.synchronize()
                generation_seconds = time.perf_counter()-generation_start
                measurements = trace.summary()
                answer = processor.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0]
                expected = case["colors"][frames[-1].frame_id]  # Evaluation ONLY, never passed to model.
                entry = {"video":case["id"],"prompt":args.prompt,"answer":answer,"expected":expected,
                         "exact_after_strip":answer.strip().lower() == expected, **evidence, **info,
                         "cutoff_seconds":frames[-1].timestamp,"input_fingerprint":fingerprint,
                         "input_tokens":inputs.input_ids.shape[1],"generated_tokens":generated.shape[1]-inputs.input_ids.shape[1],
                         "decode_resize_seconds_since_question":pending_decode,
                         "processor_positions_fingerprint_seconds":processor_seconds,
                         "transfer_seconds":transfer_seconds,"generation_seconds":generation_seconds,
                         "peak_allocated_gib":torch.cuda.max_memory_allocated()/2**30,
                         "peak_reserved_gib":torch.cuda.max_memory_reserved()/2**30, **measurements}
                if evidence["accepted_samples"] == args.every:
                    first_hashes.append(fingerprint)
                report["cases"].append(entry)
                write_json(out/"gpu_summary.json",report)
                print(json.dumps({"video":case["id"],"cutoff":entry["cutoff_seconds"],"ids":entry["frame_ids"],
                                  "answer":answer,"expected":expected},ensure_ascii=False),flush=True)
                pending_decode = 0.0
                model.model.rope_deltas = None
                del generated,inputs
        finally:
            trace.close()
            model.model.rope_deltas = None
    assert len(set(first_hashes)) == 1
    report["same_prefix_model_inputs_match"] = True
    report["exact_answers"] = sum(x["exact_after_strip"] for x in report["cases"])
    write_json(out/"gpu_summary.json",report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("assets","cpu","preflight","gpu"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--model-dir",type=Path,default=PROJECT/"models"/"Qwen3.5-0.8B")
    parser.add_argument("--assets",type=Path,default=PROJECT/"datasets"/"window_synthetic")
    parser.add_argument("--fps",type=float,default=1.0)
    parser.add_argument("--window",type=int,default=4)
    parser.add_argument("--every",type=int,default=2)
    parser.add_argument("--prompt",default=PROMPT)
    args = parser.parse_args()
    if not 1 <= args.window <= 4 or args.every < 1:
        parser.error("Small S2 experiment requires window 1..4 and positive cadence")
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"tmp").mkdir()
    tempfile.tempdir = str((args.out/"tmp").resolve())
    write_json(args.out/"provenance.json",provenance())
    write_json(args.out/"arguments.json",{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    try:
        if args.stage == "assets":
            write_json(args.out/"assets_summary.json",create_assets(args.assets))
        elif args.stage == "cpu":
            cpu_tests(args.out)
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
