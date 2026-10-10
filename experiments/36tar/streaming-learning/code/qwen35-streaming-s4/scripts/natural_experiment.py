"""Fifteen pre-labelled natural footage questions plus bounded actual PTS checks."""
import argparse
from dataclasses import replace
import hashlib
import inspect
import json
from pathlib import Path
import re
import tempfile
import time
import traceback

import torch

from scripts.baseline import (PROJECT,REPO,REVISION,check_tiny_video_adapter,
    add_video_positions,check_position_oracle,file_hash,provenance,write_json)
from scripts.stability_experiment import resource_snapshot
from streaming.inputs import make_window_inputs,tensor_fingerprint
from streaming.window import FrameWindow,read_packets
from streaming.inference import independent_question


def assets(args):
    manifest = json.loads((args.assets/"manifest.json").read_text(encoding="utf-8"))
    config = json.loads((args.assets/"cases.json").read_text(encoding="utf-8"))
    assert config["frozen_before_model_answers"] and config["questions"] == len(config["cases"]) == 15
    assert config["window"] == 4 and config["fps"] == 1 and config["max_new_tokens"] == 16
    sources = {x["id"]:x for x in manifest["videos"]+manifest["derived_videos"]}
    assert set(sources) == {"cat_play","clean_hands","hands_vfr"}
    hashes = {name:file_hash(args.assets/name) for name in ("manifest.json","cases.json")}
    for source in sources.values():
        path = (args.assets/source["file"]).resolve()
        if path.parent != args.assets.resolve() or not path.is_file():
            raise ValueError("Unexpected source path")
        assert path.stat().st_size == source["bytes"] and file_hash(path) == source["sha256"]
        hashes[source["file"]] = source["sha256"]
    return sources,config,hashes


def question_windows(args,audit):
    sources,config,_ = assets(args)
    for name in ("cat_play","clean_hands","hands_vfr"):
        window = FrameWindow(4)
        samples = []
        for index,packet in enumerate(read_packets(args.assets/sources[name]["file"],1)):
            window.append(packet)
            samples.append({"frame_id":packet.frame_id,"timestamp":packet.timestamp,
                            "delivered_decode_count":packet.decoded_count})
            if index%2 != 1:
                continue
            for specification in config["cases"]:
                if specification["video"] != name or specification["sample_index"] != index:
                    continue
                frames = window.snapshot()
                evidence = window.evidence()
                source_ids = [x.frame_id for x in frames]
                if specification.get("variant") == "reverse_pixels":
                    original = frames
                    frames = tuple(replace(slot,rgb=pixel_source.rgb) for slot,pixel_source in zip(original,reversed(original)))
                    source_ids = list(reversed(source_ids))
                evidence.update(pixel_source_frame_ids=source_ids,
                    frame_sha256=[hashlib.sha256(x.rgb.tobytes()).hexdigest() for x in frames],
                    frame_id_semantics="display slots; pixel_source_frame_ids identifies actual source pixels")
                assert max(source_ids) < packet.decoded_count and len(frames) <= 4
                yield specification,frames,evidence
        audit[name] = {"accepted_samples":len(samples),"samples":samples,"last_window":window.evidence()}
    assert [x["frame_id"] for x in audit["hands_vfr"]["samples"]] == config["vfr_expected_frame_ids"]
    assert [x["timestamp"] for x in audit["hands_vfr"]["samples"]] == config["vfr_expected_timestamps"]
    assert audit["cat_play"]["accepted_samples"] == 9 and audit["clean_hands"]["accepted_samples"] == 23


def preflight(out,args):
    import numpy as np
    from transformers import AutoProcessor
    torch.set_num_threads(1)
    assert json.loads((args.model_dir/"DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8-sig"))["revision"] == REVISION
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    tiny,model = check_tiny_video_adapter()
    _,config,hashes = assets(args)
    entries,audit = [],{}
    for spec,frames,evidence in question_windows(args,audit):
        inputs,info = make_window_inputs(processor,frames,spec["prompt"])
        # Full processor API pixel reference. Timestamp text remains the true-PTS adapter;
        # generic CFR metadata is ONLY used to compare video pixel tensors/grid.
        template = processor.apply_chat_template([{"role":"user","content":[{"type":"video"},{"type":"text","text":spec["prompt"]}]}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        reference = processor(text=[template],videos=[np.stack([x.rgb for x in frames])],do_sample_frames=False,
            video_metadata=[{"fps":1.,"total_num_frames":len(frames),"frames_indices":list(range(len(frames)))}],return_tensors="pt")
        for key in ("pixel_values_videos","video_grid_thw"):
            torch.testing.assert_close(inputs[key],reference[key],atol=0,rtol=0)
        add_video_positions(model,inputs)
        check_position_oracle(model,inputs)
        entries.append({**spec,**evidence,**info,"input_fingerprint":tensor_fingerprint(inputs),
                        "input_tokens":inputs.input_ids.shape[1],"processor_pixels_match":True})
    assert len(entries) == config["questions"]
    order = [x for x in entries if x["kind"] == "order_diagnostic"]
    assert len(order) == 4
    originals = [x for x in order if x["variant"] == "original"]
    reverses = [x for x in order if x["variant"] == "reverse_pixels"]
    assert all(a["frame_sha256"] == list(reversed(b["frame_sha256"])) and a["timestamps"] == b["timestamps"]
               for a,b in zip(originals,reverses))
    assert audit["hands_vfr"]["accepted_samples"] == 5
    assert len([x for x in entries if x["video"] == "hands_vfr"]) == 2
    write_json(out/"preflight_summary.json",{"scope":"CPU inputs/actual PTS only","tiny_adapter_test":tiny,
        "questions":15,"cases":entries,"reader_audit":audit,"asset_hashes":hashes,
        "controlled_vfr_samples_correct":True,"odd_eof_has_no_extra_question":True,
        "reversal_uses_only_arrived_pixels":True})
    print(json.dumps({"questions":15,"pixel_references_match":15,"position_oracles_pass":15,
                      "accepted_samples":{k:v["accepted_samples"] for k,v in audit.items()},"vfr_sampling_passed":True}),flush=True)


def scores(answer,expected):
    strict = answer.strip().upper() == expected
    match = re.fullmatch(r"\s*(CAT|HANDS|OTHER|FAUCET|TOWEL)[.!]?\s*",answer,re.IGNORECASE)
    normalized = match is not None and match.group(1).upper() == expected
    return {"strict_correct":strict,"normalized_correct":normalized,
            "parsed_option":None if match is None else match.group(1).upper()}


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
    report = {"model_revision":REVISION,"gpu":torch.cuda.get_device_name(),"questions":[],"reader_audit":{},
              "scope":"small pre-labelled natural closed-option diagnostic, not a benchmark or live camera",
              "score_protocol":"strict strip+upper; secondary whole option + optional terminal period/exclamation"}
    for spec,frames,evidence in question_windows(args,report["reader_audit"]):
        assert len(report["questions"]) < 15
        record,refs = independent_question(model,processor,frames,spec["prompt"])
        alive = sum(x() is not None for x in refs)
        entry = {"question":len(report["questions"])+1,**spec,**evidence,**record,
                 **scores(record["answer"],spec["expected"]),"cache_weakrefs_alive":alive,
                 "resources_after_return":resource_snapshot()}
        report["questions"].append(entry)
        write_json(out/"gpu_summary.json",report)
        print(json.dumps({"question":entry["question"],"video":spec["video"],"kind":spec["kind"],
              "variant":spec.get("variant"),"cutoff":frames[-1].timestamp,
              "answer":record["answer"],"expected":spec["expected"]}),flush=True)
    assert len(report["questions"]) == 15
    report["by_kind"] = {kind:{"questions":len(items),"strict_correct":sum(x["strict_correct"] for x in items),
         "normalized_correct":sum(x["normalized_correct"] for x in items)}
         for kind,items in ((kind,[x for x in report["questions"] if x["kind"] == kind])
                           for kind in ("native_last_subject","order_diagnostic","controlled_vfr_last_subject"))}
    report["interface_state_passed"] = all(x["cache_weakrefs_alive"] == 0 and
        next(c for c in x["calls"] if c["kind"] == "prefill")["past_tokens_before"] == 0 for x in report["questions"])
    report["order_all_variants_correct"] = all(x["normalized_correct"] for x in report["questions"] if x["kind"] == "order_diagnostic")
    report["wall_seconds_including_load"] = time.perf_counter()-start
    write_json(out/"gpu_summary.json",report)
    print(json.dumps({"by_kind":report["by_kind"],"interface_state_passed":report["interface_state_passed"],
                      "order_all_variants_correct":report["order_all_variants_correct"]}),flush=True)
    if not report["interface_state_passed"]:
        raise RuntimeError("Temporary-cache/independent-window contract failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("preflight","gpu"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--assets",type=Path,default=PROJECT/"datasets/natural_commons_v1")
    parser.add_argument("--model-dir",type=Path,default=PROJECT/"models/Qwen3.5-0.8B")
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"tmp").mkdir()
    tempfile.tempdir = str((args.out/"tmp").resolve())
    write_json(args.out/"provenance.json",provenance())
    write_json(args.out/"arguments.json",{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
    try:
        (preflight if args.stage == "preflight" else gpu)(args.out,args)
        write_json(args.out/"status.json",{"status":"completed","stage":args.stage})
    except Exception:
        (args.out/"traceback.txt").write_text(traceback.format_exc(),encoding="utf-8")
        write_json(args.out/"status.json",{"status":"failed","stage":args.stage})
        raise


if __name__ == "__main__":
    main()
