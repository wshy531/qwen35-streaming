"""Post-S3 paired diagnostic: identical pixels, original versus local timestamps."""
import argparse
from dataclasses import replace
import inspect
import json
from pathlib import Path
import tempfile
import traceback

import torch

from scripts.baseline import (PROJECT, REPO, add_video_positions, check_position_oracle,
    check_tiny_video_adapter, provenance, file_hash, write_json)
from scripts.stability_experiment import cases, infer, resource_snapshot
from scripts.window_experiment import PROMPT
from streaming.inputs import make_window_inputs, tensor_fingerprint
from streaming.window import FrameWindow, read_packets

SELECTED_QUESTIONS = [7,16,17,59]
BUDGET = {"questions":8,"max_new_tokens":16,"wall_seconds":600,"scope":"post-hoc diagnostic, not tuned accuracy"}


def variants(args):
    evidence = json.loads((args.reference/"gpu_summary.json").read_text(encoding="utf-8"))
    assert len(evidence["questions"]) == 100
    asset = cases(args.assets)["continuous"]
    for number in SELECTED_QUESTIONS:
        original = evidence["questions"][number-1]
        assert original["case"] == "continuous" and not original["exact_after_strip"]
        window = FrameWindow(4)
        reader = read_packets(args.assets/asset["file"],1)
        try:
            for packet in reader:
                window.append(packet)
                if packet.frame_id == original["frame_ids"][-1]:
                    break
            else:
                raise ValueError("Recorded cutoff not found")
        finally:
            reader.close()
        assert window.evidence()["frame_sha256"] == original["frame_sha256"]
        assert window.evidence()["frame_ids"] == original["frame_ids"]
        frames = window.snapshot()
        for variant,packets in (("original",frames),("window_relative",tuple(
                replace(frame,timestamp=frame.timestamp-frames[0].timestamp) for frame in frames))):
            yield {"source_question":number,"variant":variant,"expected":original["expected"],
                   "historical_answer":original["answer"],"historical_input_fingerprint":original["input_fingerprint"],
                   "frame_ids":original["frame_ids"],"frame_sha256":original["frame_sha256"],
                   "timestamps":[x.timestamp for x in packets]},packets


def preflight(out,args):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    torch.set_num_threads(1)
    _,model = check_tiny_video_adapter()
    entries = []
    for metadata,frames in variants(args):
        inputs,info = make_window_inputs(processor,frames,PROMPT)
        add_video_positions(model,inputs)
        check_position_oracle(model,inputs)
        fingerprint = tensor_fingerprint(inputs)
        if metadata["variant"] == "original":
            assert fingerprint == metadata["historical_input_fingerprint"]
        entries.append({**metadata,**info,"input_fingerprint":fingerprint,
                        "input_tokens":inputs.input_ids.shape[1]})
    assert len(entries) == 8
    for a,b in zip(entries[::2],entries[1::2]):
        assert a["frame_sha256"] == b["frame_sha256"] and a["input_fingerprint"] != b["input_fingerprint"]
    write_json(out/"preflight_summary.json",{"budget":BUDGET,"pairs":entries,
        "asset_manifest_sha256":file_hash(args.assets/"manifest.json"),
        "reference_summary_sha256":file_hash(args.reference/"gpu_summary.json")})
    print(json.dumps({"preflight_questions":8,"original_inputs_match_record":True,"paired_pixels_match":True}),flush=True)


def gpu(out,args):
    from transformers import AutoProcessor
    from qwen35_local import Qwen3_5ForConditionalGeneration
    torch.manual_seed(42)
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation="sdpa").to("cuda").eval()
    assert Path(inspect.getfile(type(model))).resolve().parent == REPO/"qwen35_local"
    report = {"budget":BUDGET,"cases":[],"selection":"four S3 errors, selected after observing S3; no benchmark claim"}
    for metadata,frames in variants(args):
        assert len(report["cases"]) < 8
        record,refs,_ = infer(model,processor,frames,True)
        assert all(ref() is None for ref in refs)
        if metadata["variant"] == "original":
            assert record["input_fingerprint"] == metadata["historical_input_fingerprint"]
        entry = {**metadata,**record,"correct":record["answer"].strip().lower() == metadata["expected"],
                 "resources_after_return":resource_snapshot()}
        report["cases"].append(entry)
        write_json(out/"gpu_summary.json",report)
        print(json.dumps({"source_question":metadata["source_question"],"variant":metadata["variant"],
              "answer":record["answer"],"expected":metadata["expected"]}),flush=True)
    assert len(report["cases"]) == 8
    report["original_correct"] = sum(x["correct"] for x in report["cases"] if x["variant"] == "original")
    report["relative_correct"] = sum(x["correct"] for x in report["cases"] if x["variant"] == "window_relative")
    write_json(out/"gpu_summary.json",report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("preflight","gpu"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--assets",type=Path,default=PROJECT/"datasets"/"stability_synthetic")
    parser.add_argument("--model-dir",type=Path,default=PROJECT/"models"/"Qwen3.5-0.8B")
    parser.add_argument("--reference",type=Path,default=PROJECT/"runs"/"stability_gpu_20261009_01")
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
