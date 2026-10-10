"""Same-pixel protocol audit, then original-model frame-count/computation-setting controls."""
import argparse
import hashlib
import importlib.util
import inspect
import itertools
import json
from pathlib import Path
import sys
import tempfile
import time
import traceback

import numpy as np
import psutil
import torch
from transformers import AutoProcessor

from scripts.baseline import PROJECT,REPO,REVISION,provenance,write_json,file_hash,add_video_positions,check_position_oracle
from scripts.online_experiment import reference_inputs,Q1,Q2
from scripts.fp32_regression import strict_logits
from streaming.inputs import make_window_inputs,tensor_fingerprint
from streaming.online import fp32_reference
from streaming.window import read_packets

ORIGINAL = PROJECT/"code/qwen35-streaming/qwen35_local"
PLAN = PROJECT/"同画面输入与能力控制方案.md"
LAUNCHER = PROJECT/"scripts/run_protocol_capability.py"
BUDGET = {"cases":8,"max_tokens":256,"max_new_tokens_per_answer":8,"generated_tokens_max":64,
          "language_forwards_max":72,"vision_encodings":8,"model_loads":1,"wall_seconds":600}


def original_package():
    name = "qwen35_original_reference"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name,ORIGINAL/"__init__.py",submodule_search_locations=[str(ORIGINAL)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def source_assets():
    import transformers.models.qwen3_vl.processing_qwen3_vl as processing
    import transformers.models.qwen3_vl.video_processing_qwen3_vl as video
    return {str(p.resolve()):file_hash(p) for p in (*sorted(ORIGINAL.glob("*.py")),
        Path(inspect.getfile(processing)),Path(inspect.getfile(video)))}


def packets(video):
    reader = read_packets(video,1)
    try:
        result = tuple(itertools.islice(reader,4))
    finally:
        reader.close()
    assert len(result) == 4
    return result


def build(model,processor,frames,n,q):
    model.model.rope_deltas = None
    native,info = make_window_inputs(processor,frames[:n],q)
    add_video_positions(model,native)
    check_position_oracle(model,native)
    online = reference_inputs(model,processor,frames[:n],(192,256),q)
    assert set(native) == set(online)
    comparisons = {key:torch.equal(native[key],online[key]) for key in native}
    assert all(comparisons.values()),comparisons
    assert native.input_ids.shape[1] <= 256
    return native,{**info,"all_input_tensors_bitwise_equal":True,"tensor_equalities":comparisons,
        "input_fingerprint":tensor_fingerprint(native),"input_tokens":native.input_ids.shape[1],
        "rendered_text":processor.tokenizer.decode(native.input_ids[0],skip_special_tokens=False),
        "grid":native.video_grid_thw.tolist()}


def preflight(out,args):
    from tests.test_online import small_model
    torch.set_num_threads(1)
    assert json.loads((args.model_dir/"DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8-sig"))["revision"] == REVISION
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    current = small_model(processor)
    package = original_package()
    original = package.Qwen3_5ForConditionalGeneration(package.Qwen3_5Config(**current.config.to_dict())).eval()
    original.set_attn_implementation("sdpa")
    original.load_state_dict(current.state_dict(),strict=True)
    frames = packets(args.video)
    entries = []
    with torch.inference_mode(),fp32_reference():
        for n,qname,q in ((2,"subject",Q1),(2,"hands_count",Q2),(4,"subject",Q1),(4,"hands_count",Q2)):
            inputs,record = build(original,processor,frames,n,q)
            old = original(**inputs,use_cache=True)
            new = current(**inputs,use_cache=True)
            comparison = strict_logits(new.logits,old.logits)
            assert comparison["bitwise_equal"],"Modified core must not change fresh tiny prefill"
            del old,new
            entries.append({"id":f"frames{n}_{qname}","frames":n,"question":q,"question_kind":qname,
                **record,"tiny_original_vs_modified_prefill":comparison})
    historical = json.loads((PROJECT/"runs/natural_gpu_20261009_01/gpu_summary.json").read_text())
    first_four = next(x for x in historical["questions"] if x["video"] == "clean_hands" and x["sample_index"] == 3)
    assert entries[2]["input_fingerprint"] == first_four["input_fingerprint"]
    assert first_four["frame_ids"] == [p.frame_id for p in frames]
    assert first_four["timestamps"] == [p.timestamp for p in frames]
    prior = json.loads((PROJECT/"runs/s4b_shared_preflight_20261009_02/preflight_summary.json").read_text())
    assert entries[0]["input_fingerprint"] == prior["input_fingerprints"]["question2"]
    assert entries[3]["input_fingerprint"] == prior["input_fingerprints"]["question4"]
    cases = [{"id":f"{mode}_{entry['id']}","mode":mode,"input":entry["id"]} for mode in ("bf16_sdpa","fp32_ieee_math") for entry in entries]
    report = {"cpu_controls_passed":True,"budget":BUDGET,"input_comparisons":entries,"cases":cases,
        "frame_ids":[p.frame_id for p in frames],"timestamps":[p.timestamp for p in frames],
        "rgb_shapes":[list(p.rgb.shape) for p in frames],"frame_sha256":[hashlib.sha256(p.rgb.tobytes()).hexdigest() for p in frames],
        "historical_four_frame_subject_fingerprint_matches":True,"historical_answer":first_four["answer"],
        "prior_two_frame_subject_and_four_frame_count_fingerprints_match":True,"original_reference_hashes":source_assets(),
        "video_sha256":file_hash(args.video),"model_manifest_sha256":file_hash(args.model_dir/"DOWNLOAD_MANIFEST.json"),
        "plan_sha256":file_hash(PLAN),"launcher_sha256":file_hash(LAUNCHER),"model_reference":"unaltered S1-S3 original local model",
        "scope":"input equality and tiny fresh-prefill only; no capability or pretrained run"}
    write_json(out/"preflight_summary.json",report)
    print({"four_input_pairs_bitwise_equal":True,"tiny_original_prefill_bitwise_equal":4,
        "historical_input_exactly_rebuilt":True,"budget":BUDGET},flush=True)


def top_options(logits,processor,q):
    choices = ("CAT","HANDS","OTHER") if q == Q1 else ("ONE","TWO","OTHER")
    probabilities = logits[0,-1].float().softmax(-1)
    values,indices = probabilities.topk(5)
    return {"top5_first_token":[{"id":int(i),"decoded":processor.tokenizer.decode([int(i)]),"probability":float(p)} for p,i in zip(values,indices)],
        "option_first_token_probabilities":{word:{"token_ids":processor.tokenizer(word,add_special_tokens=False).input_ids,
            "first_token_probability":float(probabilities[processor.tokenizer(word,add_special_tokens=False).input_ids[0]])} for word in choices},
        "scope":"first-token probabilities only, not full option likelihoods"}


@torch.inference_mode()
def gpu(out,args):
    from contextlib import nullcontext
    begin = time.perf_counter()
    free,total = torch.cuda.mem_get_info()
    guard = {"gpu_free_bytes":free,"gpu_total_bytes":total,"ram_available_bytes":psutil.virtual_memory().available}
    write_json(out/"resource_precheck.json",guard)
    assert free >= 5.5*2**30 and guard["ram_available_bytes"] >= 3*2**30,"Resource guard; no load"
    torch.manual_seed(42)
    package = original_package()
    model = package.Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir,local_files_only=True,dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    assert Path(inspect.getfile(type(model))).resolve() == ORIGINAL/"modeling_qwen3_5.py"
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    frames = packets(args.video)
    proof = json.loads((args.preflight/"preflight_summary.json").read_text())
    assert source_assets() == proof["original_reference_hashes"]
    counts = {"language":0,"vision":0}
    active = {"first_logits":None}
    def count_language(_m,_a,_o):
        counts["language"] += 1
    def count_vision(_m,_a,_o):
        counts["vision"] += 1
    def capture_first(_m,_a,output):
        if active["first_logits"] is None:
            active["first_logits"] = output.logits[:,-1:].detach().cpu().clone()
    hooks = [model.model.language_model.register_forward_hook(count_language),model.model.visual.register_forward_hook(count_vision),
        model.register_forward_hook(capture_first)]
    report = {"budget":BUDGET,"cases":[],"model_reference":str(ORIGINAL/"modeling_qwen3_5.py"),
        "scope":"fixed original-model same-input semantic diagnosis; no training or online-state acceptance", "resource_precheck":guard}
    generated = 0
    try:
        for mode in ("bf16_sdpa","fp32_ieee_math"):
            if mode == "fp32_ieee_math":
                assert torch.cuda.mem_get_info()[0] >= 2*2**30
                model.float()
            torch.cuda.reset_peak_memory_stats()
            context = fp32_reference() if mode == "fp32_ieee_math" else nullcontext()
            with context:
                for input_spec in proof["input_comparisons"]:
                    inputs,identity = build(model,processor,frames,input_spec["frames"],input_spec["question"])
                    assert identity["input_fingerprint"] == input_spec["input_fingerprint"]
                    model.model.rope_deltas = model.model.rope_deltas.to("cuda")
                    inputs = inputs.to("cuda")
                    active["first_logits"] = None
                    before = counts.copy()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    eos = model.generation_config.eos_token_id
                    eos_ids = set([] if eos is None else eos if isinstance(eos,list) else [eos])
                    eos_ids.add(processor.tokenizer.eos_token_id)
                    output = model.generate(**inputs,max_new_tokens=8,do_sample=False,eos_token_id=sorted(eos_ids))
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-start
                    ids = output[0,inputs.input_ids.shape[1]:].tolist()
                    answer = processor.tokenizer.decode(ids,skip_special_tokens=True)
                    generated += len(ids)
                    record = {"id":mode+"_"+input_spec["id"],"mode":mode,"frames":input_spec["frames"],
                        "question":input_spec["question"],"question_kind":input_spec["question_kind"],
                        "input_fingerprint":identity["input_fingerprint"],"generated_ids":ids,"answer":answer,
                        "stopped_on_registered_eos":bool(ids) and ids[-1] in eos_ids,
                        "forward_counts":{k:counts[k]-before[k] for k in counts},"generation_seconds":elapsed,
                        **top_options(active["first_logits"],processor,input_spec["question"])}
                    report["cases"].append(record)
                    print({"id":record["id"],"answer":answer,"generated_ids":ids},flush=True)
                    del output,inputs
                    active["first_logits"] = None
                    write_json(out/"protocol_capability_summary.json",report)
            report[mode+"_peak_memory"] = {"allocated_gib":torch.cuda.max_memory_allocated()/2**30,
                "reserved_gib":torch.cuda.max_memory_reserved()/2**30}
        assert [c["id"] for c in report["cases"]] == [c["id"] for c in proof["cases"]]
        assert counts["vision"] == 8 and counts["language"] <= 72 and generated <= 64
        report.update(actual_forwards=counts,generated_tokens=generated,wall_seconds_including_load=time.perf_counter()-begin,
            all_registered_cases_executed=True,all_input_fingerprints_verified=True)
        write_json(out/"protocol_capability_summary.json",report)
        print({k:report[k] for k in ("actual_forwards","generated_tokens","wall_seconds_including_load")},flush=True)
    finally:
        for hook in hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("preflight","gpu"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--model-dir",type=Path,default=PROJECT/"models/Qwen3.5-0.8B")
    parser.add_argument("--video",type=Path,default=PROJECT/"datasets/natural_commons_v1/clean_hands.webm")
    parser.add_argument("--preflight",type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"tmp").mkdir()
    tempfile.tempdir = str((args.out/"tmp").resolve())
    write_json(args.out/"provenance.json",provenance())
    (args.out/"script_snapshot.py").write_bytes(Path(__file__).read_bytes())
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
