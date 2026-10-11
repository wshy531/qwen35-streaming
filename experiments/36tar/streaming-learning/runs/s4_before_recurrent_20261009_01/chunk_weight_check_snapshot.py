"""Fixed S4-A original-weight numerical checks; pre-encoded vision, no answers."""
import argparse
import inspect
import itertools
import json
from pathlib import Path
import tempfile
import time
import traceback

import torch

from scripts.baseline import (PROJECT,REPO,REVISION,add_video_positions,check_position_oracle,
    check_tiny_video_adapter,cache_measure,file_hash,provenance,write_json)
from streaming.hybrid import HybridSequence
from streaming.inputs import make_window_inputs,tensor_fingerprint
from streaming.window import read_packets

FIELDS = ("key_cache","value_cache","conv_states","recurrent_states")
PROMPT = "What is the main visible subject in the LAST frame of this video window? Answer only CAT, HANDS, or OTHER."


def inputs_and_groups(args,processor,position_model):
    from transformers.feature_extraction_utils import BatchFeature
    result = {}
    for name,text in (("short_text","Explain what a video frame is."),
        ("long_text","A video consists of consecutive images captured over time. "*8+"Name the smallest temporal unit.")):
        template = processor.apply_chat_template([{"role":"user","content":[{"type":"text","text":text}]}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs = BatchFeature(processor.tokenizer([template],return_tensors="pt"))
        length = inputs.input_ids.shape[1]
        inputs["position_ids"] = torch.arange(length)[None,None,:].expand(4,1,length).clone()
        result[name] = inputs
    reader = read_packets(args.video,1)
    try:
        frames = tuple(itertools.islice(reader,4))
    finally:
        reader.close()
    assert len(frames) == 4
    video,_ = make_window_inputs(processor,frames,PROMPT)
    add_video_positions(position_model,video)
    check_position_oracle(position_model,video)
    result["video"] = video
    short,long,visual = (result[x].input_ids.shape[1] for x in ("short_text","long_text","video"))
    assert short >= 4 and 66 < long <= 256 and 66 < visual <= 256
    first_visual = int(torch.where(video.mm_token_type_ids[0] == 2)[0][0])
    inside = first_visual+1
    groups = [
        {"id":"short_2_rest","input":"short_text","chunks":[2,short-2]},
        {"id":"short_1_rest","input":"short_text","chunks":[1,short-1]},
        {"id":"long_63_1_rest","input":"long_text","chunks":[63,1,long-64]},
        {"id":"video_2_rest","input":"video","chunks":[2,visual-2]},
        {"id":"video_inside_visual","input":"video","chunks":[inside,7,visual-inside-7]},
        {"id":"video_64_rest","input":"video","chunks":[64,visual-64]},
        {"id":"video_last_single","input":"video","chunks":[visual-1,1]},
    ]
    assert len(groups)+3 == 10
    return result,groups


def preflight(out,args):
    from transformers import AutoProcessor
    torch.set_num_threads(1)
    assert json.loads((args.model_dir/"DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8-sig"))["revision"] == REVISION
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    tiny,position_model = check_tiny_video_adapter()
    prepared,groups = inputs_and_groups(args,processor,position_model)
    report = {"groups":groups,"extra_groups":["fork_isolation","reset","capacity_rejection"],"total_groups":10,
        "scope":"CPU input/position preflight; not official-weight inference","tiny_position_adapter":tiny,
        "video_sha256":file_hash(args.video),"model_manifest_sha256":file_hash(args.model_dir/"DOWNLOAD_MANIFEST.json"),
        "input_fingerprints":{k:tensor_fingerprint(v) for k,v in prepared.items()},
        "input_tokens":{k:v.input_ids.shape[1] for k,v in prepared.items()},
        "max_sequence_tokens":256,"wall_seconds":600,"generated_tokens":0}
    write_json(out/"preflight_summary.json",report)
    print(json.dumps({"groups":10,"input_tokens":report["input_tokens"],"generated_tokens":0}),flush=True)


def logits_metrics(actual,expected):
    delta = actual.float()-expected.float()
    max_abs,mean_abs,rms = delta.abs().max().item(),delta.abs().mean().item(),delta.square().mean().sqrt().item()
    return {"max_abs_diff":max_abs,"mean_abs_diff":mean_abs,"rms_diff":rms,
        "bitwise_equal":torch.equal(actual,expected),"allclose_5e3":torch.allclose(actual.float(),expected.float(),atol=5e-3,rtol=5e-3),
        "last_argmax_equal":torch.equal(actual[:,-1].argmax(-1),expected[:,-1].argmax(-1)),
        "all_positions_argmax_equal":torch.equal(actual.argmax(-1),expected.argmax(-1)),
        "registered_numeric_limits_passed":max_abs <= .25 and mean_abs <= .01 and rms <= .02}


def state_metrics(a,b):
    result = {}
    for field in FIELDS:
        differences,equalities = [],[]
        for left,right in zip(getattr(a,field),getattr(b,field)):
            if left is None or right is None:
                assert left is None and right is None
                continue
            assert left.shape == right.shape
            differences.append((left.float()-right.float()).abs().max().item())
            equalities.append(torch.equal(left,right))
        result[field] = {"populated_layers":len(differences),"max_abs_diff":max(differences,default=0.),
                         "all_tensors_bitwise_equal":all(equalities)}
    return result


@torch.inference_mode()
def encoded(model,inputs):
    embeddings = model.get_input_embeddings()(inputs.input_ids)
    if "pixel_values_videos" in inputs:
        vision = model.model.get_video_features(inputs.pixel_values_videos,inputs.video_grid_thw,return_dict=True)
        features = torch.cat(vision.pooler_output,dim=0).to(embeddings)
        _,mask = model.model.get_placeholder_mask(inputs.input_ids,inputs_embeds=embeddings,video_features=features)
        embeddings = embeddings.masked_scatter(mask,features)
    return embeddings


@torch.inference_mode()
def compare(model,embeddings,positions,layout):
    full = model(inputs_embeds=embeddings,position_ids=positions,use_cache=True)
    stream = HybridSequence(model,max_tokens=256)
    start,outputs,cache_lengths = 0,[],[]
    for length in layout:
        outputs.append(stream.append(inputs_embeds=embeddings[:,start:start+length],position_ids=positions[:,:,start:start+length]))
        start += length
        cache_lengths.append(stream.cache.get_seq_length())
    assert start == embeddings.shape[1]
    actual = torch.cat(outputs,dim=1)
    return {"chunks":layout,"cache_lengths":cache_lengths,"logits":logits_metrics(actual,full.logits),
            "final_states":state_metrics(stream.cache,full.past_key_values),"cache":cache_measure(stream.cache)}


@torch.inference_mode()
def fork_checks(model,processor,embeddings,positions):
    main = HybridSequence(model,max_tokens=256)
    main.append(inputs_embeds=embeddings,position_ids=positions)
    snapshot = main.cache.clone()
    counts = (main.seen_tokens,main.next_rope_position)
    branch = main.fork()
    suffix = processor.tokenizer(" Additional question.",add_special_tokens=False,return_tensors="pt").input_ids.to("cuda")
    actual = branch.append(input_ids=suffix)
    length = suffix.shape[1]
    text = torch.arange(main.seen_tokens,main.seen_tokens+length,device="cuda")
    rope = torch.arange(main.next_rope_position,main.next_rope_position+length,device="cuda")
    suffix_positions = torch.stack((text,rope,rope,rope))[:,None,:]
    full = model(inputs_embeds=torch.cat((embeddings,model.get_input_embeddings()(suffix)),dim=1),
        position_ids=torch.cat((positions,suffix_positions),dim=2),use_cache=True)
    unchanged = state_metrics(main.cache,snapshot)
    assert all(x["all_tensors_bitwise_equal"] for x in unchanged.values())
    assert counts == (main.seen_tokens,main.next_rope_position)
    return main,{"id":"fork_isolation","main_unchanged_bitwise":True,"prefix_tokens":main.seen_tokens,
        "branch_tokens":branch.seen_tokens,"fork_copy_tensor_bytes":cache_measure(snapshot)["total_tensor_bytes"],
        "logits":logits_metrics(actual,full.logits[:,-length:]),"final_states":state_metrics(branch.cache,full.past_key_values)}


@torch.inference_mode()
def reset_and_capacity(main,model,short_inputs):
    model.model.rope_deltas = torch.tensor([[-112]],device="cuda")
    main.reset("new_sequence")
    cleared = main.seen_tokens == main.next_rope_position == main.cache.get_seq_length() == 0 and model.model.rope_deltas is None
    assert cleared and all(x is None for field in FIELDS for x in getattr(main.cache,field))
    actual = main.append(input_ids=short_inputs.input_ids)
    full = model(input_ids=short_inputs.input_ids,position_ids=short_inputs.position_ids,use_cache=True)
    reset_record = {"id":"reset","all_state_and_positions_cleared":True,"logits":logits_metrics(actual,full.logits),
                    "final_states":state_metrics(main.cache,full.past_key_values)}
    main.max_tokens = main.seen_tokens
    before = main.cache.clone()
    counts = (main.seen_tokens,main.next_rope_position)
    rejected = False
    try:
        main.append(input_ids=short_inputs.input_ids[:,:1])
    except ValueError as error:
        rejected = "Token budget exceeded" in str(error)
    assert rejected and counts == (main.seen_tokens,main.next_rope_position)
    assert all(x["all_tensors_bitwise_equal"] for x in state_metrics(main.cache,before).values())
    return [reset_record,{"id":"capacity_rejection","rejected_before_forward":True,"all_state_unchanged_bitwise":True}]


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
    prepared,groups = inputs_and_groups(args,processor,model)
    fingerprints = {k:tensor_fingerprint(v) for k,v in prepared.items()}
    with torch.inference_mode():
        prepared = {k:v.to("cuda") for k,v in prepared.items()}
        embeddings = {k:encoded(model,v) for k,v in prepared.items()}
    report = {"model_revision":REVISION,"branch":"36tar/s4-mixed-state","gpu":torch.cuda.get_device_name(),
        "input_fingerprints":fingerprints,"groups":[],"generated_tokens":0,
        "scope":"same pre-encoded known sequence split across real hybrid state; not online vision/QA protocol"}
    torch.cuda.reset_peak_memory_stats()
    for group in groups:
        key = group["input"]
        record = {**group,**compare(model,embeddings[key],prepared[key].position_ids,group["chunks"])}
        report["groups"].append(record)
        write_json(out/"gpu_summary.json",report)
        print(json.dumps({"id":record["id"],"logits":record["logits"]}),flush=True)
    main,fork_record = fork_checks(model,processor,embeddings["video"],prepared["video"].position_ids)
    report["groups"].append(fork_record)
    report["groups"] += reset_and_capacity(main,model,prepared["short_text"])
    assert len(report["groups"]) == 10
    report["registered_numeric_limits_passed"] = all(x["logits"]["registered_numeric_limits_passed"] for x in report["groups"] if "logits" in x)
    report["all_last_argmax_equal"] = all(x["logits"]["last_argmax_equal"] for x in report["groups"] if "logits" in x)
    report["strict_allclose_all_groups"] = all(x["logits"]["allclose_5e3"] for x in report["groups"] if "logits" in x)
    report["state_isolation_reset_capacity_passed"] = True
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
    report["peak_reserved_gib"] = torch.cuda.max_memory_reserved()/2**30
    report["wall_seconds_including_load"] = time.perf_counter()-start
    write_json(out/"gpu_summary.json",report)
    print(json.dumps({k:report[k] for k in ("registered_numeric_limits_passed","all_last_argmax_equal","strict_allclose_all_groups","wall_seconds_including_load")}),flush=True)
    if not report["registered_numeric_limits_passed"]:
        raise RuntimeError("Registered BF16 numerical limits failed; retain evidence")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("preflight","gpu"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--model-dir",type=Path,default=PROJECT/"models/Qwen3.5-0.8B")
    parser.add_argument("--video",type=Path,default=PROJECT/"datasets/natural_commons_v1/clean_hands.webm")
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
