"""Fixed FP32 video/state regression; pre-encoded vision, no answer generation."""
import argparse
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import tempfile
import time
import traceback

import psutil
import torch
import scripts.chunk_weight_check as base
from scripts.baseline import PROJECT,REPO,REVISION,provenance,write_json,file_hash
from scripts.precision_fullmodel import precision,settings
from streaming.hybrid import HybridSequence
from streaming.inputs import tensor_fingerprint

PLAN = PROJECT/"FP32混合状态回归方案.md"
LAUNCHER = PROJECT/"scripts/run_fp32_regression.py"
BUDGET = {"groups":10,"max_tokens":256,"generated_tokens":0,"wall_seconds":600,
          "model_loads":1,"language_forwards":30,"vision_encodings":1}
ORIGINAL_LOGITS,ORIGINAL_STATES = base.logits_metrics,base.state_metrics


def strict_logits(actual,expected):
    result = ORIGINAL_LOGITS(actual,expected)
    result["allclose_1e4"] = torch.allclose(actual.float(),expected.float(),atol=1e-4,rtol=1e-4)
    result["fp32_reference_limits_passed"] = (result["max_abs_diff"] <= .01 and result["mean_abs_diff"] <= .0001
                                               and result["rms_diff"] <= .0003)
    return result


def strict_states(actual,expected):
    result = ORIGINAL_STATES(actual,expected)
    for field in base.FIELDS:
        left,right = getattr(actual,field),getattr(expected,field)
        assert len(left) == len(right)
        result[field]["allclose_1e4"] = all(torch.allclose(a.float(),b.float(),atol=1e-4,rtol=1e-4)
            for a,b in zip(left,right) if a is not None)
    return result


@contextmanager
def strict_metric_scope():
    assert base.logits_metrics is ORIGINAL_LOGITS and base.state_metrics is ORIGINAL_STATES
    base.logits_metrics,base.state_metrics = strict_logits,strict_states
    try:
        yield
    finally:
        base.logits_metrics,base.state_metrics = ORIGINAL_LOGITS,ORIGINAL_STATES


def passed(record):
    return (record["logits"]["fp32_reference_limits_passed"] and record["logits"]["allclose_1e4"]
            and all(v["allclose_1e4"] for v in record["final_states"].values()))


def suffix_positions(main,length):
    device = main.model.get_input_embeddings().weight.device
    text = torch.arange(main.seen_tokens,main.seen_tokens+length,device=device)
    rope = torch.arange(main.next_rope_position,main.next_rope_position+length,device=device)
    return torch.stack((text,rope,rope,rope))[:,None,:]


@torch.inference_mode()
def fork_and_continue(model,embeddings,positions,question_suffix,main_suffix):
    main = HybridSequence(model,max_tokens=256)
    main.append(inputs_embeds=embeddings,position_ids=positions)
    snapshot = main.cache.clone()
    counts = main.seen_tokens,main.next_rope_position
    branch = main.fork()
    for field in base.FIELDS:
        for a,b in zip(getattr(main.cache,field),getattr(branch.cache,field)):
            if a is not None:
                assert a.data_ptr() != b.data_ptr()
    q_positions = suffix_positions(main,question_suffix.shape[1])
    actual = branch.append(input_ids=question_suffix)
    full = model(inputs_embeds=torch.cat((embeddings,model.get_input_embeddings()(question_suffix)),1),
        position_ids=torch.cat((positions,q_positions),2),use_cache=True)
    branch_metrics = {"logits":strict_logits(actual,full.logits[:,-question_suffix.shape[1]:]),
                      "final_states":strict_states(branch.cache,full.past_key_values)}
    assert all(v["all_tensors_bitwise_equal"] for v in strict_states(main.cache,snapshot).values())
    assert counts == (main.seen_tokens,main.next_rope_position)
    del full,branch
    future_positions = suffix_positions(main,main_suffix.shape[1])
    actual = main.append(input_ids=main_suffix)
    full = model(inputs_embeds=torch.cat((embeddings,model.get_input_embeddings()(main_suffix)),1),
        position_ids=torch.cat((positions,future_positions),2),use_cache=True)
    continued = {"logits":strict_logits(actual,full.logits[:,-main_suffix.shape[1]:]),
                 "final_states":strict_states(main.cache,full.past_key_values)}
    assert main.cache.get_seq_length() == main.seen_tokens == embeddings.shape[1]+main_suffix.shape[1]
    record = {"id":"fork_isolation",**branch_metrics,"main_unchanged_before_continuation_bitwise":True,
        "cache_storage_independent":True,"prefix_tokens":counts[0],"branch_suffix_tokens":question_suffix.shape[1],
        "main_suffix_tokens":main_suffix.shape[1],"main_continuation":continued}
    return main,record


def suffixes(processor):
    q = processor.tokenizer(" Additional question.",add_special_tokens=False,return_tensors="pt").input_ids
    m = processor.tokenizer(" Continue.",add_special_tokens=False,return_tensors="pt").input_ids
    assert (q.shape[1],m.shape[1]) == (3,2)
    return q,m


@torch.inference_mode()
def cpu_controls(prepared,groups):
    from tests.test_local_model import tiny_config
    from qwen35_local import Qwen3_5Config,Qwen3_5ForConditionalGeneration
    from transformers.feature_extraction_utils import BatchFeature
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**tiny_config(),attn_implementation="sdpa")).eval()
    mini = {name:BatchFeature({"input_ids":torch.randint(1,100,value.input_ids.shape),"position_ids":value.position_ids})
            for name,value in prepared.items()}
    old = settings()
    records = []
    with precision("fp32_ieee_math"),strict_metric_scope():
        embeddings = {name:base.encoded(model,value) for name,value in mini.items()}
        for group in groups:
            record = base.compare(model,embeddings[group["input"]],mini[group["input"]].position_ids,group["chunks"])
            assert passed(record),group
            records.append({"id":group["id"],"logits":record["logits"]})
        main,fork = fork_and_continue(model,embeddings["video"],mini["video"].position_ids,
                                     torch.tensor([[51,52,53]]),torch.tensor([[4,5]]))
        assert passed(fork) and passed(fork["main_continuation"])
        reset,capacity = base.reset_and_capacity(main,model,mini["short_text"])
        assert passed(reset) and capacity["all_state_unchanged_bitwise"]
    assert settings() == old and base.logits_metrics is ORIGINAL_LOGITS and base.state_metrics is ORIGINAL_STATES
    try:
        with strict_metric_scope():
            raise ValueError("intentional restoration control")
    except ValueError:
        pass
    assert base.logits_metrics is ORIGINAL_LOGITS and base.state_metrics is ORIGINAL_STATES
    return {"passed":True,"seven_layouts":records,"fork_and_main_continuation":fork,
            "reset":reset,"capacity":capacity,"metric_context_restored_normal_and_exception":True}


def preflight(out,args):
    from transformers import AutoProcessor
    from scripts.baseline import check_tiny_video_adapter
    assert json.loads((args.model_dir/"DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8-sig"))["revision"] == REVISION
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    tiny,position_model = check_tiny_video_adapter()
    prepared,groups = base.inputs_and_groups(args,processor,position_model)
    q,m = suffixes(processor)
    fingerprints = {k:tensor_fingerprint(v) for k,v in prepared.items()}
    old = json.loads((PROJECT/"runs/s4_preflight_20261009_01/preflight_summary.json").read_text())
    assert fingerprints == old["input_fingerprints"] and groups == old["groups"]
    assert file_hash(args.video) == old["video_sha256"]
    controls = cpu_controls(prepared,groups)
    write_json(out/"preflight_summary.json",{"cpu_controls":controls,"groups":groups,"budget":BUDGET,
        "extra_groups":["fork_isolation_with_main_continuation","reset","capacity_rejection"],
        "tiny_position_adapter":tiny,"input_fingerprints":fingerprints,"question_suffix":q.tolist(),"main_suffix":m.tolist(),
        "input_tokens":{k:v.input_ids.shape[1] for k,v in prepared.items()},"video_sha256":file_hash(args.video),
        "model_manifest_sha256":file_hash(args.model_dir/"DOWNLOAD_MANIFEST.json"),
        "plan_sha256":file_hash(PLAN),"launcher_sha256":file_hash(LAUNCHER)})
    print({"cpu_controls_passed":True,"input_tokens":{k:v.input_ids.shape[1] for k,v in prepared.items()},"budget":BUDGET},flush=True)


@torch.inference_mode()
def gpu(out,args):
    from transformers import AutoProcessor
    from qwen35_local import Qwen3_5ForConditionalGeneration
    torch.manual_seed(42)
    begin = time.perf_counter()
    free,total = torch.cuda.mem_get_info()
    guard = {"gpu_free_bytes":free,"gpu_total_bytes":total,"ram_available_bytes":psutil.virtual_memory().available}
    write_json(out/"resource_precheck.json",guard)
    assert free >= 5.5*2**30 and guard["ram_available_bytes"] >= 3*2**30,"Memory guard; no loading"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation="sdpa").to("cuda").eval()
    assert Path(inspect.getfile(type(model))).resolve().parent == REPO/"qwen35_local"
    assert all(not layer.linear_attn.experimental_recurrent_reference for layer in model.model.language_model.layers
               if layer.layer_type == "linear_attention")
    torch.cuda.reset_peak_memory_stats()
    assert torch.cuda.mem_get_info()[0] >= 2*2**30,"Conversion memory guard"
    model.float()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    prepared,groups = base.inputs_and_groups(args,processor,model)
    fingerprints = {k:tensor_fingerprint(v) for k,v in prepared.items()}
    proof = json.loads((args.preflight/"preflight_summary.json").read_text())
    assert fingerprints == proof["input_fingerprints"] and groups == proof["groups"]
    q,m = suffixes(processor)
    assert q.tolist() == proof["question_suffix"] and m.tolist() == proof["main_suffix"]
    q,m = q.to("cuda"),m.to("cuda")
    prepared = {k:v.to("cuda") for k,v in prepared.items()}
    initial = settings()
    report = {"groups":[],"budget":BUDGET,"model_revision":REVISION,"input_fingerprints":fingerprints,
        "scope":"fixed FP32 pre-encoded video/text sequence and mathematical suffix branches, not online frames/QA",
        "vision_dtype":str(model.model.visual.dtype),"language_dtype":str(model.get_input_embeddings().weight.dtype),
        "initial_backend_settings":initial,"gpu":torch.cuda.get_device_name(),"resource_precheck":guard}
    with precision("fp32_ieee_math") as active,strict_metric_scope():
        report["active_backend_settings"] = active
        embeddings = {k:base.encoded(model,v) for k,v in prepared.items()}
        report["embedding_shapes"] = {k:list(v.shape) for k,v in embeddings.items()}
        for group in groups:
            key = group["input"]
            record = {**group,**base.compare(model,embeddings[key],prepared[key].position_ids,group["chunks"])}
            report["groups"].append(record)
            write_json(out/"fp32_regression_summary.json",report)
            print({"id":record["id"],"passed":passed(record),"logits":record["logits"]},flush=True)
        main,fork = fork_and_continue(model,embeddings["video"],prepared["video"].position_ids,q,m)
        report["groups"].append(fork)
        report["groups"] += base.reset_and_capacity(main,model,prepared["short_text"])
    assert len(report["groups"]) == 10 and settings() == initial and base.logits_metrics is ORIGINAL_LOGITS
    report["numeric_groups_passed"] = all(passed(g) for g in report["groups"] if "logits" in g) and passed(fork["main_continuation"])
    report["state_isolation_reset_capacity_passed"] = True
    report.update(backend_and_metrics_restored=True,wall_seconds_including_load=time.perf_counter()-begin,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    write_json(out/"fp32_regression_summary.json",report)
    print({k:report[k] for k in ("numeric_groups_passed","state_isolation_reset_capacity_passed","wall_seconds_including_load","peak_allocated_gib","peak_reserved_gib")},flush=True)
    if not report["numeric_groups_passed"]:
        raise RuntimeError("FP32 registered regression limits failed; preserve evidence")


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
    write_json(args.out/"script_snapshot.py.json",{"sha256":file_hash(Path(__file__))})
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
