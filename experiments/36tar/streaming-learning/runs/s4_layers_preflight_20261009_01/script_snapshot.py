"""Four fixed GPU controls, traced by layer; experimental norm override only."""
import argparse
from contextlib import contextmanager
import inspect
from pathlib import Path
import tempfile
import time
import traceback

import torch
import qwen35_local.modeling_qwen3_5 as local
from scripts.baseline import PROJECT,REPO,REVISION,provenance,write_json,file_hash
from scripts.chunk_weight_check import logits_metrics,state_metrics
from streaming.hybrid import HybridSequence
from streaming.inputs import tensor_fingerprint

GROUPS = [("short_text",[2,18]),("long_text",[63,1,34])]


@contextmanager
def normalization(mode,observer=None):
    original = local.l2norm
    def controlled(x,dim=-1,eps=1e-6):
        result = original(x.float() if mode == "norm_fp32" else x,dim=dim,eps=eps)
        if observer is not None:
            observer(result)
        return result
    local.l2norm = controlled
    try:
        yield
    finally:
        local.l2norm = original


def text_inputs(processor):
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
    assert [result[name].input_ids.shape[1] for name,_ in GROUPS] == [20,98]
    return result


@torch.inference_mode()
def cpu_controls():
    from tests.test_local_model import tiny_config
    torch.set_num_threads(1)
    original = local.l2norm
    torch.manual_seed(29)
    q,k,v = [torch.randn(1,20,2,16).bfloat16() for _ in range(3)]
    g,beta = -torch.rand(1,20,2)*.1,torch.rand(1,20,2).bfloat16()
    with normalization("norm_fp32"):
        normalized = local.l2norm(q)
        assert normalized.dtype == torch.float32
        assert torch.equal(normalized,torch.cat((local.l2norm(q[:,:2]),local.l2norm(q[:,2:])),1))
        kwargs = dict(output_final_state=True,use_qk_l2norm_in_kernel=True)
        full,state = local.torch_chunk_gated_delta_rule(q,k,v,g,beta,**kwargs)
        a,s = local.torch_chunk_gated_delta_rule(q[:,:2],k[:,:2],v[:,:2],g[:,:2],beta[:,:2],**kwargs)
        b,s = local.torch_chunk_gated_delta_rule(q[:,2:],k[:,2:],v[:,2:],g[:,2:],beta[:,2:],initial_state=s,**kwargs)
        error = (s-state).abs().max().item()
        assert error < 1e-5
        assert torch.allclose(torch.cat((a,b),1),full,atol=.002,rtol=.01)
    assert local.l2norm is original
    try:
        with normalization("norm_fp32"):
            raise ValueError("intentional restoration control")
    except ValueError:
        pass
    assert local.l2norm is original
    torch.manual_seed(42)
    model = local.Qwen3_5ForConditionalGeneration(local.Qwen3_5Config(**tiny_config(),attn_implementation="eager")).eval()
    tokens = torch.tensor([[1,2,3,4,5,6]])
    baseline = model(input_ids=tokens,use_cache=True)
    with normalization("norm_fp32"):
        reference = model(input_ids=tokens,use_cache=True)
        assert torch.equal(reference.logits,baseline.logits)
        sequence = HybridSequence(model)
        a,b = sequence.append(input_ids=tokens[:,:2]),sequence.append(input_ids=tokens[:,2:])
        torch.testing.assert_close(torch.cat((a,b),1),reference.logits,atol=1e-5,rtol=1e-4)
        from tests.test_mixed_state import assert_states
        import unittest
        assert_states(unittest.TestCase(),sequence.cache,reference.past_key_values)
    assert local.l2norm is original
    return {"passed":True,"bf16_norm_outputs_fp32_and_shape_invariant":True,"random_state_max_abs_diff":error,
        "override_restored_normal_and_exception":True,"tiny_fp32_prefill_unchanged_and_chunks_states_equivalent":True}


class Trace:
    def __init__(self,model):
        self.parts,self.hooks,self.kernels = {},[],[]
        self.active_layer,self.norm_call = None,0
        for i,layer in enumerate(model.model.language_model.layers):
            self.hooks.append(layer.register_forward_hook(self.hook(f"layer.{i}.output")))
            if layer.layer_type == "linear_attention":
                mixer = layer.linear_attn
                self.hooks.append(mixer.register_forward_hook(self.hook(f"linear.{i}.output")))
                for name in ("in_proj_qkv","in_proj_a","in_proj_b"):
                    self.hooks.append(getattr(mixer,name).register_forward_hook(self.hook(f"linear.{i}.{name}")))
                for name in ("chunk_gated_delta_rule","recurrent_gated_delta_rule"):
                    original = getattr(mixer,name)
                    assert original in (local.torch_chunk_gated_delta_rule,local.torch_recurrent_gated_delta_rule)
                    self.kernels.append((mixer,name,original))
                    setattr(mixer,name,self.wrap(original,i))
    def add(self,name,tensor):
        self.parts.setdefault(name,[]).append(tensor.detach().to("cpu",copy=True))
    def hook(self,name):
        def capture(_m,_args,output):
            self.add(name,output)
        return capture
    def wrap(self,function,index):
        def wrapped(q,k,v,**kwargs):
            self.active_layer,self.norm_call = index,0
            for name,tensor in (("query",q),("key",k),("value",v),("g",kwargs["g"]),("beta",kwargs["beta"])):
                self.add(f"linear.{index}.{name}",tensor)
            output,state = function(q,k,v,**kwargs)
            self.add(f"linear.{index}.core",output)
            self.active_layer = None
            return output,state
        return wrapped
    def norm_observer(self,result):
        assert self.active_layer is not None
        self.add(f"linear.{self.active_layer}.{'query_norm' if self.norm_call == 0 else 'key_norm'}",result)
        self.norm_call += 1
    def take(self):
        result = {name:torch.cat(parts,dim=1) for name,parts in self.parts.items()}
        self.parts.clear()
        return result
    def close(self):
        for hook in self.hooks:
            hook.remove()
        for module,name,function in self.kernels:
            setattr(module,name,function)


def tensor_metric(a,b):
    delta = (a.float()-b.float()).abs()
    return {"max_abs_diff":delta.max().item(),"mean_abs_diff":delta.mean().item(),
        "different_elements":int(torch.count_nonzero(delta)),"elements":a.numel(),"bitwise_equal":torch.equal(a,b)}


@torch.inference_mode()
def comparison(model,inputs,layout,mode,baseline_logits):
    trace = Trace(model)
    try:
        with normalization(mode,trace.norm_observer):
            full = model(input_ids=inputs.input_ids,position_ids=inputs.position_ids,use_cache=True)
            full_trace = trace.take()
            sequence,parts,start = HybridSequence(model),[],0
            for length in layout:
                parts.append(sequence.append(input_ids=inputs.input_ids[:,start:start+length],
                    position_ids=inputs.position_ids[:,:,start:start+length]))
                start += length
            split_trace = trace.take()
        assert full_trace.keys() == split_trace.keys()
        metrics = {name:tensor_metric(split_trace[name],value) for name,value in full_trace.items()}
        layer_metrics = [metrics[f"layer.{i}.output"] for i in range(24)]
        cpu_logits = full.logits.detach().cpu()
        record = {"mode":mode,"chunks":layout,"logits":logits_metrics(torch.cat(parts,1),full.logits),
            "final_states":state_metrics(sequence.cache,full.past_key_values),"traced_tensors":metrics,
            "first_changed_layer_output":next((i for i,m in enumerate(layer_metrics) if not m["bitwise_equal"]),None),
            "first_linear_layer":{k:v for k,v in metrics.items() if k.startswith("linear.0.")}}
        if baseline_logits is not None:
            record["whole_prefill_shift_from_original_bf16"] = logits_metrics(cpu_logits,baseline_logits)
        return record,cpu_logits
    finally:
        trace.close()


def preflight(out,args):
    import json
    from transformers import AutoProcessor
    assert json.loads((args.model_dir/"DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8-sig"))["revision"] == REVISION
    controls = cpu_controls()
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    prepared = text_inputs(processor)
    report = {"cpu_controls":controls,"input_fingerprints":{k:tensor_fingerprint(v) for k,v in prepared.items()},
        "groups":[{"mode":mode,"input":name,"chunks":layout} for mode in ("original_bf16_norm","norm_fp32") for name,layout in GROUPS],
        "model_manifest_sha256":file_hash(args.model_dir/"DOWNLOAD_MANIFEST.json"),"model_revision":REVISION,
        "budget":{"groups":4,"max_tokens":256,"generated_tokens":0,"wall_seconds":600,"model_loads":1},
        "scope":"CPU controls and input preflight; GPU effect not yet established"}
    write_json(out/"preflight_summary.json",report)
    print(report,flush=True)


def gpu(out,args):
    from transformers import AutoProcessor
    torch.manual_seed(42)
    start = time.perf_counter()
    model = local.Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation="sdpa").to("cuda").eval()
    assert Path(inspect.getfile(type(model))).resolve().parent == REPO/"qwen35_local"
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    inputs = text_inputs(processor)
    report = {"groups":[],"input_fingerprints":{k:tensor_fingerprint(v) for k,v in inputs.items()},
        "model_revision":REVISION,"generated_tokens":0,"scope":"four diagnostic comparisons only; not full S4 regression"}
    inputs = {k:v.to("cuda") for k,v in inputs.items()}
    original_function,baselines = local.l2norm,{}
    torch.cuda.reset_peak_memory_stats()
    for mode in ("original_bf16_norm","norm_fp32"):
        for name,layout in GROUPS:
            record,reference = comparison(model,inputs[name],layout,mode,baselines.get(name))
            assert local.l2norm is original_function
            if mode == "original_bf16_norm":
                baselines[name] = reference
            record["input"] = name
            report["groups"].append(record)
            write_json(out/"precision_layers_summary.json",report)
            print({"input":name,"mode":mode,"logits":record["logits"],"first_changed_layer":record["first_changed_layer_output"]},flush=True)
    report["norm_override_restored"] = local.l2norm is original_function
    report["peak_allocated_gib"] = torch.cuda.max_memory_allocated()/2**30
    report["peak_reserved_gib"] = torch.cuda.max_memory_reserved()/2**30
    report["wall_seconds_including_load"] = time.perf_counter()-start
    report["candidate_two_groups_passed"] = all(g["logits"]["registered_numeric_limits_passed"] for g in report["groups"][2:])
    write_json(out/"precision_layers_summary.json",report)
    # Diagnostic completion means evidence recorded, not numerical success.
    print({k:report[k] for k in ("candidate_two_groups_passed","wall_seconds_including_load")},flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("preflight","gpu"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--model-dir",type=Path,default=PROJECT/"models/Qwen3.5-0.8B")
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"tmp").mkdir()
    tempfile.tempdir = str((args.out/"tmp").resolve())
    write_json(args.out/"provenance.json",provenance())
    write_json(args.out/"script_snapshot.py.json",{"sha256":file_hash(Path(__file__))})
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
