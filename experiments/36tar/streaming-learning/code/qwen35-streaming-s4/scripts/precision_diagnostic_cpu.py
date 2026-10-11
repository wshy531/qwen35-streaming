"""Post-failure CPU localization; one original-weight layer, not GPU validation."""
import argparse
import json
from pathlib import Path
import time
import traceback

import psutil
from safetensors import safe_open
import torch
from transformers import AutoProcessor

from qwen35_local import Qwen3_5Config
from qwen35_local.modeling_qwen3_5 import (Qwen3_5DynamicCache,Qwen3_5GatedDeltaNet,
    Qwen3_5RMSNorm,l2norm,torch_chunk_gated_delta_rule)
from scripts.baseline import PROJECT,file_hash,provenance,write_json


def difference(a,b):
    d = (a.float()-b.float()).abs()
    return {"max_abs_diff":d.max().item(),"mean_abs_diff":d.mean().item(),
            "bitwise_equal":torch.equal(a,b)}


@torch.inference_mode()
def kernel(dtype):
    torch.manual_seed(29)
    q,k,v = [torch.randn(1,20,2,16).to(dtype) for _ in range(3)]
    g = -torch.rand(1,20,2)*.1
    beta = torch.rand(1,20,2).to(dtype)
    kwargs = dict(output_final_state=True,use_qk_l2norm_in_kernel=True)
    full,state = torch_chunk_gated_delta_rule(q,k,v,g,beta,**kwargs)
    first,s = torch_chunk_gated_delta_rule(q[:,:2],k[:,:2],v[:,:2],g[:,:2],beta[:,:2],**kwargs)
    last,s = torch_chunk_gated_delta_rule(q[:,2:],k[:,2:],v[:,2:],g[:,2:],beta[:,2:],initial_state=s,**kwargs)
    report = {"output":difference(torch.cat((first,last),1),full),"state":difference(s,state)}
    report["normalization_shape_probe"] = {name:difference(
        torch.cat((l2norm(x[:,:2]),l2norm(x[:,2:])),1),l2norm(x)) for name,x in (("query",q),("key",k))}
    # CPU diagnostic only. Do not replace the registered model path or GPU result.
    q,k,v,g,beta = l2norm(q.float()),l2norm(k.float()),v.float(),g.float(),beta.float()
    kwargs["use_qk_l2norm_in_kernel"] = False
    full,state = torch_chunk_gated_delta_rule(q,k,v,g,beta,**kwargs)
    first,s = torch_chunk_gated_delta_rule(q[:,:2],k[:,:2],v[:,:2],g[:,:2],beta[:,:2],**kwargs)
    last,s = torch_chunk_gated_delta_rule(q[:,2:],k[:,2:],v[:,2:],g[:,2:],beta[:,2:],initial_state=s,**kwargs)
    report["pre_normalized_fp32_control"] = {"output":difference(torch.cat((first,last),1),full),"state":difference(s,state)}
    return report


@torch.inference_mode()
def layer_check(config,weights,embeddings,norm_weight,dtype):
    layer = Qwen3_5GatedDeltaNet(config,0).to(dtype).eval()
    layer.load_state_dict(weights,strict=True)
    norm = Qwen3_5RMSNorm(config.hidden_size,config.rms_norm_eps).to(dtype).eval()
    norm.load_state_dict({"weight":norm_weight})
    hidden = norm(embeddings.to(dtype))
    record = {}
    hooks = [getattr(layer,name).register_forward_hook(
        lambda _m,_i,o,name=name:record.setdefault(name,[]).append(o.clone()))
        for name in ("in_proj_qkv","in_proj_a","in_proj_b")]
    original = layer.chunk_gated_delta_rule
    def traced(q,k,v,**kwargs):
        for name,value in (("query",q),("key",k),("value",v),("g",kwargs["g"]),("beta",kwargs["beta"])):
            record.setdefault(name,[]).append(value.clone())
        result = original(q,k,v,**kwargs)
        record.setdefault("core",[]).append(result[0].clone())
        return result
    layer.chunk_gated_delta_rule = traced
    def run(layout):
        record.clear()
        cache = Qwen3_5DynamicCache(config)
        # Isolated layer only: the cache readiness sentinel must refer to layer 0.
        cache.last_linear_layer = 0
        pieces,start = [],0
        for length in layout:
            pieces.append(layer(hidden[:,start:start+length],cache,
                torch.arange(start,start+length),torch.ones(1,start+length,dtype=torch.long)))
            start += length
        values = {name:torch.cat(parts,dim=1) for name,parts in record.items()}
        values["output"] = torch.cat(pieces,dim=1)
        values["conv_state"] = cache.conv_states[0].clone()
        values["recurrent_state"] = cache.recurrent_states[0].clone()
        return values
    try:
        full,split = run([20]),run([2,18])
        return {name:difference(split[name],value) for name,value in full.items()}
    finally:
        for hook in hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--model-dir",type=Path,default=PROJECT/"models/Qwen3.5-0.8B")
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    write_json(args.out/"provenance.json",{**provenance(),"diagnostic_script_sha256":file_hash(Path(__file__))})
    try:
        torch.set_num_threads(2)
        started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
        template = processor.apply_chat_template([{"role":"user","content":[{"type":"text",
            "text":"Explain what a video frame is."}]}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        tokens = processor.tokenizer([template],return_tensors="pt").input_ids
        assert tokens.shape == (1,20)
        config = Qwen3_5Config.from_pretrained(args.model_dir,local_files_only=True).text_config
        prefix = "model.language_model.layers.0.linear_attn."
        index = json.loads((args.model_dir/"model.safetensors.index.json").read_text())
        shard = args.model_dir/index["weight_map"][prefix+"in_proj_qkv.weight"]
        with safe_open(shard,framework="pt",device="cpu") as tensors:
            weights = {key[len(prefix):]:tensors.get_tensor(key) for key in tensors.keys() if key.startswith(prefix)}
            norm_weight = tensors.get_tensor("model.language_model.layers.0.input_layernorm.weight")
            embedding_slice = tensors.get_slice("model.language_model.embed_tokens.weight")
            embeddings = torch.stack([embedding_slice[int(t):int(t)+1,:][0] for t in tokens[0]])[None]
        report = {"scope":"post-failure CPU only; original first linear-attention layer, no full model/GPU/answers",
            "layout":[2,18],"input_token_ids":tokens.tolist(),
            "loaded_weight_bytes":sum(t.numel()*t.element_size() for t in weights.values()),
            "random_kernel":{},"official_first_layer":{},"model_manifest_sha256":file_hash(args.model_dir/"DOWNLOAD_MANIFEST.json")}
        for dtype in (torch.float32,torch.bfloat16):
            report["random_kernel"][str(dtype)] = kernel(dtype)
            report["official_first_layer"][str(dtype)] = layer_check(config,weights,embeddings,norm_weight,dtype)
        report["wall_seconds"] = time.perf_counter()-started
        report["process_rss_bytes_at_end"] = psutil.Process().memory_info().rss
        write_json(args.out/"precision_summary.json",report)
        write_json(args.out/"status.json",{"status":"completed","stage":"cpu_precision_diagnostic"})
        print(json.dumps(report),flush=True)
    except Exception:
        (args.out/"traceback.txt").write_text(traceback.format_exc(),encoding="utf-8")
        write_json(args.out/"status.json",{"status":"failed","stage":"cpu_precision_diagnostic"})
        raise


if __name__ == "__main__":
    main()
