"""Isolated original-weight first layer: chunk algebra versus token recurrence."""
import argparse
import json
from pathlib import Path
import time
import traceback

from safetensors import safe_open
import torch
from transformers import AutoProcessor

import qwen35_local.modeling_qwen3_5 as local
from scripts.baseline import PROJECT,provenance,write_json,file_hash
from scripts.precision_layers import text_inputs,tensor_metric


@torch.inference_mode()
def compare(config,weights,embeddings,norm_weight,dtype,rule):
    layer = local.Qwen3_5GatedDeltaNet(config,0).to(dtype).eval()
    layer.load_state_dict(weights)
    norm = local.Qwen3_5RMSNorm(config.hidden_size,config.rms_norm_eps).to(dtype).eval()
    norm.load_state_dict({"weight":norm_weight})
    hidden = norm(embeddings.to(dtype))
    record,hooks = {},[]
    def add(name,x):
        record.setdefault(name,[]).append(x.clone())
    def hook(name):
        def collect(_m,_i,o):
            add(name,o)
        return collect
    for name in ("in_proj_qkv","in_proj_a","in_proj_b","out_proj"):
        hooks.append(getattr(layer,name).register_forward_hook(hook(name)))
    def gated_norm_hook(_m,inputs,output):
        add("gated_norm_input",inputs[0].reshape(1,-1,layer.value_dim))
        add("gate",inputs[1].reshape(1,-1,layer.value_dim))
        add("gated_norm_output",output.reshape(1,-1,layer.value_dim))
    hooks.append(layer.norm.register_forward_hook(gated_norm_hook))
    function = local.torch_chunk_gated_delta_rule if rule == "chunk" else local.torch_recurrent_gated_delta_rule
    def traced(q,k,v,**kwargs):
        result = function(q,k,v,**kwargs)
        add("core",result[0].reshape(1,-1,layer.value_dim))
        return result
    layer.chunk_gated_delta_rule = traced
    def run(layout):
        record.clear()
        cache = local.Qwen3_5DynamicCache(config)
        cache.last_linear_layer = 0
        parts,start = [],0
        for length in layout:
            parts.append(layer(hidden[:,start:start+length],cache,torch.arange(start,start+length)))
            start += length
        result = {name:torch.cat(values,1) for name,values in record.items()}
        result["output"] = torch.cat(parts,1)
        result["conv_state"] = cache.conv_states[0].clone()
        result["recurrent_state"] = cache.recurrent_states[0].clone()
        return result
    try:
        full,split = run([20]),run([2,18])
        metrics = {name:tensor_metric(split[name],value) for name,value in full.items()}
        return {"dtype":str(dtype),"rule":rule,"comparisons":metrics},full["output"]
    finally:
        for handle in hooks:
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--model-dir",type=Path,default=PROJECT/"models/Qwen3.5-0.8B")
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    write_json(args.out/"provenance.json",provenance())
    (args.out/"script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    try:
        torch.set_num_threads(2)
        started = time.perf_counter()
        config = local.Qwen3_5Config.from_pretrained(args.model_dir,local_files_only=True).text_config
        processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
        tokens = text_inputs(processor)["short_text"].input_ids
        prefix = "model.language_model.layers.0.linear_attn."
        index = json.loads((args.model_dir/"model.safetensors.index.json").read_text())
        with safe_open(args.model_dir/index["weight_map"][prefix+"in_proj_qkv.weight"],framework="pt",device="cpu") as tensors:
            weights = {name[len(prefix):]:tensors.get_tensor(name) for name in tensors.keys() if name.startswith(prefix)}
            norm_weight = tensors.get_tensor("model.language_model.layers.0.input_layernorm.weight")
            embedding_slice = tensors.get_slice("model.language_model.embed_tokens.weight")
            embeddings = torch.stack([embedding_slice[int(t):int(t)+1,:][0] for t in tokens[0]])[None]
        report = {"scope":"CPU isolated first linear layer only; no full-model or GPU success claim",
            "model_manifest_sha256":file_hash(args.model_dir/"DOWNLOAD_MANIFEST.json"),"groups":[]}
        for dtype in (torch.float32,torch.bfloat16):
            references = {}
            for rule in ("chunk","recurrent"):
                result,full = compare(config,weights,embeddings,norm_weight,dtype,rule)
                report["groups"].append(result)
                references[rule] = full
            report.setdefault("whole_layer_rule_shift",{})[str(dtype)] = tensor_metric(references["recurrent"],references["chunk"])
        report["wall_seconds"] = time.perf_counter()-started
        write_json(args.out/"recurrence_summary.json",report)
        write_json(args.out/"status.json",{"status":"completed","stage":"cpu_recurrence_diagnostic"})
        print(json.dumps(report),flush=True)
    except Exception:
        (args.out/"traceback.txt").write_text(traceback.format_exc(),encoding="utf-8")
        write_json(args.out/"status.json",{"status":"failed","stage":"cpu_recurrence_diagnostic"})
        raise


if __name__ == "__main__":
    main()
