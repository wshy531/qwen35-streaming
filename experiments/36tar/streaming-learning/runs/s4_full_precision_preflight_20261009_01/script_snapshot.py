"""Four preregistered precision controls, default model path, no generated answers."""
import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
import inspect
import json
from pathlib import Path
import tempfile
import time
import traceback

import psutil
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
import qwen35_local.modeling_qwen3_5 as local
from scripts.baseline import PROJECT, REPO, REVISION, provenance, write_json, file_hash
from scripts.chunk_weight_check import logits_metrics, state_metrics, FIELDS
from scripts.precision_layers import Trace, normalization, text_inputs, tensor_metric
from streaming.hybrid import HybridSequence
from streaming.inputs import tensor_fingerprint

BUDGET = {"groups":4,"max_tokens":98,"generated_tokens":0,"wall_seconds":600,
          "model_loads":1,"full_model_forwards":13,"component_replay_calls":26}
GROUPS = [
    {"mode":"original_bf16","input":"short_text","chunks":[2,18]},
    {"mode":"bf16_no_reduced_accum","input":"short_text","chunks":[2,18]},
    {"mode":"fp32_ieee_math","input":"short_text","chunks":[2,18]},
    {"mode":"fp32_ieee_math","input":"long_text","chunks":[63,1,34]},
]
PLAN = PROJECT/"高精度整模型诊断方案.md"
LAUNCHER = PROJECT/"scripts/run_precision_fullmodel.py"


def settings():
    return {"matmul_fp32":torch.backends.cuda.matmul.fp32_precision,
            "conv_fp32":torch.backends.cudnn.conv.fp32_precision,
            "bf16_reduction":torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            "flash_sdp":torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient_sdp":torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math_sdp":torch.backends.cuda.math_sdp_enabled()}


@contextmanager
def precision(mode):
    assert mode in {g["mode"] for g in GROUPS}
    old = settings()
    try:
        if mode == "bf16_no_reduced_accum":
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        if mode == "fp32_ieee_math":
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
        with (sdpa_kernel(SDPBackend.MATH) if mode == "fp32_ieee_math" else nullcontext()):
            yield settings()
    finally:
        torch.backends.cuda.matmul.fp32_precision = old["matmul_fp32"]
        torch.backends.cudnn.conv.fp32_precision = old["conv_fp32"]
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old["bf16_reduction"]
        assert settings() == old


class ExtendedTrace(Trace):
    def __init__(self,model):
        super().__init__(model)
        mixer = model.model.language_model.layers[0].linear_attn
        self.value_dim = mixer.value_dim
        self.hooks.append(mixer.in_proj_z.register_forward_hook(self.hook("linear.0.in_proj_z")))
        self.hooks.append(mixer.norm.register_forward_pre_hook(self.norm_before))
        self.hooks.append(mixer.norm.register_forward_hook(self.norm_after))
        self.hooks.append(mixer.out_proj.register_forward_hook(self.hook("linear.0.out_proj")))
    def norm_before(self,_module,args):
        for name,tensor in zip(("norm_input","norm_gate"),args):
            self.add("linear.0."+name,tensor.reshape(1,-1,self.value_dim))
    def norm_after(self,_module,_args,output):
        self.add("linear.0.norm_output",output.reshape(1,-1,self.value_dim))


@torch.inference_mode()
def replay(model,captured,layout):
    mixer = model.model.language_model.layers[0].linear_attn
    device = mixer.out_proj.weight.device
    core,gate,normed = [captured["linear.0."+name].to(device) for name in ("norm_input","norm_gate","norm_output")]
    def norm(x,z):
        return mixer.norm(x.reshape(-1,mixer.head_v_dim),z.reshape(-1,mixer.head_v_dim)).reshape(1,-1,mixer.value_dim)
    reference = norm(core,gate)
    chunks,projection_chunks,start = [],[],0
    for size in layout:
        chunks.append(norm(core[:,start:start+size],gate[:,start:start+size]))
        projection_chunks.append(mixer.out_proj(normed[:,start:start+size]))
        start += size
    assert start == core.shape[1]
    return {"identical_input_gated_norm":tensor_metric(torch.cat(chunks,1),reference),
            "identical_input_output_projection":tensor_metric(torch.cat(projection_chunks,1),mixer.out_proj(normed)),
            "calls":2*(1+len(layout))}


@torch.inference_mode()
def comparison(model,inputs,layout):
    trace = ExtendedTrace(model)
    try:
        with normalization("original_bf16_norm",trace.norm_observer):
            full = model(input_ids=inputs.input_ids,position_ids=inputs.position_ids,use_cache=True)
            captured = trace.take()
            sequence,parts,start = HybridSequence(model),[],0
            for size in layout:
                parts.append(sequence.append(input_ids=inputs.input_ids[:,start:start+size],
                    position_ids=inputs.position_ids[:,:,start:start+size]))
                start += size
            actual = torch.cat(parts,1)
            split = trace.take()
        assert captured.keys() == split.keys()
        delta = actual.float()-full.logits.float()
        stats = logits_metrics(actual,full.logits)
        stats["allclose_1e4"] = torch.allclose(actual.float(),full.logits.float(),atol=1e-4,rtol=1e-4)
        stats["fp32_reference_limits_passed"] = stats["max_abs_diff"] <= .01 and stats["mean_abs_diff"] <= .0001 and stats["rms_diff"] <= .0003
        states = state_metrics(sequence.cache,full.past_key_values)
        for field in FIELDS:
            states[field]["allclose_1e4"] = all(torch.allclose(a.float(),b.float(),atol=1e-4,rtol=1e-4)
                for a,b in zip(getattr(sequence.cache,field),getattr(full.past_key_values,field)) if a is not None)
        metric = {name:tensor_metric(split[name],value) for name,value in captured.items()}
        record = {"logits":stats,"final_states":states,"cache_length":sequence.cache.get_seq_length(),
            "traced_tensors":metric,"same_input_component_replay":replay(model,captured,layout),
            "first_changed_layer_output":next((i for i in range(len(model.model.language_model.layers))
                if not metric[f"layer.{i}.output"]["bitwise_equal"]),None)}
        assert record["cache_length"] == inputs.input_ids.shape[1]
        return record,full.logits.detach().cpu()
    finally:
        trace.close()


@torch.inference_mode()
def cpu_controls():
    from tests.test_local_model import tiny_config
    torch.set_num_threads(1)
    original = settings()
    for mode in {g["mode"] for g in GROUPS}:
        with precision(mode):
            pass
        assert settings() == original
        try:
            with precision(mode):
                raise ValueError("intentional restoration control")
        except ValueError:
            pass
        assert settings() == original
    torch.manual_seed(42)
    model = local.Qwen3_5ForConditionalGeneration(local.Qwen3_5Config(**tiny_config(),attn_implementation="sdpa")).eval()
    assert not model.model.language_model.layers[0].linear_attn.experimental_recurrent_reference
    metrics = []
    for layout in ([2,18],[63,1,34]):
        tokens = (torch.arange(sum(layout)) % 10+1)[None,:]
        from transformers.feature_extraction_utils import BatchFeature
        inputs = BatchFeature({"input_ids":tokens,"position_ids":torch.arange(sum(layout))[None,None,:].expand(4,1,-1).clone()})
        with precision("fp32_ieee_math"):
            record,_ = comparison(model,inputs,layout)
        assert record["logits"]["allclose_1e4"]
        assert all(x["allclose_1e4"] for x in record["final_states"].values())
        metrics.append({"chunks":layout,"logits":record["logits"],"component_replay":record["same_input_component_replay"]})
    return {"passed":True,"backend_context_restores_normally_and_on_exception":True,"tiny_fp32_groups":metrics}


def preflight(out,args):
    from transformers import AutoProcessor
    assert json.loads((args.model_dir/"DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8-sig"))["revision"] == REVISION
    controls = cpu_controls()
    prepared = text_inputs(AutoProcessor.from_pretrained(args.model_dir,local_files_only=True))
    fingerprints = {k:tensor_fingerprint(v) for k,v in prepared.items()}
    assert fingerprints == json.loads((PROJECT/"runs/s4_layers_preflight_20261009_02/preflight_summary.json").read_text())["input_fingerprints"]
    write_json(out/"preflight_summary.json",{"cpu_controls":controls,"groups":GROUPS,"budget":BUDGET,
        "input_fingerprints":fingerprints,"model_manifest_sha256":file_hash(args.model_dir/"DOWNLOAD_MANIFEST.json"),
        "plan_sha256":file_hash(PLAN),"launcher_sha256":file_hash(LAUNCHER),"initial_backend_settings":settings()})
    print({"cpu_controls_passed":True,"budget":BUDGET},flush=True)


def gpu(out,args):
    from transformers import AutoProcessor
    torch.manual_seed(42)
    begin = time.perf_counter()
    free,total = torch.cuda.mem_get_info()
    ram = psutil.virtual_memory().available
    guard = {"gpu_free_bytes":free,"gpu_total_bytes":total,"ram_available_bytes":ram,
             "min_gpu_free_bytes":int(5.5*2**30),"min_ram_available_bytes":3*2**30}
    write_json(out/"resource_precheck.json",guard)
    assert free >= guard["min_gpu_free_bytes"] and ram >= guard["min_ram_available_bytes"],"Memory guard; no weight loading permitted"
    model = local.Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation="sdpa").to("cuda").eval()
    assert Path(inspect.getfile(type(model))).resolve().parent == REPO/"qwen35_local"
    assert len(model.model.language_model.layers) == 24
    assert all(not layer.linear_attn.experimental_recurrent_reference for layer in model.model.language_model.layers if layer.layer_type == "linear_attention")
    torch.cuda.reset_peak_memory_stats()
    inputs = text_inputs(AutoProcessor.from_pretrained(args.model_dir,local_files_only=True))
    fingerprints = {k:tensor_fingerprint(v) for k,v in inputs.items()}
    assert fingerprints == json.loads((args.preflight/"preflight_summary.json").read_text())["input_fingerprints"]
    inputs = {k:v.to("cuda") for k,v in inputs.items()}
    initial = settings()
    report = {"groups":[],"input_fingerprints":fingerprints,"budget":BUDGET,"model_revision":REVISION,
        "initial_backend_settings":initial,"gpu":torch.cuda.get_device_name(),"language_layers":24,
        "scope":"full language model, fixed text only; no vision/answers; BF16 checkpoints cast to FP32",
        "parameter_count":sum(p.numel() for p in model.parameters()),"resource_precheck":guard}
    reference = None
    converted = False
    for group in GROUPS:
        if group["mode"] == "fp32_ieee_math" and not converted:
            torch.cuda.empty_cache()
            conversion_free,_ = torch.cuda.mem_get_info()
            # Largest temporary cast is tied embedding/output weight (~0.95GiB FP32).
            assert conversion_free >= 2*2**30,"Insufficient GPU conversion workspace"
            report["gpu_free_before_conversion_bytes"] = conversion_free
            model.float()
            assert all(p.dtype == torch.float32 for p in model.parameters())
            converted = True
        with precision(group["mode"]) as active:
            record,whole = comparison(model,inputs[group["input"]],group["chunks"])
            record.update(group)
            record["active_backend_settings"] = active
            record["parameter_dtype_counts"] = dict(Counter(str(p.dtype) for p in model.parameters()))
            if group["mode"] == "original_bf16":
                reference = whole
            elif group["input"] == "short_text":
                record["whole_prefill_shift_from_original_bf16"] = logits_metrics(whole,reference)
        assert settings() == initial
        report["groups"].append(record)
        write_json(out/"full_precision_summary.json",report)
        print({"mode":group["mode"],"input":group["input"],"logits":record["logits"],
               "component_replay":record["same_input_component_replay"]},flush=True)
    report.update(backend_restored=True,wall_seconds_including_load=time.perf_counter()-begin,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
        fp32_two_groups_passed=all(g["logits"]["fp32_reference_limits_passed"] and all(x["allclose_1e4"] for x in g["final_states"].values()) for g in report["groups"][2:]),
        bf16_reduction_control_passed=report["groups"][1]["logits"]["registered_numeric_limits_passed"])
    write_json(out/"full_precision_summary.json",report)
    print({k:report[k] for k in ("fp32_two_groups_passed","bf16_reduction_control_passed","wall_seconds_including_load","peak_allocated_gib","peak_reserved_gib")},flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("preflight","gpu"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--model-dir",type=Path,default=PROJECT/"models/Qwen3.5-0.8B")
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
