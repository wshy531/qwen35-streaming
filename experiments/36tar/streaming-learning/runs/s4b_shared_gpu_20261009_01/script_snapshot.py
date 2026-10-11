"""S4-B causal shared ACTUAL vision features, with independent full-language controls."""
import argparse
import itertools
import json
from pathlib import Path
import tempfile
import time
import traceback

import psutil
import torch
from transformers import AutoProcessor

from scripts.baseline import PROJECT,REPO,REVISION,provenance,write_json,file_hash
from scripts.chunk_weight_check import encoded
from scripts.fp32_regression import strict_logits,strict_states
from scripts.online_experiment import reference_inputs,reference_answer,pair_record,numerical_pass,Q1,Q2
from streaming.inputs import tensor_fingerprint
from streaming.online import OnlinePairSession,fp32_reference
from streaming.window import read_packets

BUDGET = {"records":7,"strict_shared_groups":5,"max_tokens":256,"max_new_tokens_per_answer":8,
          "generated_tokens_max":32,"language_forwards_max":47,"vision_encodings":6,
          "model_loads":1,"wall_seconds":600}
PLAN = PROJECT/"共享视觉特征控制方案.md"
LAUNCHER = PROJECT/"scripts/run_shared_features.py"


class Capture:
    def __init__(self,model):
        self.counts = {"language":0,"vision":0}
        self.active = False
        self.features,self.embeddings,self.positions = [],[],[]
        self.hooks = [model.model.visual.register_forward_hook(self.vision),
            model.model.language_model.register_forward_pre_hook(self.language,with_kwargs=True)]

    def vision(self,_module,_args,output):
        self.counts["vision"] += 1
        self.features.append(output.pooler_output.detach().cpu().clone())

    def language(self,_module,_args,kwargs):
        self.counts["language"] += 1
        if self.active:
            self.embeddings.append(kwargs["inputs_embeds"].detach().cpu().clone())
            self.positions.append(kwargs["position_ids"].detach().cpu().clone())

    def close(self):
        for hook in self.hooks:
            hook.remove()


@torch.inference_mode()
def shared_input(model,processor,session,inputs,features,embeddings,positions):
    """Independently reconstruct IDs/RoPE and assert exact actual full language inputs."""
    ids = torch.tensor([session.builder.prefix_ids[0].tolist()+sum([p["token_ids"] for p in session.ledger],[])])
    assert torch.equal(ids,inputs.input_ids.cpu()),"Actual online token ledger differs from full template"
    actual = torch.cat(embeddings,1).to(inputs.input_ids.device)
    coords = torch.cat(positions,2).to(inputs.input_ids.device)
    assert torch.equal(coords,inputs.position_ids),"Actual online four-axis positions differ"
    visual = torch.cat(features,0).to(actual)
    reconstructed = model.get_input_embeddings()(inputs.input_ids)
    mask = (inputs.input_ids == processor.video_token_id).unsqueeze(-1).expand_as(reconstructed)
    assert int(mask.sum()) == visual.numel()
    reconstructed = reconstructed.masked_scatter(mask,visual)
    assert torch.equal(reconstructed,actual),"Full inputs must reuse exact actual features AND text embeddings"
    assert torch.equal(actual[0,inputs.input_ids[0] == processor.video_token_id],visual)
    assert actual.shape[1] == session.main.seen_tokens <= 256
    return actual, {"tokens_equal":True,"four_axis_positions_equal":True,"complete_embeddings_bitwise_equal":True,
        "visual_features_bitwise_equal":True,"feature_fingerprint":tensor_fingerprint({"actual_features":visual.cpu()}),
        "feature_bytes":visual.numel()*visual.element_size()}


@torch.inference_mode()
def controls(model,processor,reader,canvas,max_new=8,proof=None,out=None):
    device = model.get_input_embeddings().weight.device
    capture = Capture(model)
    session = OnlinePairSession(model,processor,canvas=canvas)
    frames,features,embeddings,positions,groups,fingerprints = [],[],[],[],[],{}
    generated = 0
    report = {"budget":BUDGET,"groups":groups,"questions":[Q1,Q2],"canvas":list(canvas),
        "scope":"fixed causal file replay, explicit FP32 IEEE/math; no live service or BF16 acceptance"}
    def prepare(name,available,question=None):
        inputs = reference_inputs(model,processor,available,canvas,question)
        fingerprints[name] = tensor_fingerprint(inputs)
        if proof:
            assert fingerprints[name] == proof["input_fingerprints"][name]
        return inputs.to(device)
    try:
        with fp32_reference():
            for question in (Q1,Q2):
                before = session.main.seen_tokens
                first,second = next(reader),next(reader)
                frames.extend((first,second))
                assert session.append(first)["logits"] is None
                try:
                    session.ask(question,max_new)
                    raise AssertionError("Pending frame must block QA")
                except ValueError:
                    pass
                capture.active = True
                try:
                    new = session.append(second)
                finally:
                    capture.active = False
                features.append(capture.features.pop())
                embeddings.extend(capture.embeddings); positions.extend(capture.positions)
                capture.embeddings.clear(); capture.positions.clear()
                name = f"prefix{len(frames)}"
                inputs = prepare(name,frames)
                actual,identity = shared_input(model,processor,session,inputs,features,embeddings,positions)
                full = model(inputs_embeds=actual,position_ids=inputs.position_ids,use_cache=True)
                start = before if before else session.builder.prefix_ids.shape[1]
                record = pair_record(new["logits"],session,full,start)
                groups.append({"id":name+"_shared","kind":"strict_shared",**record,"input_identity":identity,
                    "passed":numerical_pass(record)})
                snapshot = session.main.cache.clone()
                counts = session.main.seen_tokens,session.main.next_rope_position
                answer = session.ask(question,max_new,return_evidence=True)
                evidence = answer.pop("evidence")
                reference = reference_answer(model,processor,actual,inputs.position_ids,question,max_new)
                qinputs = prepare(f"question{len(frames)}",frames,question)
                assert torch.equal(qinputs.input_ids.cpu(),torch.cat((inputs.input_ids.cpu(),evidence["suffix_ids"]),1))
                record = {"id":f"qa{len(frames)}_shared","kind":"strict_shared","online":answer,
                    "reference_answer":reference["answer"],"reference_ids":reference["generated_ids"],
                    "logits":strict_logits(evidence["initial_logits"],reference["initial_logits"]),
                    "final_states":strict_states(evidence["branch_cache"],reference["cache"]),
                    "main_unchanged_bitwise":all(x["all_tensors_bitwise_equal"] for x in strict_states(session.main.cache,snapshot).values()),
                    "main_counters_unchanged":counts == (session.main.seen_tokens,session.main.next_rope_position)}
                record["passed"] = (numerical_pass(record) and record["main_unchanged_bitwise"] and
                    record["main_counters_unchanged"] and answer["generated_ids"] == reference["generated_ids"])
                groups.append(record)
                generated += len(answer["generated_ids"])+len(reference["generated_ids"])
                del evidence,reference,snapshot,qinputs
                if len(frames) == 4:
                    whole_embeddings = encoded(model,inputs)
                    whole_features = capture.features.pop()
                    whole = model(inputs_embeds=whole_embeddings,position_ids=inputs.position_ids,use_cache=True)
                    bias = {"id":"prefix4_vision_bias","kind":"diagnostic_different_inputs",
                        "features":strict_logits(torch.cat(features)[None],whole_features[None]),
                        "logits":strict_logits(full.logits,whole.logits),"final_states":strict_states(full.past_key_values,whole.past_key_values),
                        "online_vs_whole":pair_record(new["logits"],session,whole,start)}
                    bias["within_old_strict_language_gate"] = numerical_pass(bias)
                    bias["online_vs_whole"]["passed"] = numerical_pass(bias["online_vs_whole"])
                    groups.append(bias)
                    del whole,whole_embeddings,whole_features
                print({"prefix_frames":len(frames),"shared_prefix_pass":groups[-3 if len(frames)==4 else -2]["passed"],
                    "shared_question_pass":record["passed"],"answer":answer["answer"]},flush=True)
                del full,actual,inputs,new
                if out:
                    write_json(out/"shared_features_summary.json",report)
            session.close()
            features.clear(); embeddings.clear(); positions.clear()
            eof = OnlinePairSession(model,processor,canvas=canvas)
            capture.active = True
            try:
                eof.append(frames[0]); eof.append(frames[1]); eof.append(frames[2])
                before = eof.main.seen_tokens
                flushed = eof.finish()
            finally:
                capture.active = False
            features.extend(capture.features); embeddings.extend(capture.embeddings); positions.extend(capture.positions)
            capture.features.clear(); capture.embeddings.clear(); capture.positions.clear()
            inputs = prepare("eof3",frames[:3])
            actual,identity = shared_input(model,processor,eof,inputs,features,embeddings,positions)
            full = model(inputs_embeds=actual,position_ids=inputs.position_ids,use_cache=True)
            sealed = False
            try:
                eof.append(frames[3])
            except ValueError:
                sealed = True
            record = pair_record(flushed["logits"],eof,full,before)
            groups.append({"id":"eof3_shared","kind":"strict_shared",**record,"input_identity":identity,
                "padding_record":eof.ledger[-1],"input_sealed":sealed,
                "passed":numerical_pass(record) and sealed and eof.ledger[-1]["synthetic_padding"]})
            whole_embeddings = encoded(model,inputs)
            whole_features = capture.features.pop()
            whole = model(inputs_embeds=whole_embeddings,position_ids=inputs.position_ids,use_cache=True)
            bias = {"id":"eof3_vision_bias","kind":"diagnostic_different_inputs",
                "features":strict_logits(torch.cat(features)[None],whole_features[None]),
                "logits":strict_logits(full.logits,whole.logits),"final_states":strict_states(full.past_key_values,whole.past_key_values),
                "online_vs_whole":pair_record(flushed["logits"],eof,whole,before)}
            bias["within_old_strict_language_gate"] = numerical_pass(bias)
            bias["online_vs_whole"]["passed"] = numerical_pass(bias["online_vs_whole"])
            groups.append(bias)
            eof.close()
        assert len(groups) == BUDGET["records"]
        assert capture.counts["vision"] == BUDGET["vision_encodings"]
        assert capture.counts["language"] <= BUDGET["language_forwards_max"]
        assert generated <= BUDGET["generated_tokens_max"]
        shared = [g for g in groups if g["kind"] == "strict_shared"]
        assert len(shared) == BUDGET["strict_shared_groups"]
        report.update(input_fingerprints=fingerprints,actual_forwards=capture.counts,generated_tokens=generated,
            all_shared_groups_passed=all(g["passed"] for g in shared),frame_ids=[p.frame_id for p in frames],
            timestamps=[p.timestamp for p in frames],hooks_removed_after_return=True)
        return report
    finally:
        capture.close()
        session.close()


def preflight(out,args):
    from tests.test_online import small_model
    torch.set_num_threads(1)
    assert json.loads((args.model_dir/"DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8-sig"))["revision"] == REVISION
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    model = small_model(processor)
    before = (len(model.model.visual._forward_hooks),len(model.model.language_model._forward_pre_hooks))
    reader = read_packets(args.video,1)
    try:
        cpu = controls(model,processor,reader,(32,32))
    finally:
        reader.close()
    assert cpu["all_shared_groups_passed"]
    assert before == (len(model.model.visual._forward_hooks),len(model.model.language_model._forward_pre_hooks))
    write_json(out/"cpu_summary.json",cpu)
    reader = read_packets(args.video,1)
    try:
        frames = tuple(itertools.islice(reader,4))
    finally:
        reader.close()
    fingerprints = {name:tensor_fingerprint(reference_inputs(model,processor,subset,(192,256),q))
        for name,subset,q in (("prefix2",frames[:2],None),("prefix4",frames,None),("question2",frames[:2],Q1),
                            ("question4",frames,Q2),("eof3",frames[:3],None))}
    old = json.loads((PROJECT/"runs/s4b_online_preflight_20261009_02/preflight_summary.json").read_text())
    assert fingerprints == old["input_fingerprints"]
    write_json(out/"preflight_summary.json",{"cpu_controls_passed":True,"cpu_strict_groups":5,"hooks_cleanup_verified":True,
        "budget":BUDGET,"input_fingerprints":fingerprints,"video_sha256":file_hash(args.video),
        "model_manifest_sha256":file_hash(args.model_dir/"DOWNLOAD_MANIFEST.json"),"plan_sha256":file_hash(PLAN),
        "launcher_sha256":file_hash(LAUNCHER),"canvas":[192,256],"questions":[Q1,Q2]})
    print({"cpu_shared_groups_passed":5,"budget":BUDGET,"actual_cpu_forwards":cpu["actual_forwards"]},flush=True)


@torch.inference_mode()
def gpu(out,args):
    from qwen35_local import Qwen3_5ForConditionalGeneration
    begin = time.perf_counter()
    torch.manual_seed(42)
    free,total = torch.cuda.mem_get_info()
    guard = {"gpu_free_bytes":free,"gpu_total_bytes":total,"ram_available_bytes":psutil.virtual_memory().available}
    write_json(out/"resource_precheck.json",guard)
    assert free >= 5.5*2**30 and guard["ram_available_bytes"] >= 3*2**30,"Resource guard; no model loading"
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir,local_files_only=True,dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    assert torch.cuda.mem_get_info()[0] >= 2*2**30
    model.float()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(not x.linear_attn.experimental_recurrent_reference for x in model.model.language_model.layers if x.layer_type == "linear_attention")
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    proof = json.loads((args.preflight/"preflight_summary.json").read_text())
    reader = read_packets(args.video,1)
    try:
        report = controls(model,processor,reader,(192,256),proof=proof,out=out)
    finally:
        reader.close()
    report.update(wall_seconds_including_load=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,resource_precheck=guard)
    write_json(out/"shared_features_summary.json",report)
    print({k:report[k] for k in ("all_shared_groups_passed","actual_forwards","generated_tokens","wall_seconds_including_load",
        "peak_allocated_gib","peak_reserved_gib")},flush=True)
    if not report["all_shared_groups_passed"]:
        raise RuntimeError("Shared exact-input strict control failed; preserve evidence")


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
