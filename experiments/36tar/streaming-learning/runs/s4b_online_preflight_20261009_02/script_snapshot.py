"""S4-B independent whole-prefix reference and bounded online experiment."""
import argparse
import itertools
import json
from pathlib import Path
import tempfile
import time
import traceback
import unittest

import numpy as np
from PIL import Image
import psutil
import torch
from transformers.feature_extraction_utils import BatchFeature

from scripts.baseline import PROJECT,REPO,REVISION,provenance,write_json,file_hash,check_position_oracle,cache_measure
from scripts.chunk_weight_check import encoded
from scripts.fp32_regression import strict_logits,strict_states
from streaming.inputs import tensor_fingerprint
from streaming.online import OnlinePairSession,fp32_reference
from streaming.window import read_packets

Q1 = "What is the main visible subject in the LAST frame of this video window? Answer only CAT, HANDS, or OTHER."
Q2 = "How many hands are visible in the LAST frame? Answer only ONE, TWO, or OTHER."
BUDGET = {"groups":8,"max_tokens":256,"max_new_tokens_per_answer":8,"generated_tokens_max":32,
          "language_forwards_max":47,"vision_encodings":8,"model_loads":1,"wall_seconds":600}
PLAN = PROJECT/"在线成对帧问答验收方案.md"
LAUNCHER = PROJECT/"scripts/run_online_pairs.py"


def reference_inputs(model,processor,frames,canvas,question=None):
    """Rebuild the available prefix only; full processor, template and official RoPE."""
    frames = tuple(frames)
    assert frames
    padded = frames if len(frames)%2 == 0 else frames+(frames[-1],)
    pixels = np.stack([np.asarray(Image.fromarray(p.rgb).resize((canvas[1],canvas[0]),Image.Resampling.BICUBIC)) for p in padded])
    visual = processor.video_processor(videos=[pixels],do_resize=False,do_sample_frames=False,
        return_metadata=False,return_tensors="pt")
    per_group = int(visual.video_grid_thw[0,1:].prod())//processor.video_processor.merge_size**2
    expansion = "".join(f"<{(padded[i].timestamp+padded[i+1].timestamp)/2:.1f} seconds>"+processor.vision_start_token+
        processor.video_token*per_group+processor.vision_end_token for i in range(0,len(padded),2))
    template = processor.apply_chat_template([{"role":"user","content":[{"type":"video"},{"type":"text","text":question or "probe"}]}],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    marker = processor.vision_start_token+processor.video_token+processor.vision_end_token
    text = template.replace(marker,expansion) if question is not None else template.split(marker)[0]+expansion
    tokens = processor.tokenizer(text,add_special_tokens=False,return_tensors="pt")
    inputs = BatchFeature({**tokens,**visual,"mm_token_type_ids":(tokens.input_ids == processor.video_token_id).long()*2})
    grids = torch.repeat_interleave(inputs.video_grid_thw,inputs.video_grid_thw[:,0],0)
    grids[:,0] = 1
    kwargs = dict(inputs)
    kwargs["video_grid_thw"] = grids
    coordinates,delta = model.model.get_rope_index(**kwargs)
    inputs["position_ids"] = torch.cat((torch.arange(inputs.input_ids.shape[1])[None,None,:],coordinates),0)
    previous = model.model.rope_deltas
    try:
        model.model.rope_deltas = delta
        check_position_oracle(model,inputs)
    finally:
        model.model.rope_deltas = previous
    return inputs


@torch.inference_mode()
def reference_answer(model,processor,embeddings,positions,question,max_new_tokens):
    template = processor.apply_chat_template([{"role":"user","content":[{"type":"video"},{"type":"text","text":question}]}],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    marker = processor.vision_start_token+processor.video_token+processor.vision_end_token
    suffix = processor.tokenizer(template.split(marker)[1],add_special_tokens=False,return_tensors="pt").input_ids.to(embeddings.device)
    count = embeddings.shape[1]
    rope = int(positions[1:].max())+1
    suffix_pos = torch.stack((torch.arange(count,count+suffix.shape[1],device=embeddings.device),
        *[torch.arange(rope,rope+suffix.shape[1],device=embeddings.device) for _ in range(3)]))[:,None,:]
    result = model(inputs_embeds=torch.cat((embeddings,model.get_input_embeddings()(suffix)),1),
        position_ids=torch.cat((positions,suffix_pos),2),use_cache=True)
    initial = result.logits[:,-suffix.shape[1]:]
    logits,cache = result.logits,result.past_key_values
    count += suffix.shape[1]
    rope += suffix.shape[1]
    eos = model.generation_config.eos_token_id
    eos_ids = set([] if eos is None else eos if isinstance(eos,list) else [eos])
    if processor.tokenizer.eos_token_id is not None:
        eos_ids.add(processor.tokenizer.eos_token_id)
    generated = []
    for _ in range(max_new_tokens):
        token = int(logits[0,-1].argmax())
        generated.append(token)
        point = torch.tensor([[[count]],[[rope]],[[rope]],[[rope]]],device=embeddings.device)
        result = model(input_ids=torch.tensor([[token]],device=embeddings.device),position_ids=point,
            cache_position=torch.tensor([count],device=embeddings.device),attention_mask=torch.ones(1,count+1,device=embeddings.device,dtype=torch.long),
            past_key_values=cache,use_cache=True)
        logits,cache = result.logits,result.past_key_values
        count,rope = count+1,rope+1
        if token in eos_ids:
            break
    return {"generated_ids":generated,"answer":processor.tokenizer.decode(generated,skip_special_tokens=True),
            "initial_logits":initial,"cache":cache}


def pair_record(online_logits,session,full,prefix_tokens):
    return {"logits":strict_logits(online_logits,full.logits[:,prefix_tokens:]),
            "final_states":strict_states(session.main.cache,full.past_key_values),"status":session.status()}


def numerical_pass(record):
    return (record["logits"]["fp32_reference_limits_passed"] and record["logits"]["allclose_1e4"] and
            all(x["allclose_1e4"] for x in record["final_states"].values()))


def preflight(out,args):
    from transformers import AutoProcessor
    from tests.test_online import OnlineTests,small_model
    torch.set_num_threads(1)
    assert json.loads((args.model_dir/"DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8-sig"))["revision"] == REVISION
    with (out/"unittest.log").open("w",encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OnlineTests))
    assert result.wasSuccessful(),"CPU contract tests failed; retain log"
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    model = small_model(processor)
    packets = read_packets(args.video,1)
    try:
        frames = tuple(itertools.islice(packets,4))
    finally:
        packets.close()
    assert len(frames) == 4
    fingerprints = {}
    for name,subset,q in (("prefix2",frames[:2],None),("prefix4",frames,None),("question2",frames[:2],Q1),
                          ("question4",frames,Q2),("eof3",frames[:3],None)):
        fingerprints[name] = tensor_fingerprint(reference_inputs(model,processor,subset,(192,256),q))
    session = OnlinePairSession(model,processor)
    question_lengths = [session.builder.question_ids(q).shape[1] for q in (Q1,Q2)]
    assert max(question_lengths) <= session.max_question_tokens
    write_json(out/"preflight_summary.json",{"cpu_tests":result.testsRun,"cpu_controls_passed":True,"budget":BUDGET,
        "input_fingerprints":fingerprints,"question_lengths":question_lengths,"video_sha256":file_hash(args.video),
        "model_manifest_sha256":file_hash(args.model_dir/"DOWNLOAD_MANIFEST.json"),"plan_sha256":file_hash(PLAN),
        "launcher_sha256":file_hash(LAUNCHER),"canvas":[192,256],"questions":[Q1,Q2],
        "frame_ids":[p.frame_id for p in frames],"timestamps":[p.timestamp for p in frames]})
    print({"cpu_tests_passed":result.testsRun,"budget":BUDGET,"question_lengths":question_lengths},flush=True)


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
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir,local_files_only=True,dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    torch.cuda.reset_peak_memory_stats()
    assert torch.cuda.mem_get_info()[0] >= 2*2**30
    model.float()
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert all(not layer.linear_attn.experimental_recurrent_reference for layer in model.model.language_model.layers
               if layer.layer_type == "linear_attention")
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    proof = json.loads((args.preflight/"preflight_summary.json").read_text())
    counters = {"language":0,"vision":0}
    visual_parts = []
    def count_language(_m,_a,_o):
        counters["language"] += 1
    def capture_vision(_m,_a,output):
        counters["vision"] += 1
        visual_parts.append(output.pooler_output.detach().cpu())
    hooks = [model.model.language_model.register_forward_hook(count_language),model.model.visual.register_forward_hook(capture_vision)]
    reader = read_packets(args.video,1)
    session = OnlinePairSession(model,processor)
    frames,groups,fingerprints = [],[],{}
    generated_count = 0
    report = {"groups":groups,"budget":BUDGET,"resource_precheck":guard,"questions":[Q1,Q2],
        "scope":"causal sequential file replay with new-pair encoding and real isolated QA; not live camera/long-running service"}
    try:
        with fp32_reference():
            for pair_index,question in enumerate((Q1,Q2)):
                before = session.main.seen_tokens
                first = next(reader)
                second = next(reader)
                frames.extend((first,second))
                waiting = session.append(first)
                assert waiting["logits"] is None
                blocked = False
                try:
                    session.ask(question,8)
                except ValueError:
                    blocked = True
                assert blocked
                new = session.append(second)
                online_features = visual_parts.pop()
                inputs = reference_inputs(model,processor,frames,(192,256))
                key = f"prefix{len(frames)}"
                fingerprints[key] = tensor_fingerprint(inputs)
                assert fingerprints[key] == proof["input_fingerprints"][key]
                inputs = inputs.to("cuda")
                embeddings = encoded(model,inputs)
                whole_features = visual_parts.pop()
                full = model(inputs_embeds=embeddings,position_ids=inputs.position_ids,use_cache=True)
                count = online_features.shape[0]
                feature_diff = strict_logits(online_features[None],whole_features[-count:][None])
                feature_pass = feature_diff["max_abs_diff"] <= 1e-4 and feature_diff["allclose_1e4"]
                if pair_index == 1:
                    past_diff = strict_logits(first_features[None],whole_features[:first_features.shape[0]][None])
                    feature_pass = feature_pass and past_diff["max_abs_diff"] <= 1e-4 and past_diff["allclose_1e4"]
                else:
                    first_features = whole_features
                    first_full = full
                groups.append({"id":f"vision_pair{pair_index+1}","features":feature_diff,"passed":feature_pass,
                    **({"past_features_vs_later_full_video":past_diff} if pair_index else {})})
                start = before if before else session.builder.prefix_ids.shape[1]
                record = pair_record(new["logits"],session,full,start)
                groups.append({"id":f"prefix{len(frames)}",**record,"passed":numerical_pass(record)})
                snapshot = session.main.cache.clone()
                answer = session.ask(question,8,return_evidence=True)
                reference = reference_answer(model,processor,embeddings,inputs.position_ids,question,8)
                evidence = answer.pop("evidence")
                q_record = {"id":f"qa{len(frames)}","online":answer,"reference_answer":reference["answer"],
                    "reference_ids":reference["generated_ids"],"logits":strict_logits(evidence["initial_logits"],reference["initial_logits"]),
                    "final_states":strict_states(evidence["branch_cache"],reference["cache"]),
                    "main_unchanged_bitwise":all(v["all_tensors_bitwise_equal"] for v in strict_states(session.main.cache,snapshot).values())}
                q_record["passed"] = numerical_pass(q_record) and q_record["main_unchanged_bitwise"] and answer["generated_ids"] == reference["generated_ids"]
                groups.append(q_record)
                generated_count += len(answer["generated_ids"])+len(reference["generated_ids"])
                q_inputs = reference_inputs(model,processor,frames,(192,256),question)
                assert torch.equal(q_inputs.input_ids,torch.cat((inputs.input_ids.cpu(),evidence["suffix_ids"]),1))
                fingerprints[f"question{len(frames)}"] = tensor_fingerprint(q_inputs)
                assert fingerprints[f"question{len(frames)}"] == proof["input_fingerprints"][f"question{len(frames)}"]
                del evidence,reference,snapshot,q_inputs
                write_json(out/"online_summary.json",report)
                print({"prefix_frames":len(frames),"math_passed":q_record["passed"],"answer":answer["answer"]},flush=True)
            session.reset()
            assert session.main.seen_tokens == 0 and session.pending is None and not session.ledger and model.model.rope_deltas is None
            session.append(frames[0])
            reset = session.append(frames[1])
            visual_parts.clear()
            record = pair_record(reset["logits"],session,first_full,session.builder.prefix_ids.shape[1])
            groups.append({"id":"reset_then_pair","all_counters_cleared":True,**record,"passed":numerical_pass(record)})
            session.close()
            eof = OnlinePairSession(model,processor)
            eof.append(frames[0]); eof.append(frames[1]); eof.append(frames[2])
            old = eof.main.seen_tokens
            flushed = eof.finish()
            visual_parts.clear()
            inputs = reference_inputs(model,processor,frames[:3],(192,256))
            fingerprints["eof3"] = tensor_fingerprint(inputs)
            assert fingerprints["eof3"] == proof["input_fingerprints"]["eof3"]
            inputs = inputs.to("cuda")
            embeddings = encoded(model,inputs)
            visual_parts.clear()
            full = model(inputs_embeds=embeddings,position_ids=inputs.position_ids,use_cache=True)
            record = pair_record(flushed["logits"],eof,full,old)
            rejected = False
            try:
                eof.append(frames[3])
            except ValueError:
                rejected = True
            groups.append({"id":"eof_odd_padding","padding_record":eof.ledger[-1],"input_sealed":rejected,
                **record,"passed":numerical_pass(record) and rejected and eof.ledger[-1]["synthetic_padding"]})
            eof.close()
        assert len(groups) == 8
        assert counters["vision"] == BUDGET["vision_encodings"] and counters["language"] <= BUDGET["language_forwards_max"]
        assert generated_count <= BUDGET["generated_tokens_max"]
        report.update(input_fingerprints=fingerprints,actual_forwards=counters,generated_tokens=generated_count,
            all_interface_numeric_groups_passed=all(g["passed"] for g in groups),
            wall_seconds_including_load=time.perf_counter()-begin,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        write_json(out/"online_summary.json",report)
        print({k:report[k] for k in ("all_interface_numeric_groups_passed","actual_forwards","generated_tokens","wall_seconds_including_load","peak_allocated_gib","peak_reserved_gib")},flush=True)
        if not report["all_interface_numeric_groups_passed"]:
            raise RuntimeError("Online registered interfaces/numerics failed; retain evidence")
    finally:
        reader.close()
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
    for name,path in (("script_snapshot.py",Path(__file__)),("online_snapshot.py",REPO/"streaming/online.py"),
                       ("tests_snapshot.py",REPO/"tests/test_online.py")):
        (args.out/name).write_bytes(path.read_bytes())
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
