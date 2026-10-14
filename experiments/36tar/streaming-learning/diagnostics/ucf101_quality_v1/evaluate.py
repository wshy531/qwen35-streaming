"""Fresh-proof-gated, bounded real-weight original/streaming video QA."""
import argparse
import gc
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
CORE=ROOT/'code/qwen35-streaming-s4'
sys.path.insert(0,str(CORE))

import numpy as np
import psutil
import torch
from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from qwen35_local import Qwen3_5ForConditionalGeneration
from streaming.fair_inputs import joined_pair_inputs
from scripts.pair_demo import verify_assets
from scripts.protocol_capability_control import original_package,ORIGINAL
from streaming.live_source import LiveSource
from streaming.online import OnlinePairSession,fp32_reference
from streaming.playback import SourceDigest
from streaming.preprocessing import fit_canvas
from streaming.window import FramePacket,read_packets

QUESTION='What is the main action shown in the video? Answer in one short sentence.'
FIELDS=('key_cache','value_cache','conv_states','recurrent_states')
BUDGET={'model_loads':2,'questions_per_arm':6,'generated_per_arm':144,'answer_tokens':24,
        'original_vision_max':6,'original_language_max':160,'streaming_vision_max':8,
        'streaming_language_max':180,'max_tokens':512,'max_frames_per_video':8,
        'gpu_free_min_gib':5.5,'ram_available_after_import_min_gib':1.5,
        'conversion_free_min_gib':2,'wall_seconds':600,'precision':'fp32_ieee_math',
        'load_policy':'direct GPU device_map; each arm in a separate child process'}


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()


def sources():
    paths=[*CORE.rglob('*.py'),*(ROOT/'code/qwen35-streaming/qwen35_local').glob('*.py'),
           *Path(__file__).parent.glob('*.py'),Path(__file__).with_name('方案.md')]
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def asset_identity(args):
    manifest=read(args.data_dir/'manifest.json')
    for c in manifest['clips']:
        p=args.data_dir/c['file']
        if p.stat().st_size!=c['bytes'] or sha(p)!=c['sha256']:raise ValueError('Video fingerprint changed')
    return {'dataset':manifest,'model':verify_assets(args.model_dir),
            'model_manifest_sha256':sha(args.model_dir/'DOWNLOAD_MANIFEST.json')}


def frames(args):
    rows=[]
    for c in read(args.data_dir/'manifest.json')['clips']:
        reader=read_packets(args.data_dir/c['file'],1)
        packets=[]
        try:
            for p in reader:
                if p.timestamp>=12 or len(packets)>=8:break
                rgb=fit_canvas(p.rgb,(192,256));rgb.flags.writeable=False
                packets.append(FramePacket(p.frame_id,p.timestamp,rgb,p.decoded_count))
        finally:reader.close()
        if not 2<=len(packets)<=8:raise ValueError('Two to eight real frames required')
        rows.append((c,packets))
    if len(rows)!=3:raise ValueError('Fixed three-video subset required')
    return rows


def input_identity(rows):
    result={}
    for c,packets in rows:
        digest=SourceDigest()
        for p in packets:digest.add(p)
        result[c['file']]={'source':digest.snapshot(),'frames':[
            [p.frame_id,p.timestamp,p.decoded_count] for p in packets]}
    return result


def whole(model,processor,packets):
    padded=list(packets)
    if len(padded)%2:padded.append(padded[-1])
    visual=processor.video_processor(videos=[np.stack([p.rgb for p in padded])],
        do_resize=False,do_sample_frames=False,return_metadata=False,return_tensors='pt')
    grid=visual.video_grid_thw
    count=int(grid[0,1:].prod())//processor.video_processor.merge_size**2
    expansion=''.join(f'<{(padded[i].timestamp+padded[i+1].timestamp)/2:.1f} seconds>'+
        processor.vision_start_token+processor.video_token*count+processor.vision_end_token
        for i in range(0,len(padded),2))
    template=processor.apply_chat_template([{'role':'user','content':[
        {'type':'video'},{'type':'text','text':QUESTION}]}],tokenize=False,
        add_generation_prompt=True,enable_thinking=False)
    marker=processor.vision_start_token+processor.video_token+processor.vision_end_token
    if template.count(marker)!=1:raise ValueError('Unexpected chat template')
    ids=processor.tokenizer(template.replace(marker,expansion),add_special_tokens=False,return_tensors='pt')
    value=BatchFeature({**ids,**visual,'mm_token_type_ids':(ids.input_ids==processor.video_token_id).long()*2})
    expanded=torch.repeat_interleave(grid,grid[:,0],0);expanded[:,0]=1
    coords,_=model.model.get_rope_index(**{**value,'video_grid_thw':expanded})
    value['position_ids']=torch.cat((torch.arange(ids.input_ids.shape[1])[None,None,:],coords),0)
    if value.input_ids.shape[1]+24>512:raise ValueError('Context budget')
    return value


def cache_equal(a,b):
    for name in FIELDS:
        left,right=getattr(a,name),getattr(b,name)
        if len(left)!=len(right) or not all((x is None and y is None) or
            (x is not None and y is not None and torch.equal(x,y)) for x,y in zip(left,right)):return False
    return True


@torch.inference_mode()
def ask(session):
    before=session.status();saved=session.main.cache.clone()
    result=session.ask(QUESTION,24)
    if session.status()!=before or not cache_equal(saved,session.main.cache):
        raise AssertionError('Question changed main state')
    del saved
    return {**result,'main_unchanged_bitwise':True,'main_tokens':session.main.seen_tokens}


@torch.inference_mode()
def stream_video(model,processor,packets):
    session=OnlinePairSession(model,processor,max_tokens=512,max_question_tokens=64,
        max_answer_tokens=24,resize_mode='letterbox',video_logits='none')
    answers=[]
    try:
        for i,p in enumerate(packets):
            session.append(p)
            if i==1:answers.append({'checkpoint':'early','real_frames':2,**ask(session)})
        session.finish()
        answers.append({'checkpoint':'full','real_frames':len(packets),**ask(session)})
    finally:session.close()
    return answers


@torch.inference_mode()
def original_answer(model,processor,value):
    model.model.rope_deltas=None;device=model.device
    value=value.to(device)
    with fp32_reference():
        result=model(**value,use_cache=True,logits_to_keep=1)
        logits,cache=result.logits,result.past_key_values
        count=value.input_ids.shape[1];rope=int(value.position_ids[1:].max())+1
        eos=model.generation_config.eos_token_id
        stops=set([] if eos is None else eos if isinstance(eos,list) else [eos])
        stops.add(processor.tokenizer.eos_token_id)
        generated=[];reason='budget';del result,value
        for _ in range(24):
            token=int(logits[0,-1].argmax());generated.append(token)
            point=torch.tensor([[[count]],[[rope]],[[rope]],[[rope]]],device=device)
            result=model(input_ids=torch.tensor([[token]],device=device),position_ids=point,
                attention_mask=torch.ones((1,count+1),dtype=torch.long,device=device),
                cache_position=torch.tensor([count],device=device),past_key_values=cache,
                use_cache=True,logits_to_keep=1)
            logits,cache=result.logits,result.past_key_values;count+=1;rope+=1;del result
            if token in stops:reason='eos';break
    return {'answer':processor.tokenizer.decode(generated,skip_special_tokens=True),
            'generated_ids':generated,'stop_reason':reason}


@torch.inference_mode()
def preflight(args):
    from tests.test_online import small_model
    processor=AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    rows=frames(args);current=small_model(processor)
    package=original_package()
    reference=package.Qwen3_5ForConditionalGeneration(package.Qwen3_5Config(**current.config.to_dict())).eval()
    reference.set_attn_implementation('sdpa');reference.load_state_dict(current.state_dict(),strict=True)
    identities=input_identity(rows);checked=[]
    for c,packets in rows:
        # Real capture process verifies the exact file-source path before GPU.
        source=LiveSource('file',str(args.data_dir/c['file']),1,12,8)
        try:
            captured=list(source.start())
            d=SourceDigest()
            for p in captured:d.add(p)
            if d.snapshot()!=identities[c['file']]['source']:raise AssertionError('Real capture pixels differ')
        finally:source.close()
        for name,prefix in (('early',packets[:2]),('full',packets)):
            a=whole(reference,processor,prefix);b=joined_pair_inputs(processor,prefix,QUESTION)
            equal={k:k in b and torch.equal(v,b[k]) for k,v in a.items()}
            if set(a)!=set(b) or not all(equal.values()):raise AssertionError('Whole/pair input mismatch: '+str(equal))
            with fp32_reference():
                left=reference(**a,use_cache=False,logits_to_keep=1).logits
                right=current(**a,use_cache=False,logits_to_keep=1).logits
            if not torch.equal(left,right):raise AssertionError('Fresh tiny cores differ')
            checked.append({'file':c['file'],'checkpoint':name,'tensor_equalities':equal,
                            'input_tokens':a.input_ids.shape[1],'tiny_fresh_logits_equal':True})
        answers=stream_video(current,processor,packets)
        if len(answers)!=2 or not all(a['main_unchanged_bitwise'] for a in answers):raise AssertionError('Tiny isolation failed')
    write(args.out/'proof.json',{'sources':sources(),'assets':asset_identity(args),'inputs':identities,
          'checks':checked,'budget':BUDGET,'question':QUESTION,'passed':True})
    print(json.dumps({'preflight_pairs':len(checked),'passed':True}),flush=True)


@torch.inference_mode()
def gpu(args):
    if args.preflight is None or read(args.preflight/'status.json')!={'status':'completed','stage':'preflight'}:
        raise ValueError('Fresh completed CPU proof required')
    proof=read(args.preflight/'proof.json');rows=frames(args)
    if (proof['sources']!=sources() or proof['assets']!=asset_identity(args) or
        proof['inputs']!=input_identity(rows) or proof['budget']!=BUDGET or not proof['passed']):raise ValueError('Proof drift')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=CORE,text=True).strip()
    if branch in ('','main','master','HEAD'):raise ValueError('Personal development branch required')
    processor=AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    reports=[]
    if args.arm not in ('original','streaming'):raise ValueError('GPU child requires one arm')
    for arm in (args.arm,):
        free,total=torch.cuda.mem_get_info();ram=psutil.virtual_memory().available
        guard={'arm':arm,'gpu_free_gib':free/2**30,'ram_available_gib':ram/2**30,'branch':branch}
        write(args.out/(arm+'_precheck.json'),guard)
        if free<5.5*2**30 or ram<1.5*2**30:raise RuntimeError('Direct-GPU-load resource guard refused load')
        cls=original_package().Qwen3_5ForConditionalGeneration if arm=='original' else Qwen3_5ForConditionalGeneration
        started=time.perf_counter()
        model=cls.from_pretrained(args.model_dir,local_files_only=True,dtype=torch.bfloat16,
            device_map={'':'cuda:0'},attn_implementation='sdpa').eval()
        if model.device.type!='cuda':raise AssertionError('Weights not on GPU')
        if torch.cuda.mem_get_info()[0]<2*2**30:raise RuntimeError('FP32 conversion room unavailable')
        torch.cuda.reset_peak_memory_stats();model.float()
        driver_free_after_float=torch.cuda.mem_get_info()[0]
        model_source=str(Path(inspect.getfile(type(model))).resolve())
        if arm=='original' and Path(model_source).parent!=ORIGINAL:raise AssertionError('Wrong original core')
        counts={'vision':0,'language':0};limits={k:BUDGET[arm+'_'+k+'_max'] for k in counts}
        def counter(name):
            def hook(_m,_a):
                counts[name]+=1
                if counts[name]>limits[name]:raise RuntimeError('Forward budget')
            return hook
        hooks=[model.model.visual.register_forward_pre_hook(counter('vision')),
               model.model.language_model.register_forward_pre_hook(counter('language'))]
        answers=[]
        try:
            for c,packets in rows:
                if arm=='original':
                    batch=[]
                    for name,prefix in (('early',packets[:2]),('full',packets)):
                        torch.cuda.synchronize();begin=time.perf_counter()
                        answer=original_answer(model,processor,whole(model,processor,prefix))
                        torch.cuda.synchronize()
                        batch.append({'checkpoint':name,'real_frames':len(prefix),**answer,
                                      'wall_seconds':time.perf_counter()-begin})
                else:batch=stream_video(model,processor,packets)
                for a in batch:
                    row={'file':c['file'],'arm':arm,'question':QUESTION,**a}
                    answers.append(row)
                    print(json.dumps(row,ensure_ascii=False),flush=True)
            generated=sum(len(a['generated_ids']) for a in answers)
            if len(answers)!=6 or generated>144:raise AssertionError('QA budget')
            torch.cuda.synchronize()
            report={'arm':arm,'answers':answers,'counts':counts,'generated_tokens':generated,
                    'model_source':model_source,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                    'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
                    'driver_free_gib_after_float':driver_free_after_float/2**30,
                    'driver_resident_delta_gib_after_float':(free-driver_free_after_float)/2**30,
                    'wall_seconds_including_load':time.perf_counter()-started}
            write(args.out/(arm+'_summary.json'),report);reports.append(report)
        finally:
            for h in hooks:h.remove()
            del model;gc.collect();torch.cuda.empty_cache()
    comparisons=[]
    write(args.out/'summary.json',{'budget':BUDGET,'reports':reports,'comparisons':comparisons,
        'source_hashes':sources(),'input_identity':input_identity(rows),'scope':'3-video/6-context engineering QA; not official UCF101 accuracy'})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('preflight','gpu'),required=True)
    for name in ('model-dir','data-dir','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--preflight',type=Path)
    p.add_argument('--arm',choices=('both','original','streaming'),default='both')
    args=p.parse_args()
    for name in ('model_dir','data_dir','out','preflight'):
        if getattr(args,name) is not None:setattr(args,name,getattr(args,name).resolve())
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'tmp').mkdir();tempfile.tempdir=str(args.out/'tmp')
    snapshot=args.out/'source_snapshot';snapshot.mkdir()
    for source in Path(__file__).parent.iterdir():
        if source.suffix in ('.py','.md'):(snapshot/source.name).write_bytes(source.read_bytes())
    torch.set_num_threads(1);torch.manual_seed(42)
    write(args.out/'budget.json',BUDGET);write(args.out/'source_hashes.json',sources())
    try:
        (preflight if args.stage=='preflight' else gpu)(args)
        write(args.out/'status.json',{'status':'completed','stage':args.stage})
    except Exception:
        (args.out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        write(args.out/'status.json',{'status':'failed','stage':args.stage});raise


if __name__=='__main__':main()
