"""Same growing prefixes: repeated whole prefill versus persistent streaming."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

HERE=Path(__file__).resolve().parent;CAP=HERE.parents[1];ACTIVE=CAP/'code/qwen35-streaming-s4'
sys.path.insert(0,str(ACTIVE))
import numpy as np
import psutil
import torch
from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from scripts.baseline import file_hash,cache_measure
from scripts.pair_demo import verify_assets
from scripts.protocol_capability_control import original_package,source_assets,ORIGINAL
from scripts.video_demo import branch_allowed
from streaming.demo import FIELDS
from streaming.fair_inputs import joined_pair_inputs
from streaming.inputs import tensor_fingerprint
from streaming.online import OnlinePairSession,fp32_reference
from streaming.preprocessing import fit_canvas
from streaming.window import FramePacket,read_packets

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,v):p.write_bytes((json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
def normal(v):return json.loads(json.dumps(v))
def config():
    c=read(HERE/'cases.json')
    if c['prefix_frames']!=[2,4,6,8,10,12] or (c['answer_tokens'],c['max_tokens'],c['question_reserve'],c['warmup_rounds'],c['timed_rounds'])!=(32,512,96,1,3):
        raise ValueError('Registered prefix/output/round bounds changed')
    if len(c['videos'])!=2:raise ValueError('Two registered sources required')
    return c
def sources():
    return {**{str(p.resolve()):file_hash(p) for p in sorted(ACTIVE.rglob('*.py'))},**source_assets(),
            **{str(p.resolve()):file_hash(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.json','.md')}}
def input_assets():
    result={}
    for s in config()['videos']:
        p=Path(s['path']);digest=file_hash(p)
        if digest!=s['sha256']:raise ValueError('Source bytes changed')
        result[str(p.resolve())]=digest
    return result
def budget():return {'model_loads':1,'vision_max':48,'language_max':1700,'generated_max':1536,'answers':48,
                     'real_frames_per_source':12,'rgb_max_bytes_per_source':1769472,'original_processed_visual_frames':336,
                     'stream_processed_visual_frames':96,'max_tokens':512,'question_reserve':96,'answer_tokens':32,
                     'warmup_rounds':1,'timed_rounds':3,'wall_seconds':300,'precision':'fp32_ieee_math',
                     'available_ram_gib_min':3,'free_gpu_gib_min':5.5,'conversion_free_gib_min':2,'concurrent_models':1}
def request(a):return {'model_dir':str(a.model_dir),'allow_branch':a.allow_branch,'wall_seconds':a.wall_seconds}
def load_frames(spec):
    result=[];reader=read_packets(Path(spec['path']),1)
    try:
        for i,p in enumerate(reader):
            if i<spec['sample_start']:continue
            rgb=fit_canvas(p.rgb,(192,256));rgb.flags.writeable=False
            result.append(FramePacket(p.frame_id,p.timestamp,rgb,p.decoded_count,p.decode_seconds))
            if len(result)==12:break
    finally:reader.close()
    if len(result)!=12 or sum(p.rgb.nbytes for p in result)>1769472:raise ValueError('Bounded twelve-frame source required')
    return tuple(result)
def metadata(frames):return [{'frame_id':p.frame_id,'timestamp':p.timestamp,'rgb_sha256':hashlib.sha256(p.rgb.tobytes()).hexdigest()} for p in frames]
def whole_inputs(model,processor,frames,question):
    if len(frames) not in config()['prefix_frames']:raise ValueError('Registered even growing prefix required')
    visual=processor.video_processor(videos=[np.stack([fit_canvas(p.rgb,(192,256)) for p in frames])],
        do_resize=False,do_sample_frames=False,return_metadata=False,return_tensors='pt')
    grid=visual.video_grid_thw;unit=int(grid[0,1:].prod())//processor.video_processor.merge_size**2
    expansion=''.join(f'<{(frames[i].timestamp+frames[i+1].timestamp)/2:.1f} seconds>'+processor.vision_start_token+
        processor.video_token*unit+processor.vision_end_token for i in range(0,len(frames),2))
    template=processor.apply_chat_template([{'role':'user','content':[{'type':'video'},{'type':'text','text':question}]}],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    marker=processor.vision_start_token+processor.video_token+processor.vision_end_token
    if template.count(marker)!=1:raise ValueError('Exactly one video marker required')
    ids=processor.tokenizer(template.replace(marker,expansion),add_special_tokens=False,return_tensors='pt')
    whole=BatchFeature({**ids,**visual,'mm_token_type_ids':(ids.input_ids==processor.video_token_id).long()*2})
    grids=torch.repeat_interleave(grid,grid[:,0],0);grids[:,0]=1
    coordinates,_=model.model.get_rope_index(**{**whole,'video_grid_thw':grids})
    whole['position_ids']=torch.cat((torch.arange(ids.input_ids.shape[1])[None,None,:],coordinates),0)
    if whole.input_ids.shape[1]+32>512:raise ValueError('Whole prompt/answer exceeds registered total')
    return whole
def identity(model,processor,frames,question,source_id):
    whole=whole_inputs(model,processor,frames,question);pair=joined_pair_inputs(processor,frames,question)
    equality={k:k in pair and torch.equal(v,pair[k]) for k,v in whole.items()}
    if set(whole)!=set(pair) or not all(equality.values()):raise AssertionError('Independent whole/pair inputs differ')
    return whole,normal({'case_id':source_id+'_'+str(len(frames)),'source_id':source_id,'prefix_frames':len(frames),
                        'frames':metadata(frames),'question':question,'input_tokens':whole.input_ids.shape[1],
                        'grid':whole.video_grid_thw.tolist(),'input_fingerprint':tensor_fingerprint(whole),
                        'tensor_equalities':equality,'all_input_tensors_bitwise_equal':True})
def snapshot(session):
    rope=session.model.model.rope_deltas
    return session.status(),session.main.cache.clone(),None if rope is None else rope.clone()
def unchanged(session,before):
    status,cache,rope=before
    if session.status()!=status:raise AssertionError('Main counters changed')
    for field in FIELDS:
        a,b=getattr(cache,field),getattr(session.main.cache,field)
        if len(a)!=len(b) or not all((x is None and y is None) or (x is not None and y is not None and torch.equal(x,y)) for x,y in zip(a,b)):
            raise AssertionError('Main mixed state changed')
    current=session.model.model.rope_deltas
    if not ((rope is None and current is None) or (rope is not None and current is not None and torch.equal(rope,current))):raise AssertionError('RoPE side state changed')
def session_for(model,processor):return OnlinePairSession(model,processor,max_tokens=512,max_question_tokens=96,max_answer_tokens=32,resize_mode='letterbox',video_logits='none')
def signature(session):
    h=hashlib.sha256()
    for field in FIELDS:
        h.update(field.encode())
        for tensor in getattr(session.main.cache,field):
            if tensor is None:h.update(b'None')
            else:
                h.update(str((tensor.dtype,tuple(tensor.shape))).encode())
                h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    rope=session.model.model.rope_deltas
    if rope is not None:h.update(rope.detach().cpu().contiguous().numpy().tobytes())
    else:h.update(b'rope=None')
    return session.status(),h.hexdigest()

@torch.inference_mode()
def preflight(a):
    from tests.test_online import OnlineTests,small_model
    processor=AutoProcessor.from_pretrained(a.model_dir,local_files_only=True)
    OnlineTests.setUpClass=classmethod(lambda cls:setattr(cls,'processor',processor))
    suite=unittest.TestSuite(OnlineTests(n) for n in ('test_pending_is_copied_readonly_and_does_not_compute','test_state_only_video_writes_and_last_only_qa'))
    with (a.out/'unittest.log').open('w',encoding='utf-8') as f:r=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
    write(a.out/'cpu_tests.json',{'tests':r.testsRun,'passed':r.wasSuccessful(),'errors':len(r.errors),'failures':len(r.failures)})
    if not r.wasSuccessful():raise AssertionError('Video-write/QA projection contracts failed')
    assets=verify_assets(a.model_dir)
    current=small_model(processor);package=original_package()
    original=package.Qwen3_5ForConditionalGeneration(package.Qwen3_5Config(**current.config.to_dict())).eval()
    original.set_attn_implementation('sdpa');original.load_state_dict(current.state_dict(),strict=True)
    entries=[]
    for spec in config()['videos']:
        frames=load_frames(spec);session=session_for(current,processor)
        try:
            for size in config()['prefix_frames']:
                session.append(frames[size-2]);session.append(frames[size-1])
                whole,record=identity(original,processor,frames[:size],config()['question'],spec['id'])
                with fp32_reference():
                    left=original(**whole,use_cache=False,logits_to_keep=1).logits
                    right=current(**whole,use_cache=False,logits_to_keep=1).logits
                if not torch.equal(left,right):raise AssertionError('Tiny original/current fresh prefill differs')
                before=snapshot(session);session.ask(config()['question'],32);unchanged(session,before)
                suffix=session.builder.question_ids(config()['question']).shape[1]
                if session.main.seen_tokens!=record['input_tokens']-suffix:raise AssertionError('Streaming state prefix does not match whole input')
                record.update(main_tokens=session.main.seen_tokens,tiny_fresh_prefill_bitwise_equal=True,main_unchanged_bitwise=True,no_resets=True)
                entries.append(record);del whole,before,left,right
        finally:session.close()
    write(a.out/'preflight_summary.json',{'passed':True,'cases':entries,'scope':'Random CPU input/state contracts, not semantic evidence'})
    write(a.out/'proof.json',{'request':request(a),'budget':budget(),'source_hashes':sources(),'inputs':input_assets(),'assets':assets['files'],
                             'model_manifest_sha256':file_hash(a.model_dir/'DOWNLOAD_MANIFEST.json'),'cases':entries,
                             'summary_sha256':file_hash(a.out/'preflight_summary.json'),'tests_sha256':file_hash(a.out/'cpu_tests.json')})
    print(json.dumps({'passed':True,'tests':r.testsRun,'growing_inputs':len(entries),'max_prefix_frames':12,'no_resets':True}),flush=True)
def check(a):
    if not a.preflight or read(a.preflight/'status.json')!={'status':'completed','stage':'preflight','arm':'both'}:raise ValueError('Fresh CPU proof required')
    p=read(a.preflight/'proof.json')
    if p['request']!=request(a) or p['budget']!=budget() or p['source_hashes']!=sources():raise ValueError('Source/request/budget changed')
    if p['inputs']!=input_assets() or p['assets']!=verify_assets(a.model_dir)['files'] or p['model_manifest_sha256']!=file_hash(a.model_dir/'DOWNLOAD_MANIFEST.json'):
        raise ValueError('Source/model assets changed')
    if p['summary_sha256']!=file_hash(a.preflight/'preflight_summary.json') or p['tests_sha256']!=file_hash(a.preflight/'cpu_tests.json') or not read(a.preflight/'cpu_tests.json')['passed']:
        raise ValueError('CPU evidence changed')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=CAP,text=True).strip()
    if not branch_allowed(branch,a.allow_branch):raise ValueError('Personal branch required')
    write(a.out/'checked_branch.json',{'branch':branch});return p

@torch.inference_mode()
def original_qa(model,processor,whole,phase):
    model.model.rope_deltas=None;device=model.get_input_embeddings().weight.device
    values=whole.to(device);phase['name']='prefill'
    with fp32_reference():
        output=model(**values,use_cache=True,logits_to_keep=1);logits,cache=output.logits,output.past_key_values
        count=values.input_ids.shape[1];rope=int(values.position_ids[1:].max())+1
        eos=model.generation_config.eos_token_id;eos_ids=set([] if eos is None else eos if isinstance(eos,list) else [eos])
        eos_ids.add(processor.tokenizer.eos_token_id);ids=[];reason='budget';del output,values
        phase['name']='decode'
        for _ in range(32):
            token=int(logits[0,-1].argmax());ids.append(token)
            output=model(input_ids=torch.tensor([[token]],device=device),
                position_ids=torch.tensor([[[count]],[[rope]],[[rope]],[[rope]]],device=device),
                attention_mask=torch.ones((1,count+1),dtype=torch.long,device=device),cache_position=torch.tensor([count],device=device),
                past_key_values=cache,use_cache=True,logits_to_keep=1)
            logits,cache=output.logits,output.past_key_values;count+=1;rope+=1;del output
            if token in eos_ids:reason='eos';break
    return {'answer':processor.tokenizer.decode(ids,skip_special_tokens=True),'generated_ids':ids,'stop_reason':reason},cache

@torch.inference_mode()
def gpu(a):
    proof=check(a);free,total=torch.cuda.mem_get_info();ram=psutil.virtual_memory().available
    write(a.out/'resource_precheck.json',{'gpu_free_bytes':free,'gpu_total_bytes':total,'ram_available_bytes':ram})
    if free<5.5*2**30 or ram<3*2**30:raise RuntimeError('Resource guard refused load')
    if a.arm=='original':cls=original_package().Qwen3_5ForConditionalGeneration;expected_source=ORIGINAL/'modeling_qwen3_5.py'
    else:
        from qwen35_local import Qwen3_5ForConditionalGeneration
        cls=Qwen3_5ForConditionalGeneration;expected_source=ACTIVE/'qwen35_local/modeling_qwen3_5.py'
    start=time.perf_counter();model=cls.from_pretrained(a.model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    if Path(inspect.getfile(type(model))).resolve()!=expected_source:raise AssertionError('Wrong model source')
    if torch.cuda.mem_get_info()[0]<2*2**30:raise RuntimeError('Insufficient FP32 conversion room')
    model.float();processor=AutoProcessor.from_pretrained(a.model_dir,local_files_only=True);torch.cuda.synchronize()
    loaded=time.perf_counter()-start;counts={'vision':0,'language':0,'vision_patch_rows':0,'language_prefill_tokens':0,'language_decode_tokens':0}
    phase={'name':'outside'};report={'arm':a.arm,'cases':[],'sequences':[],'scope':config()['scope'],'timing_scope':'Preprocess/input build + GPU model/branch/decode + CPU answer decode; excludes oracle audits, weight load and playback wait'}
    def hook(kind):
        def tick(_m,args,kwargs):
            counts[kind]+=1
            if counts[kind]>budget()[kind+'_max']:raise RuntimeError('Forward budget exceeded')
            tensor=kwargs.get('inputs_embeds') if kind=='language' else kwargs.get('hidden_states')
            if tensor is None and args:tensor=args[0]
            if not isinstance(tensor,torch.Tensor):raise RuntimeError('Cannot account model input work')
            if kind=='vision':counts['vision_patch_rows']+=int(tensor.shape[0])
            else:
                length=int(tensor.shape[1]);decode=phase['name']=='decode' or (phase['name']=='qa' and length==1)
                counts['language_decode_tokens' if decode else 'language_prefill_tokens']+=length
        return tick
    hooks=[model.model.visual.register_forward_pre_hook(hook('vision'),with_kwargs=True),model.model.language_model.register_forward_pre_hook(hook('language'),with_kwargs=True)]
    try:
        with (a.out/'measurements.jsonl').open('x',encoding='utf-8') as log:
            for round_id in range(4):
                specs=config()['videos'] if round_id%2==0 else list(reversed(config()['videos']))
                for spec in specs:
                    seq_start=time.perf_counter();decode_start=time.perf_counter();frames=load_frames(spec);decode_seconds=time.perf_counter()-decode_start
                    # Full oracle comparison is outside processing timers for both arms.
                    for size in config()['prefix_frames']:
                        whole,record=identity(model,processor,frames[:size],config()['question'],spec['id'])
                        expected=next(x for x in proof['cases'] if x['case_id']==record['case_id'])
                        if any(record[k]!=expected[k] for k in record):raise AssertionError('Actual GPU input differs from CPU proof')
                        del whole
                    session=session_for(model,processor) if a.arm=='stream' else None
                    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();seq_counts=dict(counts);processing=0.;generated=0
                    try:
                        for size in config()['prefix_frames']:
                            if time.perf_counter()-start>260:raise TimeoutError('Workload budget exceeded')
                            expected=next(x for x in proof['cases'] if x['case_id']==spec['id']+'_'+str(size));prior_status=session.status() if session else None
                            torch.cuda.synchronize();begin=time.perf_counter();update_seconds=0.
                            if session:
                                phase['name']='update';session.append(frames[size-2]);session.append(frames[size-1]);torch.cuda.synchronize()
                                update_seconds=time.perf_counter()-begin
                                qa_before=signature(session)  # CPU digest; no extra GPU cache clone.
                                torch.cuda.synchronize();qa_start=time.perf_counter();phase['name']='qa'
                                answer=session.ask(config()['question'],32);torch.cuda.synchronize();qa_seconds=time.perf_counter()-qa_start
                                elapsed=update_seconds+qa_seconds
                            else:
                                whole=whole_inputs(model,processor,frames[:size],config()['question']);answer,answer_cache=original_qa(model,processor,whole,phase)
                                torch.cuda.synchronize();elapsed=time.perf_counter()-begin;qa_seconds=elapsed
                            processing+=elapsed;generated+=len(answer['generated_ids']);phase['name']='outside'
                            if session:
                                # The video update is legitimate; compare QA to its updated state.
                                if prior_status['encoded_real_frames']!=size-2:raise AssertionError('Unexpected prior prefix')
                                if session.main.seen_tokens!=expected['main_tokens']:raise AssertionError('Main prefix length differs')
                                if signature(session)!=qa_before:raise AssertionError('GPU QA changed main state/counters/RoPE')
                                state=cache_measure(session.main.cache)
                            else:state=cache_measure(answer_cache)
                            row={**expected,**answer,'round':round_id,'warmup':round_id==0,'arm':a.arm,
                                 'processing_seconds':elapsed,'update_seconds':update_seconds,'qa_seconds':qa_seconds,
                                 'cumulative_processing_seconds':processing,'state':state,'main_unchanged_bitwise':True if session else None,
                                 'cuda_allocated_bytes':torch.cuda.memory_allocated(),'cuda_reserved_bytes':torch.cuda.memory_reserved(),
                                 'rss_bytes':psutil.Process().memory_info().rss}
                            report['cases'].append(normal(row));log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
                            if not session:del whole,answer_cache
                            del answer
                        seq_work={k:counts[k]-seq_counts[k] for k in counts}
                        sequence_report={'round':round_id,'warmup':round_id==0,'source_id':spec['id'],'decode_seconds':decode_seconds,
                                         'processing_seconds':processing,'sequence_wall_seconds_including_decode_and_oracle_audits':time.perf_counter()-seq_start,
                                         'pipeline_seconds_decode_plus_processing_excluding_audits':decode_seconds+processing,
                                         'actual_work':seq_work,'processed_visual_frames':42 if a.arm=='original' else 12,
                                         'generated_tokens':generated,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                                         'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,'no_resets':True}
                        report['sequences'].append(sequence_report);write(a.out/'summary.json',report)
                        print(json.dumps({'arm':a.arm,'round':round_id,'source':spec['id'],'processing_seconds':processing,'decode_seconds':decode_seconds,'generated':generated,'work':seq_work}),flush=True)
                    finally:
                        if session:session.close()
                        del frames
        report.update(all_registered_cases_executed=len(report['cases'])==48,actual_forwards=counts,
                      generated_tokens=sum(len(x['generated_ids']) for x in report['cases']),model_source=str(expected_source),
                      model_load_seconds=loaded,wall_seconds_including_load=time.perf_counter()-start)
        if report['generated_tokens']>1536:raise RuntimeError('Generated budget exceeded')
        write(a.out/'summary.json',report)
    finally:
        for h in hooks:h.remove()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['preflight','gpu'],required=True)
    p.add_argument('--arm',choices=['both','original','stream'],required=True);p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--preflight',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-branch',action='append',default=[])
    p.add_argument('--wall-seconds',type=int,default=300);a=p.parse_args()
    if a.wall_seconds!=300 or (a.stage=='preflight')!=(a.arm=='both'):p.error('Registered stage/arm/300s required')
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'tmp').mkdir();tempfile.tempdir=str(a.out/'tmp');torch.set_num_threads(1);torch.manual_seed(42)
    write(a.out/'arguments.json',request(a));write(a.out/'source_hashes.json',sources());write(a.out/'budget.json',budget())
    try:
        (preflight if a.stage=='preflight' else gpu)(a);write(a.out/'status.json',{'status':'completed','stage':a.stage,'arm':a.arm})
    except Exception:
        (a.out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');write(a.out/'status.json',{'status':'failed','stage':a.stage,'arm':a.arm});raise
if __name__=='__main__':main()
