"""Bounded time-indexed frame memory across mixed-state capacity resets."""
import argparse
from collections import deque
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('general_stream_support',HERE.with_name('general_stream_v1')/'suite.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
sys.path.insert(0,str(HERE))
import psutil
import torch
from transformers import AutoProcessor
from memory import TemporalFrameMemory
from streaming.demo import FIELDS,PairDemo
from streaming.online import OnlinePairSession,fp32_reference
from streaming.timeline_inputs import identity
from streaming.window import FramePacket
from scripts.fair_video import stream_answer

read,write_json,file_hash=g.read,g.write_json,g.file_hash
def config():
    c=read(HERE/'cases.json')
    if len(c['orders'])!=3 or any(sorted(x)!=['cat','dog','pigeon'] for x in c['orders']) or c['frames_each']!=8:
        raise ValueError('Three fixed distinct-subject orders/eight-frame segments required')
    if (c['archive_max_pairs'],c['archive_every_pairs'],c['archive_max_bytes'])!=(3,4,884736):raise ValueError('Registered archive budget changed')
    return c
def sources():
    return {**g.sources(),**{str(p.resolve()):file_hash(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.md','.json')}}
def budget():return {'model_loads':1,'stream_vision_max':42,'stream_language_max':600,'original_vision_max':9,'original_language_max':153,
                     'generated_max':144,'answers':9,'sessions':3,'source_frames':72,'answer_tokens':16,'tokens_max':288,'question_reserve':96,
                     'archive_persistent_rgb_max_bytes':884736,'precision':'fp32_ieee_math','wall_seconds':240,
                     'ram_min_gib':3,'gpu_free_min_gib':5.5,'fp32_conversion_free_min_gib':2,'concurrent_models':1}
def request(a):return {'model_dir':str(a.model_dir),'allow_branch':a.allow_branch,'wall_seconds':a.wall_seconds}
def sequence(order):
    c=config();global_index=0
    for source_id in order:
        video=next(x for x in g.config()['videos'] if x['id']==source_id)
        start=c['sample_starts'][source_id];origin=None;count=0
        for i,p in enumerate(g.packets(video)):
            if i<start:continue
            if count>=8:break
            if origin is None:origin=p.timestamp
            stamp=(global_index//8)*8+p.timestamp-origin
            packet=FramePacket(global_index,stamp,p.rgb,global_index+1,p.decode_seconds)
            provenance={'source_id':source_id,'source_frame_id':p.frame_id,'source_pts':p.timestamp,
                        'session_frame_id':global_index,'session_pts':stamp,'rgb_sha256':g.meta((packet,))[0]['rgb_sha256']}
            yield packet,provenance
            global_index+=1;count+=1
        if count!=8:raise ValueError('Source segment lacks eight real frames')
def select_sequence(order,metadata):
    ids=[x['frame_id'] for x in metadata];frames=[];provenance=[]
    if not ids or len(ids)>6 or ids!=sorted(set(ids)):raise ValueError('Ordered <=6 real query frames required')
    for p,trace in sequence(order):
        if p.frame_id in ids:
            expected=metadata[len(frames)]
            if p.timestamp!=expected['timestamp'] or g.meta((p,))[0]['rgb_sha256']!=expected['rgb_sha256']:raise ValueError('Actual historical source/time/pixels changed')
            frames.append(p);provenance.append(trace)
        if p.frame_id>=ids[-1]:break
    if len(frames)!=len(ids):raise ValueError('Missing actual query frames')
    return tuple(frames),provenance
def immutable(row):
    return g.normal({k:row[k] for k in ('case_id','order','route','frames','source_provenance','question','expected_subject',
                                      'memory_from_pts','target_time','all_input_tensors_bitwise_equal',
                                      'tensor_equalities','input_fingerprint','input_tokens','grid')})
def snapshot(session):
    rope=session.model.model.rope_deltas
    return session.status(),session.main.cache.clone(),None if rope is None else rope.clone()
def assert_unchanged(session,before):
    status,cache,rope=before
    if session.status()!=status:raise AssertionError('Main counters changed')
    for name in FIELDS:
        left,right=getattr(cache,name),getattr(session.main.cache,name)
        if len(left)!=len(right) or not all((a is None and b is None) or (a is not None and b is not None and torch.equal(a,b)) for a,b in zip(left,right)):
            raise AssertionError('Main mixed state changed')
    current=session.model.model.rope_deltas
    if not ((rope is None and current is None) or (rope is not None and current is not None and torch.equal(rope,current))):
        raise AssertionError('Model RoPE side state changed')

@torch.inference_mode()
def stream(model,processor,out,cpu_original=None):
    c=config();rows=[];sessions=[];refusals=[]
    with (out/'events.jsonl').open('x',encoding='utf-8') as log:
        for session_id,order in enumerate(c['orders']):
            recent=deque(maxlen=6);traces={};epoch=[];pending=None;start=time.perf_counter()
            bank=TemporalFrameMemory(c['archive_max_pairs'],c['archive_every_pairs'],c['archive_max_bytes'])
            session=OnlinePairSession(model,processor,max_tokens=288,max_question_tokens=96,max_answer_tokens=16,resize_mode='letterbox',video_logits='none')
            def sink(event):
                if event['kind']=='capacity_reset':epoch.clear()
                if event['kind']=='encoded':
                    r=event['record'];epoch.extend(r['frame_ids'][:1 if r['synthetic_padding'] else 2])
                    if len(epoch)>6:raise AssertionError('Main epoch frame bound exceeded')
                log.write(json.dumps({'session_id':session_id,**event},ensure_ascii=False)+'\n');log.flush()
            demo=PairDemo(session,c['current_question'],16,sink,audit=True,first_qa_frames=1000,qa_every_frames=1000)
            try:
                for packet,trace in sequence(order):
                    recent.append(packet);traces[packet.frame_id]=trace;demo.append(packet)
                    if pending is None:pending=packet
                    else:bank.add_pair(pending,packet);pending=None
                demo.finish()
                if demo.received!=24 or demo.resets!=3 or pending is not None:raise AssertionError('Controlled 24-frame/reset schedule changed')
                current_frames=tuple(next(p for p in recent if p.frame_id==i) for i in epoch)
                historical,reason=bank.retrieve_at(c['query_time'])
                if reason!='retained_interval' or [p.frame_id for p in historical]!=[0,1]:raise AssertionError('First real evidence not retained')
                conditions=[('current',current_frames,c['current_question'],order[-1],None),
                            ('history_main',current_frames,c['historical_question'],order[0],c['query_time']),
                            ('history_retrieved',historical,c['historical_question'],order[0],c['query_time'])]
                for route,frames,question,expected,target in conditions:
                    before=snapshot(session)
                    whole,checked=identity(cpu_original or model,processor,frames,question)
                    if cpu_original is not None:
                        with fp32_reference():
                            left=cpu_original(**whole,use_cache=False,logits_to_keep=1).logits
                            right=model(**whole,use_cache=False,logits_to_keep=1).logits
                        if not torch.equal(left,right):raise AssertionError('Tiny original/current fresh prefill differs')
                    begin=time.perf_counter()
                    if route=='history_retrieved':answer=stream_answer(model,processor,frames,question)
                    else:answer=session.ask(question,16)
                    if model.get_input_embeddings().weight.device.type=='cuda':torch.cuda.synchronize()
                    assert_unchanged(session,before)
                    row={'case_id':str(session_id)+'_'+route,'order':order,'route':route,'frames':g.meta(frames),
                         'source_provenance':[traces[p.frame_id] for p in frames],'question':question,'expected_subject':expected,
                         'memory_from_pts':demo.memory_from_pts,'target_time':target,**checked,**answer,
                         'main_unchanged_bitwise':True,'main_counters_unchanged':True,'main_rope_unchanged':True,
                         'query_seconds_including_input_checks':time.perf_counter()-begin,
                         'evidence_for_target_present':route!='history_main','archive':bank.evidence()}
                    rows.append(g.normal(row));print(json.dumps({'id':row['case_id'],'expected':expected,'answer':answer['answer'],'stop':answer['stop_reason']},ensure_ascii=False),flush=True)
                    log.write(json.dumps({'kind':'query','session_id':session_id,'record':row},ensure_ascii=False)+'\n');log.flush();del before,whole
                    write_json(out/'summary.json',{'cases':rows,'sessions':sessions,'refusals':refusals,'scope':'Partial actual run'})
                for stamp,wanted in [(c['missing_query_time'],'not_retained'),(c['future_query_time'],'not_observed')]:
                    before=snapshot(session);found,why=bank.retrieve_at(stamp)
                    if found is not None or why!=wanted:raise AssertionError('Missing/future evidence substituted')
                    assert_unchanged(session,before)
                    refusals.append({'session_id':session_id,'target_time':stamp,'reason':why,'model_called':False,'answer':'Cannot answer from retained frames.'})
                sessions.append({'id':session_id,'order':order,'actual_frames':demo.received,'resets':demo.resets,
                                 'memory_from_pts':demo.memory_from_pts,'archive':bank.evidence(),'wall_seconds':time.perf_counter()-start,
                                 'rss_bytes':psutil.Process().memory_info().rss})
            finally:demo.close();recent.clear();bank.clear()
    report={'cases':rows,'sessions':sessions,'refusals':refusals,'all_registered_cases_executed':len(rows)==9,
            'scope':c['scope'],'architecture':'Bounded CPU real-frame anchors + fresh retrieval branch; primary hybrid state unchanged'}
    write_json(out/'summary.json',report);return report

@torch.inference_mode()
def preflight(a):
    from tests.test_online import small_model
    from test_memory import MemoryTests
    with (a.out/'unittest.log').open('w',encoding='utf-8') as f:r=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MemoryTests))
    write_json(a.out/'cpu_tests.json',{'tests':r.testsRun,'passed':r.wasSuccessful(),'errors':len(r.errors),'failures':len(r.failures)})
    if not r.wasSuccessful():raise AssertionError('Memory causality/ownership/eviction contracts failed')
    assets=g.verify_assets(a.model_dir);processor=AutoProcessor.from_pretrained(a.model_dir,local_files_only=True)
    current=small_model(processor);package=g.original_package()
    original=package.Qwen3_5ForConditionalGeneration(package.Qwen3_5Config(**current.config.to_dict())).eval()
    original.set_attn_implementation('sdpa');original.load_state_dict(current.state_dict(),strict=True)
    report=stream(current,processor,a.out,cpu_original=original);write_json(a.out/'preflight_summary.json',report)
    proof={'request':request(a),'budget':budget(),'source_hashes':sources(),'inputs':g.inputs_assets(),'assets':assets['files'],
           'model_manifest_sha256':file_hash(a.model_dir/'DOWNLOAD_MANIFEST.json'),'cases':[immutable(x) for x in report['cases']],
           'summary_sha256':file_hash(a.out/'preflight_summary.json'),'tests_sha256':file_hash(a.out/'cpu_tests.json'),
           'scope':'Random CPU memory/query/fresh prefill contracts, not pretrained ability'}
    write_json(a.out/'proof.json',proof);print(json.dumps({'passed':True,'memory_tests':r.testsRun,'same_input_queries':len(report['cases']),'refusals':len(report['refusals'])}),flush=True)
def check(a):
    if not a.preflight or read(a.preflight/'status.json')!={'status':'completed','stage':'preflight','arm':'both'}:raise ValueError('Completed fresh CPU proof required')
    p=read(a.preflight/'proof.json')
    if p['request']!=request(a) or p['budget']!=budget() or p['source_hashes']!=sources():raise ValueError('Source/request/budget changed')
    if p['inputs']!=g.inputs_assets() or p['assets']!=g.verify_assets(a.model_dir)['files'] or p['model_manifest_sha256']!=file_hash(a.model_dir/'DOWNLOAD_MANIFEST.json'):
        raise ValueError('Model/input assets changed')
    if p['summary_sha256']!=file_hash(a.preflight/'preflight_summary.json') or p['tests_sha256']!=file_hash(a.preflight/'cpu_tests.json') or not read(a.preflight/'cpu_tests.json')['passed']:
        raise ValueError('CPU evidence changed')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=g.CAP,text=True).strip()
    if not g.branch_allowed(branch,a.allow_branch):raise ValueError('Allowed personal branch required')
    write_json(a.out/'checked_branch.json',{'branch':branch});return p

@torch.inference_mode()
def gpu(a):
    proof=check(a);free,total=torch.cuda.mem_get_info();ram=psutil.virtual_memory().available
    write_json(a.out/'resource_precheck.json',{'gpu_free_bytes':free,'gpu_total_bytes':total,'ram_available_bytes':ram})
    if free<5.5*2**30 or ram<3*2**30:raise RuntimeError('Resource guard refused load')
    if a.arm=='original':
        if not a.stream_run or read(a.stream_run/'status.json')!={'status':'completed','stage':'gpu','arm':'stream'}:raise ValueError('Completed actual memory stream required')
        actual=read(a.stream_run/'summary.json')
        if read(a.stream_run/'source_hashes.json')!=sources() or [immutable(x) for x in actual['cases']]!=proof['cases']:
            raise ValueError('Actual memory stream source/inputs differ')
        cls=g.original_package().Qwen3_5ForConditionalGeneration;expected_source=g.ORIGINAL/'modeling_qwen3_5.py'
    else:
        from qwen35_local import Qwen3_5ForConditionalGeneration
        cls=Qwen3_5ForConditionalGeneration;expected_source=g.ACTIVE/'qwen35_local/modeling_qwen3_5.py'
    start=time.perf_counter();model=cls.from_pretrained(a.model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    if Path(inspect.getfile(type(model))).resolve()!=expected_source:raise AssertionError('Wrong model source')
    if torch.cuda.mem_get_info()[0]<2*2**30:raise RuntimeError('Insufficient conversion room')
    model.float();torch.cuda.reset_peak_memory_stats();processor=AutoProcessor.from_pretrained(a.model_dir,local_files_only=True)
    counts={'vision':0,'language':0}
    def hook(name):
        def tick(_m,_i):
            counts[name]+=1
            if counts[name]>budget()[a.arm+'_'+name+'_max']:raise RuntimeError('Forward budget exceeded')
        return tick
    hooks=[model.model.visual.register_forward_pre_hook(hook('vision')),model.model.language_model.register_forward_pre_hook(hook('language'))]
    try:
        if a.arm=='stream':
            report=stream(model,processor,a.out)
            if [immutable(x) for x in report['cases']]!=proof['cases']:raise AssertionError('Actual input differs from fresh CPU')
        else:
            report={'cases':[],'scope':'Original independent whole prefill over each exact base/retrieved context; retrieval has extra old evidence'}
            for expected,streamed in zip(proof['cases'],actual['cases']):
                if time.perf_counter()-start>200:raise TimeoutError('Workload budget exceeded')
                frames,provenance=select_sequence(expected['order'],expected['frames'])
                whole,checked=identity(model,processor,frames,expected['question'])
                row={**expected,**checked,'source_provenance':provenance}
                if immutable(row)!=expected:raise AssertionError('Actual original input differs from memory branch')
                answer=g.original_answer(model,processor,whole)
                row.update(answer,streaming_answer=streamed['answer'],streaming_generated_ids=streamed['generated_ids'],generated_ids_equal=answer['generated_ids']==streamed['generated_ids'])
                report['cases'].append(g.normal(row));write_json(a.out/'summary.json',report)
                print(json.dumps({'id':row['case_id'],'original':answer['answer'],'stream':streamed['answer'],'equal':row['generated_ids_equal']},ensure_ascii=False),flush=True)
                del whole,frames,answer
            report['all_registered_cases_executed']=len(report['cases'])==9
        generated=sum(len(x['generated_ids']) for x in report['cases'])
        if generated>144:raise RuntimeError('Generated budget exceeded')
        report.update(actual_forwards=counts,generated_tokens=generated,model_source=str(expected_source),
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                      wall_seconds_including_load=time.perf_counter()-start)
        write_json(a.out/'summary.json',report)
    finally:
        for h in hooks:h.remove()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',choices=['preflight','gpu'],required=True)
    p.add_argument('--arm',choices=['both','stream','original'],required=True);p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--preflight',type=Path);p.add_argument('--stream-run',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allow-branch',action='append',default=[]);p.add_argument('--wall-seconds',type=int,default=240)
    a=p.parse_args()
    if a.wall_seconds!=240 or (a.stage=='preflight')!=(a.arm=='both'):p.error('Registered stage/arm/240s required')
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'tmp').mkdir();tempfile.tempdir=str(a.out/'tmp');torch.set_num_threads(1);torch.manual_seed(42)
    write_json(a.out/'arguments.json',request(a));write_json(a.out/'source_hashes.json',sources());write_json(a.out/'budget.json',budget())
    try:
        (preflight if a.stage=='preflight' else gpu)(a);write_json(a.out/'status.json',{'status':'completed','stage':a.stage,'arm':a.arm})
    except Exception:
        (a.out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');write_json(a.out/'status.json',{'status':'failed','stage':a.stage,'arm':a.arm});raise
if __name__=='__main__':main()
