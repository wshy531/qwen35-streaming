"""Generic continuous file suite and independent original current-epoch controls."""
import argparse
from collections import deque
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

HERE=Path(__file__).resolve().parent
CAP=HERE.parents[1]
ACTIVE=CAP/'code/qwen35-streaming-s4'
sys.path.insert(0,str(ACTIVE))
import psutil
import torch
from transformers import AutoProcessor
from scripts.baseline import file_hash,write_json
from scripts.pair_demo import verify_assets
from scripts.protocol_capability_control import original_package,source_assets,ORIGINAL
from scripts.fair_video import original_answer
from scripts.video_demo import branch_allowed
from streaming.demo import PairDemo
from streaming.online import OnlinePairSession,fp32_reference
from streaming.timeline_inputs import identity,select
from streaming.preprocessing import fit_canvas
from streaming.window import FramePacket,read_packets

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def normal(value):return json.loads(json.dumps(value))
def config():
    c=read(HERE/'cases.json')
    if c['fps']!=1 or c['max_frames']!=48 or c['duration']!=60 or c['answer_tokens']!=16 or c['max_tokens']!=288 or c['reserve']!=96:
        raise ValueError('Registered bounds changed')
    if len(c['videos'])!=3 or len({x['id'] for x in c['videos']})!=3:raise ValueError('Three distinct registered sources required')
    return c
def sources():
    return {**{str(p.resolve()):file_hash(p) for p in sorted(ACTIVE.rglob('*.py'))},**source_assets(),
            **{str(p.resolve()):file_hash(p) for p in sorted(HERE.iterdir()) if p.suffix in ('.py','.md','.json')}}
def budget():return {'model_loads':1,'stream_vision_max':72,'stream_language_max':1000,'original_vision_max':48,'original_language_max':816,
                     'generated_max':768,'answers_max':48,'source_frames_max':144,'source_frames_each_max':48,'history_frames_max':6,
                     'answer_tokens':16,'tokens_max':288,'question_reserve':96,'precision':'fp32_ieee_math','wall_seconds':240,
                     'ram_min_gib':3,'gpu_free_min_gib':5.5,'fp32_conversion_free_min_gib':2,'concurrent_models':1}
def request(a):return {'model_dir':str(a.model_dir),'allow_branch':a.allow_branch,'wall_seconds':a.wall_seconds}
def inputs_assets():
    result={}
    for spec in config()['videos']:
        p=Path(spec['path']);digest=file_hash(p)
        if digest!=spec['sha256']:raise ValueError('Registered video bytes changed')
        result[str(p.resolve())]=digest
    return result
def packets(spec):
    reader=read_packets(Path(spec['path']),1)
    try:
        for i,p in enumerate(reader):
            if i>=48 or p.timestamp>=60:break
            rgb=fit_canvas(p.rgb,(192,256));rgb.flags.writeable=False
            yield FramePacket(p.frame_id,p.timestamp,rgb,p.decoded_count,p.decode_seconds)
    finally:reader.close()
def meta(frames):
    return [{'frame_id':p.frame_id,'timestamp':p.timestamp,'rgb_sha256':hashlib.sha256(p.rgb.tobytes()).hexdigest()} for p in frames]
def immutable(record):
    keys=('case_id','source_id','frames','question','memory_from_pts','timestamp','all_input_tensors_bitwise_equal',
          'tensor_equalities','input_fingerprint','input_tokens','grid')
    return normal({k:record[k] for k in keys})
def check_cpu(a):
    if not a.preflight or read(a.preflight/'status.json')!={'status':'completed','stage':'preflight','arm':'both'}:
        raise ValueError('Completed fresh CPU proof required')
    proof=read(a.preflight/'proof.json')
    if proof['request']!=request(a) or proof['budget']!=budget() or proof['source_hashes']!=sources():raise ValueError('Source/request/budget changed')
    if proof['inputs']!=inputs_assets() or proof['assets']!=verify_assets(a.model_dir)['files'] or proof['model_manifest_sha256']!=file_hash(a.model_dir/'DOWNLOAD_MANIFEST.json'):
        raise ValueError('Input/model assets changed')
    for name,key in [('preflight_summary.json','summary_sha256'),('cpu_tests.json','tests_sha256')]:
        if file_hash(a.preflight/name)!=proof[key]:raise ValueError('CPU evidence changed')
    if not read(a.preflight/'cpu_tests.json')['passed']:raise ValueError('CPU tests did not pass')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=CAP,text=True).strip()
    if not branch_allowed(branch,a.allow_branch):raise ValueError('Allowed personal branch required')
    write_json(a.out/'checked_branch.json',{'branch':branch})
    return proof

@torch.inference_mode()
def stream(model,processor,out,paced=False,cpu_original=None):
    c=config();entries=[];source_reports=[];started=time.perf_counter()
    event_file=(out/'events.jsonl').open('x',encoding='utf-8')
    try:
        for spec in c['videos']:
            recent=deque(maxlen=6);epoch=[];dig=hashlib.sha256();source_count=0;append_times=[]
            session=OnlinePairSession(model,processor,max_tokens=288,max_question_tokens=96,max_answer_tokens=16,resize_mode='letterbox',video_logits='none')
            source_start=time.perf_counter()
            def sink(event):
                event={'source_id':spec['id'],**event}
                if event['kind']=='capacity_reset':epoch.clear()
                if event['kind']=='encoded':
                    r=event['record'];ids=r['frame_ids'][:1 if r['synthetic_padding'] else 2]
                    epoch.extend(ids)
                    if len(epoch)>6:raise AssertionError('Current memory exceeded six-frame bound')
                if event['kind']=='answer':
                    frames=tuple(next(p for p in recent if p.frame_id==fid) for fid in epoch)
                    whole,checked=identity(cpu_original or model,processor,frames,c['question'])
                    if cpu_original is not None:
                        with fp32_reference():
                            left=cpu_original(**whole,use_cache=False,logits_to_keep=1).logits
                            right=model(**whole,use_cache=False,logits_to_keep=1).logits
                        if not torch.equal(left,right):raise AssertionError('Tiny original/current fresh prefill differs')
                    answer=event['answer']
                    row={'case_id':spec['id']+'_'+str(len([x for x in entries if x['source_id']==spec['id']])),
                         'source_id':spec['id'],'frames':meta(frames),'question':c['question'],
                         'memory_from_pts':event['status']['memory_from_pts'],'timestamp':answer['timestamp'],**checked,
                         **answer,'main_unchanged_bitwise':event['main_unchanged_bitwise'],
                         'main_counters_unchanged':event['main_counters_unchanged'],
                         'answer_available_wall_seconds':time.perf_counter()-source_start,
                         'answer_lag_seconds':time.perf_counter()-source_start-answer['timestamp'] if paced else None,
                         'cuda_allocated_bytes':torch.cuda.memory_allocated() if paced else None,
                         'rss_bytes':psutil.Process().memory_info().rss}
                    if not row['main_unchanged_bitwise'] or not row['main_counters_unchanged']:raise AssertionError('Main state changed during QA')
                    entries.append(normal(row));del whole
                    print(json.dumps({'source':spec['id'],'pts':answer['timestamp'],'answer':answer['answer'],'stop':answer['stop_reason']},ensure_ascii=False),flush=True)
                    write_json(out/'summary.json',{'cases':entries,'sources':source_reports,'scope':'Partial run; only actual recorded cases'})
                event_file.write(json.dumps(event,ensure_ascii=False)+'\n');event_file.flush()
            demo=PairDemo(session,c['question'],16,sink,audit=True,first_qa_frames=4,qa_every_frames=2)
            try:
                for packet in packets(spec):
                    if time.perf_counter()-started>200:raise TimeoutError('Stream workload budget exceeded')
                    if paced:
                        delay=packet.timestamp-(time.perf_counter()-source_start)
                        if delay>0:time.sleep(delay)
                    recent.append(packet);source_count+=1
                    dig.update(json.dumps(meta((packet,)),sort_keys=True).encode('utf-8'))
                    begin=time.perf_counter();demo.append(packet)
                    if paced:torch.cuda.synchronize()
                    append_times.append(time.perf_counter()-begin)
                demo.finish()
                if source_count!=spec['expected_source_frames']:raise AssertionError('Actual source EOF/count differs from registration')
                source_reports.append({'id':spec['id'],'actual_frames':source_count,'frame_digest':dig.hexdigest(),'final':demo.status(),
                                       'wall_seconds':time.perf_counter()-source_start,'append_seconds_including_qa':sum(append_times),
                                       'paced':paced,'dropped_samples':0,'queue_policy':'Sequential bounded input, no asynchronous producer'})
            finally:demo.close();recent.clear()
    finally:event_file.close()
    if len(entries)>48 or sum(len(x['generated_ids']) for x in entries)>768:raise RuntimeError('Answer budget exceeded')
    report={'cases':entries,'sources':source_reports,'all_registered_sources_executed':len(source_reports)==3,
            'scope':'Three fixed short sources; generic current-epoch QA, not a benchmark or long-memory proof'}
    write_json(out/'summary.json',report);return report

@torch.inference_mode()
def preflight(a):
    from tests.test_timeline_inputs import TimelineInputTests
    from tests.test_online import small_model
    with (a.out/'unittest.log').open('w',encoding='utf-8') as f:
        result=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TimelineInputTests))
    write_json(a.out/'cpu_tests.json',{'tests':result.testsRun,'passed':result.wasSuccessful(),'errors':len(result.errors),'failures':len(result.failures)})
    if not result.wasSuccessful():raise AssertionError('Epoch contracts failed')
    asset=verify_assets(a.model_dir);processor=AutoProcessor.from_pretrained(a.model_dir,local_files_only=True)
    current=small_model(processor);package=original_package()
    original=package.Qwen3_5ForConditionalGeneration(package.Qwen3_5Config(**current.config.to_dict())).eval()
    original.set_attn_implementation('sdpa');original.load_state_dict(current.state_dict(),strict=True)
    report=stream(current,processor,a.out,cpu_original=original)
    proof={'request':request(a),'budget':budget(),'source_hashes':sources(),'inputs':inputs_assets(),'assets':asset['files'],
           'model_manifest_sha256':file_hash(a.model_dir/'DOWNLOAD_MANIFEST.json'),
           'cases':[immutable(x) for x in report['cases']],
           'source_digests':{x['id']:x['frame_digest'] for x in report['sources']},
           'summary_sha256':file_hash(a.out/'summary.json'),'tests_sha256':file_hash(a.out/'cpu_tests.json'),
           'scope':'Random CPU state/input/fresh prefill contracts, no semantic evidence'}
    write_json(a.out/'preflight_summary.json',report)
    proof['summary_sha256']=file_hash(a.out/'preflight_summary.json')
    write_json(a.out/'proof.json',proof)
    print(json.dumps({'passed':True,'epoch_tests':result.testsRun,'qa_inputs':len(report['cases']),'source_frames':sum(x['actual_frames'] for x in report['sources'])}),flush=True)

@torch.inference_mode()
def gpu(a):
    proof=check_cpu(a);free,total=torch.cuda.mem_get_info();ram=psutil.virtual_memory().available
    write_json(a.out/'resource_precheck.json',{'gpu_free_bytes':free,'gpu_total_bytes':total,'ram_available_bytes':ram})
    if free<5.5*2**30 or ram<3*2**30:raise RuntimeError('Resource guard refused load')
    if a.arm=='original':
        if not a.stream_run or read(a.stream_run/'status.json')!={'status':'completed','stage':'gpu','arm':'stream'}:raise ValueError('Completed actual stream required')
        actual=read(a.stream_run/'summary.json')
        if [immutable(x) for x in actual['cases']]!=proof['cases']:raise ValueError('Stream contexts differ from fresh CPU proof')
        if read(a.stream_run/'source_hashes.json')!=sources():raise ValueError('Actual stream source changed')
        cls=original_package().Qwen3_5ForConditionalGeneration;expected_source=ORIGINAL/'modeling_qwen3_5.py'
    else:
        from qwen35_local import Qwen3_5ForConditionalGeneration
        cls=Qwen3_5ForConditionalGeneration;expected_source=ACTIVE/'qwen35_local/modeling_qwen3_5.py'
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
            report=stream(model,processor,a.out,paced=True)
            if [immutable(x) for x in report['cases']]!=proof['cases'] or {x['id']:x['frame_digest'] for x in report['sources']}!=proof['source_digests']:
                raise AssertionError('Actual source/QA input differs from CPU proof')
        else:
            report={'cases':[],'scope':'Original unmodified whole prefill over each actual current epoch, not full discarded history'}
            with (a.out/'answers.jsonl').open('x',encoding='utf-8') as f:
                for expected,streamed in zip(proof['cases'],actual['cases']):
                    if time.perf_counter()-start>200:raise TimeoutError('Original workload budget exceeded')
                    spec=next(x for x in config()['videos'] if x['id']==expected['source_id'])
                    frames,provenance=select(Path(spec['path']),expected['frames'])
                    whole,checked=identity(model,processor,frames,expected['question'])
                    record={**expected,**checked,'frames':provenance}
                    if immutable(record)!=expected:raise AssertionError('Original input differs from CPU/actual stream')
                    begin=time.perf_counter();answer=original_answer(model,processor,whole);torch.cuda.synchronize()
                    record.update(answer,model_seconds=time.perf_counter()-begin,streaming_answer=streamed['answer'],
                                  streaming_generated_ids=streamed['generated_ids'],generated_ids_equal=answer['generated_ids']==streamed['generated_ids'])
                    report['cases'].append(normal(record));f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();write_json(a.out/'summary.json',report)
                    print(json.dumps({'case':record['case_id'],'original':answer['answer'],'stream':streamed['answer'],'equal':record['generated_ids_equal']},ensure_ascii=False),flush=True)
                    del frames,whole,answer
            report['all_registered_sources_executed']=len(report['cases'])==len(proof['cases'])
        generated=sum(len(x['generated_ids']) for x in report['cases'])
        if generated>768:raise RuntimeError('Generated budget exceeded')
        report.update(actual_forwards=counts,generated_tokens=generated,model_source=str(expected_source),
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                      wall_seconds_including_load=time.perf_counter()-start)
        write_json(a.out/'summary.json',report)
    finally:
        for h in hooks:h.remove()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['preflight','gpu'],required=True);p.add_argument('--arm',choices=['both','stream','original'],required=True)
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--preflight',type=Path);p.add_argument('--stream-run',type=Path)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-branch',action='append',default=[]);p.add_argument('--wall-seconds',type=int,default=240)
    a=p.parse_args()
    if a.wall_seconds!=240 or (a.stage=='preflight')!=(a.arm=='both'):p.error('Registered stage/arm and 240s bound required')
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'tmp').mkdir();tempfile.tempdir=str(a.out/'tmp')
    torch.set_num_threads(1);torch.manual_seed(42)
    write_json(a.out/'arguments.json',request(a));write_json(a.out/'source_hashes.json',sources());write_json(a.out/'budget.json',budget())
    try:
        (preflight if a.stage=='preflight' else gpu)(a)
        write_json(a.out/'status.json',{'status':'completed','stage':a.stage,'arm':a.arm})
    except Exception:
        (a.out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');write_json(a.out/'status.json',{'status':'failed','stage':a.stage,'arm':a.arm});raise
if __name__=='__main__':main()
