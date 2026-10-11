"""Original-model controls for actual current epochs of a continuous file stream."""
import argparse
import inspect
import json
from pathlib import Path
import subprocess
import tempfile
import time
import traceback
import unittest
import psutil
import torch
from transformers import AutoProcessor
from scripts.baseline import PROJECT,REPO,file_hash,write_json
from scripts.fair_video import original_answer
from scripts.protocol_capability_control import original_package,source_assets,ORIGINAL
from scripts.pair_demo import verify_assets
from scripts.live_demo import source_hashes as stream_sources
from scripts.video_demo import branch_allowed
from streaming.online import fp32_reference
from streaming.playback import SourceDigest
from streaming.preprocessing import fit_canvas
from streaming.timeline_inputs import contexts,select,identity
from streaming.window import FramePacket,read_packets

CONFIG=PROJECT/'configs/timeline_action_cases.json'
PLAN=PROJECT/'连续动作同输入对照方案.md'
LAUNCHER=PROJECT/'scripts/run_timeline_reference.py'
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sources():
    return {**stream_sources(),'@timeline_launcher':file_hash(LAUNCHER),'@timeline_plan':file_hash(PLAN),
            '@timeline_cases':file_hash(CONFIG),**{'@reference:'+n:v for n,v in source_assets().items()}}
def request(args):
    return {k:str(v.resolve()) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('stage','out','preflight','stream_run')}
def budget(args):
    return {'model_loads':1,'cases':7,'frames_per_case_max':6,'vision_max':7,'language_max':119,'generated_max':112,
            'answer_tokens':16,'total_tokens_max':288,'question_reserve':96,'precision':'fp32_ieee_math',
            'wall_seconds':args.wall_seconds,'free_gpu_gib_min':5.5,'available_ram_gib_min':3,'simultaneous_real_models':1}
def config():
    c=read(CONFIG)
    if len(c['cases'])!=7 or [x['sample_index'] for x in c['cases']]!=[3,5,9,11,15,17,21] or c['model_answer_budget']!=16:
        raise ValueError('Registered seven probe configuration changed')
    return c
def source_digest(video):
    digest=SourceDigest();reader=read_packets(video,1)
    try:
        for p in reader:
            if p.timestamp>=23.1:break
            digest.add(FramePacket(p.frame_id,p.timestamp,fit_canvas(p.rgb,(192,256)),p.decoded_count))
            if digest.count>=24:break
    finally:reader.close()
    return digest.snapshot()
def meta(row):return {k:row[k] for k in ('frames','question','timestamp','memory_from_pts')}
def stream_check(args,run,stage):
    c=config();proof=read(args.stream_preflight/'proof.json')
    if read(args.stream_preflight/'status.json')!={'status':'completed','stage':'preflight'} or not read(args.stream_preflight/'cpu_tests.json')['passed']:
        raise ValueError('Completed fresh stream CPU proof required')
    if proof['source_hashes']!=stream_sources() or file_hash(args.stream_preflight/'summary.json')!=proof['summary_sha256']:
        raise ValueError('Stream CPU source or evidence differs')
    if read(run/'status.json')!={'status':'completed','stage':stage} or read(run/'source_hashes.json')!=stream_sources():
        raise ValueError('Actual stream source/stage incomplete')
    if file_hash(args.video)!=c['source_sha256'] or proof['video_sha256']!=c['source_sha256']:
        raise ValueError('Registered source bytes changed')
    r=read(run/'summary.json');a=read(run/'arguments.json')
    expected={'input_kind':'file','video':str(args.video),'model_dir':str(args.model_dir),'fps':1.,'duration':23.1,
              'max_frames':24,'question':c['question'],'answer_tokens':16,'max_tokens':288,'qa_every':2,'first_qa_frames':4,'wall_seconds':90}
    if any(a.get(k)!=v for k,v in expected.items()) or a!=proof['request']:
        raise ValueError('Stream parameters differ from timeline registration')
    if r['source_fingerprint']!=proof['source_fingerprint'] or r['source_fingerprint']!=source_digest(args.video):
        raise ValueError('Complete actual source differs from registered CPU source')
    if r['playback']['dropped_samples'] or r['final']['received_real_frames']!=23 or r['final']['questions']!=7 or r['final']['resets']!=3:
        raise ValueError('Actual stream schedule differs; fixed seven-case scoring suspended')
    rows=contexts(run/'events.jsonl')
    if len(rows)!=7 or any(row['question']!=c['question'] or row['frames'][-1]['frame_id']!=spec['sample_index']*30 or not row['main_unchanged_bitwise'] for row,spec in zip(rows,c['cases'])):
        raise ValueError('Actual answer contexts differ from registration')
    return rows

@torch.inference_mode()
def preflight(args):
    from tests.test_timeline_inputs import TimelineInputTests
    from tests.test_online import small_model
    with (args.out/'unittest.log').open('w',encoding='utf-8') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TimelineInputTests))
    write_json(args.out/'cpu_tests.json',{'tests':result.testsRun,'passed':result.wasSuccessful(),'errors':len(result.errors),'failures':len(result.failures)})
    if not result.wasSuccessful():raise AssertionError('Timeline contracts failed')
    rows=stream_check(args,args.stream_preflight,'preflight')
    assets=verify_assets(args.model_dir)
    processor=AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    current=small_model(processor)
    package=original_package()
    original=package.Qwen3_5ForConditionalGeneration(package.Qwen3_5Config(**current.config.to_dict())).eval()
    original.set_attn_implementation('sdpa');original.load_state_dict(current.state_dict(),strict=True)
    entries=[]
    for row,spec in zip(rows,config()['cases']):
        frames,provenance=select(args.video,row['frames'])
        whole,checked=identity(original,processor,frames,row['question'])
        with fp32_reference():
            left=original(**whole,use_cache=False,logits_to_keep=1).logits
            right=current(**whole,use_cache=False,logits_to_keep=1).logits
            if not torch.equal(left,right):raise AssertionError('Original/modified tiny fresh prefill differs')
        entries.append({'context':meta(row),'frames':provenance,**checked,'expected':spec['expected'],'tiny_fresh_prefill_bitwise_equal':True})
        del frames,whole,left,right
    write_json(args.out/'preflight_summary.json',{'cases':entries,'passed':True,'scope':'Random CPU input and fresh prefill contracts; not semantic accuracy'})
    write_json(args.out/'proof.json',{'request':request(args),'budget':budget(args),'source_hashes':sources(),'assets':assets['files'],
        'model_manifest_sha256':file_hash(args.model_dir/'DOWNLOAD_MANIFEST.json'),'video_sha256':file_hash(args.video),
        'stream_proof_sha256':file_hash(args.stream_preflight/'proof.json'),'stream_events_sha256':file_hash(args.stream_preflight/'events.jsonl'),
        'summary_sha256':file_hash(args.out/'preflight_summary.json'),'cpu_tests_sha256':file_hash(args.out/'cpu_tests.json'),'cases':entries})
    print(json.dumps({'cases':len(entries),'tests':result.testsRun,'passed':True}),flush=True)

def check(args):
    if args.preflight is None or args.stream_run is None or read(args.preflight/'status.json')!={'status':'completed','stage':'preflight'}:
        raise ValueError('Fresh timeline preflight and completed actual stream required')
    p=read(args.preflight/'proof.json')
    if p['request']!=request(args) or p['budget']!=budget(args) or p['source_hashes']!=sources():raise ValueError('Timeline source/parameters/budget differs')
    for name,key in (('preflight_summary.json','summary_sha256'),('cpu_tests.json','cpu_tests_sha256')):
        if file_hash(args.preflight/name)!=p[key]:raise ValueError('Timeline evidence changed')
    if not read(args.preflight/'cpu_tests.json')['passed'] or p['video_sha256']!=file_hash(args.video):raise ValueError('CPU/source invalid')
    if p['assets']!=verify_assets(args.model_dir)['files'] or p['model_manifest_sha256']!=file_hash(args.model_dir/'DOWNLOAD_MANIFEST.json'):
        raise ValueError('Model assets changed')
    if p['stream_proof_sha256']!=file_hash(args.stream_preflight/'proof.json') or p['stream_events_sha256']!=file_hash(args.stream_preflight/'events.jsonl'):
        raise ValueError('Stream CPU evidence changed')
    rows=stream_check(args,args.stream_run,'gpu')
    if [meta(row) for row in rows]!=[e['context'] for e in p['cases']]:raise ValueError('Actual QA epochs differ; no unfair original control')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    if not branch_allowed(branch,args.allow_branch):raise ValueError('Allowed personal branch required')
    write_json(args.out/'checked_branch.json',{'branch':branch})
    return p,rows

@torch.inference_mode()
def gpu(args):
    proof,rows=check(args)
    free,total=torch.cuda.mem_get_info();ram=psutil.virtual_memory().available
    write_json(args.out/'resource_precheck.json',{'gpu_free_bytes':free,'gpu_total_bytes':total,'ram_available_bytes':ram})
    if free<5.5*2**30 or ram<3*2**30:raise RuntimeError('Resource guard refused original model load')
    begin=time.perf_counter()
    model=original_package().Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    if Path(inspect.getfile(type(model))).resolve()!=ORIGINAL/'modeling_qwen3_5.py':raise AssertionError('Original source not loaded')
    if torch.cuda.mem_get_info()[0]<2*2**30:raise RuntimeError('Insufficient FP32 conversion room')
    model.float();torch.cuda.reset_peak_memory_stats()
    processor=AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    counts={'vision':0,'language':0};b=budget(args)
    def hook(name):
        def count(_m,_i):
            counts[name]+=1
            if counts[name]>b[name+'_max']:raise RuntimeError('Registered original forward budget exceeded')
        return count
    hooks=[model.model.visual.register_forward_pre_hook(hook('vision')),model.model.language_model.register_forward_pre_hook(hook('language'))]
    report={'scope':'Original unmodified core over same actual current epoch frames, not full discarded history',
            'budget':b,'model_source':inspect.getfile(type(model)),'cases':[]};generated=0
    try:
        with (args.out/'answers.jsonl').open('x',encoding='utf-8') as stream:
            for row,expected in zip(rows,proof['cases']):
                if time.perf_counter()-begin>=args.wall_seconds:raise TimeoutError('Registered original wall budget exceeded')
                frames,provenance=select(args.video,row['frames']);whole,checked=identity(model,processor,frames,row['question'])
                if {'context':meta(row),'frames':provenance,**checked,'expected':expected['expected'],'tiny_fresh_prefill_bitwise_equal':True}!=expected:
                    raise AssertionError('Actual original input differs from CPU proof')
                start=time.perf_counter();answer=original_answer(model,processor,whole);torch.cuda.synchronize()
                generated+=len(answer['generated_ids'])
                if generated>112:raise RuntimeError('Registered generated budget exceeded')
                record={'context':meta(row),'frames':provenance,**checked,**answer,'expected':expected['expected'],
                    'strict_correct':answer['answer'].strip()==expected['expected'],'streaming_answer':row['answer'],
                    'streaming_generated_ids':row['generated_ids'],'generated_ids_equal':answer['generated_ids']==row['generated_ids'],
                    'streaming_strict_correct':row['answer'].strip()==expected['expected'],'model_seconds':time.perf_counter()-start}
                report['cases'].append(record);stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
                write_json(args.out/'summary.json',report)
                print(json.dumps({'timestamp':row['timestamp'],'original':answer['answer'],'streaming':row['answer'],'expected':expected['expected'],'ids_equal':record['generated_ids_equal']},ensure_ascii=False),flush=True)
                del whole,frames,answer
        report.update(actual_forwards=counts,generated_tokens=generated,all_registered_cases_executed=len(report['cases'])==7,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
            wall_seconds_including_load=time.perf_counter()-begin)
        write_json(args.out/'summary.json',report)
    finally:
        for h in hooks:h.remove()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['preflight','gpu'],required=True)
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--video',type=Path,required=True)
    p.add_argument('--stream-preflight',type=Path,required=True);p.add_argument('--stream-run',type=Path)
    p.add_argument('--preflight',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--wall-seconds',type=int,default=90);p.add_argument('--allow-branch',action='append',default=[])
    args=p.parse_args()
    if not 1<=args.wall_seconds<=180:p.error('Wall budget 1..180 required')
    for k,v in vars(args).items():
        if isinstance(v,Path):setattr(args,k,v.resolve())
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'tmp').mkdir();tempfile.tempdir=str(args.out/'tmp')
    torch.set_num_threads(1);torch.manual_seed(42)
    write_json(args.out/'arguments.json',request(args));write_json(args.out/'source_hashes.json',sources());write_json(args.out/'budget.json',budget(args))
    try:
        (preflight if args.stage=='preflight' else gpu)(args);write_json(args.out/'status.json',{'status':'completed','stage':args.stage})
    except Exception:
        (args.out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');write_json(args.out/'status.json',{'status':'failed','stage':args.stage});raise

if __name__=='__main__':main()
