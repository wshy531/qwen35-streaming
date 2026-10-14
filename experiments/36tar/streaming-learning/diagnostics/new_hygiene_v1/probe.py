"""Portable original-model fixed-frame category-order and wording controls."""
import argparse,inspect,json,subprocess,sys,tempfile,time,traceback,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
CAP=next((p for p in HERE.parents if p.name=='streaming-learning' and (p/'code/qwen35-streaming-s4').is_dir()),None)
if CAP is None:CAP=HERE.parents[1]/'publish/qwen35-streaming/experiments/36tar/streaming-learning'
ACTIVE=CAP/'code/qwen35-streaming-s4'
sys.path.insert(0,str(ACTIVE))
import psutil,torch
from transformers import AutoProcessor
from scripts.baseline import file_hash,write_json
from scripts.pair_demo import verify_assets
from scripts.protocol_capability_control import original_package,source_assets,ORIGINAL
from scripts.fair_video import original_answer
from scripts.video_demo import branch_allowed
from streaming.timeline_inputs import select,identity
from streaming.online import fp32_reference

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sources():
    s={str(p.resolve()):file_hash(p) for p in sorted(ACTIVE.rglob('*.py'))}
    s.update(source_assets())
    s.update({str(p.resolve()):file_hash(p) for p in (HERE/'probe.py',HERE/'run_probe.py',HERE/'方案.md',HERE/'cases.json')})
    return s
def budget():return {'model_loads':1,'cases':6,'vision_max':6,'language_max':102,'generated_max':96,'answer_tokens':16,'input_with_answer_max':288,'frames_per_case':2,'precision':'fp32_ieee_math','wall_seconds':90,'available_ram_gib_min':3,'free_gpu_gib_min':5.5}
def request(a):return {'model_dir':str(a.model_dir),'video':str(a.video),'wall_seconds':a.wall_seconds,'allow_branch':a.allow_branch}
def specs():
    config=read(HERE/'cases.json')
    if [p['id'] for p in config['pairs']]!=['standing','dispensing','rubbing'] or [q['id'] for q in config['questions']]!=['numeric','action'] or any(len(p['frames'])!=2 for p in config['pairs']):raise ValueError('Registered new-source three-by-two cases changed')
    return [{**p,'question_id':q['id'],'question':q['text'],'case_id':p['id']+'_'+q['id']} for p in config['pairs'] for q in config['questions']]
def inputs(model,processor,a,spec):
    metadata=spec['frames']
    frames,provenance=select(a.video,metadata);whole,check=identity(model,processor,frames,spec['question'])
    return whole,{'case_id':spec['case_id'],'question':spec['question'],'question_id':spec['question_id'],'frames':provenance,**check,'expected':spec['expected_by_question'][spec['question_id']],'rubric':spec['rubric']}
def asset_check(a):
    if file_hash(a.video)!=read(HERE/'cases.json')['source_sha256']:raise ValueError('Source bytes changed')
    return verify_assets(a.model_dir)
@torch.inference_mode()
def preflight(a):
    from tests.test_timeline_inputs import TimelineInputTests
    from tests.test_online import small_model
    with (a.out/'unittest.log').open('w',encoding='utf-8') as f:r=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TimelineInputTests))
    write_json(a.out/'cpu_tests.json',{'tests':r.testsRun,'passed':r.wasSuccessful(),'errors':len(r.errors),'failures':len(r.failures)})
    if not r.wasSuccessful():raise AssertionError('Epoch contracts failed')
    assets=asset_check(a);processor=AutoProcessor.from_pretrained(a.model_dir,local_files_only=True);current=small_model(processor)
    package=original_package();original=package.Qwen3_5ForConditionalGeneration(package.Qwen3_5Config(**current.config.to_dict())).eval()
    original.set_attn_implementation('sdpa');original.load_state_dict(current.state_dict(),strict=True)
    records=[];pixels={}
    for spec in specs():
        whole,record=inputs(original,processor,a,spec)
        with fp32_reference():
            left=original(**whole,use_cache=False,logits_to_keep=1).logits;right=current(**whole,use_cache=False,logits_to_keep=1).logits
            if not torch.equal(left,right):raise AssertionError('Tiny fresh prefill differs')
        visual=whole.pixel_values_videos.clone()
        if spec['id'] in pixels and not torch.equal(pixels[spec['id']],visual):raise AssertionError('Question changed source pixels')
        pixels[spec['id']]=visual;records.append(record);del whole,left,right
    write_json(a.out/'preflight_summary.json',{'passed':True,'cases':records,'same_group_pixels_equal':True,'tiny_fresh_prefill_bitwise_equal':True,'scope':'CPU random models, not semantic evidence'})
    write_json(a.out/'proof.json',{'request':request(a),'budget':budget(),'source_hashes':sources(),'assets':assets['files'],'model_manifest_sha256':file_hash(a.model_dir/'DOWNLOAD_MANIFEST.json'),'video_sha256':file_hash(a.video),'cases':records,'summary_sha256':file_hash(a.out/'preflight_summary.json'),'cpu_tests_sha256':file_hash(a.out/'cpu_tests.json')})
    print(json.dumps({'passed':True,'epoch_tests':r.testsRun,'independent_inputs':len(records)}),flush=True)
def check(a):
    p=read(a.preflight/'proof.json')
    if read(a.preflight/'status.json')!={'status':'completed','stage':'preflight'}:raise ValueError('Completed fresh proof required')
    if p['request']!=request(a) or p['source_hashes']!=sources() or p['budget']!=budget():raise ValueError('Source/request/budget changed')
    if p['assets']!=asset_check(a)['files'] or p['model_manifest_sha256']!=file_hash(a.model_dir/'DOWNLOAD_MANIFEST.json'):raise ValueError('Assets changed')
    if p['summary_sha256']!=file_hash(a.preflight/'preflight_summary.json') or p['cpu_tests_sha256']!=file_hash(a.preflight/'cpu_tests.json') or not read(a.preflight/'cpu_tests.json')['passed']:raise ValueError('CPU evidence changed')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=CAP,text=True).strip()
    if not branch_allowed(branch,a.allow_branch):raise ValueError('Registered personal branch required')
    write_json(a.out/'checked_branch.json',{'branch':branch});return p
@torch.inference_mode()
def gpu(a):
    proof=check(a);free,total=torch.cuda.mem_get_info();ram=psutil.virtual_memory().available
    write_json(a.out/'resource_precheck.json',{'gpu_free_bytes':free,'gpu_total_bytes':total,'ram_available_bytes':ram})
    if free<5.5*2**30 or ram<3*2**30:raise RuntimeError('Resource guard refused original model load')
    start=time.perf_counter();model=original_package().Qwen3_5ForConditionalGeneration.from_pretrained(a.model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    if Path(inspect.getfile(type(model))).resolve()!=ORIGINAL/'modeling_qwen3_5.py':raise AssertionError('Frozen original class required')
    if torch.cuda.mem_get_info()[0]<2*2**30:raise RuntimeError('Insufficient FP32 conversion room')
    model.float();torch.cuda.reset_peak_memory_stats();processor=AutoProcessor.from_pretrained(a.model_dir,local_files_only=True)
    counts={'vision':0,'language':0};generated=0
    def hook(name):
        def tick(_m,_i):
            counts[name]+=1
            if counts[name]>budget()[name+'_max']:raise RuntimeError('Forward budget exceeded')
        return tick
    hooks=[model.model.visual.register_forward_pre_hook(hook('vision')),model.model.language_model.register_forward_pre_hook(hook('language'))]
    report={'scope':'Fixed original two-frame question controls; not streaming/long-term memory/benchmark','cases':[],'model_source':inspect.getfile(type(model))}
    try:
        with (a.out/'answers.jsonl').open('x',encoding='utf-8') as f:
            for spec,expected in zip(specs(),proof['cases']):
                if time.perf_counter()-start>90:raise TimeoutError('Wall budget exceeded')
                whole,record=inputs(model,processor,a,spec)
                if record!=expected:raise AssertionError('Actual input differs from CPU proof')
                answer=original_answer(model,processor,whole);generated+=len(answer['generated_ids'])
                if generated>96:raise RuntimeError('Generated budget exceeded')
                record.update(answer,strict_choice_correct=None if record['expected'] is None else answer['answer'].strip()==record['expected'])
                report['cases'].append(record);f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();write_json(a.out/'summary.json',report)
                print(json.dumps({'id':record['case_id'],'answer':record['answer'],'stop':record['stop_reason']},ensure_ascii=False),flush=True);del whole,answer
        report.update(all_registered_cases_executed=len(report['cases'])==6,actual_forwards=counts,generated_tokens=generated,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,wall_seconds_including_load=time.perf_counter()-start)
        write_json(a.out/'summary.json',report)
    finally:
        for h in hooks:h.remove()
def main():
    p=argparse.ArgumentParser();p.add_argument('--allow-branch',action='append',default=[]);p.add_argument('--stage',choices=['preflight','gpu'],required=True);p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--video',type=Path,required=True);p.add_argument('--preflight',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--wall-seconds',type=int,default=90)
    a=p.parse_args()
    if a.wall_seconds!=90 or (a.stage=='gpu' and a.preflight is None):p.error('Registered90s and fresh CPU proof required')
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'tmp').mkdir();tempfile.tempdir=str(a.out/'tmp');torch.set_num_threads(1);torch.manual_seed(42)
    write_json(a.out/'arguments.json',request(a));write_json(a.out/'budget.json',budget());write_json(a.out/'source_hashes.json',sources())
    try:(preflight if a.stage=='preflight' else gpu)(a);write_json(a.out/'status.json',{'status':'completed','stage':a.stage})
    except Exception:
        (a.out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');write_json(a.out/'status.json',{'status':'failed','stage':a.stage});raise
if __name__=='__main__':main()
