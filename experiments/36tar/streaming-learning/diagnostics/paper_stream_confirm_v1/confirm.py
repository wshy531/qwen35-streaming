"""Three same-input numeric probes on the unchanged streaming implementation."""
import argparse,importlib.util,inspect,json,subprocess,tempfile,time,traceback,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('paper_choice_reference',HERE.with_name('paper_choice_v1')/'probe.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
import psutil,torch
from transformers import AutoProcessor
from scripts.fair_video import stream_answer
from streaming.timeline_inputs import select
from streaming.online import fp32_reference
def read(f):return json.loads(f.read_text(encoding='utf-8-sig'))
def sources():return {**p.sources(),**{str(f.resolve()):p.file_hash(f) for f in (HERE/'confirm.py',HERE/'run_confirm.py',HERE/'方案.md')}}
def budget():return {'model_loads':1,'cases':3,'vision_max':3,'language_max':63,'generated_max':48,'answer_tokens':16,'main_tokens_max':256,'question_reserve':96,'input_with_answer_max':288,'frames_per_case':2,'precision':'fp32_ieee_math','wall_seconds':90,'available_ram_gib_min':3,'free_gpu_gib_min':5.5}
def request(a):return {'model_dir':str(a.model_dir),'video':str(a.video),'original_run':str(a.original_run),'wall_seconds':a.wall_seconds,'allow_branch':a.allow_branch}
def choices():return [s for s in p.specs() if s['question_id']=='numeric']
def frames_for(a,s):return select(a.video,[{'frame_id':i*30,'timestamp':i*1.001} for i in s['indices']])[0]
def original_records(a):
    if read(a.original_run/'status.json')!={'status':'completed','stage':'gpu'}:raise ValueError('Completed original controls required')
    r=read(a.original_run/'summary.json')
    if not r['all_registered_cases_executed']:raise ValueError('Original controls incomplete')
    rows=[x for x in r['cases'] if x['question_id']=='numeric']
    if len(rows)!=3:raise ValueError('Three registered numeric originals required')
    return rows
def equal_input(x,y):
    keys=('case_id','question','question_id','frames','all_input_tensors_bitwise_equal','tensor_equalities','input_fingerprint','input_tokens','grid','expected','rubric')
    if any(x[k]!=y[k] for k in keys):raise AssertionError('Streaming control sees different input')
@torch.inference_mode()
def preflight(a):
    from tests.test_timeline_inputs import TimelineInputTests
    from tests.test_online import small_model
    with (a.out/'unittest.log').open('w',encoding='utf-8') as f:r=unittest.TextTestRunner(stream=f,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TimelineInputTests))
    p.write_json(a.out/'cpu_tests.json',{'tests':r.testsRun,'passed':r.wasSuccessful(),'errors':len(r.errors),'failures':len(r.failures)})
    if not r.wasSuccessful():raise AssertionError('Epoch contracts failed')
    assets=p.asset_check(a);processor=AutoProcessor.from_pretrained(a.model_dir,local_files_only=True);current=small_model(processor)
    package=p.original_package();original=package.Qwen3_5ForConditionalGeneration(package.Qwen3_5Config(**current.config.to_dict())).eval();original.set_attn_implementation('sdpa');original.load_state_dict(current.state_dict(),strict=True)
    originals=original_records(a);entries=[]
    for s,expected in zip(choices(),originals):
        whole,record=p.inputs(original,processor,a,s);equal_input(record,expected)
        with fp32_reference():
            if not torch.equal(original(**whole,use_cache=False,logits_to_keep=1).logits,current(**whole,use_cache=False,logits_to_keep=1).logits):raise AssertionError('Tiny fresh prefill differs')
        answer=stream_answer(current,processor,frames_for(a,s),s['question'])
        if not answer['main_unchanged_bitwise'] or not answer['main_counters_unchanged'] or answer['resets']!=0:raise AssertionError('Streaming QA changed main state')
        entries.append(record);del whole,answer
    p.write_json(a.out/'preflight_summary.json',{'passed':True,'cases':entries,'tiny_fresh_prefill_equal':True,'tiny_streaming_main_unchanged':True,'scope':'Random CPU contracts, not semantic evidence'})
    p.write_json(a.out/'proof.json',{'request':request(a),'budget':budget(),'source_hashes':sources(),'assets':assets['files'],'model_manifest_sha256':p.file_hash(a.model_dir/'DOWNLOAD_MANIFEST.json'),'original_summary_sha256':p.file_hash(a.original_run/'summary.json'),'video_sha256':p.file_hash(a.video),'cases':entries,'summary_sha256':p.file_hash(a.out/'preflight_summary.json'),'cpu_tests_sha256':p.file_hash(a.out/'cpu_tests.json')})
    print(json.dumps({'passed':True,'epoch_tests':r.testsRun,'same_input_cases':3,'streaming_main_unchanged':True}),flush=True)
def check(a):
    proof=read(a.preflight/'proof.json')
    if read(a.preflight/'status.json')!={'status':'completed','stage':'preflight'} or proof['request']!=request(a) or proof['budget']!=budget() or proof['source_hashes']!=sources():raise ValueError('Fresh streaming proof/source required')
    if proof['assets']!=p.asset_check(a)['files'] or proof['model_manifest_sha256']!=p.file_hash(a.model_dir/'DOWNLOAD_MANIFEST.json') or proof['original_summary_sha256']!=p.file_hash(a.original_run/'summary.json'):raise ValueError('Assets/original results changed')
    if proof['summary_sha256']!=p.file_hash(a.preflight/'preflight_summary.json') or proof['cpu_tests_sha256']!=p.file_hash(a.preflight/'cpu_tests.json') or not read(a.preflight/'cpu_tests.json')['passed']:raise ValueError('CPU evidence changed')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=p.CAP,text=True).strip()
    if not p.branch_allowed(branch,a.allow_branch):raise ValueError('Personal branch required')
    p.write_json(a.out/'checked_branch.json',{'branch':branch});return proof
@torch.inference_mode()
def gpu(a):
    proof=check(a);free,total=torch.cuda.mem_get_info();ram=psutil.virtual_memory().available
    p.write_json(a.out/'resource_precheck.json',{'gpu_free_bytes':free,'gpu_total_bytes':total,'ram_available_bytes':ram})
    if free<5.5*2**30 or ram<3*2**30:raise RuntimeError('Resource guard refused streaming model load')
    from qwen35_local import Qwen3_5ForConditionalGeneration
    start=time.perf_counter();model=Qwen3_5ForConditionalGeneration.from_pretrained(a.model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    if Path(inspect.getfile(type(model))).resolve()!=p.ACTIVE/'qwen35_local/modeling_qwen3_5.py':raise AssertionError('Unchanged current source required')
    if torch.cuda.mem_get_info()[0]<2*2**30:raise RuntimeError('Insufficient FP32 conversion room')
    model.float();torch.cuda.reset_peak_memory_stats();processor=AutoProcessor.from_pretrained(a.model_dir,local_files_only=True)
    counts={'vision':0,'language':0};generated=0
    def hook(name):
        def tick(_m,_i):
            counts[name]+=1
            if counts[name]>budget()[name+'_max']:raise RuntimeError('Forward budget exceeded')
        return tick
    hooks=[model.model.visual.register_forward_pre_hook(hook('vision')),model.model.language_model.register_forward_pre_hook(hook('language'))]
    originals=original_records(a);report={'scope':'Three fresh two-frame streaming controls, not continuous video or general accuracy','cases':[],'model_source':inspect.getfile(type(model))}
    try:
        with (a.out/'answers.jsonl').open('x',encoding='utf-8') as f:
            for s,expected,original in zip(choices(),proof['cases'],originals):
                if time.perf_counter()-start>90:raise TimeoutError('Wall budget exceeded')
                whole,record=p.inputs(model,processor,a,s);equal_input(record,expected);equal_input(record,original)
                answer=stream_answer(model,processor,frames_for(a,s),s['question']);generated+=len(answer['generated_ids'])
                if generated>48:raise RuntimeError('Generated budget exceeded')
                if not answer['main_unchanged_bitwise'] or not answer['main_counters_unchanged'] or answer['resets']!=0:raise AssertionError('Main state changed')
                record.update(answer,strict_choice_correct=answer['answer'].strip()==record['expected'],original_answer=original['answer'],original_generated_ids=original['generated_ids'],generated_ids_equal=answer['generated_ids']==original['generated_ids'])
                report['cases'].append(record);f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();p.write_json(a.out/'summary.json',report)
                print(json.dumps({'id':record['case_id'],'answer':record['answer'],'original':original['answer'],'ids_equal':record['generated_ids_equal']},ensure_ascii=False),flush=True);del whole,answer
        report.update(all_registered_cases_executed=len(report['cases'])==3,actual_forwards=counts,generated_tokens=generated,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,wall_seconds_including_load=time.perf_counter()-start)
        p.write_json(a.out/'summary.json',report)
    finally:
        for h in hooks:h.remove()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['preflight','gpu'],required=True);parser.add_argument('--model-dir',type=Path,required=True);parser.add_argument('--video',type=Path,required=True);parser.add_argument('--original-run',type=Path,required=True);parser.add_argument('--allow-branch',action='append',default=[]);parser.add_argument('--preflight',type=Path);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--wall-seconds',type=int,default=90)
    a=parser.parse_args()
    if a.wall_seconds!=90 or (a.stage=='gpu' and a.preflight is None):parser.error('Registered90s and CPU proof required')
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'tmp').mkdir();tempfile.tempdir=str(a.out/'tmp');torch.set_num_threads(1);torch.manual_seed(42)
    p.write_json(a.out/'arguments.json',request(a));p.write_json(a.out/'budget.json',budget());p.write_json(a.out/'source_hashes.json',sources())
    try:(preflight if a.stage=='preflight' else gpu)(a);p.write_json(a.out/'status.json',{'status':'completed','stage':a.stage})
    except Exception:
        (a.out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8');p.write_json(a.out/'status.json',{'status':'failed','stage':a.stage});raise
if __name__=='__main__':main()
