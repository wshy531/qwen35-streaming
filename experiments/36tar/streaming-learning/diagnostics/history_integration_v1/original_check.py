"""Independent original-core controls for the three exact recorded frame pairs."""
import argparse
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback

HERE = Path(__file__).resolve().parent
CAP = HERE.parents[1]
ACTIVE = CAP/'code/qwen35-streaming-s4'
sys.path.insert(0,str(ACTIVE))
import psutil
import torch
from transformers import AutoProcessor
from scripts.baseline import file_hash,write_json
from scripts.pair_demo import verify_assets
from scripts.protocol_capability_control import original_package,source_assets,ORIGINAL
from scripts.fair_video import original_answer
from scripts.video_demo import branch_allowed
from streaming.online import fp32_reference
from streaming.timeline_inputs import identity
from streaming.window import read_packets


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sources():
    return {**{str(p.resolve()):file_hash(p) for p in sorted(ACTIVE.rglob('*.py'))},**source_assets(),
            **{str(p.resolve()):file_hash(p) for p in sorted(HERE.glob('*.py'))},
            '@plan':file_hash(HERE/'方案.md')}
def budget(): return {'model_loads':1,'unique_pairs':3,'frames':6,'vision_max':3,'language_max':51,
                     'generated_max':48,'answer_tokens':16,'tokens_max':200,'wall_seconds':240,
                     'ram_min_gib':3,'gpu_free_min_gib':5.5,'fp32_conversion_free_min_gib':2}
def request(args): return {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('stage','out','preflight')}
def frames(args):
    packets = tuple(read_packets(args.video,1.))
    if len(packets)!=6 or [p.timestamp for p in packets]!=list(range(6)):
        raise ValueError('Registered six-frame controlled source required')
    return [packets[i:i+2] for i in range(0,6,2)]
def metadata(pair):
    import hashlib
    return [{'frame_id':p.frame_id,'timestamp':p.timestamp,'rgb_sha256':hashlib.sha256(p.rgb.tobytes()).hexdigest()} for p in pair]
def check_stream(args,pairs):
    if read(args.stream_run/'status.json')!={'status':'completed','stage':'gpu'}:raise ValueError('Completed stream required')
    actual = read(args.stream_run/'summary.json')
    recorded = [read_line['answer'] for line in (args.stream_run/'events.jsonl').read_text(encoding='utf-8').splitlines()
                if (read_line:=json.loads(line))['kind']=='answer']
    if len(recorded)!=3 or any(x['question']!=args.question for x in recorded):raise ValueError('Actual current queries differ')
    if read(args.stream_run/'arguments.json')['video'] != str(args.video):raise ValueError('Wrong recorded video')
    for answer,pair in zip(recorded,pairs):
        if answer['frame_id']!=pair[-1].frame_id or answer['timestamp']!=pair[-1].timestamp:raise ValueError('Wrong current evidence pair')
    for answer in actual['historical_answers']:
        if answer['model_called']:
            pair = next((p for p in pairs if [x.frame_id for x in p]==answer['evidence_frame_ids']),None)
            if pair is None or answer['question']!=args.question:raise ValueError('Unexpected history evidence')
    return recorded,actual['historical_answers']


@torch.inference_mode()
def preflight(args):
    from tests.test_online import small_model
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    current = small_model(processor)
    package = original_package()
    original = package.Qwen3_5ForConditionalGeneration(package.Qwen3_5Config(**current.config.to_dict())).eval()
    original.set_attn_implementation('sdpa')
    original.load_state_dict(current.state_dict(),strict=True)
    pairs = frames(args);check_stream(args,pairs);cases = []
    for pair in pairs:
        whole,checked = identity(original,processor,pair,args.question)
        if whole.input_ids.shape[1]+16>200:raise ValueError('Input exceeds registered token budget')
        with fp32_reference():
            left = original(**whole,use_cache=False,logits_to_keep=1).logits
            right = current(**whole,use_cache=False,logits_to_keep=1).logits
        if not torch.equal(left,right):raise AssertionError('Tiny fresh prefill differs')
        cases.append({'frames':metadata(pair),**checked,'tiny_fresh_prefill_bitwise_equal':True})
    proof = {'request':request(args),'budget':budget(),'sources':sources(),'video_sha256':file_hash(args.video),
             'model_manifest_sha256':file_hash(args.model_dir/'DOWNLOAD_MANIFEST.json'),
             'assets':verify_assets(args.model_dir)['files'],'cases':cases,
             'stream_summary_sha256':file_hash(args.stream_run/'summary.json'),
             'stream_events_sha256':file_hash(args.stream_run/'events.jsonl')}
    write_json(args.out/'proof.json',proof)
    print(json.dumps({'passed':True,'cases':len(cases),'scope':'CPU same-input/original-current tiny prefill, no semantics'}))


@torch.inference_mode()
def gpu(args):
    if args.preflight is None or read(args.preflight/'status.json')!={'status':'completed','stage':'preflight'}:
        raise ValueError('Fresh CPU proof required')
    proof=read(args.preflight/'proof.json')
    if proof['request']!=request(args) or proof['budget']!=budget() or proof['sources']!=sources():raise ValueError('Source/request/budget changed')
    if (proof['video_sha256']!=file_hash(args.video) or proof['assets']!=verify_assets(args.model_dir)['files'] or
        proof['model_manifest_sha256']!=file_hash(args.model_dir/'DOWNLOAD_MANIFEST.json') or
        proof['stream_summary_sha256']!=file_hash(args.stream_run/'summary.json') or
        proof['stream_events_sha256']!=file_hash(args.stream_run/'events.jsonl')):raise ValueError('Assets/evidence changed')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=CAP,text=True).strip()
    if not branch_allowed(branch,[]):raise ValueError('Personal branch required')
    free,total=torch.cuda.mem_get_info();ram=psutil.virtual_memory().available
    write_json(args.out/'resource_precheck.json',{'gpu_free_bytes':free,'gpu_total_bytes':total,'ram_available_bytes':ram,'branch':branch})
    if free<5.5*2**30 or ram<3*2**30:raise RuntimeError('Resource guard refused load')
    start=time.perf_counter()
    model=original_package().Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    if Path(inspect.getfile(type(model))).resolve()!=ORIGINAL/'modeling_qwen3_5.py':raise AssertionError('Not original core')
    if torch.cuda.mem_get_info()[0]<2*2**30:raise RuntimeError('No FP32 conversion room')
    model.float();torch.cuda.reset_peak_memory_stats()
    processor=AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    counts={'vision':0,'language':0}
    def guard(name,limit):
        def tick(_m,_i):
            counts[name]+=1
            if counts[name]>limit:raise RuntimeError('Forward budget exceeded')
        return tick
    hooks=[model.model.visual.register_forward_pre_hook(guard('vision',3)),model.model.language_model.register_forward_pre_hook(guard('language',51))]
    try:
        pairs=frames(args);recorded,historical=check_stream(args,pairs);answers=[]
        for index,pair in enumerate(pairs):
            whole,checked=identity(model,processor,pair,args.question)
            case={'frames':metadata(pair),**checked,'tiny_fresh_prefill_bitwise_equal':True}
            if case!=proof['cases'][index]:raise AssertionError('Actual input differs from CPU')
            answer=original_answer(model,processor,whole)
            history=[h for h in historical if h['model_called'] and h['evidence_frame_ids']==[p.frame_id for p in pair]]
            answers.append({**case,**answer,'same_current_ids':answer['generated_ids']==recorded[index]['generated_ids'],
                            'same_history_ids':[answer['generated_ids']==h['generated_ids'] for h in history]})
            print(json.dumps({'frame_ids':[p.frame_id for p in pair],'answer':answer['answer'],'same_current_ids':answers[-1]['same_current_ids']},ensure_ascii=False),flush=True)
        if sum(len(a['generated_ids']) for a in answers)>48:raise RuntimeError('Generation budget exceeded')
        write_json(args.out/'summary.json',{'cases':answers,'counts':counts,'scope':'Three exact pair original controls; not general accuracy',
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
            'wall_seconds_including_load':time.perf_counter()-start})
    finally:
        for hook in hooks:hook.remove()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('preflight','gpu'),required=True)
    for name in ('video','model-dir','stream-run','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--question',default='Name the main visible animal.')
    parser.add_argument('--preflight',type=Path)
    args=parser.parse_args()
    for name in ('video','model_dir','stream_run','out','preflight'):
        value=getattr(args,name)
        if value is not None:setattr(args,name,value.resolve())
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'tmp').mkdir();tempfile.tempdir=str(args.out/'tmp')
    torch.set_num_threads(1);torch.manual_seed(42)
    write_json(args.out/'budget.json',budget());write_json(args.out/'source_hashes.json',sources())
    try:
        (preflight if args.stage=='preflight' else gpu)(args)
        write_json(args.out/'status.json',{'status':'completed','stage':args.stage})
    except Exception:
        (args.out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        write_json(args.out/'status.json',{'status':'failed','stage':args.stage});raise


if __name__=='__main__':main()
