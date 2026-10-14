"""Optional video-file strict-time query entry; fresh CPU proof required for GPU."""
import argparse,hashlib,html,json,math,subprocess,sys,tempfile,time,traceback,unittest
from pathlib import Path
sys.modules.setdefault('time_file_entry',sys.modules[__name__])
from dependencies import frozen
from file_session import TimeFileSession
import torch,psutil
from transformers import AutoProcessor
from scripts.baseline import file_hash,write_json
from scripts.pair_demo import verify_assets
from streaming.window import read_packets
from streaming.preprocessing import fit_canvas
HERE=Path(__file__).resolve().parent
CAP=HERE.parents[1]


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sources():return {**frozen.sources(),**{str(p.resolve()):file_hash(p) for p in HERE.iterdir() if p.suffix in ('.py','.md','.json')}}


def plan(args):
    rows=[] if args.query_plan is None else read(args.query_plan)
    if not isinstance(rows,list):raise ValueError('Query plan must be a list')
    rows=rows+[{'after_pairs':'eof','question':q} for q in args.question]
    if not 1<=len(rows)<=24:raise ValueError('One to24 planned questions required')
    for row in rows:
        if not isinstance(row,dict) or set(row)!={'after_pairs','question'}:raise ValueError('Plan fields: after_pairs, question')
        trigger=row['after_pairs']
        if trigger!='eof' and (type(trigger) is not int or not 0<=trigger<=args.max_frames//2):raise ValueError('Plan trigger outside configured pair budget')
        if not isinstance(row['question'],str) or len(row['question'])>512:raise ValueError('Bounded text question required')
    return rows


def request(args):
    return {**{k:str(getattr(args,k)) if isinstance(getattr(args,k),Path) else getattr(args,k)
               for k in ('model_dir','reference_run','video','fps','duration','max_frames','max_pairs','max_tokens','reset_after_pairs','no_preview','branch')},
            'queries':plan(args),'video_sha256':file_hash(args.video),
            'query_plan_sha256':None if args.query_plan is None else file_hash(args.query_plan)}


def budget(args):
    pairs=args.max_frames//2;attempts=len(plan(args))
    return {'model_loads':1,'vision_max':pairs,'questions_max':attempts,'generated_max':24*attempts,
            'language_max':2*pairs+25*attempts+16,'answer_tokens':24,'wall_seconds':360,
            'main_max_tokens':args.max_tokens,'qa_max_tokens':768,'index_visual_tokens_max':48*args.max_pairs,
            'external_working_conservative_max_bytes':8*2**20,'rgb_and_preview_max_bytes':2*2**20,
            'output_preview_max_bytes':6*2**20,'ram_min_gib':3,'gpu_free_min_gib':5.5,'conversion_free_min_gib':2}


def packets(args):
    source=read_packets(args.video,args.fps)
    try:
        for i,packet in enumerate(source):
            if packet.timestamp>args.duration:break
            yield packet
            if i+1>=args.max_frames:break
    finally:source.close()


def identity(packet):
    return {'source_frame_id':packet.frame_id,'source_pts':packet.timestamp,
            'source_rgb_sha256':hashlib.sha256(packet.rgb.tobytes()).hexdigest(),
            'canvas_sha256':hashlib.sha256(fit_canvas(packet.rgb,(192,256)).tobytes()).hexdigest()}


def render_report(args,answers,final,pending):
    def esc(v):return html.escape(str(v),quote=True)
    rows=[]
    for a in answers:
        result=a['outcome'];selection=result.get('selection',{})
        answer=result['result']['answer'] if result['model_called'] else result['reason_text']
        span=selection.get('selected_interval','无对应证据')
        images=''.join('<img width="256" src="'+esc(p)+'" alt="实际采样帧预览">' for p in a['previews'])
        rows.append('<article><h2>'+esc(a['question'])+'</h2><p>触发：'+esc(a['trigger'])+
                    '；帧对：'+esc(selection.get('selected_pair','无'))+'；证据区间：'+esc(span)+'；主记忆轮次：'+esc(a['status']['main']['epoch'])+
                    '</p><p>当时保留：'+esc([r['interval'] for r in a['status']['retained_intervals']])+ '</p>'+images+
                    '<pre>'+esc(answer)+'</pre></article>')
    note='CPU机制重放：随机小模型回答，不用于能力判断。' if args.stage=='preflight' else '真实权重读取结果；请结合所选采样帧核对。'
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>视频历史时间查询</title><style>body{max-width:1000px;margin:32px auto;padding:0 20px;background:#f4f6f8;color:#182332;font:16px/1.7 system-ui}article{background:white;padding:20px;margin:20px 0;border-radius:12px}h2{font-size:19px}img{max-width:48%;margin:0 8px 8px 0}pre{white-space:pre-wrap;font:inherit;background:#edf3f7;padding:14px}small{color:#46566a}</style>
<h1>视频历史时间查询</h1><p>'''+esc(args.video.name)+'</p><p>'+esc(note)+'''</p>
<p>按问题时间读取已保存的完整采样帧对。帧对区间不代表连续录像，也不是该中间时刻的一张精确画面。</p>
<small>支持：Name the animal visible at T seconds.<br>Is the animal's coat mostly light-colored or mostly dark-colored at T seconds? Answer light or dark.</small>
'''+''.join(rows)+'<p>结束保留区间：'+esc([r['interval'] for r in final['retained_intervals']])+ '</p><p>未达触发点：'+esc(pending)+'</p></html>'
    (args.out/'report.html').write_text(document,encoding='utf-8')


@torch.inference_mode()
def replay(model,processor,args,expected=None):
    wrapper=TimeFileSession(model,processor,args.max_pairs,args.max_tokens,not args.no_preview)
    limits=budget(args);counts={'vision':0,'language':0};answers=[];inputs=[];updates=[];handled=set()
    generated=peak_features=peak_rgb=peak_cache=0;preview_manifest={};requests=plan(args)
    def guard(kind):
        def hook(_m,_a):
            counts[kind]+=1
            if counts[kind]>limits[kind+'_max']:raise RuntimeError('Forward budget')
        return hook
    hooks=[model.model.visual.register_forward_pre_hook(guard('vision')),model.model.language_model.register_forward_pre_hook(guard('language'))]
    log=(args.out/'events.jsonl').open('x',encoding='utf-8')
    def emit(row):
        log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
        if row['kind']=='query':
            r=row['outcome'];status=row['status']
            display={'kind':'query','question':row['question'],'trigger':row['trigger'],
                     'model_called':r['model_called'],'evidence_interval':r.get('selection',{}).get('selected_interval'),
                     'answer':r['result']['answer'] if r['model_called'] else r['reason_text'],
                     'retained_intervals':[v['interval'] for v in status['retained_intervals']],
                     'main_epoch':status['main']['epoch']}
            print(json.dumps(display,ensure_ascii=False),flush=True)
        elif row['kind'] in ('reset','finish'):
            status=row['status']
            print(json.dumps({'kind':row['kind'],'received_frames':status['main']['received_frames'],
                              'encoded_pairs':status['main']['encoded_pairs'],'main_epoch':status['main']['epoch'],
                              'retained_intervals':[v['interval'] for v in status['retained_intervals']],
                              'odd_eof':status['main']['dropped_odd_eof_frame']},ensure_ascii=False),flush=True)
    def query(i,trigger):
        nonlocal generated
        question=requests[i]['question'];before=wrapper.status();vision=counts['vision'];language=counts['language']
        outcome=wrapper.ask(question)
        if wrapper.status()!=before or counts['vision']!=vision:raise AssertionError('Query polluted input/state')
        if not outcome['model_called'] and counts['language']!=language:raise AssertionError('Refusal invoked model')
        previews=[]
        if outcome['model_called']:
            generated+=len(outcome['result']['generated_ids'])
            pair=outcome['selection']['selected_pair']
            for j,data in enumerate(wrapper.previews(pair)):
                name=f'previews/pair-{pair}-sample-{j}.jpg';p=args.out/name;p.parent.mkdir(exist_ok=True)
                if p.exists():assert p.read_bytes()==data
                else:p.write_bytes(data)
                preview_manifest[name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()};previews.append(name)
        if generated>limits['generated_max'] or len(answers)+1>limits['questions_max']:raise RuntimeError('Answer budget')
        if sum(v['bytes'] for v in preview_manifest.values())>6*2**20:raise RuntimeError('Output preview bound')
        row={'kind':'query','plan_index':i,'trigger':trigger,'question':question,'outcome':outcome,
             'previews':previews,'status':before,'main_and_index_unchanged':True,
             'no_query_vision':True,'language_calls':counts['language']-language}
        answers.append(row);handled.add(i);emit(row)
    def due(trigger):
        for i,r in enumerate(requests):
            if i not in handled and r['after_pairs']==trigger:query(i,trigger)
    try:
        due(0)
        for packet in packets(args):
            inputs.append(identity(packet));result=wrapper.feed(packet)
            status=wrapper.status();bank=wrapper.inner.bank.status()['tensor_bytes']
            # Includes index, legacy capture bank, transient raw/selected copies and native prefill staging.
            peak_features=max(peak_features,status['index_tensor_bytes']+bank+10*197952+512*1024)
            peak_rgb=max(peak_rgb,packet.rgb.nbytes+4*192*256*3+status['preview_bytes'])
            peak_cache=max(peak_cache,frozen.runtime_ref.main_storage(wrapper.inner)['main_cache_tensor_bytes'])
            if peak_features>8*2**20 or peak_rgb>2*2**20:raise RuntimeError('Small resource estimate bound')
            if result['encoded']:
                count=wrapper.inner.encoded_pairs
                updates.append({'encoded_pairs':count,'feature_sha256':result['feature_sha256'],
                                'ledger':wrapper.inner.session.ledger[-1],'capacity_reset':result['reset'],
                                'retained_intervals':status['retained_intervals']})
                emit({'kind':'pair','update':updates[-1]})
                if count in args.reset_after_pairs:emit({'kind':'reset','reset':wrapper.reset_main(),'status':wrapper.status()})
                due(count)
        final=wrapper.finish();emit({'kind':'finish','status':final});due('eof')
        pending=[{'plan_index':i,**r} for i,r in enumerate(requests) if i not in handled]
        if expected is not None and (inputs!=expected['input_frames'] or [r['ledger'] for r in updates]!=expected['layout']):raise AssertionError('Fresh CPU/GPU pixel or layout drift')
        render_report(args,answers,final,pending)
        return {'passed':True,'answers':answers,'counts':counts,'generated_tokens':generated,
                'model_questions':sum(a['outcome']['model_called'] for a in answers),'request_attempts':len(answers),
                'input_frames':inputs,'updates':updates,'layout':[r['ledger'] for r in updates],
                'final':final,'pending_unreached':pending,'preview_manifest':preview_manifest,
                'external_working_conservative_bytes':peak_features,'rgb_preview_conservative_bytes':peak_rgb,
                'peak_main_cache_tensor_bytes':peak_cache,'capability_scope':'Mechanism acceptance; all semantic errors retained separately.'}
    finally:
        wrapper.close();log.close()
        for h in hooks:h.remove()


@torch.inference_mode()
def preflight(args):
    from tests.test_online import small_model
    from contracts import Contracts
    processor=AutoProcessor.from_pretrained(args.model_dir,local_files_only=True);model=small_model(processor)
    Contracts.model,Contracts.processor=model,processor
    with (args.out/'unittest.log').open('x',encoding='utf-8') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not result.wasSuccessful():raise AssertionError('File-session contracts failed')
    report=replay(model,processor,args)
    report.update(cpu_tests=result.testsRun,source_hashes=sources(),request=request(args),budget=budget(args),
                  model_assets=verify_assets(args.model_dir)['files'],
                  model_manifest_sha256=file_hash(args.model_dir/'DOWNLOAD_MANIFEST.json'),
                  reference_sha256=file_hash(args.reference_run/'summary.json'))
    write_json(args.out/'proof.json',report)


@torch.inference_mode()
def gpu(args):
    if args.preflight is None or read(args.preflight/'status.json')!={'status':'completed','stage':'preflight'}:raise ValueError('Fresh successful CPU proof required')
    p=read(args.preflight/'proof.json')
    if (p['source_hashes']!=sources() or p['request']!=request(args) or p['budget']!=budget(args)
            or p['model_assets']!=verify_assets(args.model_dir)['files']
            or p['model_manifest_sha256']!=file_hash(args.model_dir/'DOWNLOAD_MANIFEST.json')
            or p['reference_sha256']!=file_hash(args.reference_run/'summary.json')):raise ValueError('Proof drift')
    if [identity(packet) for packet in packets(args)]!=p['input_frames']:raise ValueError('Pixel drift before load')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=CAP,text=True).strip()
    if args.branch in ('','main','master','HEAD') or branch!=args.branch:raise ValueError('Configured personal branch required')
    free,total=torch.cuda.mem_get_info();ram=psutil.virtual_memory().available
    write_json(args.out/'resource_precheck.json',{'ram_available_bytes':ram,'gpu_free_bytes':free,'gpu_total_bytes':total,'branch':branch})
    if ram<3*2**30 or free<5.5*2**30:raise RuntimeError('Resource gate refused load')
    from qwen35_local import Qwen3_5ForConditionalGeneration
    started=time.perf_counter()
    model=Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    if torch.cuda.mem_get_info()[0]<2*2**30:raise RuntimeError('No FP32 conversion room')
    model.float();torch.cuda.reset_peak_memory_stats();processor=AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    result=replay(model,processor,args,p)
    result.update(source_hashes=sources(),request=request(args),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,wall_seconds_including_load=time.perf_counter()-started)
    write_json(args.out/'summary.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('preflight','gpu'),required=True)
    for name in ('model-dir','video','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reference-run',type=Path,default=CAP/'runs/event_memory_gpu_20261013_01')
    parser.add_argument('--preflight',type=Path);parser.add_argument('--query-plan',type=Path)
    parser.add_argument('--question',action='append',default=[]);parser.add_argument('--branch',required=True)
    parser.add_argument('--fps',type=float,default=1.);parser.add_argument('--duration',type=float,default=30.)
    parser.add_argument('--max-frames',type=int,default=16);parser.add_argument('--max-pairs',type=int,default=3)
    parser.add_argument('--max-tokens',type=int,default=256);parser.add_argument('--reset-after-pairs',type=int,nargs='*',default=[])
    parser.add_argument('--no-preview',action='store_true');args=parser.parse_args()
    for name in ('model_dir','video','out','reference_run','preflight','query_plan'):
        if getattr(args,name) is not None:setattr(args,name,getattr(args,name).resolve())
    if (not math.isfinite(args.fps) or not .1<=args.fps<=4 or not math.isfinite(args.duration) or not 0<args.duration<=60
            or not 2<=args.max_frames<=32 or not 1<=args.max_pairs<=3 or not 256<=args.max_tokens<=512
            or any(not 1<=i<=args.max_frames//2 for i in args.reset_after_pairs)):parser.error('Outside bounded demo budget')
    if not args.video.is_file() or args.video.stat().st_size>64*2**20:parser.error('Video must be a file≤64MiB')
    if args.query_plan is not None and args.query_plan.stat().st_size>16*1024:parser.error('Plan≤16KiB required')
    args.reset_after_pairs=sorted(set(args.reset_after_pairs));plan(args)
    expected={p.name:file_hash(p) for p in HERE.iterdir() if p.suffix in ('.py','.md','.json')}
    if {p.name for p in args.out.iterdir()}!={'source_snapshot','bootstrap.json'} or read(args.out/'bootstrap.json')!={'prepared_by':'time_file_v1_watchdog','files':expected}:raise ValueError('Fresh frozen watchdog output required')
    (args.out/'tmp').mkdir();tempfile.tempdir=str(args.out/'tmp');torch.set_num_threads(1);torch.manual_seed(42)
    write_json(args.out/'source_hashes.json',sources());write_json(args.out/'budget.json',budget(args))
    try:
        (preflight if args.stage=='preflight' else gpu)(args)
        write_json(args.out/'status.json',{'status':'completed','stage':args.stage})
    except Exception as error:
        write_json(args.out/'status.json',{'status':'failed','stage':args.stage,'error':type(error).__name__,'message':str(error)})
        (args.out/'error.log').write_text(traceback.format_exc(),encoding='utf-8');raise

if __name__=='__main__':main()
