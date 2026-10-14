"""Fixed tier split controls and persistent external memory across main reset."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

HERE=Path(__file__).resolve().parent;CAP=HERE.parents[1]
ACTIVE=CAP/'code/qwen35-streaming-s4';VISUAL=HERE.parent/'visual_memory_v1'
sys.path.insert(0,str(VISUAL));sys.path.insert(0,str(ACTIVE))
import psutil
import torch
from transformers import AutoProcessor
import experiment as reference
from scripts.baseline import file_hash,write_json,cache_measure
from scripts.pair_demo import verify_assets
from streaming.online import OnlinePairSession
from streaming.window import FramePacket
from features import raw_block
from memory import SplitMemory,TieredSession

CURRENT='Name the animal visible most recently.'
HISTORY='Name the main visible animal in this memory interval.'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def budget():return {'model_loads':1,'vision_max':8,'questions_max':14,'generated_max':224,
    'language_max':280,'answer_tokens':16,'max_tokens':512,'wall_seconds':300,
    'ram_min_gib':3,'gpu_free_min_gib':5.5,'conversion_free_min_gib':2,
    'bank_tokens_max':144,'bank_tensor_bytes_max':592704,'banks':3,
    'current_question':CURRENT,'history_question':HISTORY,'current_trigger_pts':[3.,7.,11.],
    'explicit_reset_after_encoded_pts':11.,'controlled_repeat_to_encoded_pts':15.}
def sources():
    return {**reference.sources(),**{str(p.resolve()):file_hash(p) for p in HERE.glob('*.py')},
            '@tiered_plan':file_hash(HERE/'方案.md'),
            **{str(p.resolve()):file_hash(p) for p in (HERE.parent/'feature_window_v1').glob('*.py')}}
def request(args):return {'model_dir':str(args.model_dir),'reference_run':str(args.reference_run)}
def inputs():
    pairs,rows,assets=reference.inputs();extras=[]
    for i,old in enumerate([p for pair in pairs[2:4] for p in pair]):
        new_id=12+i
        extras.append(FramePacket(new_id,float(new_id),old.rgb.copy(),new_id+1,old.decode_seconds))
        rows.append({**rows[4+i],'session_frame_id':new_id,'session_pts':float(new_id),
                     'controlled_repeat_of_session_frame_id':old.frame_id})
    return pairs+[extras[:2],extras[2:]],rows,assets
def make(model,processor):
    return TieredSession(OnlinePairSession(model,processor,max_tokens=512,max_question_tokens=64,
                        max_answer_tokens=16,resize_mode='letterbox',video_logits='none'))


class Contracts(unittest.TestCase):
    def test_same_three_slot_budget_and_long_mass(self):
        banks=[SplitMemory(i) for i in (1,2,3)];expected=torch.zeros(16)
        for i in range(512):
            x=torch.randn((48,16),generator=torch.Generator().manual_seed(600+i));expected+=x.sum(0)
            block=raw_block(x,i,2*i,2*i+1)
            for bank in banks:
                bank.append(block);state=bank.status()
                self.assertLessEqual(state['tokens'],144)
                self.assertLessEqual(state['tensor_bytes'],144*(16*4+4+16))
        for bank in banks[:2]:
            rows=[*bank.roots,*bank.recent];mass=sum(float(b.mass.sum()) for b in rows)
            total=sum((b.features*b.mass).sum(0) for b in rows)
            self.assertEqual(mass,512*48)
            torch.testing.assert_close(total/mass,expected/mass,rtol=1e-5,atol=1e-5)
        self.assertEqual(banks[2].discarded,509)
        for bank in banks:bank.clear();self.assertEqual(bank.status()['tensor_bytes'],0)

    def test_reset_preserves_banks_and_no_stale_current_or_future(self):
        wrapper=make(self.model,self.processor)
        try:
            for pair in self.pairs[:6]:
                for packet in pair:wrapper.append(packet)
            memory=wrapper.snapshot();wrapper.reset_main()
            self.assertEqual(wrapper.snapshot(),memory)
            self.assertEqual(wrapper.session.main.seen_tokens,0)
            self.assertEqual(cache_measure(wrapper.session.main.cache)['total_tensor_bytes'],0)
            self.assertFalse(wrapper.current(2,CURRENT)['model_called'])
            before=reference.cache_fingerprint(wrapper.session.main)
            self.assertTrue(wrapper.coarse(2,HISTORY)['model_called'])
            self.assertEqual(reference.cache_fingerprint(wrapper.session.main),before)
            wrapper.append(self.pairs[6][0])
            self.assertEqual(wrapper.current(2,CURRENT)['reason'],'waiting_for_complete_pair')
            self.assertEqual(wrapper.exact_available(12.)['reason'],'beyond_encoded_horizon')
            wrapper.append(self.pairs[6][1])
            for packet in self.pairs[7]:wrapper.append(packet)
            self.assertTrue(wrapper.current(2,CURRENT)['model_called'])
            self.assertEqual(wrapper.session.last_encoded_pts,15.)
            self.assertEqual(wrapper.banks[2].last_end,15.)
        finally:
            wrapper.close()
            self.assertTrue(all(b.status()['tensor_bytes']==0 for b in wrapper.banks.values()))


@torch.inference_mode()
def replay(model,processor,args,expected_ledger=None):
    pairs,metadata,assets=inputs();wrapper=make(model,processor)
    counts={'vision':0,'language':0};events=[];called=0;generated=0;epochs=[]
    def guard(name,limit):
        def hook(_m,_a):
            counts[name]+=1
            if counts[name]>limit:raise RuntimeError('Forward budget exceeded')
        return hook
    hooks=[model.model.visual.register_forward_pre_hook(guard('vision',8)),
           model.model.language_model.register_forward_pre_hook(guard('language',280))]
    def query(kind,operation):
        nonlocal called,generated
        state=reference.cache_fingerprint(wrapper.session.main);status=wrapper.session.status();ledger=list(wrapper.session.ledger)
        result=operation();called+=int(result.get('model_called',False));generated+=len(result.get('generated_ids',[]))
        if reference.cache_fingerprint(wrapper.session.main)!=state or wrapper.session.status()!=status or wrapper.session.ledger!=ledger:
            raise AssertionError('Query polluted main sequence')
        row={'kind':kind,'encoded_pts':wrapper.session.last_encoded_pts,'main_seen_tokens':wrapper.session.main.seen_tokens,
             'result':result,'main_unchanged':True,'banks':{i:b.status() for i,b in wrapper.banks.items()}}
        events.append(row)
        print(json.dumps({'kind':kind,'encoded_pts':row['encoded_pts'],'answer':result.get('answer'),
                          'model_called':result.get('model_called'),'reason':result.get('reason'),
                          'evidence_interval':result.get('evidence_interval')},ensure_ascii=False),flush=True)
        return result
    try:
        for pair in pairs[:6]:
            for packet in pair:wrapper.append(packet)
            if wrapper.session.last_encoded_pts in (3.,7.,11.):
                for size in (1,2,3):query('current_window_'+str(size),lambda n=size:wrapper.current(n,CURRENT))
        query('coarse_short1_long2',lambda:wrapper.coarse(1,HISTORY))
        old_coarse=query('coarse_short2_long1',lambda:wrapper.coarse(2,HISTORY))
        query('exact_old_0.5',lambda:wrapper.exact_available(.5))
        query('future_99',lambda:wrapper.exact_available(99.))
        epochs.append(list(wrapper.session.ledger));before=wrapper.snapshot()
        wrapper.reset_main()
        if wrapper.session.main.seen_tokens!=0 or cache_measure(wrapper.session.main.cache)['total_tensor_bytes']!=0:
            raise AssertionError('Main hybrid cache not cleared')
        if wrapper.snapshot()!=before:raise AssertionError('External feature memory erased by main reset')
        query('current_after_reset',lambda:wrapper.current(2,CURRENT))
        after=query('coarse_after_reset',lambda:wrapper.coarse(2,HISTORY))
        same_ids=after['generated_ids']==old_coarse['generated_ids']
        for pair in pairs[6:]:
            wrapper.append(pair[0])
            query('unencoded_time_'+str(pair[0].timestamp),lambda t=pair[0].timestamp:wrapper.exact_available(t))
            wrapper.append(pair[1])
        query('current_after_resume',lambda:wrapper.current(2,CURRENT))
        query('coarse_after_resume',lambda:wrapper.coarse(2,HISTORY))
        epochs.append(list(wrapper.session.ledger))
        if expected_ledger is not None and epochs!=expected_ledger:raise ValueError('Main epoch input ledger drift')
        if counts['vision']!=8 or called!=14 or generated>224:raise RuntimeError('QA/vision/token budget exceeded')
        if any(b.status()['tokens']>144 or b.status()['tensor_bytes']>592704 for b in wrapper.banks.values()):
            raise RuntimeError('Feature bank budget exceeded')
        expected=[b['feature_sha256'] for b in read(args.reference_run/'summary.json')['feature_blocks']]
        expected+=expected[2:4]
        return {'passed':True,'events':events,'counts':counts,'model_called_queries':called,'generated_tokens':generated,
                'input_frames':metadata,'source_assets':assets,'epoch_ledgers':epochs,
                'banks':{i:b.status() for i,b in wrapper.banks.items()},'main_reset_kept_memory_bitwise':True,
                'main_cache_empty_after_reset':True,'coarse_before_after_reset_same_generated_ids':same_ids,
                'same_projected_features_as_gpu74':[a==b for a,b in zip(wrapper.hashes,expected)],
                'controlled_repeat_frames':4,'continued_after_reset_to':15.,'default_integration':False,
                'scope':'Finite controlled montage plus explicit frame reuse; compressed range evidence, not exact historical frames or natural long run'}
    finally:
        for hook in hooks:hook.remove()
        wrapper.close()


@torch.inference_mode()
def preflight(args):
    from tests.test_online import small_model
    processor=AutoProcessor.from_pretrained(args.model_dir,local_files_only=True);model=small_model(processor)
    pairs,_,_=inputs();Contracts.model,Contracts.processor,Contracts.pairs=model,processor,pairs
    with (args.out/'unittest.log').open('x',encoding='utf-8') as log:
        tests=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not tests.wasSuccessful():raise AssertionError('Tiered memory CPU contracts failed')
    report=replay(model,processor,args)
    report.update(cpu_tests=tests.testsRun,request=request(args),budget=budget(),source_hashes=sources(),
        assets=verify_assets(args.model_dir)['files'],model_manifest_sha256=file_hash(args.model_dir/'DOWNLOAD_MANIFEST.json'),
        reference_summary_sha256=file_hash(args.reference_run/'summary.json'))
    write_json(args.out/'proof.json',report)
    print(json.dumps({'passed':True,'cpu_tests':tests.testsRun,'vision':report['counts']['vision'],
                      'queries':report['model_called_queries'],'resumed_to':15.,'gpu_loads':0}),flush=True)


@torch.inference_mode()
def gpu(args):
    if args.preflight is None or read(args.preflight/'status.json')!={'status':'completed','stage':'preflight'}:
        raise ValueError('Fresh successful CPU proof required')
    proof=read(args.preflight/'proof.json')
    if (proof['request']!=request(args) or proof['budget']!=budget() or proof['source_hashes']!=sources() or
        proof['assets']!=verify_assets(args.model_dir)['files'] or
        proof['model_manifest_sha256']!=file_hash(args.model_dir/'DOWNLOAD_MANIFEST.json') or
        proof['reference_summary_sha256']!=file_hash(args.reference_run/'summary.json')):
        raise ValueError('Proof source/parameters/assets drift')
    _,metadata,assets=inputs()
    if metadata!=proof['input_frames'] or assets!=proof['source_assets']:raise ValueError('Input changed')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=CAP,text=True).strip()
    if branch!='36tar/general-video-demo':raise ValueError('Registered personal branch required')
    free,total=torch.cuda.mem_get_info();ram=psutil.virtual_memory().available
    write_json(args.out/'resource_precheck.json',{'ram_available_bytes':ram,'gpu_free_bytes':free,'gpu_total_bytes':total,'branch':branch})
    if free<5.5*2**30 or ram<3*2**30:raise RuntimeError('Resource guard refused load')
    from qwen35_local import Qwen3_5ForConditionalGeneration
    begin=time.perf_counter()
    model=Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    if torch.cuda.mem_get_info()[0]<2*2**30:raise RuntimeError('No FP32 conversion room')
    model.float();torch.cuda.reset_peak_memory_stats()
    processor=AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    report=replay(model,processor,args,proof['epoch_ledgers'])
    report.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,wall_seconds_including_load=time.perf_counter()-begin)
    write_json(args.out/'summary.json',report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('preflight','gpu'),required=True)
    for name in ('model-dir','reference-run','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--preflight',type=Path)
    args=parser.parse_args()
    for name in ('model_dir','reference_run','out','preflight'):
        value=getattr(args,name)
        if value is not None:setattr(args,name,value.resolve())
    args.out.mkdir(parents=True,exist_ok=False);(args.out/'tmp').mkdir();tempfile.tempdir=str(args.out/'tmp')
    torch.set_num_threads(1);torch.manual_seed(42)
    write_json(args.out/'budget.json',budget());write_json(args.out/'source_hashes.json',sources())
    try:
        (preflight if args.stage=='preflight' else gpu)(args)
        write_json(args.out/'status.json',{'status':'completed','stage':args.stage})
    except Exception as error:
        write_json(args.out/'status.json',{'status':'failed','stage':args.stage,'error':type(error).__name__,'message':str(error)})
        (args.out/'error.log').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
