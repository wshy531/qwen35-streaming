"""Mid-input current-window routing with coarse-history safety and continuation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent;CAP=HERE.parents[1]
ACTIVE=CAP/'code/qwen35-streaming-s4';VISUAL=HERE.parent/'visual_memory_v1'
sys.path.insert(0,str(VISUAL));sys.path.insert(0,str(ACTIVE))
import psutil
import torch
from transformers import AutoProcessor
import experiment as reference
from scripts.baseline import file_hash,write_json
from scripts.pair_demo import verify_assets
from streaming.online import OnlinePairSession
from routing import FeatureWindowSession

CURRENT='Name the animal visible most recently.'
HISTORY='Name the main visible animal in this memory interval.'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sources():
    return {**reference.sources(),**{str(p.resolve()):file_hash(p) for p in HERE.glob('*.py')},
            '@routing_plan':file_hash(HERE/'方案.md')}
def budget():return {'model_loads':1,'vision_max':6,'questions_max':7,'generated_max':112,
    'language_max':160,'answer_tokens':16,'max_tokens':512,'wall_seconds':300,
    'ram_min_gib':3,'gpu_free_min_gib':5.5,'conversion_free_min_gib':2,
    'root_tokens_max':144,'root_tensor_bytes_max':592704,'current_question':CURRENT,'history_question':HISTORY,
    'trigger_encoded_pts':[7.,11.],'exact_target':.5,'future_target':99.}
def request(args):return {'model_dir':str(args.model_dir),'reference_run':str(args.reference_run)}
def make(model,processor):
    return FeatureWindowSession(OnlinePairSession(model,processor,max_tokens=512,
        max_question_tokens=64,max_answer_tokens=16,resize_mode='letterbox',video_logits='none'))


class Contracts(unittest.TestCase):
    def test_pending_future_queries_do_not_call_language(self):
        wrapper=make(self.model,self.processor)
        try:
            wrapper.append(self.pairs[0][0])
            with mock.patch.object(self.model.model.language_model,'forward',side_effect=AssertionError('No language expected')):
                self.assertEqual(wrapper.current(CURRENT)['reason'],'waiting_for_complete_pair')
                self.assertFalse(wrapper.at(99.,HISTORY)['model_called'])
            self.assertEqual(wrapper.memory.status()['events'],0)
        finally:wrapper.close()

    def test_failed_main_write_does_not_create_memory_and_hook_is_removed(self):
        wrapper=make(self.model,self.processor)
        initial=len(self.model.model.visual._forward_hooks)
        wrapper.append(self.pairs[0][0])
        try:
            with mock.patch.object(wrapper.session.main,'append',side_effect=RuntimeError('deliberate CPU failure')):
                with self.assertRaisesRegex(RuntimeError,'deliberate'):wrapper.append(self.pairs[0][1])
            self.assertEqual(wrapper.memory.status()['events'],0)
            self.assertEqual(wrapper.hashes,[])
            self.assertTrue(wrapper.invalid)
            self.assertEqual(len(self.model.model.visual._forward_hooks),initial)
        finally:wrapper.close()

    def test_current_and_compressed_history_query_preserve_main_then_continue(self):
        wrapper=make(self.model,self.processor)
        try:
            for pair in self.pairs[:4]:
                for packet in pair:wrapper.append(packet)
            before=reference.cache_fingerprint(wrapper.session.main)
            result=wrapper.current(CURRENT)
            self.assertTrue(result['model_called']);self.assertEqual(result['evidence_interval'],[6.,7.])
            root=next(b for b in wrapper.memory.export() if b.leaves>1)
            self.assertEqual(wrapper.at(root.center,HISTORY)['reason'],'compressed_only_no_exact_pair')
            self.assertTrue(wrapper.coarse(0,HISTORY)['model_called'])
            self.assertEqual(reference.cache_fingerprint(wrapper.session.main),before)
            for pair in self.pairs[4:]:
                for packet in pair:wrapper.append(packet)
            self.assertEqual(wrapper.session.last_encoded_pts,11.)
            self.assertEqual(wrapper.memory.status()['events'],6)
            self.assertTrue(wrapper.current(CURRENT)['model_called'])
            copy=wrapper.memory.export();copy[-1].features.zero_()
            self.assertFalse(torch.equal(copy[-1].features,wrapper.memory.export()[-1].features))
        finally:
            wrapper.close();self.assertEqual(wrapper.memory.status()['tensor_bytes'],0)


@torch.inference_mode()
def replay(model,processor,args,expected_ledger=None):
    pairs,metadata,assets=reference.inputs();wrapper=make(model,processor)
    counts={'vision':0,'language':0}
    def guard(name,limit):
        def hook(_m,_a):
            counts[name]+=1
            if counts[name]>limit:raise RuntimeError('Forward budget exceeded')
        return hook
    hooks=[model.model.visual.register_forward_pre_hook(guard('vision',6)),
           model.model.language_model.register_forward_pre_hook(guard('language',160))]
    events=[];generated=0;called=0
    def query(kind,operation,pts):
        nonlocal generated,called
        before=reference.cache_fingerprint(wrapper.session.main)
        state=wrapper.session.status();ledger=list(wrapper.session.ledger)
        result=operation()
        generated+=len(result.get('generated_ids',[]));called+=int(result.get('model_called',True))
        if reference.cache_fingerprint(wrapper.session.main)!=before or wrapper.session.status()!=state or wrapper.session.ledger!=ledger:
            raise AssertionError('Main sequence changed during query')
        event={'kind':kind,'encoded_pts':pts,'result':result,'main_unchanged':True,'memory':wrapper.memory.status()}
        events.append(event)
        print(json.dumps({'kind':kind,'encoded_pts':pts,'answer':result.get('answer'),
                          'model_called':result.get('model_called',True),'reason':result.get('reason'),
                          'evidence_interval':result.get('evidence_interval')},ensure_ascii=False),flush=True)
    try:
        for pair in pairs:
            for packet in pair:wrapper.append(packet)
            pts=wrapper.session.last_encoded_pts
            if pts in (7.,11.):
                query('main_current',lambda:dict(wrapper.session.ask(CURRENT,16),model_called=True),pts)
                query('window_current',lambda:wrapper.current(CURRENT),pts)
                query('coarse_first_interval',lambda:wrapper.coarse(0,HISTORY),pts)
                query('exact_pair_0.5',lambda:wrapper.at(.5,HISTORY),pts)
                query('future_99',lambda:wrapper.at(99.,HISTORY),pts)
        query('coarse_middle_interval',lambda:wrapper.coarse(1,HISTORY),11.)
        ledger=wrapper.session.ledger
        if expected_ledger is not None and ledger!=expected_ledger:raise ValueError('Actual main input ledger drift')
        if counts['vision']!=6 or generated>112 or called>7:raise RuntimeError('QA/vision/generated budget exceeded')
        if wrapper.memory.status()['tokens']>144 or wrapper.memory.status()['tensor_bytes']>592704:
            raise RuntimeError('Feature memory budget exceeded')
        prior=read(args.reference_run/'summary.json')
        feature_parity=[digest==row['feature_sha256'] for digest,row in zip(wrapper.hashes,prior['feature_blocks'])]
        summary={'passed':True,'events':events,'counts':counts,'model_called_queries':called,
                 'generated_tokens':generated,'input_frames':metadata,'source_assets':assets,
                 'ledger':ledger,'memory':wrapper.memory.status(),
                 'captured_feature_sha256':list(wrapper.hashes),'same_projected_features_as_gpu74':feature_parity,
                 'query_after_encoded_pts':7.,'continued_to_encoded_pts':11.,
                 'default_integration':False,'semantic_accuracy_evaluated':False,
                 'scope':'Finite controlled PTS ingestion; current unmerged pair, coarse interval not exact historical frame'}
        return summary
    finally:
        for hook in hooks:hook.remove()
        wrapper.close()


@torch.inference_mode()
def preflight(args):
    from tests.test_online import small_model
    processor=AutoProcessor.from_pretrained(args.model_dir,local_files_only=True);model=small_model(processor)
    pairs,metadata,assets=reference.inputs()
    Contracts.model,Contracts.processor,Contracts.pairs=model,processor,pairs
    with (args.out/'unittest.log').open('x',encoding='utf-8') as log:
        tests=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not tests.wasSuccessful():raise AssertionError('Feature window CPU contracts failed')
    report=replay(model,processor,args)
    report.update(cpu_tests=tests.testsRun,request=request(args),budget=budget(),source_hashes=sources(),
        model_manifest_sha256=file_hash(args.model_dir/'DOWNLOAD_MANIFEST.json'),
        assets=verify_assets(args.model_dir)['files'],reference_summary_sha256=file_hash(args.reference_run/'summary.json'))
    write_json(args.out/'proof.json',report)
    print(json.dumps({'passed':True,'cpu_tests':tests.testsRun,'vision':report['counts']['vision'],
                      'queries':report['model_called_queries'],'continued_to':11.,'gpu_loads':0}),flush=True)


@torch.inference_mode()
def gpu(args):
    if args.preflight is None or read(args.preflight/'status.json')!={'status':'completed','stage':'preflight'}:
        raise ValueError('Successful fresh CPU proof required')
    proof=read(args.preflight/'proof.json')
    if (proof['request']!=request(args) or proof['budget']!=budget() or proof['source_hashes']!=sources() or
        proof['assets']!=verify_assets(args.model_dir)['files'] or
        proof['model_manifest_sha256']!=file_hash(args.model_dir/'DOWNLOAD_MANIFEST.json') or
        proof['reference_summary_sha256']!=file_hash(args.reference_run/'summary.json')):
        raise ValueError('Proof source/parameters/assets/reference drift')
    pairs,metadata,assets=reference.inputs()
    if metadata!=proof['input_frames'] or assets!=proof['source_assets']:raise ValueError('Input changed')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=CAP,text=True).strip()
    if branch!='36tar/general-video-demo':raise ValueError('Registered personal branch required')
    free,total=torch.cuda.mem_get_info();ram=psutil.virtual_memory().available
    write_json(args.out/'resource_precheck.json',{'ram_available_bytes':ram,'gpu_free_bytes':free,'gpu_total_bytes':total,'branch':branch})
    if free<5.5*2**30 or ram<3*2**30:raise RuntimeError('Resource guard refused full model load')
    from qwen35_local import Qwen3_5ForConditionalGeneration
    begin=time.perf_counter()
    model=Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    if torch.cuda.mem_get_info()[0]<2*2**30:raise RuntimeError('No FP32 conversion room')
    model.float();torch.cuda.reset_peak_memory_stats()
    processor=AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    report=replay(model,processor,args,proof['ledger'])
    report.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  wall_seconds_including_load=time.perf_counter()-begin)
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
