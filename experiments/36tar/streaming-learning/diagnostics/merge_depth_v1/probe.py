"""A fixed paired compression ladder; conditional same-budget balanced merge."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

HERE=Path(__file__).resolve().parent
CAP=HERE.parents[1]
HISTORY=HERE.parent/'history_causal_v1'
sys.path.insert(0,str(HISTORY))
spec=importlib.util.spec_from_file_location('depth_history_reference',HISTORY/'probe.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
forest_ref=prior.prior;runtime_ref=forest_ref.prior;events=prior.events
import torch
import psutil
from transformers import AutoProcessor
from forest import raw,merge,ForestBank,RecentBank
from features import raw_block
from reader import packed_context,packed_answer,native_answer
from scripts.baseline import file_hash,write_json
from scripts.pair_demo import verify_assets
from streaming.window import FramePacket

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def config():return read(HERE/'cases.json')
def sources():
    return {**prior.sources(),**{str(p.resolve()):file_hash(p) for p in HERE.iterdir()
                               if p.suffix in ('.py','.md','.json')}}
def budget():
    return {'model_loads':1,'vision_max':16,'questions_max':56,'generated_max':1344,'language_max':1500,
            'answer_tokens':24,'wall_seconds':480,'main_max_tokens':256,'qa_max_tokens':768,
            'reference_visual_tokens_max':384,'bank_visual_tokens_max':192,'rgb_max_bytes':12*2**20,
            'external_working_conservative_max_bytes':16*2**20,'ram_min_gib':3,'gpu_free_min_gib':5.5,
            'conversion_free_min_gib':2,'precision':'fp32_ieee_math'}
def request(args):
    return {'model_dir':str(args.model_dir),'reference_run':str(args.reference_run),'branch':args.branch,'cases':config()}
def inputs():
    cfg=config();registered=events.read(CAP/'diagnostics/general_stream_v1/cases.json')['videos']
    registered+=events.read(events.HERE/'cases.json')['natural_sources']
    suffix_source=next(r for r in registered if r['id']==cfg['suffix']['source'])
    suffix,suffix_id=events.segment(suffix_source,cfg['suffix']['start'],cfg['suffix']['frames'],True)
    assets={suffix_source['path']:suffix_source['sha256']};cases=[]
    for animal,start in cfg['pairs'][0]['prefix'].items():
        source=next(r for r in registered if r['id']==animal)
        prefix,identity=events.segment(source,start,4,False);assets[source['path']]=source['sha256']
        pairs=[]
        for i,pair in enumerate(prefix+suffix):
            pairs.append([FramePacket(2*i+j,float(2*i+j),p.rgb,2*i+j+1,p.decode_seconds) for j,p in enumerate(pair)])
        identities=[{**r,'session_frame_id':i,'session_pts':float(i),'session_time_is_source_time':False}
                    for i,r in enumerate(identity+suffix_id)]
        cases.append({'name':animal,'pairs':pairs,'identity':identities})
    assert cases[0]['identity'][4:]==cases[1]['identity'][4:]
    rgb=sum(p.rgb.nbytes for c in cases for pair in c['pairs'] for p in pair)
    if rgb>12*2**20 or sum(len(c['pairs']) for c in cases)!=16:raise ValueError('Input budget')
    return cases,assets,rgb


def compress(nodes,balanced=False):
    """Adjacent fixed two-pair segments; no labels or model answers in merging."""
    if not 2<=len(nodes)<=7:raise ValueError('Fixed compression ladder only')
    pieces=[];traces=[]
    for i in range(0,len(nodes),2):
        chunk=nodes[i:i+2]
        if len(chunk)==2:
            value=merge(chunk[0],chunk[1],within=True)
            traces.append({'kind':'within_control_segment','left':chunk[0].metadata(),
                           'right':chunk[1].metadata(),'parent':value.metadata()})
        else:value=chunk[0].copy()
        pieces.append(value)
    if balanced:
        while len(pieces)>1:
            next_level=[]
            for i in range(0,len(pieces),2):
                if i+1==len(pieces):next_level.append(pieces[i]);continue
                value=merge(pieces[i],pieces[i+1])
                traces.append({'kind':'balanced_cross_segment','left':pieces[i].metadata(),
                               'right':pieces[i+1].metadata(),'parent':value.metadata()})
                next_level.append(value)
            pieces=next_level
        value=pieces[0]
    else:
        value=pieces[0]
        for right in pieces[1:]:
            parent=merge(value,right)
            traces.append({'kind':'left_fold_cross_segment','left':value.metadata(),
                           'right':right.metadata(),'parent':parent.metadata()})
            value=parent
    return value,traces


def slim(result):
    result=dict(result)
    ids=result.pop('input_ids');positions=result.pop('four_axis_positions')
    result['position_shape']=[4,1,len(ids)]
    result['trace_note']='IDs and four-axis SHA retained; reconstruct from frozen reader/source nodes, no repeated full list'
    return result


class Contracts(unittest.TestCase):
    def test_ladder_and_balanced_mass_span_and_reading_positions(self):
        width=self.model.get_input_embeddings().embedding_dim
        nodes=[raw(raw_block(torch.randn((48,width),generator=torch.Generator().manual_seed(i+211)),i,2*i,2*i+1)) for i in range(8)]
        before=prior.node_hashes(nodes)
        for count,depth in zip((2,4,6,7),(1,2,3,4)):
            root,trace=compress(nodes[:count]);self.assertEqual(root.depth,depth)
            self.assertEqual(root.metadata()['mass'],48*count);self.assertEqual(len(root.features),48)
            original=sum((n.features*n.mass).sum(0) for n in nodes[:count])
            torch.testing.assert_close((root.features*root.mass).sum(0),original,atol=1e-4,rtol=1e-4)
            for suffix in ([],nodes[-1:]):
                a=packed_context(self.model,self.wrapper.session.builder,nodes[:count]+suffix,config()['question'],24)
                b=packed_context(self.model,self.wrapper.session.builder,[root]+suffix,config()['question'],24)
                self.assertEqual(a[3]['label'],b[3]['label'])
                self.assertEqual(a[3]['question_three_axis_positions'],b[3]['question_three_axis_positions'])
        balanced,_=compress(nodes[:7],True)
        self.assertEqual(balanced.depth,3);self.assertEqual(balanced.metadata()['mass'],336)
        self.assertEqual(before,prior.node_hashes(nodes))
    def test_fixed_pairs_and_real_long_merge_schedule(self):
        cases,_assets,rgb=inputs();self.assertLess(rgb,12*2**20)
        bank=ForestBank()
        for i in range(8):
            block=raw_block(torch.randn((48,16),generator=torch.Generator().manual_seed(i)),i,2*i,2*i+1)
            bank.append(block,boundary=(i>0 and i%2==0))
        self.assertGreater(bank.long_merges,0)
        self.assertLessEqual(bank.status()['tokens'],192)
        self.assertEqual(cases[0]['identity'][4:],cases[1]['identity'][4:])


@torch.inference_mode()
def replay(model,processor,args,expected=None,save_photos=False):
    cases,assets,rgb=inputs()
    if save_photos:forest_ref.photos(cases,args.out)
    counts={'vision':0,'language':0};answers=[];layouts={};timelines={};stored={};generated=0
    peak_external=peak_main=peak_temp=0
    def guard(kind):
        def hook(_m,_a):
            counts[kind]+=1
            if counts[kind]>budget()[kind+'_max']:raise RuntimeError('Forward budget exceeded')
        return hook
    handles=[model.model.visual.register_forward_pre_hook(guard('vision')),
             model.model.language_model.register_forward_pre_hook(guard('language'))]
    log=(args.out/'answers.jsonl').open('x',encoding='utf-8')
    def query(wrapper,banks,case,condition,nodes,layout='packed',question=None,horizon=15):
        nonlocal generated
        q=question or config()['question'];state=forest_ref.protected(wrapper,banks)
        hashes=prior.node_hashes(nodes);vision=counts['vision']
        if any(n.end>horizon for n in nodes):raise AssertionError('Future read')
        result=slim((packed_answer if layout=='packed' else native_answer)(model,wrapper.session.builder,nodes,q,24))
        if state!=forest_ref.protected(wrapper,banks) or hashes!=prior.node_hashes(nodes) or counts['vision']!=vision:
            raise AssertionError('Query contaminated protected state')
        generated+=len(result['generated_ids'])
        if len(answers)+1>56 or generated>1344:raise RuntimeError('QA budget')
        row={'case':case['name'],'condition':condition,'layout':layout,'result':result,
             'selected_feature_sha256':hashes,'main_unchanged':True,'banks_unchanged':True,'no_query_vision':True,
             'encoded_horizon':horizon}
        answers.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
        print(json.dumps({'case':case['name'],'condition':condition,'answer':result['answer']},ensure_ascii=False),flush=True)
        return len(answers)-1
    try:
        for case in cases:
            wrapper=runtime_ref.make(model,processor)
            banks={'auto':ForestBank(),'control_segments':ForestBank(),'recent':RecentBank()};raw_nodes=[]
            layouts[case['name']]=[];timelines[case['name']]=[]
            try:
                for i,pair in enumerate(case['pairs']):
                    before=forest_ref.fingerprints(banks)
                    assert not wrapper.feed(pair[0])['encoded']
                    row=wrapper.feed(pair[1]);assert row['encoded']
                    assert before==forest_ref.fingerprints(banks)
                    block=wrapper.bank.current;raw_nodes.append(raw(block))
                    updates={'auto':banks['auto'].append(block,row['group_update']['closed_old_group']),
                             'control_segments':banks['control_segments'].append(block,i>0 and i%2==0)}
                    banks['recent'].append(block)
                    layouts[case['name']].append(wrapper.session.ledger[-1])
                    storage=runtime_ref.main_storage(wrapper)
                    peak_main=max(peak_main,storage['main_cache_tensor_bytes'])
                    peak_temp=max(peak_temp,storage['fingerprint_cpu_temporary_conservative_bytes'])
                    resident=wrapper.bank.status()['tensor_bytes']+sum(b.status()['tensor_bytes'] for b in banks.values())
                    resident+=prior.node_bytes(raw_nodes)+sum(prior.node_bytes(v) for v in stored.values())
                    peak_external=max(peak_external,resident)
                    if resident>8*2**20:raise RuntimeError('Resident feature budget')
                    timelines[case['name']].append({'pair':i,'feature_sha256':row['feature_sha256'],
                        'auto_boundary':row['group_update']['closed_old_group'],'forced_boundary':i>0 and i%2==0,
                        'reset':row['reset'],'updates':updates,'auto':banks['auto'].status(),
                        'control_segments':banks['control_segments'].status()})
                    if i<2:query(wrapper,banks,case,'single_'+str(i),raw_nodes[-1:],'native',config()['single_question'],2*i+1)
                query(wrapper,banks,case,'full_native',raw_nodes,'native',config()['early_question'])
                query(wrapper,banks,case,'auto_native',banks['auto'].export(),'native',config()['early_question'])
                query(wrapper,banks,case,'auto_packed',banks['auto'].export(),'packed',config()['early_question'])
                query(wrapper,banks,case,'control_segments_native',banks['control_segments'].export(),'native',config()['early_question'])
                query(wrapper,banks,case,'recent_only',banks['recent'].export(),'native',config()['early_question'])
                for count in config()['merge_prefix_pairs']:
                    root,trace=compress(raw_nodes[:count])
                    timeline={'compression_pairs':count,'root':root.metadata(),'trace':trace,'queries':{}}
                    for label,selected in [('raw_alone',raw_nodes[:count]),('merged_alone',[root]),
                                          ('raw_with_recent',raw_nodes[:count]+raw_nodes[-1:]),
                                          ('merged_with_recent',[root]+raw_nodes[-1:])]:
                        index=query(wrapper,banks,case,f'n{count}_'+label,selected)
                        timeline['queries'][label]=index
                    for a,b in [('raw_alone','merged_alone'),('raw_with_recent','merged_with_recent')]:
                        left,right=[answers[timeline['queries'][k]]['result'] for k in (a,b)]
                        if left['label']!=right['label'] or left['question_three_axis_positions']!=right['question_three_axis_positions']:
                            raise AssertionError('Same-span packed controls differ')
                    timelines[case['name']].append(timeline)
                if banks['control_segments'].long_merges==0:raise AssertionError('Forced online long merge not exercised')
                stored[case['name']]=raw_nodes
            finally:
                wrapper.close()
                for b in banks.values():b.clear()
        def correct(a):
            return set(re.findall(r'\b(dog|cat)\b',a['result']['answer'].lower()))=={a['case']}
        triggers=[]
        for count in config()['merge_prefix_pairs']:
            for suffix in ('alone','with_recent'):
                raw_rows=[a for a in answers if a['condition']==f'n{count}_raw_'+suffix]
                merged_rows=[a for a in answers if a['condition']==f'n{count}_merged_'+suffix]
                if len(raw_rows)==2 and all(correct(a) for a in raw_rows) and not all(correct(a) for a in merged_rows):
                    triggers.append({'pairs':count,'read':suffix})
        activate=bool(triggers) or args.stage=='preflight'
        if activate:
            for case in cases:
                wrapper=runtime_ref.make(model,processor)
                try:
                    balanced,trace=compress(stored[case['name']][:7],True)
                    for name,selection in [('balanced_alone',[balanced]),('balanced_with_recent',[balanced]+stored[case['name']][-1:])]:
                        query(wrapper,{},case,name,selection)
                    timelines[case['name']].append({'balanced_candidate':balanced.metadata(),'trace':trace})
                finally:wrapper.close()
        if counts['vision']!=16 or len(answers)>56 or generated>1344:raise RuntimeError('Final budget')
        if expected is not None and layouts!=expected:raise AssertionError('CPU/GPU video layout drift')
        return {'passed':True,'answers':answers,'layout':layouts,'timelines':timelines,'counts':counts,
                'generated_tokens':generated,'model_questions':len(answers),'assets':assets,'rgb_bytes':rgb,
                'input_frames':{c['name']:c['identity'] for c in cases},'candidate_executed':activate,
                'candidate_trigger':triggers,'candidate_cpu_forced':args.stage=='preflight',
                'peak_external_resident_tensor_bytes':peak_external,'external_working_conservative_bytes':16*2**20,
                'peak_main_cache_tensor_bytes':peak_main,'peak_fingerprint_cpu_temporary_bytes':peak_temp,
                'scope':'One known, previously answerable paired prefix; controlled merge depth, no generalization'}
    finally:
        log.close()
        for h in handles:h.remove()


@torch.inference_mode()
def preflight(args):
    from tests.test_online import small_model
    processor=AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    model=small_model(processor);wrapper=runtime_ref.make(model,processor)
    Contracts.model,Contracts.wrapper=model,wrapper
    try:
        with (args.out/'unittest.log').open('x',encoding='utf-8') as stream:
            tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
        if not tests.wasSuccessful():raise AssertionError('Compression ladder contract failure')
    finally:wrapper.close()
    report=replay(model,processor,args,save_photos=True)
    report.update(cpu_tests=tests.testsRun,source_hashes=sources(),request=request(args),budget=budget(),
        model_assets=verify_assets(args.model_dir)['files'],
        model_manifest_sha256=file_hash(args.model_dir/'DOWNLOAD_MANIFEST.json'),
        reference_sha256=file_hash(args.reference_run/'summary.json'),scope='Random CPU mechanism and candidate coverage; no semantic score')
    write_json(args.out/'proof.json',report)


@torch.inference_mode()
def gpu(args):
    if args.preflight is None or read(args.preflight/'status.json') != {'status':'completed','stage':'preflight'}:
        raise ValueError('Fresh successful CPU proof required')
    proof = read(args.preflight/'proof.json')
    if (proof['source_hashes'] != sources() or proof['request'] != request(args) or proof['budget'] != budget()
            or proof['model_assets'] != verify_assets(args.model_dir)['files']
            or proof['model_manifest_sha256'] != file_hash(args.model_dir/'DOWNLOAD_MANIFEST.json')
            or proof['reference_sha256'] != file_hash(args.reference_run/'summary.json')):
        raise ValueError('Proof drift')
    if read(args.reference_run/'summary.json')['calibration']['threshold'] != config()['threshold']:
        raise ValueError('Frozen grouping threshold drift')
    cases, assets, _ = inputs()
    if assets != proof['assets'] or {c['name']:c['identity'] for c in cases} != proof['input_frames']:
        raise ValueError('Input pixel identity drift')
    del cases
    branch = subprocess.check_output(['git','branch','--show-current'], cwd=CAP, text=True).strip()
    if args.branch in ('', 'main', 'master', 'HEAD') or branch != args.branch:
        raise ValueError('Configured personal branch required')
    free, total = torch.cuda.mem_get_info(); ram = psutil.virtual_memory().available
    write_json(args.out/'resource_precheck.json', {'ram_available_bytes':ram,'gpu_free_bytes':free,'gpu_total_bytes':total,'branch':branch})
    if ram < 3*2**30 or free < 5.5*2**30: raise RuntimeError('Resource gate refused load')
    from qwen35_local import Qwen3_5ForConditionalGeneration
    started = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    if torch.cuda.mem_get_info()[0] < 2*2**30: raise RuntimeError('No FP32 conversion room')
    model.float(); torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    result = replay(model, processor, args, proof['layout'])
    result.update(source_hashes=sources(), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  wall_seconds_including_load=time.perf_counter()-started)
    write_json(args.out/'summary.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preflight','gpu'), required=True)
    for name in ('model-dir','reference-run','out'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--branch', required=True)
    args = parser.parse_args()
    for name in ('model_dir','reference_run','out','preflight'):
        if getattr(args,name) is not None: setattr(args,name,getattr(args,name).resolve())
    if {p.name for p in args.out.iterdir()} != {'source_snapshot','bootstrap.json'}:
        raise ValueError('Fresh watchdog bootstrap required')
    expected = {p.name:file_hash(p) for p in HERE.iterdir() if p.suffix in ('.py','.md','.json')}
    if read(args.out/'bootstrap.json') != {'prepared_by':'merge_depth_v1_watchdog','files':expected}:
        raise ValueError('Bootstrap source identity drift')
    (args.out/'tmp').mkdir(); tempfile.tempdir = str(args.out/'tmp')
    torch.set_num_threads(1); torch.manual_seed(42)
    write_json(args.out/'source_hashes.json',sources()); write_json(args.out/'budget.json',budget())
    try:
        (preflight if args.stage == 'preflight' else gpu)(args)
        write_json(args.out/'status.json',{'status':'completed','stage':args.stage})
    except Exception as error:
        write_json(args.out/'status.json',{'status':'failed','stage':args.stage,'error':type(error).__name__,'message':str(error)})
        (args.out/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise


if __name__ == '__main__': main()
