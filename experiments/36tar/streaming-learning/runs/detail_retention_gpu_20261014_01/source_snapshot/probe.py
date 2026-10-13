"""Fixed binary visual-detail readout; conditional balanced schedule."""
import argparse
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

HERE=Path(__file__).resolve().parent;CAP=HERE.parents[1]
DEPTH=HERE.parent/'merge_depth_v1';sys.path.insert(0,str(DEPTH))
spec=importlib.util.spec_from_file_location('detail_depth_reference',DEPTH/'probe.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
history=prior.prior;forest_ref=prior.forest_ref;runtime_ref=prior.runtime_ref;events=prior.events
import torch
import psutil
from transformers import AutoProcessor
from forest import raw,ForestBank,RecentBank
from features import raw_block
from reader import packed_context,packed_answer,native_answer
from scripts.baseline import file_hash,write_json
from scripts.pair_demo import verify_assets

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def config():return read(HERE/'cases.json')
def sources():
    return {**prior.sources(),**{str(p.resolve()):file_hash(p) for p in HERE.iterdir()
                               if p.suffix in ('.py','.md','.json')}}
def inputs():return prior.inputs()
def budget():
    return {'model_loads':1,'vision_max':16,'questions_max':88,'generated_max':2112,'language_max':2600,
            'answer_tokens':24,'wall_seconds':480,'main_max_tokens':256,'qa_max_tokens':768,
            'reference_visual_tokens_max':384,'bank_visual_tokens_max':192,'rgb_max_bytes':12*2**20,
            'external_working_conservative_max_bytes':16*2**20,'ram_min_gib':3,'gpu_free_min_gib':5.5,
            'conversion_free_min_gib':2,'precision':'fp32_ieee_math'}
def request(args):return {'model_dir':str(args.model_dir),'reference_run':str(args.reference_run),'branch':args.branch,'cases':config()}


def property_value(topic,text):
    text=text.lower()
    if topic=='coat':
        values=set(re.findall(r'\b(light|dark)\b',text))
    elif topic=='surface':
        values=set()
        if re.search(r'\bgrass\b',text):values.add('grass')
        if re.search(r'\b(pavement|paved|concrete|stone|cobblestones?)\b',text):values.add('pavement')
    elif topic=='people':
        values=set(re.findall(r'\b(yes|no)\b',text))
    else:raise ValueError('Frozen property required')
    return next(iter(values)) if len(values)==1 else None


class Contracts(unittest.TestCase):
    def test_detail_readout_reservations_positions_and_balanced_span(self):
        width=self.model.get_input_embeddings().embedding_dim
        nodes=[raw(raw_block(torch.randn((48,width),generator=torch.Generator().manual_seed(i+333)),i,2*i,2*i+1)) for i in range(8)]
        before=history.node_hashes(nodes)
        left,_=prior.compress(nodes[:7]);balanced,_=prior.compress(nodes[:7],True)
        self.assertEqual(left.metadata()['mass'],balanced.metadata()['mass'])
        self.assertEqual(len(left.features),len(balanced.features))
        for question in config()['detail_questions']:
            refs=[packed_context(self.model,self.wrapper.session.builder,selected,question['text'],24)
                  for selected in (nodes,[left,nodes[-1]],[balanced,nodes[-1]])]
            self.assertEqual(len({a[3]['label'] for a in refs}),1)
            self.assertEqual(refs[0][3]['question_three_axis_positions'],refs[1][3]['question_three_axis_positions'])
            self.assertEqual(refs[1][3]['question_three_axis_positions'],refs[2][3]['question_three_axis_positions'])
            self.assertEqual(refs[1][3]['input_tokens'],refs[2][3]['input_tokens'])
            self.assertLessEqual(refs[0][3]['input_tokens']+24,768)
        self.assertEqual(before,history.node_hashes(nodes))


@torch.inference_mode()
def replay(model,processor,args,expected=None,save_photos=False):
    cases,assets,rgb=inputs()
    if save_photos:forest_ref.photos(cases,args.out)
    counts={'vision':0,'language':0};answers=[];layouts={};timelines={};stored={};generated=0
    peak_external=peak_main=peak_temp=0
    def guard(kind):
        def hook(_m,_a):
            counts[kind]+=1
            if counts[kind]>budget()[kind+'_max']:raise RuntimeError('Forward budget')
        return hook
    handles=[model.model.visual.register_forward_pre_hook(guard('vision')),
             model.model.language_model.register_forward_pre_hook(guard('language'))]
    stream=(args.out/'answers.jsonl').open('x',encoding='utf-8')
    def query(wrapper,banks,case,condition,selected,topic,question,layout='packed'):
        nonlocal generated
        before=forest_ref.protected(wrapper,banks);hashes=history.node_hashes(selected);vision=counts['vision']
        if any(n.end>15 for n in selected):raise AssertionError('Future evidence')
        result=prior.slim((packed_answer if layout=='packed' else native_answer)(model,wrapper.session.builder,selected,question,24))
        if before!=forest_ref.protected(wrapper,banks) or hashes!=history.node_hashes(selected) or counts['vision']!=vision:
            raise AssertionError('Query contamination')
        generated+=len(result['generated_ids'])
        if len(answers)+1>88 or generated>2112:raise RuntimeError('QA budget')
        row={'case':case['name'],'condition':condition,'topic':topic,'layout':layout,'result':result,
             'selected_feature_sha256':hashes,'main_unchanged':True,'banks_unchanged':True,'no_query_vision':True}
        answers.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
        print(json.dumps({'case':case['name'],'condition':condition,'topic':topic,'answer':result['answer']},ensure_ascii=False),flush=True)
        return len(answers)-1
    try:
        for case in cases:
            wrapper=runtime_ref.make(model,processor);banks={'auto':ForestBank(),'recent':RecentBank()};raw_nodes=[]
            layouts[case['name']]=[];timelines[case['name']]=[]
            try:
                for i,pair in enumerate(case['pairs']):
                    before=forest_ref.fingerprints(banks)
                    assert not wrapper.feed(pair[0])['encoded']
                    row=wrapper.feed(pair[1]);assert row['encoded'] and before==forest_ref.fingerprints(banks)
                    raw_nodes.append(raw(wrapper.bank.current))
                    updates=banks['auto'].append(wrapper.bank.current,row['group_update']['closed_old_group'])
                    banks['recent'].append(wrapper.bank.current)
                    layouts[case['name']].append(wrapper.session.ledger[-1])
                    storage=runtime_ref.main_storage(wrapper)
                    peak_main=max(peak_main,storage['main_cache_tensor_bytes'])
                    peak_temp=max(peak_temp,storage['fingerprint_cpu_temporary_conservative_bytes'])
                    resident=wrapper.bank.status()['tensor_bytes']+sum(b.status()['tensor_bytes'] for b in banks.values())
                    resident+=history.node_bytes(raw_nodes)+sum(history.node_bytes(n) for n in stored.values())
                    peak_external=max(peak_external,resident)
                    if resident>8*2**20:raise RuntimeError('Resident diagnostic feature budget')
                    timelines[case['name']].append({'pair':i,'feature_sha256':row['feature_sha256'],
                        'boundary':row['group_update']['closed_old_group'],'reset':row['reset'],
                        'updates':updates,'auto':banks['auto'].status()})
                oldest=banks['auto'].export()[:1]
                if oldest[0].begin>0 or oldest[0].end<3:raise AssertionError('Oldest node fails requested0..3 coverage')
                for q in config()['detail_questions']:
                    for name,selection in [('early_raw',raw_nodes[:2]),('auto_all',banks['auto'].export()),
                                            ('selected_early_root',oldest),('recent_only',banks['recent'].export())]:
                        query(wrapper,banks,case,name,selection,q['id'],q['text'])
                    for count in config()['merge_prefix_pairs']:
                        root,trace=prior.compress(raw_nodes[:count])
                        ids=[]
                        for name,selection in [('raw',raw_nodes[:count]+raw_nodes[-1:]),('merged',[root]+raw_nodes[-1:])]:
                            ids.append(query(wrapper,banks,case,f'n{count}_'+name,selection,q['id'],q['text']))
                        a,b=[answers[i]['result'] for i in ids]
                        if a['label']!=b['label'] or a['question_three_axis_positions']!=b['question_three_axis_positions']:
                            raise AssertionError('Same-span detail controls differ')
                        timelines[case['name']].append({'topic':q['id'],'pairs':count,'root':root.metadata(),'trace':trace,'queries':ids})
                deepest,_=prior.compress(raw_nodes[:7])
                query(wrapper,banks,case,'coarse_raw',raw_nodes,'coarse',config()['early_question'],'native')
                query(wrapper,banks,case,'coarse_merged',[deepest]+raw_nodes[-1:],'coarse',config()['early_question'])
                stored[case['name']]=raw_nodes
            finally:
                wrapper.close()
                for b in banks.values():b.clear()
        truth=read(HERE/'rubric.json')['expected']
        def correct(a):return property_value(a['topic'],a['result']['answer'])==truth[a['case']][a['topic']]
        triggers=[]
        for q in config()['detail_questions']:
            for count in config()['merge_prefix_pairs']:
                original=[a for a in answers if a['topic']==q['id'] and a['condition']==f'n{count}_raw']
                compressed=[a for a in answers if a['topic']==q['id'] and a['condition']==f'n{count}_merged']
                if len(original)==2 and all(correct(a) for a in original) and not all(correct(a) for a in compressed):
                    triggers.append({'topic':q['id'],'pairs':count})
        activate=bool(triggers) or args.stage=='preflight'
        if activate:
            for case in cases:
                wrapper=runtime_ref.make(model,processor)
                try:
                    balanced,trace=prior.compress(stored[case['name']][:7],True)
                    for q in config()['detail_questions']:
                        query(wrapper,{},case,'balanced', [balanced]+stored[case['name']][-1:],q['id'],q['text'])
                    timelines[case['name']].append({'balanced_root':balanced.metadata(),'trace':trace})
                finally:wrapper.close()
        if counts['vision']!=16 or len(answers)>88 or generated>2112:raise RuntimeError('Final budget')
        if expected is not None and layouts!=expected:raise AssertionError('CPU/GPU layout drift')
        return {'passed':True,'answers':answers,'layout':layouts,'timelines':timelines,'counts':counts,
            'generated_tokens':generated,'model_questions':len(answers),'assets':assets,'rgb_bytes':rgb,
            'input_frames':{c['name']:c['identity'] for c in cases},'candidate_executed':activate,
            'candidate_trigger':triggers,'candidate_cpu_forced':args.stage=='preflight',
            'peak_external_resident_tensor_bytes':peak_external,'external_working_conservative_bytes':16*2**20,
            'peak_main_cache_tensor_bytes':peak_main,'peak_fingerprint_cpu_temporary_bytes':peak_temp,
            'scope':'Known pair, fixed binary details; species/property correlation and guessing retained'}
    finally:
        stream.close()
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
        if not tests.wasSuccessful():raise AssertionError('Detail reader contract failure')
    finally:wrapper.close()
    report=replay(model,processor,args,save_photos=True)
    report.update(cpu_tests=tests.testsRun,source_hashes=sources(),request=request(args),budget=budget(),
        model_assets=verify_assets(args.model_dir)['files'],
        model_manifest_sha256=file_hash(args.model_dir/'DOWNLOAD_MANIFEST.json'),
        reference_sha256=file_hash(args.reference_run/'summary.json'),scope='Random CPU workflow and candidate coverage, no semantic accuracy')
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
    if read(args.out/'bootstrap.json') != {'prepared_by':'detail_retention_v1_watchdog','files':expected}:
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
