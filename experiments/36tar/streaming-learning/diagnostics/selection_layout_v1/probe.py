"""Matched time/geometry selection experiment, unchanged model core."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
HERE=Path(__file__).resolve().parent;CAP=HERE.parents[1]
FIRST=HERE.parent/'first_fusion_v1';sys.path.insert(0,str(FIRST))
spec=importlib.util.spec_from_file_location('selection_first_reference',FIRST/'probe.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
history=prior.history;forest_ref=prior.forest_ref;runtime_ref=prior.runtime_ref;events=prior.events
import torch
import psutil
from transformers import AutoProcessor
from forest import raw
from features import raw_block
from selection_reader import variants,context,answer
from intervention import context as frozen_context
from scripts.baseline import file_hash,write_json
from scripts.pair_demo import verify_assets
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def config():return read(HERE/'cases.json')
def sources():
    return {**prior.sources(),**{str(p.resolve()):file_hash(p) for p in HERE.iterdir() if p.suffix in ('.py','.md','.json')}}
def inputs():return prior.inputs()
def budget():
    return {'model_loads':1,'vision_max':6,'questions_max':96,'generated_max':2304,'language_max':2800,
            'answer_tokens':24,'wall_seconds':360,'main_max_tokens':256,'qa_max_tokens':512,
            'reference_visual_tokens_max':96,'candidate_visual_tokens':48,'retained_rgb_max_bytes':4*2**20,
            'photo_decode_conservative_max_bytes':24*2**20,'external_working_conservative_max_bytes':16*2**20,
            'ram_min_gib':3,'gpu_free_min_gib':5.5,'conversion_free_min_gib':2,'precision':'fp32_ieee_math'}
def request(args):return {'model_dir':str(args.model_dir),'reference_run':str(args.reference_run),'branch':args.branch,'cases':config()}

class Contracts(unittest.TestCase):
    def make_nodes(self):
        width=self.model.get_input_embeddings().embedding_dim
        return [raw(raw_block(torch.randn((48,width),generator=torch.Generator().manual_seed(i+1822)),i,2*i,2*i+1)) for i in range(2)]
    def test_geometry_coverage_ownership_and_no_label_dependency(self):
        nodes=self.make_nodes();before=history.node_hashes(nodes);arms=variants(nodes)
        for name in ('first','last','odd','even','checker','merged'):
            node=arms[name+'_source'][0][0]
            self.assertEqual(len(node.features),48)
            self.assertEqual(len(torch.unique(node.coords,dim=0)),48)
        self.assertEqual(len(torch.unique(arms['checker_source'][0][0].coords[:,1:],dim=0)),48)
        self.assertEqual(len(torch.unique(arms['odd_source'][0][0].coords[:,1:],dim=0)),24)
        self.assertEqual(len(torch.unique(arms['even_source'][0][0].coords[:,1:],dim=0)),24)
        self.assertEqual(arms['first_source'][0][0].end,1)
        self.assertEqual(arms['last_source'][0][0].begin,2)
        arms['checker_source'][0][0].features.zero_()
        self.assertEqual(before,history.node_hashes(nodes))
    def test_matched_template_frozen_identity_and_value_position_controls(self):
        nodes=self.make_nodes();arms=variants(nodes);builder=self.wrapper.session.builder;q=config()['questions'][0]['text']
        out={k:context(self.model,builder,n,q,m) for k,(n,m) in arms.items()}
        for name in ('raw96','first_actual','legacy_merged','legacy_odd'):
            selection,mode=arms[name]
            old=frozen_context(self.model,builder,selection,q,'destination')
            for x,y in zip(out[name][:3],old[:3]):self.assertTrue(torch.equal(x,y))
        common=[out[n+'_grid'] for n in ('first','last','odd','even','checker','merged')]
        for other in common[1:]:
            self.assertTrue(torch.equal(common[0][1],other[1]));self.assertTrue(torch.equal(common[0][2],other[2]))
        actual,matched=out['first_actual'],out['first_source']
        self.assertTrue(torch.equal(actual[1],matched[1]))
        self.assertFalse(torch.equal(actual[2],matched[2]))
        self.assertEqual(actual[3]['visual_values_sha256'],matched[3]['visual_values_sha256'])
        for name in ('first','last','odd','even','checker','merged'):
            a,b=out[name+'_source'],out[name+'_grid']
            self.assertTrue(torch.equal(a[0],b[0]));self.assertTrue(torch.equal(a[2],b[2]))


@torch.inference_mode()
def replay(model,processor,args,expected=None,save_photos=False):
    cases,assets,rgb=inputs()
    if save_photos:forest_ref.photos(cases,args.out)
    counts={'vision':0,'language':0};answers=[];layouts={};timelines={};generated=0
    peak_external=peak_main=peak_temp=0
    def guard(kind):
        def hook(_m,_a):
            counts[kind]+=1
            if counts[kind]>budget()[kind+'_max']:raise RuntimeError('Forward budget')
        return hook
    handles=[model.model.visual.register_forward_pre_hook(guard('vision')),
             model.model.language_model.register_forward_pre_hook(guard('language'))]
    stream=(args.out/'answers.jsonl').open('x',encoding='utf-8')
    try:
        for case in cases:
            wrapper=runtime_ref.make(model,processor);nodes=[]
            layouts[case['name']]=[];timelines[case['name']]=[]
            try:
                for i,pair in enumerate(case['pairs']):
                    assert not wrapper.feed(pair[0])['encoded']
                    row=wrapper.feed(pair[1]);assert row['encoded']
                    nodes.append(raw(wrapper.bank.current));layouts[case['name']].append(wrapper.session.ledger[-1])
                    storage=runtime_ref.main_storage(wrapper)
                    peak_main=max(peak_main,storage['main_cache_tensor_bytes'])
                    peak_temp=max(peak_temp,storage['fingerprint_cpu_temporary_conservative_bytes'])
                    timelines[case['name']].append({'pair':i,'feature_sha256':row['feature_sha256'],
                                                  'reset':row['reset'],'storage':storage})
                if case['static'] and not torch.equal(nodes[0].features,nodes[1].features):
                    raise AssertionError('Repeated still did not produce identical pair features')
                arms=variants(nodes)
                resident=wrapper.bank.status()['tensor_bytes']+history.node_bytes(nodes)+sum(history.node_bytes(v[0]) for v in arms.values())
                peak_external=max(peak_external,resident)
                if resident>8*2**20:raise RuntimeError('Small diagnostic feature bound')
                for arm in config()['arms']:
                    selection,mode=arms[arm]
                    if any(n.end>wrapper.bank.last_end for n in selection):raise AssertionError('Future read')
                    for q in config()['questions']:
                        before=forest_ref.protected(wrapper,{});hashes=history.node_hashes(nodes);selected=history.node_hashes(selection)
                        vision=counts['vision'];result=answer(model,wrapper.session.builder,selection,q['text'],mode)
                        if before!=forest_ref.protected(wrapper,{}) or hashes!=history.node_hashes(nodes) or selected!=history.node_hashes(selection) or vision!=counts['vision']:
                            raise AssertionError('Value/position query changed protected state')
                        generated+=len(result['generated_ids'])
                        if len(answers)+1>96 or generated>2304:raise RuntimeError('Question budget')
                        row={'case':case['name'],'static_repeated_control':case['static'],'condition':arm,'topic':q['id'],
                             'result':result,'selected_feature_sha256':selected,
                             'main_unchanged':True,'bank_unchanged':True,'no_query_vision':True}
                        answers.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                        print(json.dumps({'case':case['name'],'condition':arm,'topic':q['id'],'answer':result['answer']},ensure_ascii=False),flush=True)
            finally:wrapper.close()
        if counts['vision']!=6:raise AssertionError('Not once per registered pair')
        if expected is not None and layouts!=expected:raise AssertionError('CPU/GPU layout drift')
        return {'passed':True,'answers':answers,'layout':layouts,'timelines':timelines,'counts':counts,
            'generated_tokens':generated,'model_questions':len(answers),'assets':assets,'rgb_bytes':rgb,
            'input_frames':{c['name']:c['identity'] for c in cases},'peak_external_resident_tensor_bytes':peak_external,
            'external_working_conservative_bytes':16*2**20,'photo_decode_conservative_bytes':24*2**20,
            'peak_main_cache_tensor_bytes':peak_main,'peak_fingerprint_cpu_temporary_bytes':peak_temp,
            'scope':'Old moving-cat diagnostic plus two new repeated-still same-species controls; no natural-video generalization'}
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
        if not tests.wasSuccessful():raise AssertionError('Independent value/position contract failed')
    finally:wrapper.close()
    report=replay(model,processor,args,save_photos=True)
    report.update(cpu_tests=tests.testsRun,source_hashes=sources(),request=request(args),budget=budget(),
        model_assets=verify_assets(args.model_dir)['files'],
        model_manifest_sha256=file_hash(args.model_dir/'DOWNLOAD_MANIFEST.json'),
        reference_sha256=file_hash(args.reference_run/'summary.json'),scope='Random CPU value/position/workflow proof, not recognition')
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
    if read(args.out/'bootstrap.json') != {'prepared_by':'selection_layout_v1_watchdog','files':expected}:
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
