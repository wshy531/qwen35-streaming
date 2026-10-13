"""Frozen paired early-memory intervention and bounded readout diagnostics."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

HERE = Path(__file__).resolve().parent
CAP = HERE.parents[1]
FROZEN = HERE.parent/'forest_runtime_v1'
sys.path.insert(0,str(FROZEN))
spec = importlib.util.spec_from_file_location('history_forest_reference',FROZEN/'probe.py')
prior = importlib.util.module_from_spec(spec); spec.loader.exec_module(prior)
events = prior.events
import torch
import psutil
from transformers import AutoProcessor
from forest import raw, merge, fingerprint
from features import raw_block
from reader import native_context, packed_context, native_answer, packed_answer
from scripts.baseline import write_json,file_hash
from scripts.pair_demo import verify_assets
from streaming.window import FramePacket


def read(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def config(): return read(HERE/'cases.json')
def sources():
    return {**prior.sources(), **{str(p.resolve()):file_hash(p) for p in HERE.iterdir()
                                  if p.suffix in ('.py','.md','.json')}}
def budget():
    return {'model_loads':1,'vision_max':32,'questions_max':64,'generated_max':1536,'language_max':2000,
            'answer_tokens':24,'wall_seconds':600,'main_max_tokens':256,'qa_max_tokens':768,
            'reference_visual_tokens_max':384,'bank_visual_tokens_max':192,'rgb_max_bytes':12*2**20,
            'external_working_conservative_max_bytes':16*2**20,'ram_min_gib':3,
            'gpu_free_min_gib':5.5,'conversion_free_min_gib':2,'precision':'fp32_ieee_math'}
def request(args):
    return {'model_dir':str(args.model_dir),'reference_run':str(args.reference_run),
            'branch':args.branch,'cases':config()}


def inputs():
    registered = events.read(CAP/'diagnostics/general_stream_v1/cases.json')['videos']
    registered += events.read(events.HERE/'cases.json')['natural_sources']
    cfg = config(); rows=[]; assets={}
    suffix_source = next(r for r in registered if r['id'] == cfg['suffix']['source'])
    suffix, suffix_identity = events.segment(suffix_source,cfg['suffix']['start'],cfg['suffix']['frames'],True)
    assets[suffix_source['path']] = suffix_source['sha256']
    for matched in cfg['pairs']:
        for animal,start in matched['prefix'].items():
            source = next(r for r in registered if r['id']==animal)
            prefix,identity = events.segment(source,start,4,False)
            assets[source['path']]=source['sha256']
            sequence=prefix+suffix; identities=identity+suffix_identity
            pairs=[]; truth=[]
            for index,pair in enumerate(sequence):
                pairs.append([FramePacket(2*index+j,float(2*index+j),p.rgb,2*index+j+1,p.decode_seconds)
                              for j,p in enumerate(pair)])
            for i,entry in enumerate(identities):
                truth.append({**entry,'session_frame_id':i,'session_pts':float(i),
                              'session_time_is_source_time':False})
            rows.append({'name':matched['id']+'_'+animal,'match':matched['id'],'expected_early':animal,
                         'pairs':pairs,'identity':truth})
    for index in (0,2):
        a,b=rows[index:index+2]
        if a['identity'][4:] != b['identity'][4:]: raise AssertionError('Matched suffix differs')
        if [r['session_pts'] for r in a['identity']] != [r['session_pts'] for r in b['identity']]:
            raise AssertionError('Matched session timing differs')
    for animal in ('dog','cat'):
        a=next(r for r in rows if r['name']=='development_'+animal)
        b=next(r for r in rows if r['name']=='confirmation_'+animal)
        if {r['source_frame_id'] for r in a['identity'][:4]} & {r['source_frame_id'] for r in b['identity'][:4]}:
            raise AssertionError('Confirmation early frames overlap development')
    rgb=sum(p.rgb.nbytes for case in rows for pair in case['pairs'] for p in pair)
    if rgb>budget()['rgb_max_bytes'] or sum(len(c['pairs']) for c in rows)!=32:
        raise ValueError('Registered input budget changed')
    return rows,assets,rgb


def node_hashes(nodes):
    return [[file_tensor_sha(t) for t in (n.features,n.mass,n.coords)] for n in nodes]
def file_tensor_sha(t): return hashlib.sha256(t.contiguous().cpu().numpy().tobytes()).hexdigest()
def node_bytes(nodes): return sum(n.metadata()['tensor_bytes'] for n in nodes)


class Contracts(unittest.TestCase):
    def make_nodes(self):
        width=self.model.get_input_embeddings().embedding_dim
        return [raw(raw_block(torch.randn((48,width),generator=torch.Generator().manual_seed(i+70)),
                              i,2*i,2*i+1)) for i in range(8)]
    def test_native_reference_and_packed_position_control(self):
        builder=self.wrapper.session.builder; nodes=self.make_nodes(); q=config()['early_question']
        a=native_context(self.model,builder,nodes[:1],q,24)
        b=prior.context(self.model,builder,nodes[:1],q,24)
        for x,y in zip(a[:3],b[:3]): self.assertTrue(torch.equal(x,y))
        compressed=merge(nodes[0],nodes[1],within=True)
        a=packed_context(self.model,builder,nodes[:2],q,24)
        b=packed_context(self.model,builder,[compressed],q,24)
        self.assertEqual(a[3]['label'],b[3]['label'])
        self.assertEqual(a[3]['question_three_axis_positions'],b[3]['question_three_axis_positions'])
        self.assertNotEqual(a[3]['input_tokens'],b[3]['input_tokens'])
        self.assertEqual(node_hashes(nodes),node_hashes(nodes))
        full=native_context(self.model,builder,nodes,q,24)
        self.assertEqual(full[3]['visual_tokens'],384)
        with self.assertRaises(ValueError): packed_context(self.model,builder,[nodes[1],nodes[0]],q,24)
        with self.assertRaises(ValueError): native_context(self.model,builder,nodes+nodes[-1:],q,24)
    def test_inputs_are_paired_and_not_labelled_in_model_prompt(self):
        cases,_assets,rgb=inputs(); self.assertLess(rgb,12*2**20)
        for case in cases:
            self.assertNotIn(case['expected_early'],config()['early_question'].lower())
            self.assertEqual([r['session_pts'] for r in case['identity']],list(map(float,range(16))))


@torch.inference_mode()
def replay(model,processor,args,expected=None,save_photos=False):
    cases,assets,rgb=inputs()
    if save_photos: prior.photos(cases,args.out)
    counts={'vision':0,'language':0}; answers=[]; layouts={}; timelines={}; stored={}; swaps=[]
    generated=peak_main=peak_temp=peak_external=0
    def guard(kind):
        def hook(_m,_a):
            counts[kind]+=1
            if counts[kind]>budget()[kind+'_max']: raise RuntimeError('Forward budget exceeded')
        return hook
    handles=[model.model.visual.register_forward_pre_hook(guard('vision')),
             model.model.language_model.register_forward_pre_hook(guard('language'))]
    stream=(args.out/'answers.jsonl').open('x',encoding='utf-8')
    def query(wrapper,banks,case,condition,selected,layout='native',question=None):
        nonlocal generated,peak_external
        q=question or config()['early_question']
        before=prior.protected(wrapper,banks); vision=counts['vision']; hashes=node_hashes(selected)
        if any(n.end>wrapper.bank.last_end for n in selected): raise AssertionError('Future evidence')
        result=(native_answer if layout=='native' else packed_answer)(model,wrapper.session.builder,selected,q,24)
        if before!=prior.protected(wrapper,banks) or vision!=counts['vision'] or hashes!=node_hashes(selected):
            raise AssertionError('Query changed state, evidence or vision count')
        generated+=len(result['generated_ids'])
        if len(answers)+1>64 or generated>1536: raise RuntimeError('QA budget exceeded')
        row={'case':case['name'],'condition':condition,'layout':layout,'result':result,
             'selected_feature_sha256':hashes,'encoded_horizon':wrapper.bank.last_end,
             'main_epoch':wrapper.epoch,'main_unchanged':True,'banks_unchanged':True,'no_query_vision':True}
        answers.append(row); stream.write(json.dumps(row,ensure_ascii=False)+'\n'); stream.flush()
        print(json.dumps({'case':case['name'],'condition':condition,'answer':result['answer']},ensure_ascii=False),flush=True)
        return len(answers)-1
    try:
        for case in cases:
            wrapper=prior.prior.make(model,processor); banks=prior.banks(); raw_nodes=[]
            layouts[case['name']]=[]; timelines[case['name']]=[]
            try:
                for i,pair in enumerate(case['pairs']):
                    row,updates=prior.feed(wrapper,banks,pair)
                    original=raw(wrapper.bank.current); raw_nodes.append(original)
                    layouts[case['name']].append(wrapper.session.ledger[-1])
                    storage=prior.prior.main_storage(wrapper)
                    peak_main=max(peak_main,storage['main_cache_tensor_bytes'])
                    peak_temp=max(peak_temp,storage['fingerprint_cpu_temporary_conservative_bytes'])
                    resident=wrapper.bank.status()['tensor_bytes']+sum(b.status()['tensor_bytes'] for b in banks.values())
                    resident+=node_bytes(raw_nodes)+sum(node_bytes(v['forest']) for v in stored.values())
                    peak_external=max(peak_external,resident)
                    if resident>8*2**20: raise RuntimeError('Resident diagnostic feature cap')
                    timelines[case['name']].append({'pair':i,'feature_sha256':row['feature_sha256'],
                        'boundary':row['group_update']['closed_old_group'],'reset':row['reset'],
                        'forest_updates':updates['forest'],'forest':banks['forest'].status()})
                    if i<2: query(wrapper,banks,case,'single_'+str(i),[original],question=config()['single_question'])
                forest=banks['forest'].export(); recent=banks['recent4'].export()
                if recent[0].first<=1: raise AssertionError('Recent includes early prefix')
                for condition,selected,layout in [('full_native',raw_nodes,'native'),('full_packed',raw_nodes,'packed'),
                    ('forest_native',forest,'native'),('forest_packed',forest,'packed'),('recent_only',recent,'native')]:
                    query(wrapper,banks,case,condition,selected,layout)
                root=forest[0]; same_range=raw_nodes[root.first:root.last+1]; latest=raw_nodes[-1:]
                if root.last>=7: raise AssertionError('No distinct old root for quartet')
                quartet={}
                for name,selected in [('raw_alone',same_range),('compressed_alone',[root]),
                                      ('raw_with_recent',same_range+latest),('compressed_with_recent',[root]+latest)]:
                    quartet[name]=query(wrapper,banks,case,'root_'+name,selected,'packed',config()['segment_question'])
                for left,right in [('raw_alone','compressed_alone'),('raw_with_recent','compressed_with_recent')]:
                    a,b=[answers[quartet[k]]['result'] for k in (left,right)]
                    if a['label']!=b['label'] or a['question_three_axis_positions']!=b['question_three_axis_positions']:
                        raise AssertionError('Packed matched-span position control failed')
                prefix_root=merge(raw_nodes[0],raw_nodes[1],within=True)
                query(wrapper,banks,case,'prefix_compressed_alone',[prefix_root],'packed',config()['segment_question'])
                query(wrapper,banks,case,'prefix_compressed_with_recent',[prefix_root]+latest,'packed',config()['segment_question'])
                before=prior.fingerprints(banks); wrapper.reset_main()
                if before!=prior.fingerprints(banks): raise AssertionError('Main reset changed memory')
                reset=query(wrapper,banks,case,'forest_native_after_reset',forest)
                native=next(a for a in answers if a['case']==case['name'] and a['condition']=='forest_native')
                if answers[reset]['result']['generated_ids']!=native['result']['generated_ids']:
                    raise AssertionError('Reset altered external read')
                stored[case['name']]={'forest':forest,'recent':recent,'root_range':[root.first,root.last],
                                      'root_interval':[root.begin,root.end]}
            finally:
                wrapper.close()
                for b in banks.values(): b.clear()
                del raw_nodes
        # Only evidence is exchanged; unchanged root spans prevent timestamp/position leakage.
        for matched in config()['pairs']:
            a,b=[stored[matched['id']+'_'+animal] for animal in ('dog','cat')]
            if a['root_range']!=b['root_range'] or a['root_interval']!=b['root_interval']:
                swaps.append({'match':matched['id'],'executed':False,'reason':'different_root_scopes',
                              'ranges':[a['root_range'],b['root_range']]}); continue
            for animal,own,other in [('dog',a,b),('cat',b,a)]:
                selected=[other['forest'][0],*own['forest'][1:]]
                # Read-only swap does not reinsert into any bank or main state.
                builder=prior.prior.make(model,processor)
                try:
                    before=events.old_visual.cache_fingerprint(builder.session.main)
                    hashes=node_hashes(selected); vision=counts['vision']
                    result=packed_answer(model,builder.session.builder,selected,config()['early_question'],24)
                    if hashes!=node_hashes(selected) or counts['vision']!=vision or before!=events.old_visual.cache_fingerprint(builder.session.main):
                        raise AssertionError('Swap query mutation')
                    generated+=len(result['generated_ids'])
                    record={'case':matched['id']+'_'+animal,'condition':'swapped_old_root','layout':'packed','result':result,
                            'selected_feature_sha256':hashes,'main_unchanged':True,'banks_unchanged':True,'no_query_vision':True}
                    answers.append(record); stream.write(json.dumps(record,ensure_ascii=False)+'\n'); stream.flush()
                    swaps.append({'match':matched['id'],'case':record['case'],'executed':True,'root_range':own['root_range']})
                finally: builder.close()
        if len(answers)>64 or generated>1536 or counts['vision']!=32: raise RuntimeError('Final budget check')
        if expected is not None and layouts!=expected: raise AssertionError('CPU/GPU layout drift')
        return {'passed':True,'answers':answers,'layout':layouts,'timelines':timelines,'swaps':swaps,'counts':counts,
                'generated_tokens':generated,'model_questions':len(answers),'assets':assets,'rgb_bytes':rgb,
                'input_frames':{c['name']:c['identity'] for c in cases},
                'peak_external_resident_tensor_bytes':peak_external,'external_working_conservative_bytes':16*2**20,
                'peak_main_cache_tensor_bytes':peak_main,'peak_fingerprint_cpu_temporary_bytes':peak_temp,
                'scope':'Known sources, two matched prefix controls, packed readout candidate, no general accuracy'}
    finally:
        stream.close()
        for h in handles: h.remove()


@torch.inference_mode()
def preflight(args):
    from tests.test_online import small_model
    processor=AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    model=small_model(processor); wrapper=prior.prior.make(model,processor)
    Contracts.model,Contracts.wrapper=model,wrapper
    try:
        with (args.out/'unittest.log').open('x',encoding='utf-8') as stream:
            tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
        if not tests.wasSuccessful(): raise AssertionError('New paired/reader contract failed')
    finally: wrapper.close()
    report=replay(model,processor,args,save_photos=True)
    report.update(cpu_tests=tests.testsRun,source_hashes=sources(),request=request(args),budget=budget(),
        model_assets=verify_assets(args.model_dir)['files'],
        model_manifest_sha256=file_hash(args.model_dir/'DOWNLOAD_MANIFEST.json'),
        reference_sha256=file_hash(args.reference_run/'summary.json'),scope='Random CPU mechanism, no semantic scoring')
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
    if read(args.out/'bootstrap.json') != {'prepared_by':'history_causal_v1_watchdog','files':expected}:
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
