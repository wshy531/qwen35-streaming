"""Bounded explicit-time question routing and mixed-context diagnosis."""
import argparse,importlib.util,json,subprocess,sys,tempfile,time,traceback,unittest
from pathlib import Path
from dataclasses import replace
HERE=Path(__file__).resolve().parent;CAP=HERE.parents[1]
OLD=HERE.parent/'geometry_transfer_v1';sys.path.insert(0,str(OLD))
spec=importlib.util.spec_from_file_location('time_frozen_transfer',OLD/'probe.py')
prior=importlib.util.module_from_spec(spec);spec.loader.exec_module(prior)
history=prior.history;forest_ref=prior.forest_ref;runtime_ref=prior.runtime_ref;events=prior.events
import torch,psutil
from transformers import AutoProcessor
from forest import raw,merge
from features import raw_block
from reader import native_context,native_answer
from time_index import RawTimeIndex
from streaming.window import FramePacket
from scripts.baseline import file_hash,write_json
from scripts.pair_demo import verify_assets
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def config():return read(HERE/'cases.json')
def sources():return {**prior.sources(),**{str(p.resolve()):file_hash(p) for p in HERE.iterdir() if p.suffix in ('.py','.md','.json')}}
def budget():return {'model_loads':1,'vision_max':12,'questions_max':72,'generated_max':1728,'language_max':2200,
    'answer_tokens':24,'wall_seconds':360,'main_max_tokens':256,'qa_max_tokens':768,
    'reference_visual_tokens_max':144,'bank_visual_tokens_max':144,'candidate_visual_tokens':48,
    'retained_rgb_max_bytes':4*2**20,'external_working_conservative_max_bytes':16*2**20,
    'ram_min_gib':3,'gpu_free_min_gib':5.5,'conversion_free_min_gib':2,'precision':'fp32_ieee_math'}
def request(args):return {'model_dir':str(args.model_dir),'reference_run':str(args.reference_run),'branch':args.branch,'cases':config()}
def inputs():
    registered=events.read(CAP/'diagnostics/general_stream_v1/cases.json')['videos']+events.read(CAP/'diagnostics/event_memory_v1/cases.json')['natural_sources']
    source={r['id']:r for r in registered};cases=[];assets={}
    for c in config()['cases']:
        packets=[];identity=[]
        for name,start in c['parts']:
            row=source[name];pairs,rows=events.segment(row,start,2,False);assets[row['path']]=row['sha256']
            for packet,ident in zip(pairs[0],rows):
                i=len(packets);packets.append(FramePacket(i,float(i),packet.rgb,i+1,packet.decode_seconds))
                identity.append({**ident,'session_frame_id':i,'session_pts':float(i),'session_time_is_source_time':False,'constructed_transition':True})
        cases.append({'name':c['id'],'pairs':[packets[i:i+2] for i in range(0,6,2)],'identity':identity,'questions':c['questions']})
    rgb=sum(p.rgb.nbytes for c in cases for pair in c['pairs'] for p in pair)
    if rgb>4*2**20:raise ValueError('RGB bound')
    return cases,assets,rgb
def choose(nodes,index,q,arm):
    target=q['target'];route=None;donor=None
    if arm=='routed':
        result=index.query(q['text']);route={k:v for k,v in result.items() if k!='node'}
        if not result['ok']:raise AssertionError('Registered supported query refused')
        selected=[result['node']]
    elif arm=='single_old':selected=[nodes[target].copy()]
    elif arm=='old_plus_recent':selected=[nodes[target].copy(),nodes[-1].copy()]
    elif arm=='all_history':selected=[n.copy() for n in nodes]
    elif arm=='recent_only':selected=[nodes[-1].copy()]
    elif arm=='swap_values':
        other=nodes[1-target];selected=[replace(nodes[target],features=other.features.clone(),mass=nodes[target].mass.clone(),coords=nodes[target].coords.clone())]
        donor=other.metadata()
    else:raise ValueError('Unknown condition')
    return selected,route,donor
class Contracts(unittest.TestCase):
    def nodes(self):
        width=self.model.get_input_embeddings().embedding_dim
        return [raw(raw_block(torch.randn((48,width),generator=torch.Generator().manual_seed(i+2601)),i,2*i,2*i+1)) for i in range(4)]
    def test_grammar_bounds_eviction_compressed_refusal(self):
        nodes=self.nodes();index=RawTimeIndex(3)
        for n in nodes[:3]:index.append(n)
        self.assertEqual(index.query('Name the animal visible at 2.5 seconds.')['node'].first,1)
        for q in ('What happened?','Name the animal visible at -1 seconds.','Name the animal visible at nan seconds.','Name the animal visible at 9.0 seconds.','Name the animal visible at 1.5 seconds.','Name the animal visible at 0.5 seconds and at 2.5 seconds.'):
            self.assertFalse(index.query(q)['ok'])
        index.append(nodes[3]);self.assertEqual(len(index.nodes),3)
        self.assertFalse(index.query('Name the animal visible at 0.5 seconds.')['ok'])
        with self.assertRaises(ValueError):index.append(merge(nodes[0],nodes[1],within=True))
        with self.assertRaises(ValueError):index.append(nodes[3])
    def test_oracle_route_and_value_only_swap(self):
        nodes=self.nodes()[:3];before=history.node_hashes(nodes);index=RawTimeIndex(3)
        for n in nodes:index.append(n)
        for q in config()['questions']:
            a,_,_=choose(nodes,index,q,'single_old');b,route,_=choose(nodes,index,q,'routed');c,_,donor=choose(nodes,index,q,'swap_values')
            self.assertEqual(history.node_hashes(a),history.node_hashes(b));self.assertEqual(route['selected_pair'],q['target'])
            x,y=[native_context(self.model,self.wrapper.session.builder,ns,q['text'],24) for ns in (a,c)]
            self.assertTrue(torch.equal(x[1],y[1]));self.assertTrue(torch.equal(x[2],y[2]));self.assertFalse(torch.equal(x[0],y[0]))
        b[0].features.zero_();self.assertEqual(before,history.node_hashes(nodes));self.assertEqual(len(index.nodes),3)
    def test_paired_suffix_and_source_time_separation(self):
        cases,assets,rgb=inputs();self.assertEqual(len(cases),4)
        for i in (0,2):
            a,b=cases[i:i+2];self.assertEqual(a['identity'][4:],b['identity'][4:])
            self.assertEqual([r['session_pts'] for r in a['identity']],list(map(float,range(6))))

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
    handles=[model.model.visual.register_forward_pre_hook(guard('vision')),model.model.language_model.register_forward_pre_hook(guard('language'))]
    stream=(args.out/'answers.jsonl').open('x',encoding='utf-8')
    try:
        for case in cases:
            wrapper=runtime_ref.make(model,processor);nodes=[];index=RawTimeIndex(3)
            layouts[case['name']]=[];timelines[case['name']]=[]
            try:
                for i,pair in enumerate(case['pairs']):
                    assert not wrapper.feed(pair[0])['encoded'];r=wrapper.feed(pair[1]);assert r['encoded']
                    node=raw(wrapper.bank.current);nodes.append(node);index.append(node)
                    layouts[case['name']].append(wrapper.session.ledger[-1]);storage=runtime_ref.main_storage(wrapper)
                    peak_main=max(peak_main,storage['main_cache_tensor_bytes']);peak_temp=max(peak_temp,storage['fingerprint_cpu_temporary_conservative_bytes'])
                    timelines[case['name']].append({'pair':i,'feature_sha256':r['feature_sha256'],'storage':storage})
                fingerprint=index.fingerprint();reset=wrapper.reset_main();assert index.fingerprint()==fingerprint
                for arm in config()['arms']:
                    for q in case['questions']:
                        selected,route,donor=choose(nodes,index,q,arm)
                        before=forest_ref.protected(wrapper,{});hashes=history.node_hashes(nodes);selected_hash=history.node_hashes(selected);index_hash=index.fingerprint()
                        resident=wrapper.bank.status()['tensor_bytes']+history.node_bytes(nodes)+history.node_bytes(index.nodes)+history.node_bytes(selected)
                        peak_external=max(peak_external,resident)
                        if resident>4*2**20:raise RuntimeError('Feature resident bound')
                        vision=counts['vision'];result=native_answer(model,wrapper.session.builder,selected,q['text'],24)
                        result['visual_values_sha256']=history.file_tensor_sha(torch.cat([n.features for n in selected]))
                        if before!=forest_ref.protected(wrapper,{}) or hashes!=history.node_hashes(nodes) or selected_hash!=history.node_hashes(selected) or index_hash!=index.fingerprint() or vision!=counts['vision']:raise AssertionError('Query changed protected evidence')
                        generated+=len(result['generated_ids'])
                        if len(answers)+1>72 or generated>1728:raise RuntimeError('QA budget')
                        row={'case':case['name'],'condition':arm,'topic':q['id'],'result':result,'route':route,
                             'counterfactual_value_donor':donor,'artificial_counterfactual':donor is not None,
                             'selected_feature_sha256':selected_hash,'target_index_diagnostic_only':q['target'],
                             'main_reset_before_queries':reset,'encoded_horizon':wrapper.bank.last_end,
                             'main_unchanged':True,'bank_unchanged':True,'no_query_vision':True}
                        answers.append(row);stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                        print(json.dumps({'case':case['name'],'condition':arm,'topic':q['id'],'answer':result['answer']},ensure_ascii=False),flush=True)
            finally:index.clear();wrapper.close()
        if counts['vision']!=12:raise AssertionError('Vision count')
        if expected is not None and layouts!=expected:raise AssertionError('Layout drift')
        return {'passed':True,'answers':answers,'layout':layouts,'timelines':timelines,'counts':counts,'generated_tokens':generated,
                'model_questions':len(answers),'assets':assets,'rgb_bytes':rgb,'input_frames':{c['name']:c['identity'] for c in cases},
                'peak_external_resident_tensor_bytes':peak_external,'external_working_conservative_bytes':16*2**20,
                'peak_main_cache_tensor_bytes':peak_main,'peak_fingerprint_cpu_temporary_bytes':peak_temp,
                'scope':'Known-source bounded explicit-time routing; no semantic or new-source generalization claim'}
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
    if read(args.out/'bootstrap.json') != {'prepared_by':'time_query_v1_watchdog','files':expected}:
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
