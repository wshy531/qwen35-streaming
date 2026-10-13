"""Guarded real projected-feature compression controls, no default integration."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
import weakref

HERE=Path(__file__).resolve().parent
CAP=HERE.parents[1];ACTIVE=CAP/'code/qwen35-streaming-s4'
sys.path.insert(0,str(ACTIVE))
import psutil
import torch
from transformers import AutoProcessor
from scripts.baseline import file_hash,write_json
from scripts.pair_demo import verify_assets
from streaming.online import PairInputBuilder,OnlinePairSession,fp32_reference
from streaming.hybrid import HybridSequence
from streaming.window import FramePacket,read_packets
from streaming.preprocessing import fit_canvas
from features import raw_block,reduce,ShortLongMemory,context,answer,variants

QUESTIONS=('Name the animal visible at the beginning.',
           'Name the animal visible most recently.',
           'List the animals in order of first appearance.')


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def tensor_sha(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def sources():
    return {**{str(p.resolve()):file_hash(p) for p in sorted(ACTIVE.rglob('*.py'))},
            **{str(p.resolve()):file_hash(p) for p in sorted(HERE.glob('*.py'))},
            '@plan':file_hash(HERE/'方案.md'),
            '@cases':file_hash(CAP/'diagnostics/general_stream_v1/cases.json'),
            '@pinned_manager':file_hash(HERE.parent/'event_forest_v1/reference/memory_manager.py')}
def budget():
    return {'model_loads':1,'vision_max':6,'questions_max':18,'generated_max':576,
            'language_max':610,'answer_tokens':32,'max_tokens':512,'wall_seconds':360,
            'ram_min_gib':3,'gpu_free_min_gib':5.5,'conversion_free_min_gib':2,
            'owned_control_tensor_bytes_max':8*2**20,'short_long_tokens_max':144,
            'short_long_feature_bytes_max':589824,'precision':'fp32_ieee_math',
            'questions':list(QUESTIONS),'input':'12 controlled frames, synthetic session PTS'}
def request(args):return {'model_dir':str(args.model_dir)}


def inputs():
    config=read(CAP/'diagnostics/general_stream_v1/cases.json')
    packets=[];metadata=[];assets={}
    for name,start in [('dog',12),('pigeon',0),('cat',0)]:
        row=next(r for r in config['videos'] if r['id']==name)
        path=Path(row['path']);digest=file_hash(path)
        if digest!=row['sha256']:raise ValueError('Existing source SHA changed')
        assets[str(path)]=digest
        reader=read_packets(path,1.)
        try:
            for _ in range(start):next(reader)
            for _ in range(4):
                packet=next(reader);i=len(packets)
                pixels=fit_canvas(packet.rgb,(192,256))
                metadata.append({'session_frame_id':i,'session_pts':float(i),'source_id':name,
                    'source_frame_id':packet.frame_id,'source_pts':packet.timestamp,'source_sha256':digest,
                    'source_rgb_sha256':hashlib.sha256(packet.rgb.tobytes()).hexdigest(),
                    'canvas_rgb_sha256':hashlib.sha256(pixels.tobytes()).hexdigest()})
                packets.append(FramePacket(i,float(i),pixels,i+1,packet.decode_seconds))
        finally:reader.close()
    return [packets[i:i+2] for i in range(0,12,2)],metadata,assets


@torch.inference_mode()
def encode(model,builder,pairs):
    blocks=[];layout=[]
    device=model.get_input_embeddings().weight.device
    for i,pair in enumerate(pairs):
        inp,record=builder.pair(*pair,0,0)
        if inp.video_grid_thw.tolist()!=[[1,12,16]]:raise ValueError('Unexpected registered real grid')
        with fp32_reference():
            visual=model.model.get_video_features(inp.pixel_values_videos.to(device),inp.video_grid_thw.to(device),return_dict=True)
            features=torch.cat(visual.pooler_output)
        if features.shape!=(48,model.get_input_embeddings().embedding_dim):raise ValueError('Unexpected projected shape')
        blocks.append(raw_block(features,i,pair[0].timestamp,pair[1].timestamp))
        layout.append({'frame_ids':record['frame_ids'],'timestamps':record['timestamps'],'grid':record['grid'],
                       'input_ids':record['token_ids'],'pixel_sha256':tensor_sha(inp.pixel_values_videos)})
    return blocks,layout


def cache_fingerprint(sequence):
    return {'seen':sequence.seen_tokens,'next_rope':sequence.next_rope_position,
            'cache':{name:[None if t is None else tensor_sha(t) for t in getattr(sequence.cache,name)]
                     for name in ('key_cache','value_cache','conv_states','recurrent_states')}}


class Contracts(unittest.TestCase):
    @staticmethod
    def block(i,dim=16):
        x=torch.randn((48,dim),generator=torch.Generator().manual_seed(300+i))
        return raw_block(x,i,2*i,2*i+1)

    def test_single_spatial_merge_against_pinned_reference(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('pinned_reference',HERE.parent/'event_forest_v1/reference/memory_manager.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        block=self.block(0);actual=reduce(block,'tome')
        merge,_=module.bipartite_soft_matching(block.features[None],24)
        expected,mass=module.merge_wavg(merge,block.features[None])
        torch.testing.assert_close(actual.features,expected[0],rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(actual.mass,mass[0],rtol=0,atol=0)
        torch.testing.assert_close((actual.features*actual.mass).sum(0),block.features.sum(0),atol=1e-5,rtol=1e-5)

    def test_zero_vectors_are_finite(self):
        block=raw_block(torch.zeros((48,16)),0,0,1)
        result=reduce(block,'tome')
        self.assertTrue(torch.isfinite(result.features).all());self.assertEqual(float(result.mass.sum()),48)

    def test_long_short_budget_and_weighted_mean(self):
        bank=ShortLongMemory();expected=torch.zeros(16)
        for i in range(512):
            block=self.block(i);expected+=block.features.sum(0);bank.append(block)
            state=bank.status()
            self.assertLessEqual(state['tokens'],144);self.assertLessEqual(state['long_roots'],2)
            self.assertLessEqual(state['tensor_bytes'],144*(16*4+4+16))
        rows=bank.export();mass=sum(float(b.mass.sum()) for b in rows)
        total=sum((b.features*b.mass).sum(0) for b in rows)
        self.assertEqual(mass,512*48)
        torch.testing.assert_close(total/mass,expected/mass,atol=1e-5,rtol=1e-5)
        self.assertEqual(rows[-1].first,511);self.assertEqual(rows[-1].leaves,1)
        self.assertEqual(sum(b.leaves for b in rows),512)
        bank.clear();self.assertEqual(bank.status()['tensor_bytes'],0)

    def test_ownership_and_removed_tensor_release(self):
        bank=ShortLongMemory();source=self.block(0);bank.append(source)
        saved=bank.export()[0].features.clone();source.features.zero_()
        export=bank.export();export[0].features.fill_(99)
        torch.testing.assert_close(bank.export()[0].features,saved,rtol=0,atol=0)
        bank.append(self.block(1));bank.append(self.block(2))
        weak=[weakref.ref(b.features) for b in [*bank._roots,bank._recent]]
        records=bank.append(self.block(3));gc.collect()
        self.assertEqual(len(records),1);self.assertGreaterEqual(sum(ref() is None for ref in weak),2)

    def test_future_duplicate_invalid_inputs_are_rejected(self):
        bank=ShortLongMemory();bank.append(self.block(0));before=bank.status()
        from dataclasses import replace
        invalid=[self.block(0),self.block(2),replace(self.block(1),begin=.5),
                 replace(self.block(1),features=torch.full((48,16),float('nan')))]
        for block in invalid:
            with self.assertRaises(ValueError):bank.append(block)
            self.assertEqual(bank.status(),before)

    def test_temporal_pooling_is_lossy(self):
        from dataclasses import replace
        bank=ShortLongMemory()
        for i,sign in enumerate((1.,-1.,0.,0.)):
            bank.append(replace(self.block(i),features=torch.full((48,16),sign)))
        # Whichever adjacent pair was selected, provenance is a compressed span.
        parent=next(b for b in bank.export() if b.leaves>1)
        self.assertGreater(parent.last,parent.first)
        self.assertFalse(parent.metadata()['exact_pair'])
        a=torch.ones((48,16));b=-a
        self.assertEqual(float(((a+b)/2).abs().sum()),0)

    def test_full_context_native_rope(self):
        embeds,pos,ids,meta=context(self.model,self.builder,self.blocks,QUESTIONS[0],2)
        mm=(ids==self.builder.processor.video_token_id).long()*2
        native,_=self.model.model.get_rope_index(ids,mm,video_grid_thw=torch.tensor([[1,12,16]]*6))
        torch.testing.assert_close(pos[1:],native,rtol=0,atol=0)
        self.assertEqual(meta['visual_tokens'],288)

    def test_compressed_coordinates_and_suffix_positions(self):
        _,full,_,_=context(self.model,self.builder,self.blocks,QUESTIONS[0],2)
        odd=[reduce(b,'odd') for b in self.blocks];tome=[reduce(b,'tome') for b in self.blocks]
        _,left,ids,_=context(self.model,self.builder,odd,QUESTIONS[0],2)
        _,right,_,_=context(self.model,self.builder,tome,QUESTIONS[0],2)
        torch.testing.assert_close(left,right,rtol=0,atol=0)
        torch.testing.assert_close(left[0,0],torch.arange(ids.shape[1]),rtol=0,atol=0)
        size=self.builder.question_ids(QUESTIONS[0]).shape[1]
        torch.testing.assert_close(left[1:,:,-size:],full[1:,:,-size:],rtol=0,atol=0)
        for a,b in zip(self.blocks,tome):torch.testing.assert_close(b.xy,a.xy[1::2],rtol=0,atol=0)

    def test_full_prefill_matches_online_session_and_all_queries_are_isolated(self):
        session=OnlinePairSession(self.model,self.builder.processor,max_tokens=512,max_question_tokens=64,
                                  max_answer_tokens=32,resize_mode='letterbox',video_logits='none')
        for pair in self.pairs:
            for packet in pair:session.append(packet)
        before=cache_fingerprint(session.main);counts=session.status()
        with fp32_reference():
            branch=session.main.fork()
            reference=branch.append(input_ids=self.builder.question_ids(QUESTIONS[0]),logits_mode='last')
            seq=HybridSequence(self.model,max_tokens=512)
            emb,pos,_,_=context(self.model,self.builder,self.blocks,QUESTIONS[0],2)
            actual=seq.append(inputs_embeds=emb,position_ids=pos,logits_mode='last')
        torch.testing.assert_close(actual,reference,rtol=1e-4,atol=1e-5)
        for label,blocks in variants(self.blocks)[0].items():
            for question in QUESTIONS:
                result=answer(self.model,self.builder,blocks,question,2)
                self.assertLessEqual(len(result['generated_ids']),2)
                self.assertEqual(cache_fingerprint(session.main),before)
                self.assertEqual(session.status(),counts)
        session.close()


@torch.inference_mode()
def preflight(args):
    from tests.test_online import small_model
    processor=AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    model=small_model(processor);builder=PairInputBuilder(processor,resize_mode='letterbox')
    pairs,metadata,assets=inputs();blocks,layout=encode(model,builder,pairs)
    Contracts.model,Contracts.builder,Contracts.blocks,Contracts.pairs=model,builder,blocks,pairs
    with (args.out/'unittest.log').open('x',encoding='utf-8') as log:
        tests=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not tests.wasSuccessful():raise AssertionError('CPU visual memory contract failed')
    arms,memory,trace=variants(blocks);rows=[]
    for label,events in arms.items():
        for question in QUESTIONS:
            _,pos,ids,item=context(model,builder,events,question,32)
            rows.append({'method':label,'question':question,'input_tokens':item['input_tokens'],
                         'visual_tokens':item['visual_tokens'],'positions_sha256':tensor_sha(pos)})
    report={'passed':True,'cpu_tests':tests.testsRun,'input_frames':metadata,'layout':layout,
        'request':request(args),'budget':budget(),'source_hashes':sources(),'source_assets':assets,
        'model_manifest_sha256':file_hash(args.model_dir/'DOWNLOAD_MANIFEST.json'),
        'assets':verify_assets(args.model_dir)['files'],'query_layouts':rows,'memory':memory,'merges':trace,
        'scope':'CPU tiny random weights, real processor/pixels, no semantic evidence; content-dependent merge may differ on GPU'}
    write_json(args.out/'proof.json',report)
    print(json.dumps({'passed':True,'cpu_tests':tests.testsRun,'pairs':6,'query_layouts':len(rows),'gpu_loads':0}),flush=True)


@torch.inference_mode()
def gpu(args):
    if args.preflight is None or read(args.preflight/'status.json')!={'status':'completed','stage':'preflight'}:
        raise ValueError('Fresh successful CPU preflight required')
    proof=read(args.preflight/'proof.json')
    if (proof['request']!=request(args) or proof['budget']!=budget() or proof['source_hashes']!=sources() or
        proof['assets']!=verify_assets(args.model_dir)['files'] or
        proof['model_manifest_sha256']!=file_hash(args.model_dir/'DOWNLOAD_MANIFEST.json')):
        raise ValueError('Frozen source/arguments/weight proof drift')
    pairs,metadata,assets=inputs()
    if proof['input_frames']!=metadata or proof['source_assets']!=assets:raise ValueError('Input identity changed')
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=CAP,text=True).strip()
    if branch in ('main','master','HEAD','') or branch!='36tar/general-video-demo':raise ValueError('Registered personal branch required')
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
    builder=PairInputBuilder(processor,resize_mode='letterbox');counts={'vision':0,'language':0}
    def guard(name,limit):
        def hook(_m,_a):
            counts[name]+=1
            if counts[name]>limit:raise RuntimeError('Forward budget exceeded')
        return hook
    hooks=[model.model.visual.register_forward_pre_hook(guard('vision',6)),
           model.model.language_model.register_forward_pre_hook(guard('language',610))]
    try:
        blocks,layout=encode(model,builder,pairs)
        if layout!=proof['layout']:raise ValueError('Real preprocessed input differs from proof')
        if any(b.features.shape!=(48,1024) for b in blocks):raise ValueError('Unexpected real projection')
        arms,memory,trace=variants(blocks)
        owned=sum(b.metadata()['tensor_bytes'] for b in blocks)+sum(b.metadata()['tensor_bytes'] for arm in arms.values() for b in arm)
        if owned>8*2**20:raise RuntimeError('Owned feature controls exceed byte budget')
        anchor=HybridSequence(model,max_tokens=512,session_id='protected-reference')
        with fp32_reference():anchor.append(input_ids=builder.prefix_ids.to('cuda'),logits_mode='none')
        before=cache_fingerprint(anchor);answers=[];generated=0
        with (args.out/'answers.jsonl').open('x',encoding='utf-8') as log:
            for method,events in arms.items():
                for question in QUESTIONS:
                    torch.cuda.synchronize();start=time.perf_counter()
                    result=answer(model,builder,events,question,32)
                    torch.cuda.synchronize();result['query_seconds']=time.perf_counter()-start
                    generated+=len(result['generated_ids'])
                    if generated>576:raise RuntimeError('Generation budget exceeded')
                    if cache_fingerprint(anchor)!=before:raise AssertionError('Protected main state changed')
                    row={'method':method,**result,'protected_main_unchanged':True}
                    answers.append(row);log.write(json.dumps(row,ensure_ascii=False)+'\n');log.flush()
                    print(json.dumps({'method':method,'question':question,'answer':row['answer'],
                                      'visual_tokens':row['visual_tokens'],'stop_reason':row['stop_reason']},ensure_ascii=False),flush=True)
        summary={'passed':True,'answers':answers,'questions':18,'counts':counts,'generated_tokens':generated,
                 'feature_blocks':[dict(b.metadata(),feature_sha256=tensor_sha(b.features)) for b in blocks],
                 'memory':memory,'merges':trace,'owned_cpu_feature_tensor_bytes':owned,
                 'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                 'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
                 'wall_seconds_including_load':time.perf_counter()-begin,
                 'qa_accuracy_evaluated':False,'scope':'Six shared projected pairs and independent QA; not official full-video baseline or end-to-end acceleration'}
        write_json(args.out/'summary.json',summary)
        anchor.cache.reset();arms.clear();blocks.clear()
    finally:
        for hook in hooks:hook.remove()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('preflight','gpu'),required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--preflight',type=Path)
    args=parser.parse_args()
    for name in ('model_dir','out','preflight'):
        value=getattr(args,name)
        if value is not None:setattr(args,name,value.resolve())
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'tmp').mkdir();tempfile.tempdir=str(args.out/'tmp')
    torch.set_num_threads(1);torch.manual_seed(42)
    write_json(args.out/'budget.json',budget());write_json(args.out/'source_hashes.json',sources())
    try:
        (preflight if args.stage=='preflight' else gpu)(args)
        write_json(args.out/'status.json',{'status':'completed','stage':args.stage})
    except Exception as error:
        write_json(args.out/'status.json',{'status':'failed','stage':args.stage,'error':type(error).__name__,'message':str(error)})
        (args.out/'error.log').write_text(traceback.format_exc(),encoding='utf-8');raise


if __name__=='__main__':main()
