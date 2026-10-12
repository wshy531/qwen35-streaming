"""CPU synthetic event features: merge parity, bounded memory and lossy evidence."""
import argparse
from collections import deque
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import unittest
import weakref

import torch
from prototype import EventForest,soft_merge

HERE=Path(__file__).resolve().parent


def official():
    source=HERE/'reference/memory_manager.py'
    spec=importlib.util.spec_from_file_location('pinned_streamforest_reference',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def event(i,tokens=16,dim=16):
    generator=torch.Generator().manual_seed(100+i)
    return torch.randn((tokens,dim),generator=generator)


class ForestTests(unittest.TestCase):
    def test_one_level_matches_pinned_tome_nonzero_features(self):
        ref=official()
        for n in (8,16,17):
            x=event(n,tokens=n);size=torch.ones((n,1))
            actual,mass,_=soft_merge(x,size,4)
            metric=x[None].reshape(1,n,4,4).mean(2)
            merge,_=ref.bipartite_soft_matching(metric,n//2)
            expected,weights=ref.merge_wavg(merge,x[None],size[None])
            torch.testing.assert_close(actual,expected[0],rtol=1e-6,atol=1e-6)
            torch.testing.assert_close(mass,weights[0],rtol=0,atol=0)

    def test_mass_and_weighted_mean_across_levels(self):
        bank=EventForest();expected=torch.zeros(16)
        for i in range(16):
            x=event(i);expected+=x.sum(0);bank.append(i,i,i+.5,x)
        outputs=bank.export();total_mass=sum(float(m.sum()) for _,x,m in outputs)
        actual=sum((x*m).sum(0) for _,x,m in outputs)
        self.assertEqual(total_mass,256)
        torch.testing.assert_close(actual/total_mass,expected/256,atol=1e-6,rtol=1e-6)

    def test_streaming_and_metadata_hard_limits(self):
        bank=EventForest(metadata_budget=512)
        for i in range(128):
            state=bank.append(i,i,i+.5,event(i))
            self.assertLessEqual(state['tokens'],64)
            self.assertLessEqual(state['tensor_bytes'],64*17*4)
            self.assertLessEqual(state['metadata_serialized_bytes'],512)
            self.assertLessEqual(len(state['roots']),64)
        self.assertGreater(bank.merges,0)

    def test_ownership_export_and_clear(self):
        x=event(0);bank=EventForest();bank.append(0,0,.5,x)
        original=bank.export()[0][1].clone();x.fill_(99)
        exported=bank.export()[0][1];exported.zero_()
        torch.testing.assert_close(bank.export()[0][1],original,rtol=0,atol=0)
        bank.clear();self.assertEqual(bank.tensor_bytes(),0);self.assertEqual(bank.last_id,-1)

    def test_removed_child_tensor_released(self):
        bank=EventForest(token_budget=16)
        bank.append(0,0,.5,event(0));old=weakref.ref(bank._roots[0].features)
        bank.append(1,1,1.5,event(1));gc.collect()
        self.assertIsNone(old())

    def test_invalid_inputs_do_not_mutate_state(self):
        bank=EventForest();bank.append(0,0,.5,event(0));before=bank.status()
        bad=event(1);bad[0,0]=float('nan')
        cases=[(2,1,1.5,event(1)),(1,.4,1.5,event(1)),(1,1,float('inf'),event(1)),
               (1,1,1.5,bad),(1,1,1.5,event(1).double()),(1,1,1.5,event(1,tokens=17))]
        for args in cases:
            with self.assertRaises(ValueError):bank.append(*args)
            self.assertEqual(bank.status(),before)

    def test_zero_vectors_and_information_loss(self):
        a=torch.ones((1,16));x=torch.cat((a,-a));merged,mass,_=soft_merge(x,torch.ones((2,1)),4)
        self.assertTrue(torch.isfinite(merged).all())
        self.assertEqual(float(merged.abs().sum()),0)
        self.assertEqual(float(mass.sum()),2)
        self.assertGreater(float((merged[0]-a[0]).square().mean()),0)
        bank=EventForest();bank.append(0,0,.5,torch.zeros((16,16)))
        bank.append(1,1,1.5,torch.zeros((16,16)))

    def test_temporal_only_prefers_old_adjacent_events(self):
        bank=EventForest(token_budget=32,weights=(0,0,1));records=[];bank.sink=records.append
        for i in range(3):bank.append(i,i,i+.5,event(i))
        merged=next(r for r in records if r['kind']=='merge')
        self.assertEqual((merged['left']['begin_id'],merged['right']['begin_id']),(0,1))


def reference_update_probe():
    ref=official();manager=ref.MemoryManager(16,4,st_memory_windows=[1,2],st_memory_tokens=[16,8],
        event_split_window=2,long_memory_tokens_per_frame=4,long_memory_tokens_quota=16)
    for i in range(16):manager.update(event(i))
    before=sum(t.numel()*t.element_size() for t in manager.memory_now)
    current_frames=len(manager.memory_now)
    output=manager.get_memory_tokens()
    return {'scope':'Only pinned helper update→get path, not every official demo path',
            'updates_without_get':16,'current_buffer_frames_before_get':current_frames,
            'current_buffer_bytes_before_get':before,'output_tokens_after_get':output.shape[1],
            'long_tokens_after_get':manager.long_memory_current_tokens}


def retention_controls():
    """Same synthetic input and feature+mass budget, not the RGB frame budget."""
    fifo=deque(maxlen=4)
    anchor=[]
    for i in range(24):
        x=event(i//4)
        fifo.append((i,x.clone(),torch.ones((16,1))))
        anchor.append((i,x.clone(),torch.ones((16,1))))
        if len(anchor)>4:del anchor[1]
        for rows in (fifo,anchor):
            assert sum(len(x) for _,x,m in rows)<=64
            assert sum(t.numel()*t.element_size() for _,x,m in rows for t in (x,m))<=4352
    controls=[]
    for label,rows in [('fifo_recent_four',fifo),('first_anchor_recent_three',anchor)]:
        ids=[i for i,x,m in rows]
        assert ids==([20,21,22,23] if label=='fifo_recent_four' else [0,21,22,23])
        for i,x,m in rows:torch.testing.assert_close(x,event(i//4),rtol=0,atol=0)
        controls.append({'scenario':label,'exact_event_ids_retained':ids,
                         'discarded_event_count':24-len(ids),'tokens':64,
                         'tensor_bytes_including_matching_unit_mass':4352,
                         'exact_features_preserved_for_retained_events':True,
                         'all_event_spans_represented':False})
    return controls


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    begin=time.perf_counter()
    with (args.out/'unittest.log').open('w',encoding='utf-8') as log:
        tests=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ForestTests))
    if not tests.wasSuccessful():raise AssertionError('CPU forest contracts failed')
    scenarios=[]
    with (args.out/'trace.jsonl').open('x',encoding='utf-8') as log:
        for label,weights in [('combined',(.4,.4,.2)),('similarity_only',(1,0,0)),
                              ('count_only',(0,1,0)),('time_only',(0,0,1))]:
            def sink(record):log.write(json.dumps({'scenario':label,**record})+'\n')
            bank=EventForest(weights=weights,sink=sink)
            for i in range(24):
                # Repeated feature groups, not learned or recognized events.
                vector=event(i//4)
                bank.append(i,i,i+.5,vector)
            outputs=bank.export()
            scenarios.append({'scenario':label,'weights':weights,'final':bank.status(),
                              'total_original_token_mass':sum(float(m.sum()) for _,x,m in outputs),
                              'all_event_spans_represented':outputs[0][0]['begin_id']==0 and outputs[-1][0]['end_id']==23,
                              'exact_frames_preserved':False})
            bank.clear();assert bank.tensor_bytes()==0
    stress=EventForest()
    for i in range(512):stress.append(i,i,i+.5,event(i))
    state=stress.status();stress.clear()
    controls=retention_controls()
    report={'passed':True,'tests':tests.testsRun,'one_level_tome_controls':3,'scenarios':scenarios,
            'same_budget_retention_controls':controls,
            'information_loss_counterexample':{'input_vectors':'ones and negative ones',
                'merged_vector':'zero','original_token_mass':2,
                'exact_reconstruction_possible':False,
                'caveat':'Provenance spans and preserved weighted mean do not prove QA accuracy or detail retention'},
            'stress':state,'reference_update_probe':reference_update_probe(),
            'budget':{'gpu_loads':0,'real_model_loads':0,'feature_dim':16,'heads':4,'root_token_budget':64,
                      'event_tokens_max':16,'root_feature_and_mass_bytes_max':4352,
                      'metadata_serialized_bytes_max':16384,'stress_events':512},
            'wall_seconds':time.perf_counter()-begin,
            'source_hashes':{p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts},
            'scope':'CPU synthetic event-token scheduler, not Qwen visual compression or QA ability'}
    (args.out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'tests':tests.testsRun,'tome_controls':3,'stress_events':512,
                      'gpu_loads':0,'wall_seconds':report['wall_seconds']}))


if __name__=='__main__':main()
