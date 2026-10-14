"""Same-budget retention coverage on existing videos and tiny model isolation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
CAP=HERE.parents[1];ACTIVE=CAP/'code/qwen35-streaming-s4'
sys.path.insert(0,str(ACTIVE));sys.path.insert(0,str(HERE))
import numpy as np
import torch
from transformers import AutoProcessor
from streaming.frame_memory import TemporalFrameMemory
from streaming.online import OnlinePairSession
from streaming.preprocessing import fit_canvas
from streaming.remembering import RememberingPairDemo
from streaming.window import FramePacket,read_packets
from tests.test_online import small_model
from policy import AnchorRecentMemory


def packet(i,shape=(192,256,3)):
    return FramePacket(i,float(i),np.full(shape,i%256,np.uint8),i+1)


class PolicyTests(unittest.TestCase):
    def test_same_total_coverage_but_early_recent_tradeoff(self):
        fifo=TemporalFrameMemory(every_pairs=1);anchor=AnchorRecentMemory()
        for i in range(0,12,2):
            for bank in (fifo,anchor):bank.add_pair(packet(i),packet(i+1))
        self.assertEqual([r['frame_ids'][0] for r in fifo.evidence()['intervals']],[6,8,10])
        self.assertEqual([r['frame_ids'][0] for r in anchor.evidence()['intervals']],[0,8,10])
        self.assertEqual(fifo.bytes,anchor.bytes)
        self.assertEqual(anchor.bytes,884736)
        self.assertEqual(anchor.retrieve_at(6.5)[1],'not_retained')
        self.assertEqual(fifo.retrieve_at(.5)[1],'not_retained')

    def test_byte_cap_can_release_anchor_without_dropping_latest(self):
        bank=AnchorRecentMemory(max_bytes=400000)
        bank.add_pair(packet(0),packet(1))
        bank.add_pair(packet(2),packet(3))
        self.assertFalse(bank.evidence()['anchor_retained'])
        self.assertEqual(bank.anchor_released,1)
        self.assertEqual(bank.retrieve_at(2.5)[1],'retained_interval')
        self.assertLessEqual(bank.bytes,bank.max_bytes)

    def test_readonly_ownership_and_no_future_substitution(self):
        bank=AnchorRecentMemory();a,b=packet(0),packet(1)
        bank.add_pair(a,b);a.rgb.fill(255)
        found,_=bank.retrieve_at(.5)
        self.assertTrue(np.all(found[0].rgb==0));self.assertFalse(found[0].rgb.flags.writeable)
        self.assertEqual(bank.retrieve_at(2.5),(None,'not_observed'))
        self.assertEqual(bank.retrieve_at(1.5),(None,'not_observed'))

    def test_rejection_and_clear(self):
        bank=AnchorRecentMemory();bank.add_pair(packet(0),packet(1));before=bank.evidence()
        with self.assertRaises(ValueError):bank.add_pair(packet(0),packet(1))
        self.assertEqual(bank.evidence(),before)
        bank.clear();self.assertEqual(bank.bytes,0);self.assertIsNone(bank.anchor_seed_id)
        with self.assertRaises(ValueError):AnchorRecentMemory(max_pairs=1)

    def test_sparse_sampling_matches_before_capacity_fills(self):
        banks=[TemporalFrameMemory(every_pairs=4),AnchorRecentMemory(every_pairs=4)]
        for i in range(0,24,2):
            for b in banks:b.add_pair(packet(i),packet(i+1))
        self.assertEqual(banks[0].evidence()['intervals'],banks[1].evidence()['intervals'])


def coverage(bank,targets):
    results=[]
    for t in targets:
        frames,reason=bank.retrieve_at(t)
        results.append({'time':t,'retained':frames is not None,'reason':reason,
                        'frame_ids':[] if frames is None else [p.frame_id for p in frames]})
    return results


def real_sources():
    config=json.loads((HERE.with_name('general_stream_v1')/'cases.json').read_text(encoding='utf-8'))
    rows=[]
    for spec in config['videos']:
        path=Path(spec['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:raise AssertionError('Source SHA changed')
        start=12 if spec['id']=='dog' else 0
        limit={'dog':24,'pigeon':16,'cat':8}[spec['id']]
        reader=read_packets(path,1.);pending=None;targets=[]
        banks=[TemporalFrameMemory(every_pairs=1),AnchorRecentMemory()]
        try:
            for _ in range(start):next(reader)
            for _ in range(limit):
                p=next(reader);rgb=fit_canvas(p.rgb,(192,256))
                current=FramePacket(p.frame_id,p.timestamp,rgb,p.decoded_count,p.decode_seconds)
                if pending is None:pending=current
                else:
                    targets.append((pending.timestamp+current.timestamp)/2)
                    for bank in banks:bank.add_pair(pending,current)
                    pending=None
        finally:reader.close()
        controls=[]
        for bank in banks:
            retained=coverage(bank,targets)
            controls.append({'memory':bank.evidence(),'queries':retained,
                             'all_targets_retained':sum(q['retained'] for q in retained),
                             'earliest_retained':retained[0]['retained'],
                             'latest_three_retained':sum(q['retained'] for q in retained[-3:])})
        assert controls[0]['all_targets_retained']==controls[1]['all_targets_retained']==3
        assert controls[0]['latest_three_retained']==3 and controls[1]['latest_three_retained']==2
        assert not controls[0]['earliest_retained'] and controls[1]['earliest_retained']
        rows.append({'source_id':spec['id'],'source_sha256':spec['sha256'],'real_frames':limit,
                     'pairs':len(targets),'controls':controls,'scope':'Retained pixel/time coverage only, no animal recognition'})
        for bank in banks:bank.clear()
    return rows


@torch.inference_mode()
def tiny_control(model_dir):
    processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True);model=small_model(processor)
    rows=[]
    for cls in (TemporalFrameMemory,AnchorRecentMemory):
        session=OnlinePairSession(model,processor,max_tokens=200,max_question_tokens=96,max_answer_tokens=2,
                                  resize_mode='letterbox',video_logits='none')
        demo=RememberingPairDemo(session,'Describe the subject.',2,first_qa_frames=100,memory_every_pairs=1)
        demo.frame_memory=cls(every_pairs=1)
        try:
            for i in range(8):demo.append(packet(i,(32,32,3)))
            before=session.status();answer=demo.ask_at(.5,'Describe the subject.',2)
            assert session.status()==before
            for i in range(8,10):demo.append(packet(i,(32,32,3)))
            rows.append({'policy':demo.frame_memory.evidence()['policy'],'early_model_called':answer['model_called'],
                         'early_reason':answer['retrieval_reason'],'main_unchanged_bitwise':answer['main_unchanged_bitwise'],
                         'continued_after_query':demo.received==10,'resets':demo.resets})
            demo.finish()
        finally:demo.close()
        assert demo.frame_memory.bytes==0
    assert not rows[0]['early_model_called'] and rows[1]['early_model_called'] and rows[1]['main_unchanged_bitwise']
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    with (args.out/'unittest.log').open('w',encoding='utf-8') as log:
        test=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PolicyTests))
    if not test.wasSuccessful():raise AssertionError('Policy contracts failed')
    report={'passed':True,'policy_tests':test.testsRun,'sources':real_sources(),'tiny_controls':tiny_control(args.model_dir),
            'budget':{'real_gpu_loads':0,'persistent_bytes_per_policy':884736,'max_pairs':3,'source_frames_total':48,
                      'simultaneous_banks':2,'semantic_accuracy_evaluated':False},
            'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*')) if p.is_file()},
            'scope':'Independent CPU candidate; default core untouched; tradeoff, not a universal improvement'}
    (args.out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'passed':True,'policy_tests':test.testsRun,'real_source_frames':48,'sources':3,
                      'tiny_isolation_controls':len(report['tiny_controls']),'gpu_loads':0}))


if __name__=='__main__':main()
