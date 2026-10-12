"""Evidence ownership, bounded storage, reset and isolated historical queries."""
import unittest
from unittest import mock
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor
from scripts.baseline import PROJECT
from streaming.demo import PairDemo
from streaming.frame_memory import TemporalFrameMemory
from streaming.online import OnlinePairSession
from streaming.remembering import RememberingPairDemo
from streaming.window import FramePacket
from tests.test_mixed_state import snapshot,assert_states
from tests.test_online import small_model


def frame(i,shape=(32,32,3)):
    return FramePacket(i,float(i),np.full(shape,i%256,np.uint8),i+1)


class FrameMemoryTests(unittest.TestCase):
    def test_copies_sources_and_query_results(self):
        bank = TemporalFrameMemory(every_pairs=1)
        a,b = frame(0),frame(1)
        bank.add_pair(a,b)
        a.rgb.fill(255)
        found,reason = bank.retrieve_at(.5)
        self.assertEqual(reason,'retained_interval')
        self.assertTrue(np.all(found[0].rgb==0))
        found[0].rgb.flags.writeable=True
        found[0].rgb.fill(99)
        self.assertTrue(np.all(bank.retrieve_at(.5)[0][0].rgb==0))

    def test_byte_and_pair_limits_with_large_square_frames(self):
        bank = TemporalFrameMemory(every_pairs=1)
        for i in range(0,10,2): bank.add_pair(frame(i,(256,256,3)),frame(i+1,(256,256,3)))
        self.assertLessEqual(bank.bytes,884736)
        self.assertEqual(len(bank.evidence()['intervals']),2)
        self.assertEqual(bank.evicted_pairs,3)
        self.assertEqual(bank.retrieve_at(.5)[1],'not_retained')
        self.assertEqual(bank.retrieve_at(10)[1],'not_observed')

    def test_no_nearest_frame_substitution_in_sampling_gap(self):
        bank = TemporalFrameMemory(every_pairs=2)
        for i in range(0,6,2): bank.add_pair(frame(i),frame(i+1))
        self.assertEqual(bank.retrieve_at(2.5),(None,'not_retained'))
        self.assertEqual(bank.retrieve_at(3.5),(None,'not_retained'))
        self.assertEqual(bank.observed_pairs,3)
        for value in (-1,float('nan'),float('inf'),True,'1'):
            with self.assertRaises(ValueError): bank.retrieve_at(value)

    def test_rejected_pair_does_not_change_bank(self):
        bank = TemporalFrameMemory(every_pairs=1)
        bank.add_pair(frame(0),frame(1))
        before = bank.evidence()
        for pair in ((frame(2),frame(2)),(frame(0),frame(1)),(frame(2),frame(3,(256,257,3)))):
            with self.assertRaises(ValueError): bank.add_pair(*pair)
            self.assertEqual(bank.evidence(),before)

    def test_oversize_copy_failure_before_counters_and_clear(self):
        bank = TemporalFrameMemory(max_bytes=10)
        with self.assertRaises(ValueError): bank.add_pair(frame(0),frame(1))
        self.assertEqual(bank.observed_pairs,0)
        bank.clear()
        self.assertEqual(bank.bytes,0)


class RememberingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        default = next((p/'models/Qwen3.5-0.8B' for p in (PROJECT,*PROJECT.parents)
                        if (p/'models/Qwen3.5-0.8B/config.json').is_file()),PROJECT/'models/Qwen3.5-0.8B')
        model_dir = Path(getattr(cls,'model_dir',os.environ.get('QWEN35_MODEL_DIR',str(default))))
        cls.processor = AutoProcessor.from_pretrained(model_dir,local_files_only=True)

    def setUp(self):
        self.model = small_model(self.processor)
        self.session = OnlinePairSession(self.model,self.processor,max_tokens=200,max_question_tokens=96,
            max_answer_tokens=2,resize_mode='letterbox',video_logits='none')
        self.events = []
        self.demo = RememberingPairDemo(self.session,'Describe the subject.',2,self.events.append,
                                        first_qa_frames=100,memory_every_pairs=1)

    def tearDown(self): self.demo.close()

    @torch.inference_mode()
    def test_old_evidence_survives_capacity_reset_without_main_change(self):
        for i in range(6): self.demo.append(frame(i))
        self.demo.finish()
        self.assertGreaterEqual(self.demo.resets,1)
        self.assertGreater(self.demo.memory_from_pts,1)
        before,cache = self.session.status(),snapshot(self.session.main.cache)
        self.model.model.rope_deltas = torch.tensor([[17]])
        answer = self.demo.ask_at(.5,'Describe the subject.',2)
        self.assertTrue(answer['model_called'])
        self.assertEqual(answer['evidence_frame_ids'],[0,1])
        self.assertEqual(answer['evidence_interval'],[0.,1.])
        self.assertTrue(answer['main_unchanged_bitwise'])
        self.assertEqual(self.session.status(),before)
        assert_states(self,self.session.main.cache,cache,exact=True)
        self.assertEqual(self.model.model.rope_deltas.item(),17)

    def test_missing_and_future_evidence_never_call_model(self):
        for i in range(8): self.demo.append(frame(i))
        with mock.patch.object(self.model.model,'get_video_features',side_effect=AssertionError('No model call')):
            missing = self.demo.ask_at(.5,'Describe the subject.')
            future = self.demo.ask_at(99.,'Describe the subject.')
        self.assertEqual(missing['retrieval_reason'],'not_retained')
        self.assertEqual(future['retrieval_reason'],'not_observed')
        self.assertFalse(missing['model_called'] or future['model_called'])
        self.assertEqual(self.demo.history_model_calls,0)

    def test_eof_padding_does_not_create_memory_pair(self):
        for i in range(3): self.demo.append(frame(i))
        self.demo.finish()
        self.assertEqual(self.demo.frame_memory.observed_pairs,1)
        answer = self.demo.ask_at(2.,'Describe the subject.')
        self.assertEqual(answer['retrieval_reason'],'not_retained')
        self.assertFalse(answer['model_called'])

    @torch.inference_mode()
    def test_failed_history_generation_restores_main_and_rope(self):
        for i in range(2): self.demo.append(frame(i))
        before,cache = self.session.status(),snapshot(self.session.main.cache)
        self.model.model.rope_deltas = torch.tensor([[19]])
        with mock.patch.object(OnlinePairSession,'ask',side_effect=RuntimeError('Injected generation failure')):
            with self.assertRaisesRegex(RuntimeError,'Injected'): self.demo.ask_at(.5,'Describe the subject.')
        self.assertEqual(self.session.status(),before)
        assert_states(self,self.session.main.cache,cache,exact=True)
        self.assertEqual(self.model.model.rope_deltas.item(),19)

    def test_invalid_query_and_closed_demo(self):
        for stamp in (-1,float('inf'),True):
            with self.assertRaises(ValueError): self.demo.ask_at(stamp,'Describe the subject.')
        with self.assertRaises(ValueError): self.demo.ask_at(0.,'Describe the subject.',True)
        self.demo.close()
        self.assertEqual(self.demo.frame_memory.bytes,0)
        with self.assertRaises(RuntimeError): self.demo.ask_at(0.,'Describe the subject.')

    def test_legacy_demo_has_no_external_memory(self):
        legacy = PairDemo(self.session,'Describe the subject.',2,first_qa_frames=100)
        self.assertNotIn('external_frame_memory',legacy.status())
        legacy.close()


if __name__ == '__main__': unittest.main()
