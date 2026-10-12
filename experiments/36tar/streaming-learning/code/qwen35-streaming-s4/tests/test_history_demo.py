"""Full generic replay with optional history and the original default path."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from scripts.video_demo import replay,budget
from streaming.viewer import Viewer
from tests import test_frame_memory as memory_tests
from tests.test_online import small_model


class HistoryDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        memory_tests.RememberingTests.setUpClass()
        cls.processor = memory_tests.RememberingTests.processor

    @torch.inference_mode()
    def test_optional_replay_cross_reset_queries_and_viewer(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(stage='preflight',out=Path(tmp),video=Path(tmp)/'not_used',
                playback='fast',fps=1.,duration=10.,max_frames=6,wall_seconds=120,
                question='Describe the subject.',answer_tokens=2,max_tokens=200,first_qa_frames=2,
                qa_every=2,ui_port=-1,consumer_delay_seconds=0.,remember_history=True,
                memory_every_pairs=1,history_time=[.5,2.5,99.],history_question='Describe the subject.')
            with mock.patch('scripts.video_demo.read_packets',side_effect=lambda *_:(memory_tests.frame(i) for i in range(6))):
                report = replay(small_model(self.processor),self.processor,args)
            self.assertGreater(report['final']['resets'],0)
            self.assertEqual(report['final']['history_queries'],3)
            self.assertEqual(report['final']['history_model_calls'],2)
            self.assertEqual(report['historical_answers'][0]['evidence_frame_ids'],[0,1])
            self.assertEqual(report['historical_answers'][-1]['retrieval_reason'],'not_observed')
            self.assertTrue(report['historical_answers'][0]['main_unchanged_bitwise'])
            self.assertEqual(report['counts']['vision'],5)
            self.assertLessEqual(report['counts']['language'],budget(args)['language_forwards_max'])
            self.assertIn('external_frame_memory',(Path(tmp)/'viewer.html').read_text(encoding='utf-8'))
            events = [json.loads(x) for x in (Path(tmp)/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            self.assertEqual(sum(x['kind']=='history_answer' for x in events),3)
            self.assertEqual(events[-1]['status']['external_frame_memory']['persistent_rgb_bytes'],0)

    @torch.inference_mode()
    def test_default_replay_does_not_allocate_frame_bank(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = Namespace(stage='preflight',out=Path(tmp),video=Path(tmp)/'not_used',
                playback='fast',fps=1.,duration=10.,max_frames=2,wall_seconds=120,
                question='Describe the subject.',answer_tokens=2,max_tokens=200,first_qa_frames=2,
                qa_every=2,ui_port=-1,consumer_delay_seconds=0.)
            with mock.patch('scripts.video_demo.read_packets',side_effect=lambda *_:(memory_tests.frame(i) for i in range(2))):
                report = replay(small_model(self.processor),self.processor,args)
            self.assertFalse(report['external_history_enabled'])
            self.assertEqual(report['historical_answers'],[])
            self.assertNotIn('external_frame_memory',report['final'])
            self.assertEqual(report['counts']['vision'],1)

    def test_history_viewer_labels_refusal_without_fabricated_basis(self):
        viewer = Viewer('CPU')
        state = {'last_received_pts':5.,'memory_from_pts':4.,'resets':2,'received_real_frames':6,
                 'encoded_real_frames':6,'session':{'seen_tokens':70},'finished':True,
                 'external_frame_memory':{'persistent_rgb_bytes':0,'max_bytes':884736,'evicted_pairs':3,'intervals':[]}}
        viewer.event({'kind':'history_answer','status':state,'answer':{'route':'external_frame_memory',
            'timestamp':.5,'target_time':.5,'model_called':False,'question':'Old subject?',
            'evidence_interval':None,'answer':'No retained evidence.'}})
        self.assertEqual(len(viewer.snapshot()['answers']),1)
        self.assertFalse(viewer.snapshot()['answers'][0]['model_called'])
        viewer.close()


if __name__ == '__main__': unittest.main()
