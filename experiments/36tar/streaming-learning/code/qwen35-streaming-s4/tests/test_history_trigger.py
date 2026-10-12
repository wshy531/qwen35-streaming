"""Trigger on encoded PTS, continue input, and never silently fall back at EOF."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch
from scripts.video_demo import replay
from tests import test_frame_memory as memory_tests
from tests.test_online import small_model


class HistoryTriggerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        memory_tests.RememberingTests.setUpClass()
        cls.processor = memory_tests.RememberingTests.processor

    def arguments(self,out,threshold,max_frames=6):
        return Namespace(stage='preflight',out=out,video=out/'not_used',playback='fast',fps=1.,
            duration=10.,max_frames=max_frames,wall_seconds=120,question='Describe the subject.',
            answer_tokens=2,max_tokens=200,first_qa_frames=2,qa_every=2,ui_port=-1,
            consumer_delay_seconds=0.,remember_history=True,memory_every_pairs=1,
            history_time=[.5],history_question='Describe the subject.',history_query_after=threshold)

    @torch.inference_mode()
    def run_case(self,threshold,max_frames=6):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);args=self.arguments(out,threshold,max_frames)
            with mock.patch('scripts.video_demo.read_packets',side_effect=lambda *_:(memory_tests.frame(i) for i in range(max_frames))):
                report=replay(small_model(self.processor),self.processor,args)
            events=[json.loads(x) for x in (out/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            return report,events

    def test_query_midstream_after_reset_then_continue(self):
        report,events=self.run_case(3.)
        history=[e for e in events if e['kind']=='history_answer']
        self.assertEqual(len(history),1)
        answer=history[0]['answer']
        self.assertEqual(answer['trigger_encoded_pts'],3.)
        self.assertFalse(answer['trigger_finished'])
        self.assertTrue(answer['main_unchanged_bitwise'])
        index=events.index(history[0])
        self.assertTrue(any(e['kind']=='encoded' and e['record']['frame_ids']==[4,5] for e in events[index+1:]))
        self.assertEqual(report['final']['resets'],2)
        self.assertEqual(report['pending_history_times'],[])

    def test_pending_frame_does_not_trigger_query(self):
        report,events=self.run_case(2.)
        history=next(e for e in events if e['kind']=='history_answer')
        self.assertEqual(history['answer']['trigger_received_pts'],3.)
        self.assertEqual(history['answer']['trigger_encoded_pts'],3.)
        self.assertNotEqual(history['status']['received_real_frames'],3)

    def test_unreached_threshold_no_hidden_eof_query(self):
        report,events=self.run_case(8.)
        self.assertFalse(report['history_query_triggered'])
        self.assertEqual(report['pending_history_times'],[.5])
        self.assertEqual(report['historical_answers'],[])
        self.assertFalse(any(e['kind']=='history_answer' for e in events))
        self.assertEqual(report['counts']['vision'],3)

    def test_odd_eof_encoding_marks_finish_phase(self):
        report,events=self.run_case(2.,max_frames=3)
        history=next(e for e in events if e['kind']=='history_answer')
        self.assertEqual(history['answer']['trigger_encoded_pts'],2.)
        self.assertTrue(history['answer']['trigger_finished'])
        self.assertTrue(report['history_query_triggered'])


if __name__=='__main__':unittest.main()
