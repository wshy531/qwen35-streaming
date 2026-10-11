"""Policy contracts with the actual processor and tiny FP32 model, no semantics."""
import unittest
from unittest import mock

import numpy as np
import torch
from transformers import AutoProcessor

from scripts.baseline import PROJECT
from streaming.demo import PairDemo
from streaming.online import OnlinePairSession, PairInputBuilder
from streaming.window import FramePacket
from tests.test_online import small_model


def packet(i, timestamp=None, side=32):
    return FramePacket(i, float(i) if timestamp is None else timestamp,
                       np.full((side, side, 3), 30 + i % 180, np.uint8), i + 1)


class DemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.processor = AutoProcessor.from_pretrained(PROJECT / "models/Qwen3.5-0.8B", local_files_only=True)

    def setUp(self):
        self.model = small_model(self.processor)
        self.events = []
        self.demo = self.make()
        self.addCleanup(lambda: self.demo.close())

    def make(self, max_tokens=256):
        session = OnlinePairSession(self.model, self.processor, canvas=(32, 32), max_tokens=max_tokens)
        return PairDemo(session, "Name the subject.", 2, self.events.append)

    def test_generic_single_frame_eof_can_answer_custom_question(self):
        self.demo.close()
        self.events.clear()
        session = OnlinePairSession(self.model,self.processor,canvas=(32,32),max_answer_tokens=2,
                                    resize_mode="letterbox",video_logits="none")
        self.demo = PairDemo(session,"What is visible?",2,self.events.append,first_qa_frames=1)
        self.demo.append(packet(7,timestamp=2.5))
        self.demo.finish()
        self.assertEqual(self.demo.received,1)
        self.assertEqual(self.demo.questions,1)
        self.assertEqual(self.demo.memory_from_pts,2.5)
        answer = next(e for e in self.events if e["kind"] == "answer")
        self.assertEqual(answer["answer"]["question"],"What is visible?")
        self.assertEqual(answer["status"]["memory_from_pts"],2.5)

    def test_skipped_source_frames_preserve_actual_pts_and_qa_context(self):
        for i in (0,5,10,15):
            self.demo.append(packet(i))
        self.assertEqual(self.demo.received,4)
        self.assertEqual(self.demo.session.ledger[-1]["timestamps"],[10.,15.])
        self.assertEqual(self.demo.status()["memory_from_pts"],0.)
        answer = next(e for e in self.events if e["kind"] == "answer")
        self.assertEqual(answer["answer"]["timestamp"],15.)
        self.assertEqual(answer["answer"]["frame_id"],15)

    def test_capacity_reset_exposes_new_memory_origin(self):
        self.limited()
        for i in range(6):
            self.demo.append(packet(i,timestamp=float(i)+10))
        resets = [e for e in self.events if e["kind"] == "capacity_reset"]
        self.assertGreaterEqual(len(resets),1)
        for event in resets:
            self.assertEqual(event["status"]["memory_from_pts"],event["retained_frame_ids"][0]+10.)
        self.assertEqual(self.demo.status()["memory_from_pts"],14.)
        self.assertFalse(self.demo.status()["history_across_reset"])

    def limited(self):
        builder = PairInputBuilder(self.processor, (32, 32))
        sizes = [builder.pair(packet(i), packet(i + 1), 0, 0)[0].input_ids.shape[1] for i in (0, 2)]
        self.demo.close()
        self.events.clear()
        self.demo = self.make(80 + builder.prefix_ids.shape[1] + sum(sizes))

    def test_first_four_then_each_two_and_pending_no_computation(self):
        p = packet(0)
        with mock.patch.object(self.model.model, "get_video_features", side_effect=AssertionError("No early encode")):
            self.demo.append(p)
            p.rgb.fill(255)
            self.assertTrue(np.all(self.demo.pending.rgb == 30))
            self.assertFalse(self.demo.pending.rgb.flags.writeable)
        self.assertEqual(self.demo.session.main.seen_tokens, 0)
        for i in range(1, 6):
            self.demo.append(packet(i))
            self.assertEqual(self.demo.questions, 0 if i < 3 else 1 if i < 5 else 2)
        answers = [e for e in self.events if e["kind"] == "answer"]
        self.assertEqual([e["answer"]["frame_id"] for e in answers], [3, 5])
        self.assertTrue(all(e["main_unchanged_bitwise"] for e in answers))
        self.assertEqual(self.demo.session.vision_encodings, 3)

    def test_capacity_reset_retains_both_new_frames_and_restarts_four_frame_trigger(self):
        self.limited()
        for i in range(8):
            self.demo.append(packet(i))
        self.assertEqual(self.demo.resets, 1)
        reset = next(e for e in self.events if e["kind"] == "capacity_reset")
        self.assertEqual(reset["retained_frame_ids"], [4, 5])
        self.assertEqual([e["answer"]["frame_id"] for e in self.events if e["kind"] == "answer"], [3, 7])
        self.assertEqual([r["frame_ids"] for r in self.demo.session.ledger], [[4, 5], [6, 7]])
        self.assertEqual(self.demo.encoded, 8)
        self.assertEqual(self.demo.session.accepted_frames, 4)

    def test_three_real_frames_at_eof_do_not_become_four(self):
        for i in range(3):
            self.demo.append(packet(i))
        self.demo.finish()
        self.assertEqual((self.demo.encoded, self.demo.questions), (3, 0))
        self.assertEqual(self.demo.session.ledger[-1]["frame_ids"], [2, 2])
        self.assertTrue(self.demo.session.ledger[-1]["synthetic_padding"])
        self.assertEqual(self.demo.session.encoded_frames, 3)
        with self.assertRaises(RuntimeError):
            self.demo.append(packet(3))

    def test_eof_capacity_reset_keeps_one_real_frame(self):
        self.limited()
        for i in range(5):
            self.demo.append(packet(i))
        self.demo.finish()
        self.assertEqual((self.demo.encoded, self.demo.questions, self.demo.resets), (5, 1, 1))
        self.assertEqual(self.demo.session.encoded_frames, 1)
        self.assertEqual(self.demo.session.ledger[0]["frame_ids"], [4, 4])
        self.assertTrue(next(e for e in self.events if e["kind"] == "capacity_reset")["synthetic_padding"])

    def test_odd_after_first_answer_does_not_trigger_extra_eof_answer(self):
        for i in range(5):
            self.demo.append(packet(i))
        self.demo.finish()
        self.assertEqual((self.demo.encoded, self.demo.questions), (5, 1))

    def test_global_order_and_shape_survive_capacity_reset_and_rejection_is_reversible(self):
        self.limited()
        for i in range(6):
            self.demo.append(packet(i))
        before = self.demo.status()
        for p in (packet(1, 99.), packet(6, 4.), packet(6, 6., 64)):
            with self.assertRaises(ValueError):
                self.demo.append(p)
            self.assertEqual(before, self.demo.status())
        self.demo.append(packet(6))
        self.assertEqual(self.demo.received, 7)

    def test_actual_timestamp_tokens_are_checked_before_vision(self):
        self.demo.close()
        builder = PairInputBuilder(self.processor, (32, 32))
        short = builder.pair(packet(0), packet(1), 0, 0)[0].input_ids.shape[1]
        long = builder.pair(packet(0, 100000.), packet(1, 100001.), 0, 0)[0].input_ids.shape[1]
        self.assertGreater(long, short)
        self.demo = self.make(80 + builder.prefix_ids.shape[1] + short)
        self.demo.append(packet(0, 100000.))
        with mock.patch.object(self.model.model, "get_video_features", side_effect=AssertionError("No oversized GPU encode")):
            with self.assertRaisesRegex(ValueError, "fresh epoch"):
                self.demo.append(packet(1, 100001.))
        self.assertEqual(self.demo.session.main.seen_tokens, 0)
        self.assertEqual(self.demo.session.vision_encodings, 0)
        self.assertTrue(self.demo.invalid)

    def test_failure_is_logged_and_close_clears_active_state(self):
        self.demo.append(packet(0))
        with mock.patch.object(self.model.model, "get_video_features", side_effect=RuntimeError("Injected failure")):
            with self.assertRaisesRegex(RuntimeError, "Injected"):
                self.demo.append(packet(1))
        self.assertTrue(self.demo.invalid)
        self.assertEqual(self.events[-1]["kind"], "error")
        with self.assertRaises(RuntimeError):
            self.demo.append(packet(2))
        self.demo.close()
        self.assertEqual(self.demo.session.main.seen_tokens, 0)
        self.assertIsNone(self.demo.pending)
        self.assertTrue(self.demo.closed)
        n = len(self.events)
        self.demo.close()
        self.assertEqual(len(self.events), n)

    def test_empty_and_even_eof_release_and_seal(self):
        self.demo.finish()
        self.assertEqual((self.demo.pairs, self.demo.questions), (0, 0))
        self.demo.close()
        self.demo = self.make()
        for i in range(4):
            self.demo.append(packet(i))
        self.demo.finish()
        self.assertEqual((self.demo.pairs, self.demo.questions), (2, 1))
        self.assertFalse(self.demo.session.ledger[-1]["synthetic_padding"])


if __name__ == "__main__":
    unittest.main()
