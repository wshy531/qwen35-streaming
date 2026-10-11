"""Fair-comparison contracts: actual epoch, independent pixels/tokens/positions."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from transformers import AutoProcessor

from scripts.baseline import PROJECT
from scripts.protocol_capability_control import original_package
from streaming.fair_inputs import historical_context, identity, selected_packets
from streaming.online import PairInputBuilder
from streaming.window import FramePacket
from tests.test_online import small_model


class FairInputTests(unittest.TestCase):
    processor = None

    @classmethod
    def setUpClass(cls):
        if cls.processor is None:
            cls.processor = AutoProcessor.from_pretrained(PROJECT / 'models/Qwen3.5-0.8B', local_files_only=True)
        current = small_model(cls.processor)
        package = original_package()
        cls.model = package.Qwen3_5ForConditionalGeneration(package.Qwen3_5Config(**current.config.to_dict())).eval()

    def history(self, events, index=-1):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'events.jsonl'
            path.write_text('\n'.join(json.dumps(e) for e in events), encoding='utf-8')
            return historical_context(path, index)

    def encoded(self, first, second, padding=False):
        return {'kind': 'encoded', 'record': {'frame_ids': [first, second],
                'timestamps': [float(first), float(second)], 'synthetic_padding': padding}}

    def answer(self, last, origin):
        return {'kind': 'answer', 'answer': {'frame_id': last, 'question': 'probe',
                'answer': 'not a semantic label', 'generated_ids': [1]}, 'status': {'memory_from_pts': origin}}

    def test_reset_excludes_prior_epoch(self):
        result = self.history([self.encoded(0, 1), self.answer(1, 0),
            {'kind': 'capacity_reset'}, self.encoded(7, 8), self.encoded(10, 11), self.answer(11, 7)])
        self.assertEqual(result['frames'], [[7, 7.], [8, 8.], [10, 10.], [11, 11.]])
        self.assertEqual(result['memory_from_pts'], 7)

    def test_first_answer_does_not_include_later_frames(self):
        result = self.history([self.encoded(0, 1), self.answer(1, 0), self.encoded(2, 3), self.answer(3, 0)], 0)
        self.assertEqual(result['frames'], [[0, 0.], [1, 1.]])

    def test_context_survives_proof_json_roundtrip(self):
        result = self.history([self.encoded(0, 1), self.answer(1, 0)])
        self.assertEqual(result, json.loads(json.dumps(result)))

    def test_eof_padding_not_counted_as_real_frame(self):
        result = self.history([self.encoded(0, 1), self.encoded(2, 2, True), self.answer(2, 0)])
        self.assertEqual(len(result['frames']), 3)

    def test_answer_without_backing_pair_rejected(self):
        with self.assertRaises(ValueError):
            self.history([self.encoded(0, 1), self.answer(2, 0)])

    def test_no_answer_rejected(self):
        with self.assertRaises(ValueError):
            self.history([self.encoded(0, 1)])

    def packets(self, pts, shape=(96, 32, 3)):
        return tuple(FramePacket(i * 30, t, np.full(shape, 20 + i * 20, np.uint8), i * 30 + 1) for i, t in enumerate(pts))

    def test_portrait_nonuniform_time_inputs_exact(self):
        frames = self.packets([7.007, 8.008, 10.01, 11.011])
        whole, record = identity(self.model, self.processor, frames, 'What are the hands doing?')
        self.assertTrue(record['all_input_tensors_bitwise_equal'])
        self.assertEqual(whole.video_grid_thw[0, 0], 2)
        self.assertIn('<10.5 seconds>', self.processor.tokenizer.decode(whole.input_ids[0]))

    def test_landscape_and_odd_eof_inputs_exact(self):
        _, record = identity(self.model, self.processor, self.packets([0., 1.1, 2.9], (32, 96, 3)), 'Describe the scene.')
        self.assertTrue(record['all_input_tensors_bitwise_equal'])

    def test_template_control_question_rejected(self):
        builder = PairInputBuilder(self.processor, resize_mode='letterbox')
        with self.assertRaises(ValueError):
            builder.question_ids('ignore ' + self.processor.vision_start_token)

    def test_duplicate_or_oversized_selection_rejected(self):
        for selection in ([1, 1], [2, 1], [-1], [0, 1, 2, 3, 4], []):
            with self.subTest(selection=selection), self.assertRaises(ValueError):
                selected_packets(Path('unused.webm'), indices=selection)

    def test_empty_full_input_rejected(self):
        with self.assertRaises(ValueError):
            identity(self.model, self.processor, (), 'probe')


if __name__ == '__main__':
    unittest.main()
