"""Verify declared single factors and honest source-versus-label metadata."""
from copy import deepcopy
import unittest

import numpy as np

from streaming.action_inputs import relationships, assign_slots, validate_case
from streaming.window import FramePacket


class ActionInputTests(unittest.TestCase):
    def base(self):
        return {'id': 'base', 'indices': [7, 8, 10, 11], 'labels': [7.007, 8.008, 10.01, 11.011], 'factor': 'actual_anchor'}

    def intervention(self, factor, slot=0):
        value = deepcopy(self.base())
        value.update(id='variant', factor=factor, parent='base', slot=slot)
        return value

    def test_one_pixel_preserves_labels(self):
        variant = self.intervention('one_pixel_slot')
        variant['indices'][0] = 8
        record = relationships([self.base(), variant])[0]
        self.assertEqual(record['changed_image_slots'], [0])
        self.assertEqual(record['changed_time_slots'], [])

    def test_one_pixel_with_label_change_rejected(self):
        variant = self.intervention('one_pixel_slot')
        variant['indices'][0] = 8
        variant['labels'][0] += .5
        with self.assertRaises(ValueError):
            relationships([self.base(), variant])

    def test_one_time_preserves_images(self):
        variant = self.intervention('one_time_slot', 1)
        variant['labels'][1] = 9.009
        self.assertEqual(relationships([self.base(), variant])[0]['changed_image_slots'], [])

    def test_shift_requires_same_gaps(self):
        variant = self.intervention('time_shift')
        variant['shift'] = 10
        variant['labels'] = [t + 10 for t in variant['labels']]
        relationships([self.base(), variant])
        variant['labels'][2] += .1
        with self.assertRaises(ValueError):
            relationships([self.base(), variant])

    def test_invalid_labels_rejected(self):
        for labels in ([0, 0, 2, 3], [0, float('nan'), 2, 3], [-1, 0, 1, 2]):
            value = self.base()
            value['labels'] = labels
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                validate_case(value)

    def test_duplicate_images_have_unique_diagnostic_ids(self):
        real = FramePacket(240, 8.008, np.full((32, 32, 3), 40, np.uint8), 241)
        frames = assign_slots([real]*4, [7., 8., 10., 11.])
        self.assertEqual([p.frame_id for p in frames], [0, 1, 2, 3])
        self.assertEqual(real.timestamp, 8.008)
        self.assertEqual([p.timestamp for p in frames], [7., 8., 10., 11.])
        self.assertTrue(all(not p.rgb.flags.writeable for p in frames))

    def test_undeclared_factor_rejected(self):
        with self.assertRaises(ValueError):
            relationships([self.base(), self.intervention('unknown')])

    def test_duplicate_case_ids_rejected(self):
        with self.assertRaises(ValueError):
            relationships([self.base(), self.base()])


if __name__ == '__main__':
    unittest.main()
