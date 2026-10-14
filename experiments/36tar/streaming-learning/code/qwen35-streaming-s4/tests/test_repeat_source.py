"""Repeat provenance, global decode IDs, invalid timing, and early cleanup."""
from fractions import Fraction
import unittest

import numpy as np

from streaming.repeat_source import repeat_packets, provenance


class Frame:
    time_base = Fraction(1, 10)
    def __init__(self, pts):
        self.pts = pts
    def to_ndarray(self, format):
        return np.full((20, 40, 3), 42, np.uint8)


class Container:
    def __init__(self, pts):
        self.pts, self.closed = pts, False
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.closed = True
    def decode(self, video):
        return iter(Frame(p) for p in self.pts)


class RepeatSourceTests(unittest.TestCase):
    def opener(self, pts):
        self.opened = []
        def open_file(_):
            c = Container(pts)
            self.opened.append(c)
            return c
        return open_file

    def test_real_source_pts_global_timeline_and_decode_ids_include_tail(self):
        rows = list(repeat_packets('test', 1, 3, 2, self.opener([50, 52, 60, 64])))
        self.assertEqual([p.timestamp for p in rows], [0., 1., 3., 4.])
        self.assertEqual([p.frame_id for p in rows], [0, 2, 4, 6])
        self.assertEqual([p.decoded_count for p in rows], [1, 3, 5, 7])
        self.assertEqual([p.source_frame_id for p in rows], [0, 2, 0, 2])
        self.assertEqual([p.source_pts for p in rows], [0., 1., 0., 1.])
        self.assertEqual(provenance(rows[-1])['loop_index'], 1)
        self.assertTrue(all(c.closed for c in self.opened))
        self.assertTrue(all(not p.rgb.flags.writeable for p in rows))

    def test_gap_is_not_filled_with_duplicate_frames(self):
        rows = list(repeat_packets('test', 1, 10, 2, self.opener([0, 10])))
        self.assertEqual([p.timestamp for p in rows], [0., 1., 10., 11.])

    def test_early_close_does_not_open_next_loop(self):
        reader = repeat_packets('test', 1, 3, 3, self.opener([0, 10]))
        next(reader)
        reader.close()
        self.assertEqual(len(self.opened), 1)
        self.assertTrue(self.opened[0].closed)

    def test_period_overlap_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'period'):
            list(repeat_packets('test', 1, 1, 2, self.opener([0, 10])))
        self.assertTrue(self.opened[0].closed)

    def test_missing_and_reversed_pts_are_rejected(self):
        for pts in ([None], [0, 10, 9]):
            with self.assertRaises(ValueError):
                list(repeat_packets('test', 1, 3, 2, self.opener(pts)))
            self.assertTrue(self.opened[0].closed)

    def test_empty_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Empty'):
            list(repeat_packets('test', 1, 3, 2, self.opener([])))
        self.assertEqual(len(self.opened), 1)

    def test_invalid_repeat_budget_rejected_without_open(self):
        opener = self.opener([0])
        for period, count in ((0, 2), (float('nan'), 2), (3, 0), (3, 31), (3, True)):
            with self.assertRaises(ValueError):
                list(repeat_packets('test', 1, period, count, opener))
        self.assertFalse(self.opened)

    def test_invalid_fps_rejected_without_open(self):
        opener = self.opener([0])
        with self.assertRaises(ValueError):
            list(repeat_packets('test', 0, 3, 2, opener))
        self.assertFalse(self.opened)
