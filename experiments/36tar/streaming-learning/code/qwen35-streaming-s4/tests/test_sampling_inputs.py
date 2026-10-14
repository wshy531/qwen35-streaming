import unittest

from streaming.sampling_inputs import select_due, targets


class SamplingInputTests(unittest.TestCase):
    def test_declared_targets(self):
        self.assertEqual(targets(4, 0.5), [4.5, 5.5, 6.5, 7.5])

    def test_invalid_start(self):
        for value in (-1, float('nan'), float('inf'), True):
            with self.assertRaises(ValueError):
                targets(value, 0)

    def test_unregistered_phase(self):
        for value in (-0.5, 1, float('nan'), True):
            with self.assertRaises(ValueError):
                targets(0, value)

    def test_first_real_pts_after_target(self):
        records = [(i, i / 10, None) for i in range(40)]
        selected = list(select_due(records, [0.05, 1.05, 2.05, 3.05]))
        self.assertEqual([i for i, *_ in selected], [1, 11, 21, 31])
        self.assertEqual([t for _, t, *_ in selected], [0.1, 1.1, 2.1, 3.1])

    def test_does_not_read_after_fourth(self):
        def records():
            for i in range(4):
                yield i, float(i), None
            raise AssertionError('Future read')
        self.assertEqual(len(list(select_due(records(), [0, 1, 2, 3]))), 4)

    def test_missing_slot_not_duplicated(self):
        with self.assertRaises(ValueError):
            list(select_due([(0, 0, None), (60, 2, None)], [0, 1, 2, 3]))

    def test_short_source_rejected(self):
        with self.assertRaises(ValueError):
            list(select_due([(0, 0, None)], [0, 1, 2, 3]))

    def test_invalid_delivered_pts(self):
        for timestamp in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                list(select_due([(0, 0, None), (1, timestamp, None)], [0, 1, 2, 3]))


if __name__ == '__main__':
    unittest.main()
