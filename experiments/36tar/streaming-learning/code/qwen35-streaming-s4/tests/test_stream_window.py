"""Meaningful causal/transactional checks without pretrained models or GPU."""
import unittest
import numpy as np
from streaming.window import CausalSampler, FramePacket, FrameWindow, iter_windows


def packet(index, timestamp=None):
    return FramePacket(index, float(index) if timestamp is None else timestamp,
                       np.full((8,8,3), index, dtype=np.uint8), index+1)


class WindowTests(unittest.TestCase):
    def test_question_does_not_pull_next_frame(self):
        visited = []
        def source():
            for i in range(8):
                visited.append(i)
                yield packet(i)
        windows = iter_windows(source(), capacity=4, every=2)
        for ids in ([0,1], [0,1,2,3], [2,3,4,5], [4,5,6,7]):
            frames, evidence = next(windows)
            self.assertEqual(evidence["frame_ids"], ids)
            self.assertEqual(visited, list(range(ids[-1]+1)))
            self.assertEqual(evidence["delivered_decode_count"], ids[-1]+1)
        self.assertRaises(StopIteration, next, windows)

    def test_invalid_append_leaves_window_unchanged(self):
        window = FrameWindow(4)
        window.append(packet(0))
        before = window.evidence()
        bad = [packet(0), packet(1,-1), packet(1,float("nan")), packet(1,0),
               FramePacket(1,1,np.zeros((300,8,3),dtype=np.uint8),2),
               FramePacket(1,1,np.zeros((8,8,3),dtype=np.float32),2)]
        for value in bad:
            with self.assertRaises(ValueError):
                window.append(value)
            self.assertEqual(window.evidence(), before)

    def test_window_owns_pixels_and_reset_restarts_identity(self):
        window = FrameWindow(4)
        value = packet(0)
        window.append(value)
        value.rgb[:] = 99
        self.assertTrue((window.snapshot()[0].rgb == 0).all())
        self.assertFalse(window.snapshot()[0].rgb.flags.writeable)
        window.reset()
        fresh = FrameWindow(4)
        window.append(packet(0)); fresh.append(packet(0))
        self.assertEqual(window.evidence(), fresh.evidence())

    def test_sampler_uses_arrived_pts_and_skips_missed_deadlines(self):
        sampler = CausalSampler(1)
        stamps = [0,0.2,0.9,1.1,1.9,2.0,5.4,5.5,6.0]
        self.assertEqual([t for t in stamps if sampler.accept(t)], [0,1.1,2.0,5.4,6.0])
        before = sampler.next_due, sampler.last_time
        for bad in (6.0,5.0,float("nan")):
            self.assertRaises(ValueError, sampler.accept, bad)
            self.assertEqual((sampler.next_due,sampler.last_time),before)

    def test_window_is_bounded_after_many_frames(self):
        window = FrameWindow(4)
        for i in range(100):
            window.append(packet(i))
            self.assertLessEqual(len(window.snapshot()),4)
            self.assertLessEqual(window.evidence()["window_pixel_bytes"],4*8*8*3)
        self.assertEqual(window.evidence()["frame_ids"],[96,97,98,99])

    def test_same_prefix_different_future_has_same_early_window(self):
        def source(suffix):
            yield packet(0)
            yield packet(1)
            for i in range(2,6):
                value = packet(i)
                value.rgb[:] = suffix
                yield value
        a = iter_windows(source(50))
        b = iter_windows(source(200))
        self.assertEqual(next(a)[1], next(b)[1])
        self.assertNotEqual(next(a)[1]["frame_sha256"],next(b)[1]["frame_sha256"])


if __name__ == "__main__":
    unittest.main()
