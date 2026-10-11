"""Deterministic overload, PTS causality, bounded state, EOF and cancellation."""
import threading
import time
import unittest

import numpy as np

from streaming.playback import ClockPlayback,LatestMailbox,Pacer,SourceDigest
from streaming.window import FramePacket


def packet(i, pts=None):
    return FramePacket(i,float(i) if pts is None else pts,np.full((32,32,3),i%255,np.uint8),i+1)


class PlaybackTests(unittest.TestCase):
    def test_overload_replaces_only_unclaimed_frame_and_counts_losses(self):
        box = LatestMailbox()
        box.put(packet(0))
        inflight = box.take()
        self.assertIsNone(box.put(packet(1)))
        self.assertEqual(box.put(packet(2)),1)
        self.assertEqual(box.put(packet(3)),2)
        self.assertEqual(inflight.frame_id,0)
        self.assertEqual(box.take().frame_id,3)
        box.finish()
        with self.assertRaises(StopIteration):
            box.take()
        s = box.status()
        self.assertEqual((s["produced_samples"],s["delivered_samples"],s["dropped_samples"]),(4,2,2))
        self.assertEqual(s["max_buffered_samples"],1)

    def test_repeated_overload_has_constant_pixel_storage(self):
        box = LatestMailbox()
        for i in range(2000):
            box.put(packet(i))
        self.assertEqual(box.status()["buffered_rgb_bytes"],32*32*3)
        self.assertEqual(box.status()["dropped_samples"],1999)
        self.assertEqual(box.take().frame_id,1999)

    def test_eof_drains_last_slot_and_error_is_not_silently_eof(self):
        box = LatestMailbox()
        box.put(packet(7))
        box.finish("controlled decode failure")
        self.assertEqual(box.take().frame_id,7)
        with self.assertRaisesRegex(RuntimeError,"decode failure"):
            box.take()

    def test_cancel_wakes_waiter_and_releases_buffer(self):
        box = LatestMailbox()
        done = threading.Event()
        def wait():
            try:
                box.take()
            except StopIteration:
                done.set()
        worker = threading.Thread(target=wait)
        worker.start()
        box.cancel()
        self.assertTrue(done.wait(1))
        worker.join(timeout=1)
        other = LatestMailbox()
        other.put(packet(1))
        other.cancel()
        self.assertEqual(other.status()["cancelled_samples"],1)
        self.assertEqual(other.status()["buffered_rgb_bytes"],0)

    def test_pacer_does_not_deliver_future_and_uses_actual_vfr_pts(self):
        stop = threading.Event()
        now = [100.]
        pacer = Pacer(stop,100.,clock=lambda:now[0],wait=lambda seconds:now.__setitem__(0,now[0]+seconds))
        for pts in (0.,0.013,0.257,1.4):
            self.assertTrue(pacer.until(pts))
            self.assertGreaterEqual(now[0]-100.,pts)
            self.assertLess(now[0]-100.-pts,1e-6)
        stop.set()
        self.assertFalse(pacer.until(2.))

    def test_close_interrupts_future_frame_wait_and_closes_generator(self):
        entered,closed = threading.Event(),threading.Event()
        def reader():
            try:
                entered.set()
                yield packet(0,10.)
            finally:
                closed.set()
        stream = ClockPlayback(reader(),20.,2).start()
        self.assertTrue(entered.wait(1))
        start = time.monotonic()
        stream.close()
        self.assertLess(time.monotonic()-start,1.)
        self.assertTrue(closed.is_set())
        self.assertFalse(stream.status()["producer_alive"])
        self.assertEqual(stream.status()["produced_samples"],0)

    def test_real_thread_all_source_fingerprint_includes_dropped_frames(self):
        closed = threading.Event()
        rows = [packet(i,0.) for i in range(5)]
        def reader():
            try:
                yield from rows
            finally:
                closed.set()
        seen = []
        stream = ClockPlayback(reader(),1.,5,lambda p,s:seen.append((p.frame_id,s["wall_seconds"]))).start()
        self.assertTrue(closed.wait(1))
        stream.thread.join(timeout=1)
        values = list(stream)
        expected = SourceDigest()
        for p in rows:
            expected.add(p)
        self.assertEqual([p.frame_id for p in values],[4])
        self.assertEqual(stream.status()["source_fingerprint"],expected.snapshot())
        self.assertEqual(len(seen),5)
        self.assertTrue(all(wall >= 0 for _,wall in seen))
        stream.close()

    def test_source_exception_propagates_and_stops_producer(self):
        def reader():
            raise ValueError("bad PTS")
            yield
        stream = ClockPlayback(reader(),1.,1).start()
        try:
            with self.assertRaisesRegex(RuntimeError,"bad PTS"):
                next(stream)
        finally:
            stream.close()
        self.assertFalse(stream.status()["producer_alive"])

    def test_threaded_vfr_source_never_publishes_before_pts(self):
        arrivals = []
        rows = [packet(i,pts) for i,pts in enumerate((0.,0.03,0.081))]
        stream = ClockPlayback(iter_reader(rows),1.,3,lambda p,s:arrivals.append((p.timestamp,s["wall_seconds"]))).start()
        stream.thread.join(timeout=2)
        try:
            self.assertFalse(stream.thread.is_alive())
            self.assertEqual(len(arrivals),3)
            self.assertTrue(all(wall >= pts for pts,wall in arrivals))
            self.assertEqual(stream.status()["dropped_samples"],2)
            self.assertEqual([p.frame_id for p in stream],[2])
        finally:
            stream.close()

    def test_duration_limit_is_not_counted_as_a_delivered_frame(self):
        stream = ClockPlayback(iter_reader([packet(0,0.),packet(1,2.)]),1.,5).start()
        stream.thread.join(timeout=1)
        self.assertEqual([p.frame_id for p in stream],[0])
        self.assertEqual(stream.reason,"video_duration_limit")
        self.assertEqual(stream.status()["source_fingerprint"]["samples"],1)
        stream.close()

    def test_invalid_input_budgets_rejected(self):
        for duration,maximum in ((0.,5),(float("nan"),2),(1.,0),(1.,True)):
            with self.assertRaises(ValueError):
                ClockPlayback(iter_reader([]),duration,maximum)


def iter_reader(rows):
    yield from rows
