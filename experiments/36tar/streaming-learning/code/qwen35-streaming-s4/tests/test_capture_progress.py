"""Sparse real file pacing and mocked camera decode progress regressions."""
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from scripts import capture_worker
from streaming.live_source import LiveSource
from streaming.playback import SourceDigest
from streaming.preprocessing import fit_canvas
from streaming.window import FramePacket, read_packets


class CaptureProgressTests(unittest.TestCase):
    def test_close_interrupts_a_long_scheduled_wait(self):
        import time
        source = LiveSource('synthetic','unused',.01,30,2)
        try:
            self.assertEqual(next(source.start()).frame_id,0)
            start = time.monotonic()
            source.close()
            self.assertLess(time.monotonic()-start,2.)
            self.assertEqual(source.digest.count,1)
            self.assertFalse(source.status()['worker_alive'])
            self.assertFalse(source.status()['producer_alive'])
        finally:
            source.close()

    def test_duration_limit_ends_before_a_distant_next_sample(self):
        import time
        source = LiveSource('synthetic','unused',.01,1.5,2)
        start = time.monotonic()
        try:
            rows = list(source.start())
            self.assertEqual([packet.frame_id for packet in rows],[0])
            self.assertEqual(source.reason,'live_duration_limit')
            self.assertGreater(source.progress_messages,0)
            self.assertLess(time.monotonic()-start,3.)
        finally:
            source.close()

    def test_real_sparse_file_survives_low_sampling_and_pts_gaps(self):
        import av
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'sparse.mkv'
            with av.open(str(path),'w') as container:
                video = container.add_stream('ffv1',rate=10)
                video.width,video.height,video.pix_fmt = 48,32,'bgr0'
                for i,pts in enumerate((0,50)):
                    frame = av.VideoFrame.from_ndarray(np.full((32,48,3),60+i*100,np.uint8),format='rgb24')
                    frame.pts,frame.time_base = pts,Fraction(1,10)
                    for encoded in video.encode(frame):
                        container.mux(encoded)
                for encoded in video.encode():
                    container.mux(encoded)
            for fps in (.2,1.):
                with self.subTest(fps=fps):
                    expected = SourceDigest()
                    for packet in read_packets(path,fps):
                        expected.add(FramePacket(packet.frame_id,packet.timestamp,
                            fit_canvas(packet.rgb,(192,256)),packet.decoded_count))
                    source = LiveSource('file',str(path),fps,8,3,idle_seconds=3.)
                    try:
                        rows = list(source.start())
                        self.assertEqual([p.timestamp for p in rows],[0.,5.])
                        self.assertEqual(source.reason,'source_eof')
                        self.assertEqual(source.digest.snapshot(),expected.snapshot())
                        self.assertEqual(source.status()['produced_samples'],2)
                        self.assertEqual(source.status()['dropped_samples'],0)
                        self.assertGreater(source.progress_messages,0)
                        self.assertFalse(source.worker_torch_loaded)
                    finally:
                        source.close()
                    self.assertFalse(source.status()['worker_alive'])
                    self.assertFalse(source.status()['producer_alive'])

    def test_camera_reports_real_decode_progress_between_selected_frames(self):
        # Simulate the camera decoder; no physical camera is opened.
        clock, messages = [100.], []
        def emit(meta,rgb=None):
            messages.append(dict(meta))
        class Camera:
            def __enter__(self): return self
            def __exit__(self,*_): return False
            def decode(self,video=0):
                for i in range(5):
                    clock[0] = 100.+i*.2
                    yield SimpleNamespace(width=96,height=64,pts=i*2,time_base=Fraction(1,10),
                        to_ndarray=lambda format:np.zeros((64,96,3),np.uint8))
        camera_open = mock.Mock(return_value=Camera())
        fake_av = SimpleNamespace(formats_available={'dshow'},open=camera_open)
        real_progress = capture_worker.CaptureProgress
        argv = ['capture_worker','--kind','dshow','--device','mock-camera','--fps','1',
                '--duration','3','--max-frames','2','--origin','100','--owner-pid',str(os.getpid())]
        with mock.patch.object(sys,'argv',argv), mock.patch.dict(sys.modules,{'av':fake_av}), \
             mock.patch.object(capture_worker.time,'monotonic',side_effect=lambda:clock[0]), \
             mock.patch.object(capture_worker.threading,'Thread'), \
             mock.patch.object(capture_worker,'send',side_effect=emit), \
             mock.patch.object(capture_worker,'CaptureProgress',side_effect=lambda kind:
                 real_progress(kind,clock=lambda:clock[0],emit=emit)):
            self.assertEqual(capture_worker.main(),0)
        frames = [m for m in messages if m['kind']=='frame']
        progress = [m for m in messages if m['kind']=='progress']
        self.assertEqual([m['frame_id'] for m in frames],[0])
        self.assertEqual(len(progress),5)
        self.assertTrue(all(m['source_kind']=='dshow' and m['reason']=='decoded_frame' for m in progress))
        self.assertEqual(messages[-1]['kind'],'end')
        self.assertEqual(messages[-1]['frames'],1)
        self.assertEqual(camera_open.call_args.args[0],'video=mock-camera')

    def test_progress_from_wrong_source_is_rejected_without_counting_frames(self):
        meta = {'kind':'progress','source_kind':'file','reason':'scheduled_wait'}
        data = json.dumps(meta).encode('utf-8')
        source = LiveSource('synthetic','unused',1,5,2)
        source.process = SimpleNamespace(stdout=io.BytesIO(struct.pack('!I',len(data))+data))
        source._drain()
        with self.assertRaisesRegex(RuntimeError,'progress source differs'):
            next(source)
        self.assertEqual(source.digest.count,0)
        self.assertEqual(source.progress_messages,0)
        self.assertEqual(source.mailbox.status()['produced_samples'],0)
        self.assertEqual(source.reason,'source_error')


if __name__=='__main__':
    unittest.main()
