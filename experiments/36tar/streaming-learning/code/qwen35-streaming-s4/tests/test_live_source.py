"""Bounded wire protocol, real child cancellation/stall/error, and parent death."""
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np
import psutil

from streaming.live_source import LiveSource,read_message,validate_packet


def meta(**changes):
    value = {'kind':'frame','shape':[32,32,3],'frame_id':0,'decoded_count':1,
             'timestamp':.1,'device_pts':None,'source_kind':'synthetic','timestamp_basis':'receive_monotonic'}
    return {**value,**changes}


def wire(value,pixels=b''):
    raw = json.dumps(value).encode('utf-8')
    return io.BytesIO(struct.pack('!I',len(raw))+raw+pixels)


class LiveSourceTests(unittest.TestCase):
    def source(self,fixture='normal',**kwargs):
        return LiveSource('synthetic','fixture',10,10,3,fixture=fixture,**kwargs)

    def test_packet_wire_roundtrip_and_readonly_rgb(self):
        rgb = np.zeros((32,32,3),np.uint8)
        value,received = read_message(wire(meta(),rgb.tobytes()))
        p = validate_packet(value,received)
        np.testing.assert_array_equal(received,rgb)
        self.assertFalse(received.flags.writeable)
        self.assertEqual(p.timestamp_basis,'receive_monotonic')
        self.assertIsNone(p.device_pts)

    def test_oversized_header_is_refused_before_allocation(self):
        stream = io.BytesIO(struct.pack('!I',4097))
        with self.assertRaisesRegex(ValueError,'header budget'):
            read_message(stream)
        self.assertEqual(stream.tell(),4)

    def test_oversized_pixel_shape_is_refused_before_payload(self):
        with self.assertRaisesRegex(ValueError,'shape'):
            read_message(wire(meta(shape=[10000,10000,3])))

    def test_truncated_header_and_pixels_rejected(self):
        for stream in (io.BytesIO(b'\x00\x00'),io.BytesIO(struct.pack('!I',5)+b'{}'),wire(meta(),b'1')):
            with self.assertRaises(ValueError):
                read_message(stream)

    def test_unknown_control_kind_refused(self):
        with self.assertRaises(ValueError):
            read_message(wire({'kind':'unrecognized'}))

    def test_ids_time_and_shape_strict_across_packets(self):
        rgb = np.zeros((32,32,3),np.uint8)
        first = validate_packet(meta(),rgb)
        for changes in ({'frame_id':0,'decoded_count':2,'timestamp':.2},
                        {'frame_id':1,'decoded_count':2,'timestamp':.1},
                        {'timestamp':float('nan')},{'decoded_count':0},{'timestamp':-.1}):
            with self.assertRaises(ValueError):
                validate_packet(meta(**changes),rgb,first)
        with self.assertRaises(ValueError):
            validate_packet(meta(frame_id=1,decoded_count=2,timestamp=.2),rgb[:16],first)

    def test_timestamp_basis_must_be_explicit_and_device_pts_metadata_valid(self):
        rgb = np.zeros((32,32,3),np.uint8)
        for change in ({'timestamp_basis':'guessed_exposure'},{'source_kind':'unknown'},{'device_pts':float('nan')}):
            with self.assertRaises(ValueError):
                validate_packet(meta(**change),rgb)
        self.assertEqual(validate_packet(meta(device_pts=12345.),rgb).device_pts,12345.)

    def test_invalid_budget_and_camera_fault_fixture_rejected_before_spawn(self):
        for change in ({'kind':'bad'},{'fps':0},{'duration':float('inf')},{'max_frames':121},{'idle_seconds':0}):
            args = dict(kind='synthetic',device='test',fps=1,duration=10,max_frames=3)
            args.update(change)
            with self.assertRaises(ValueError):
                LiveSource(**args)
        with self.assertRaises(ValueError):
            LiveSource('dshow','test',1,10,3,fixture='stall')

    def test_file_requires_source_pts_and_preserves_pts(self):
        rgb = np.zeros((32,32,3),np.uint8)
        with self.assertRaises(ValueError):
            validate_packet(meta(source_kind='file'),rgb)
        packet = validate_packet(meta(source_kind='file',timestamp_basis='source_pts',timestamp=1.25,device_pts=1.25),rgb)
        self.assertEqual((packet.timestamp,packet.device_pts,packet.timestamp_basis),(1.25,1.25,'source_pts'))

    def test_real_file_worker_preserves_complete_source_and_bounded_latest_slot(self):
        import av
        from streaming.window import read_packets
        from streaming.preprocessing import fit_canvas
        from streaming.playback import SourceDigest
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fixture.mkv'
            with av.open(str(path),'w') as container:
                video = container.add_stream('ffv1',rate=10)
                video.width,video.height,video.pix_fmt = 48,32,'bgr0'
                for i in range(3):
                    for encoded in video.encode(av.VideoFrame.from_ndarray(np.full((32,48,3),i*32,np.uint8),format='rgb24')):
                        container.mux(encoded)
                for encoded in video.encode():
                    container.mux(encoded)
            expected = SourceDigest()
            for p in read_packets(path,10):
                from streaming.window import FramePacket
                expected.add(FramePacket(p.frame_id,p.timestamp,fit_canvas(p.rgb,(192,256)),p.decoded_count))
            source = LiveSource('file',str(path),10,2,3)
            try:
                source.start()
                source.process.wait(timeout=10)
                source.thread.join(timeout=2)
                self.assertEqual(source.digest.snapshot(),expected.snapshot())
                status = source.status()
                self.assertEqual((status['produced_samples'],status['dropped_samples'],status['buffered_samples']),(3,2,1))
                self.assertEqual(next(source).timestamp,.2)
                self.assertEqual(status['timestamp_basis'],'source_pts')
                self.assertFalse(source.worker_torch_loaded)
                self.assertLessEqual(status['peak_buffered_rgb_bytes'],192*256*3)
            finally:
                source.close()
            self.assertFalse(source.status()['worker_alive'])
            self.assertFalse(source.status()['producer_alive'])

    def test_missing_file_worker_reports_error_and_closes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = LiveSource('file',str(Path(directory)/'missing.mp4'),1,2,3)
            try:
                source.start()
                with self.assertRaises(RuntimeError):
                    next(source)
                self.assertEqual(source.reason,'source_error')
            finally:
                source.close()
            self.assertFalse(source.status()['worker_alive'])

    def test_real_synthetic_worker_completes_and_closes_without_torch(self):
        s = self.source()
        try:
            rows = list(s.start())
            self.assertEqual(len(rows),3)
            self.assertTrue(all(a.timestamp<b.timestamp for a,b in zip(rows,rows[1:])))
            self.assertEqual(s.reason,'frame_budget_limit')
            self.assertFalse(s.worker_torch_loaded)
        finally:
            s.close()
        self.assertFalse(s.status()['worker_alive'])
        self.assertFalse(s.status()['producer_alive'])

    def test_real_burst_overload_keeps_only_latest_and_conserves_counts(self):
        s = LiveSource('synthetic','test',10,10,80,fixture='burst')
        try:
            s.start()
            s.process.wait(timeout=10)
            s.thread.join(timeout=2)
            status = s.status()
            self.assertEqual((status['produced_samples'],status['dropped_samples'],status['buffered_samples']),(80,79,1))
            self.assertEqual(next(s).frame_id,79)
            with self.assertRaises(StopIteration):
                next(s)
            self.assertEqual(s.status()['peak_buffered_rgb_bytes'],64*96*3)
        finally:
            s.close()

    def test_real_stalled_native_style_read_times_out_and_worker_stops(self):
        s = self.source('stall',idle_seconds=1.)
        start = time.monotonic()
        try:
            s.start()
            with self.assertRaisesRegex(RuntimeError,'idle timeout'):
                next(s)
            self.assertEqual(s.reason,'source_idle_timeout')
        finally:
            s.close()
        self.assertLess(time.monotonic()-start,4.)
        self.assertFalse(s.status()['worker_alive'])

    def test_real_disconnect_error_is_not_silent_eof(self):
        s = self.source('error')
        try:
            s.start()
            with self.assertRaisesRegex(RuntimeError,'synthetic disconnect'):
                next(s)
            self.assertEqual(s.reason,'source_error')
        finally:
            s.close()

    def test_early_close_releases_pipe_and_cancels_slot(self):
        s = self.source('stall')
        s.start()
        s.close()
        s.close()
        self.assertFalse(s.status()['worker_alive'])
        self.assertEqual(s.status()['buffered_rgb_bytes'],0)

    @unittest.skipUnless(os.name=='nt','Windows owner-handle guard')
    def test_capture_worker_exits_when_parent_dies(self):
        cwd = Path(__file__).resolve().parents[1]
        code = "from streaming.live_source import LiveSource; import os,time; s=LiveSource('synthetic','test',1,30,30,fixture='stall').start(); print(s.process.pid,flush=True); time.sleep(1); os._exit(0)"
        parent = subprocess.Popen([sys.executable,'-c',code],cwd=cwd,stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),text=True)
        worker = None
        try:
            worker = int(parent.stdout.readline().strip())
            parent.wait(timeout=8)
            deadline = time.monotonic()+3
            while psutil.pid_exists(worker) and time.monotonic()<deadline:
                time.sleep(.05)
            self.assertFalse(psutil.pid_exists(worker))
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=2)
            parent.stdout.close()
            if worker and psutil.pid_exists(worker):
                psutil.Process(worker).terminate()


if __name__=='__main__':
    unittest.main()
