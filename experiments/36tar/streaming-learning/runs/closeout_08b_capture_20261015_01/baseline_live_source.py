"""Process-isolated capture transport; no torch, retained clip, or device open here."""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import threading
import time

import numpy as np

from streaming.playback import LatestMailbox, SourceDigest
from streaming.window import FramePacket

MAX_HEADER = 4096
MAX_PIXELS = 256*256*3


@dataclass(frozen=True)
class LivePacket(FramePacket):
    device_pts: float | None = None
    source_kind: str = 'synthetic'
    timestamp_basis: str = 'receive_monotonic'


def read_exact(stream, size):
    chunks = bytearray()
    while len(chunks) < size:
        value = stream.read(size-len(chunks))
        if not value:
            if not chunks:
                return None
            raise ValueError('Truncated capture transport')
        chunks.extend(value)
    return bytes(chunks)


def read_message(stream):
    prefix = read_exact(stream,4)
    if prefix is None:
        return None
    size = struct.unpack('!I',prefix)[0]
    if not 1 <= size <= MAX_HEADER:
        raise ValueError('Capture header budget exceeded')
    raw = read_exact(stream,size)
    if raw is None:
        raise ValueError('Truncated capture header')
    meta = json.loads(raw.decode('utf-8'))
    if meta.get('kind') != 'frame':
        if meta.get('kind') not in ('end','error'):
            raise ValueError('Invalid capture message kind')
        return meta,None
    shape = meta.get('shape')
    if not isinstance(shape,list) or len(shape)!=3 or shape[2]!=3 or any(type(n)!=int for n in shape) or not all(1<=n<=256 for n in shape[:2]):
        raise ValueError('Invalid or oversized capture RGB shape')
    count = math.prod(shape)
    raw = read_exact(stream,count)
    if raw is None:
        raise ValueError('Truncated capture pixels')
    rgb = np.frombuffer(raw,dtype=np.uint8).reshape(shape)
    rgb.flags.writeable = False
    return meta,rgb


def validate_packet(meta,rgb,last=None):
    fid, decoded, pts = meta['frame_id'],meta['decoded_count'],meta['timestamp']
    if type(fid)!=int or type(decoded)!=int or fid<0 or decoded<=fid or not isinstance(pts,(int,float)) or not math.isfinite(pts) or pts<0:
        raise ValueError('Invalid live frame identity/time')
    if last is not None and (fid<=last.frame_id or pts<=last.timestamp or rgb.shape!=last.rgb.shape):
        raise ValueError('Live identity/time must increase and shape must stay fixed')
    device = meta.get('device_pts')
    if device is not None and (not isinstance(device,(int,float)) or not math.isfinite(device)):
        raise ValueError('Invalid device PTS metadata')
    expected_basis = 'source_pts' if meta.get('source_kind')=='file' else 'receive_monotonic'
    if meta.get('timestamp_basis')!=expected_basis or meta.get('source_kind') not in ('synthetic','dshow','file'):
        raise ValueError('Live timestamp/source basis must be explicit')
    return LivePacket(fid,float(pts),rgb,decoded,0.,device,meta['source_kind'],expected_basis)


class LiveSource:
    """Pipe drainer never touches GPU; bounded latest-frame slot feeds one owner.

    Idle detection terminates a worker blocked in native read. Explicit close
    wakes the consumer, terminates/joins the worker and joins its pipe drainer.
    """
    def __init__(self,kind,device,fps,duration,max_frames,on_frame=None,idle_seconds=3.,
                 size='640x480',capture_fps=15,stderr_path=None,fixture='normal'):
        if kind not in ('synthetic','dshow','file') or not math.isfinite(fps) or not 0<fps<=10 or not math.isfinite(duration) or not 0<duration<=120 or type(max_frames)!=int or not 1<=max_frames<=120 or not math.isfinite(idle_seconds) or not .2<=idle_seconds<=10:
            raise ValueError('Invalid bounded live source configuration')
        if fixture not in ('normal','stall','error','burst') or (kind!='synthetic' and fixture!='normal'):
            raise ValueError('Fault fixtures are synthetic only')
        self.kind,self.device,self.fps,self.duration,self.max_frames = kind,device,fps,duration,max_frames
        self.idle_seconds,self.size,self.capture_fps,self.fixture = idle_seconds,size,capture_fps,fixture
        self.on_frame,self.stderr_path = on_frame,stderr_path
        self.mailbox,self.digest = LatestMailbox(),SourceDigest()
        self.process,self.thread,self.origin,self.reason = None,None,None,None
        self.stop = threading.Event()
        self.last_activity = None
        self.stderr = None
        self.last_metadata = None
        self.maximum_packet_bytes = 0
        self.worker_torch_loaded = None

    def start(self):
        if self.process is not None:
            raise RuntimeError('Live source already started')
        self.origin = self.last_activity = time.monotonic()
        command = [sys.executable,'-m','scripts.capture_worker','--kind',self.kind,'--device',self.device,
            '--fps',str(self.fps),'--duration',str(self.duration),'--max-frames',str(self.max_frames),
            '--origin',str(self.origin),'--owner-pid',str(os.getpid()),'--size',self.size,
            '--capture-fps',str(self.capture_fps),'--fixture',self.fixture]
        env = dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
        self.stderr = open(self.stderr_path,'xb') if self.stderr_path else subprocess.DEVNULL
        try:
            self.process = subprocess.Popen(command,cwd=Path(__file__).resolve().parents[1],env=env,
                stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=self.stderr)
        except Exception:
            if self.stderr_path:
                self.stderr.close()
            raise
        self.thread = threading.Thread(target=self._drain,name='live-pipe-drainer',daemon=True)
        self.thread.start()
        return self

    def _drain(self):
        problem, last = None,None
        self.reason = 'unexpected_worker_eof'
        try:
            while not self.stop.is_set():
                item = read_message(self.process.stdout)
                if item is None:
                    if not self.stop.is_set():
                        raise RuntimeError('Capture worker ended without completion message')
                    break
                self.last_activity = time.monotonic()
                meta,rgb = item
                if meta['kind']=='error':
                    raise RuntimeError(meta.get('message','Capture worker error'))
                if meta['kind']=='end':
                    self.worker_torch_loaded = meta.get('torch_loaded')
                    self.reason = meta['reason']
                    break
                packet = validate_packet(meta,rgb,last)
                if packet.source_kind!=self.kind or packet.timestamp>=self.duration or self.digest.count>=self.max_frames:
                    raise ValueError('Capture worker exceeded registered source/time/frame budget')
                last = packet
                self.last_metadata = {k:v for k,v in meta.items() if k!='shape'}
                self.maximum_packet_bytes = max(self.maximum_packet_bytes,rgb.nbytes)
                self.digest.add(packet)
                discarded = self.mailbox.put(packet)
                if self.on_frame:
                    self.on_frame(packet,{'wall_seconds':time.monotonic()-self.origin,
                        'timestamp_basis':packet.timestamp_basis,'source_kind':self.kind,'device_pts':packet.device_pts,
                        'discarded_frame_id':discarded,**self.mailbox.status()})
        except Exception as error:
            if not self.stop.is_set():
                self.reason,problem = 'source_error',type(error).__name__+': '+str(error)
        finally:
            with self.mailbox.condition:
                # Preserve a consumer-side idle error/cancellation instead of
                # replacing it with a clean pipe EOF after terminating worker.
                if not self.mailbox.done:
                    self.mailbox.finish(problem)

    def __iter__(self):
        return self

    def __next__(self):
        with self.mailbox.condition:
            while self.mailbox.latest is None and not self.mailbox.done:
                if time.monotonic()-self.last_activity>self.idle_seconds:
                    self.reason = 'source_idle_timeout'
                    self.stop.set()
                    self.mailbox.finish('Source idle timeout; capture worker terminated')
                    if self.process.poll() is None:
                        self.process.terminate()
                    break
                self.mailbox.condition.wait(timeout=.05)
        return self.mailbox.take()

    def status(self):
        return {**self.mailbox.status(),'source_fingerprint':self.digest.snapshot(),'end_reason':self.reason,
            'producer_alive':self.thread is not None and self.thread.is_alive(),
            'worker_alive':self.process is not None and self.process.poll() is None,'queue_capacity':1,
            'max_packet_rgb_bytes':self.maximum_packet_bytes,'timestamp_basis':'source_pts' if self.kind=='file' else 'receive_monotonic',
            'sensor_to_answer_latency_measured':False,'source_kind':self.kind,'last_metadata':self.last_metadata,
            'worker_torch_loaded':self.worker_torch_loaded}

    def close(self):
        self.stop.set()
        self.mailbox.cancel()
        if self.process:
            if self.process.poll() is None:
                self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        if self.thread:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                raise RuntimeError('Capture drainer failed to exit')
        if self.process and self.process.stdout:
            self.process.stdout.close()
        if self.stderr_path and self.stderr:
            self.stderr.close()
