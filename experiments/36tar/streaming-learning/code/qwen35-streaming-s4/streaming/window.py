"""Bounded frame windows and sequential PTS sampling, without model imports."""
from collections import deque
from dataclasses import dataclass
import hashlib
import math
import time

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    frame_id: int
    timestamp: float
    rgb: np.ndarray
    decoded_count: int
    decode_seconds: float = 0.0


class CausalSampler:
    def __init__(self, fps=1.0):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Sampling FPS must be finite and positive")
        self.fps = fps
        self.next_due = 0.0
        self.last_time = None

    def accept(self, timestamp):
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Invalid timestamp")
        if self.last_time is not None and timestamp <= self.last_time:
            raise ValueError("Delivered PTS must strictly increase")
        self.last_time = timestamp
        if timestamp + 1e-9 < self.next_due:
            return False
        self.next_due = (math.floor(timestamp*self.fps + 1e-9)+1)/self.fps
        return True


class FrameWindow:
    def __init__(self, max_frames=4):
        if not isinstance(max_frames, int) or max_frames < 1:
            raise ValueError("Window capacity must be a positive integer")
        self.max_frames = max_frames
        self.reset()

    def reset(self):
        self.frames = deque(maxlen=self.max_frames)
        self.accepted = 0
        self.last_id = self.last_time = None

    def append(self, packet):
        if packet.frame_id < 0 or packet.decoded_count <= packet.frame_id:
            raise ValueError("Invalid delivered frame count")
        if not math.isfinite(packet.timestamp) or packet.timestamp < 0:
            raise ValueError("Invalid frame timestamp")
        if self.last_id is not None and (packet.frame_id <= self.last_id or packet.timestamp <= self.last_time):
            raise ValueError("Frame IDs and timestamps must strictly increase")
        rgb = packet.rgb
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("RGB frame must be HxWx3 uint8")
        if min(rgb.shape[:2]) < 1 or max(rgb.shape[:2]) > 256:
            raise ValueError("Window pixels must have maximum side <=256")
        copied = rgb.copy()
        copied.flags.writeable = False
        frozen = FramePacket(packet.frame_id, packet.timestamp, copied, packet.decoded_count, packet.decode_seconds)
        self.frames.append(frozen)
        self.accepted += 1
        self.last_id, self.last_time = packet.frame_id, packet.timestamp

    def snapshot(self):
        return tuple(self.frames)

    def evidence(self):
        return {"accepted_samples": self.accepted,
                "frame_ids": [p.frame_id for p in self.frames],
                "timestamps": [p.timestamp for p in self.frames],
                "frame_sha256": [hashlib.sha256(p.rgb.tobytes()).hexdigest() for p in self.frames],
                "window_pixel_bytes": sum(p.rgb.nbytes for p in self.frames),
                "delivered_decode_count": self.frames[-1].decoded_count if self.frames else 0}


def read_packets(path, fps=1.0):
    """Yield selected delivered frames. No list decode, seek, duration or total frames."""
    import av
    from PIL import Image
    sampler = CausalSampler(fps)
    with av.open(str(path)) as container:
        decoder = iter(container.decode(video=0))
        origin = None
        index = 0
        pending_decode_seconds = 0.0
        while True:
            start = time.perf_counter()
            try:
                frame = next(decoder)
            except StopIteration:
                return
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Frame PTS is required; no guessed timestamps")
            actual_time = float(frame.pts * frame.time_base)
            if origin is None:
                origin = actual_time
            timestamp = actual_time-origin
            selected = sampler.accept(timestamp)
            if selected:
                image = Image.fromarray(frame.to_ndarray(format="rgb24"))
                image.thumbnail((256, 256))
                rgb = np.asarray(image)
            pending_decode_seconds += time.perf_counter()-start
            if selected:
                yield FramePacket(index, timestamp, rgb, index+1, pending_decode_seconds)
                pending_decode_seconds = 0.0
            index += 1


def iter_windows(packets, capacity=4, every=2):
    if not isinstance(every, int) or every < 1:
        raise ValueError("Question cadence must be a positive integer")
    window = FrameWindow(capacity)
    for packet in packets:
        window.append(packet)
        if window.accepted % every == 0:
            yield window.snapshot(), window.evidence()
