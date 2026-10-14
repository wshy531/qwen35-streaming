"""Explicit bounded repeat stress input; session time is not natural source time."""
from dataclasses import dataclass
import math
import time

import numpy as np

from streaming.window import CausalSampler, FramePacket


@dataclass(frozen=True)
class RepeatPacket(FramePacket):
    loop_index: int = 0
    source_frame_id: int = 0
    source_pts: float = 0.0


def provenance(packet):
    if not isinstance(packet, RepeatPacket):
        return None
    return {"synthetic_repeat": True, "loop_index": packet.loop_index,
            "source_frame_id": packet.source_frame_id, "source_pts": packet.source_pts,
            "session_pts": packet.timestamp}


def repeat_packets(path, fps, period, count, opener=None):
    """Reopen one file per loop, never retain a clip or fill the inter-loop gap.

    Session PTS = loop * declared period + real normalized source PTS.
    Global frame IDs count all actual decoded frames, including unsampled tails.
    Reject a source reaching the period rather than silently overlap its timeline.
    """
    import av
    from PIL import Image
    if not math.isfinite(period) or period <= 0 or not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 30:
        raise ValueError("Positive finite repeat period and count 1..30 required")
    CausalSampler(fps)  # Validate before opening anything.
    opener = av.open if opener is None else opener
    decoded = 0
    for loop in range(count):
        sampler = CausalSampler(fps)
        origin, selected = None, 0
        with opener(str(path)) as container:
            decoder = iter(container.decode(video=0))
            source_id, seconds = 0, 0.0
            while True:
                begin = time.perf_counter()
                try:
                    frame = next(decoder)
                except StopIteration:
                    break
                if frame.pts is None or frame.time_base is None:
                    raise ValueError("Actual source PTS required")
                actual = float(frame.pts * frame.time_base)
                if origin is None:
                    origin = actual
                source_pts = actual - origin
                if not math.isfinite(source_pts) or source_pts >= period:
                    raise ValueError("Source reaches declared repeat period; overlapping loops refused")
                accepted = sampler.accept(source_pts)
                global_id = decoded
                decoded += 1
                if accepted:
                    image = Image.fromarray(frame.to_ndarray(format="rgb24"))
                    image.thumbnail((256, 256))
                    rgb = np.asarray(image)
                    rgb.flags.writeable = False
                seconds += time.perf_counter() - begin
                if accepted:
                    selected += 1
                    yield RepeatPacket(global_id, loop * period + source_pts, rgb, decoded,
                                       seconds, loop, source_id, source_pts)
                    seconds = 0.0
                source_id += 1
        if not selected:
            raise ValueError("Empty repeat source")
