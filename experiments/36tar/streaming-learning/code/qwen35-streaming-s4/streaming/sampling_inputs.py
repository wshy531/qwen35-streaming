"""Four real PTS frames selected at declared targets, without future lookahead."""
import math

from streaming.fair_inputs import rgb_sha
from streaming.preprocessing import fit_canvas
from streaming.window import FramePacket


def targets(start, phase):
    if isinstance(start, bool) or not isinstance(start, (int, float)) or not math.isfinite(start) or start < 0:
        raise ValueError('Finite nonnegative start required')
    if isinstance(phase, bool) or phase not in (0, 0.5):
        raise ValueError('Registered phase 0 or 0.5 required')
    return [float(start + phase + i) for i in range(4)]


def select_due(records, due):
    """Yield only each target's first delivered record; stop at the fourth selection."""
    if len(due) != 4 or any(not math.isfinite(t) or t < 0 for t in due) or any(a >= b for a, b in zip(due, due[1:])):
        raise ValueError('Four increasing nonnegative finite targets required')
    slot, last_time = 0, None
    for frame_id, timestamp, payload in records:
        if not math.isfinite(timestamp) or timestamp < 0 or (last_time is not None and timestamp <= last_time):
            raise ValueError('Delivered PTS must strictly increase')
        last_time = timestamp
        if timestamp + 1e-9 < due[slot]:
            continue
        if timestamp >= due[slot] + 1 - 1e-9:
            raise ValueError('Decoder skipped an entire target slot')
        yield frame_id, timestamp, payload, due[slot]
        slot += 1
        if slot == 4:
            return
    raise ValueError('Source lacks four requested real frames')


def load_case(spec, video_dir):
    import av
    import numpy as np
    from PIL import Image
    if spec['video'] not in ('clean_hands', 'cat_play'):
        raise ValueError('Registered source required')
    due = targets(spec['start'], spec['phase'])
    frames, metadata = [], []
    with av.open(str(video_dir / (spec['video'] + '.webm'))) as container:
        def records():
            origin = None
            for index, frame in enumerate(container.decode(video=0)):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError('Actual PTS required')
                actual = float(frame.pts * frame.time_base)
                if origin is None:
                    origin = actual
                yield index, actual - origin, frame
        reader = records()
        selections = select_due(reader, due)
        try:
            for index, timestamp, frame, target in selections:
                image = Image.fromarray(frame.to_ndarray(format='rgb24'))
                image.thumbnail((256, 256))
                canvas = fit_canvas(np.asarray(image), (192, 256))
                canvas.flags.writeable = False
                frames.append(FramePacket(index, timestamp, canvas, index + 1))
                metadata.append({'frame_id': index, 'source_pts': timestamp, 'target_pts': target,
                    'target_error_seconds': timestamp - target, 'canonical_rgb_sha256': rgb_sha(canvas)})
        finally:
            selections.close()
            reader.close()
    return tuple(frames), {'id': spec['id'], 'video': spec['video'], 'start': spec['start'], 'phase': spec['phase'],
        'question': spec['question'], 'frames': metadata, 'synthetic_time_labels': False,
        'rendered_pair_times': [f'{(frames[i].timestamp + frames[i+1].timestamp)/2:.1f}' for i in (0, 2)],
        'retained_rgb_bytes': sum(p.rgb.nbytes for p in frames)}
