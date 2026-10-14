"""Keep image source time distinct from diagnostic timestamp labels."""
import math

from streaming.fair_inputs import rgb_sha, selected_packets
from streaming.preprocessing import fit_canvas
from streaming.window import FramePacket


def validate_case(spec):
    indices, labels = spec['indices'], spec['labels']
    if len(indices) != 4 or len(labels) != 4:
        raise ValueError('Exactly four input slots required')
    if any(not isinstance(i, int) or isinstance(i, bool) or not 0 <= i <= 11 for i in indices):
        raise ValueError('Bounded source sample indices required')
    if any(not isinstance(t, (float, int)) or isinstance(t, bool) or not math.isfinite(t) or t < 0 for t in labels):
        raise ValueError('Finite nonnegative diagnostic labels required')
    if any(a >= b for a, b in zip(labels, labels[1:])):
        raise ValueError('Diagnostic labels must strictly increase')


def relationships(specs):
    by_id = {spec['id']: spec for spec in specs}
    if len(by_id) != len(specs):
        raise ValueError('Case IDs must be unique')
    result = []
    for spec in specs:
        validate_case(spec)
        if spec['factor'] == 'actual_anchor':
            if 'parent' in spec:
                raise ValueError('Anchor has no intervention parent')
            continue
        parent = by_id[spec['parent']]
        pixels = [i for i in range(4) if spec['indices'][i] != parent['indices'][i]]
        times = [i for i in range(4) if spec['labels'][i] != parent['labels'][i]]
        factor = spec['factor']
        if factor == 'one_pixel_slot':
            valid = pixels == [spec['slot']] and not times
        elif factor == 'one_time_slot':
            valid = times == [spec['slot']] and not pixels
        elif factor == 'pixel_combination':
            valid = bool(pixels) and not times
        elif factor in ('time_combination', 'time_origin_and_gaps'):
            valid = bool(times) and not pixels
        elif factor == 'time_shift':
            valid = not pixels and all(abs(a - b - spec['shift']) < 1e-9 for a, b in zip(spec['labels'], parent['labels']))
        else:
            valid = False
        if not valid:
            raise ValueError('Intervention changed undeclared factors: ' + spec['id'])
        result.append({'id': spec['id'], 'parent': parent['id'], 'factor': factor,
                       'changed_image_slots': pixels, 'changed_time_slots': times})
    return result


def assign_slots(source_packets, labels):
    if len(source_packets) != 4 or len(labels) != 4 or any(a >= b for a, b in zip(labels, labels[1:])):
        raise ValueError('Four slots and increasing labels required')
    result = []
    for i, (packet, label) in enumerate(zip(source_packets, labels)):
        canvas = fit_canvas(packet.rgb, (192, 256))
        canvas.flags.writeable = False
        result.append(FramePacket(i, float(label), canvas, i + 1))
    return tuple(result)


def load_case(spec, video):
    validate_case(spec)
    unique = sorted(set(spec['indices']))
    source = selected_packets(video, indices=unique)
    selected = dict(zip(unique, source))
    ordered = [selected[index] for index in spec['indices']]
    frames = assign_slots(ordered, spec['labels'])
    metadata = []
    for index, real, assigned in zip(spec['indices'], ordered, frames):
        metadata.append({'sample_index': index, 'source_frame_id': real.frame_id, 'source_pts': real.timestamp,
            'source_rgb_sha256': rgb_sha(real.rgb), 'diagnostic_id': assigned.frame_id,
            'diagnostic_label': assigned.timestamp, 'canonical_rgb_sha256': rgb_sha(assigned.rgb)})
    synthetic = any(abs(real.timestamp - assigned.timestamp) > 1e-9 for real, assigned in zip(ordered, frames))
    if spec['factor'] == 'actual_anchor' and synthetic:
        raise ValueError('Actual anchor labels differ from source PTS')
    return frames, {'id': spec['id'], 'frames': metadata, 'factor': spec['factor'],
        'synthetic_time_labels': synthetic, 'duplicate_source_images': len(unique) < 4,
        'rendered_pair_times': [f'{(spec["labels"][i] + spec["labels"][i+1])/2:.1f}' for i in (0, 2)],
        'retained_rgb_bytes': sum(p.rgb.nbytes for p in frames)}
