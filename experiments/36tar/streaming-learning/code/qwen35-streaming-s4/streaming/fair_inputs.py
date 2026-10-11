"""Independent whole-video inputs and exact epoch selection for bounded controls."""
import hashlib
import json

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature

from streaming.inputs import tensor_fingerprint
from streaming.online import PairInputBuilder
from streaming.preprocessing import fit_canvas
from streaming.window import FramePacket, read_packets


def rgb_sha(rgb):
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def historical_context(path, answer_index):
    """Reconstruct encoded pairs in the answer's epoch, excluding discarded history."""
    current, answers = [], []
    for line in path.read_text(encoding='utf-8').splitlines():
        event = json.loads(line)
        if event['kind'] == 'capacity_reset':
            current = []
        elif event['kind'] == 'encoded':
            current.append(event['record'])
        elif event['kind'] == 'answer':
            frames = []
            for record in current:
                pairs = [[frame_id, timestamp] for frame_id, timestamp in zip(record['frame_ids'], record['timestamps'])]
                frames.extend(pairs[:1] if record['synthetic_padding'] else pairs)
            if not frames or frames[-1][0] != event['answer']['frame_id']:
                raise ValueError('Historical answer not backed by current encoded epoch')
            answers.append({'frames': frames, 'question': event['answer']['question'],
                            'historical_answer': event['answer']['answer'],
                            'generated_ids': event['answer']['generated_ids'],
                            'memory_from_pts': event['status']['memory_from_pts']})
    if not answers:
        raise ValueError('History has no encoded answers')
    return answers[answer_index]


def selected_packets(video, indices=None, exact=None):
    if (indices is None) == (exact is None):
        raise ValueError('Specify one selection rule')
    chosen = list(indices if indices is not None else [int(i) for i, _ in exact])
    if not 1 <= len(chosen) <= 4 or chosen != sorted(set(chosen)) or chosen[0] < 0:
        raise ValueError('One to four increasing selections required')
    result = []
    reader = read_packets(video, 1)
    try:
        for sample_index, packet in enumerate(reader):
            key = sample_index if indices is not None else packet.frame_id
            if key in chosen:
                if exact is not None:
                    position = chosen.index(key)
                    if abs(packet.timestamp - exact[position][1]) > 1e-9:
                        raise ValueError('Decoded PTS differs from historical selection')
                result.append(packet)
            if key >= chosen[-1]:
                break
    finally:
        reader.close()
    if len(result) != len(chosen):
        raise ValueError('Video lacks requested samples')
    return result


def load_case(spec, video_dir, histories):
    provenance = []
    historical = None
    if 'selection' in spec:
        key = spec['selection']
        historical = historical_context(histories[key], 0 if key == 'historical_coffee' else -1)
        if historical['question'] != spec['question']:
            raise ValueError('Historical question changed')
        expected_ids = [0, 30] if key == 'historical_coffee' else [210, 240, 300, 330]
        if [i for i, _ in historical['frames']] != expected_ids:
            raise ValueError('Fixed failure context differs from recorded experiment')
        groups = [('clean_hands', selected_packets(video_dir / 'clean_hands.webm', exact=historical['frames']))]
    elif 'splice' in spec:
        groups = [(name, selected_packets(video_dir / (name + '.webm'), indices=[0, 1])) for name in spec['splice']]
    else:
        groups = [(spec['video'], selected_packets(video_dir / (spec['video'] + '.webm'), indices=spec['indices']))]
    result = []
    for name, packets in groups:
        for p in packets:
            index = len(result)
            frame_id, pts, decoded = (index, float(index), index + 1) if 'splice' in spec else (p.frame_id, p.timestamp, p.decoded_count)
            canvas = fit_canvas(p.rgb, (192, 256))
            canvas.flags.writeable = False
            result.append(FramePacket(frame_id, pts, canvas, decoded))
            provenance.append({'video': name, 'source_frame_id': p.frame_id, 'source_pts': p.timestamp,
                'source_rgb_sha256': rgb_sha(p.rgb), 'frame_id': frame_id, 'timestamp': pts,
                'canonical_rgb_sha256': rgb_sha(canvas), 'canonical_shape': list(canvas.shape)})
    return tuple(result), {'id': spec['id'], 'frames': provenance, 'controlled_splice': 'splice' in spec,
                           'historical_context': historical, 'retained_rgb_bytes': sum(p.rgb.nbytes for p in result)}


def whole_inputs(model, processor, frames, question):
    """Full processor + original RoPE; never calls the pair builder."""
    if not frames or len(frames) > 4:
        raise ValueError('Bounded nonempty prefix required')
    padded = tuple(frames) if len(frames) % 2 == 0 else tuple(frames) + (frames[-1],)
    visual = processor.video_processor(videos=[np.stack([fit_canvas(p.rgb, (192, 256)) for p in padded])],
        do_resize=False, do_sample_frames=False, return_metadata=False, return_tensors='pt')
    grid = visual.video_grid_thw
    per_pair = int(grid[0, 1:].prod()) // processor.video_processor.merge_size**2
    expansion = ''.join(f'<{(padded[i].timestamp + padded[i+1].timestamp)/2:.1f} seconds>' +
        processor.vision_start_token + processor.video_token * per_pair + processor.vision_end_token
        for i in range(0, len(padded), 2))
    template = processor.apply_chat_template([{'role': 'user', 'content': [
        {'type': 'video'}, {'type': 'text', 'text': question}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    marker = processor.vision_start_token + processor.video_token + processor.vision_end_token
    if template.count(marker) != 1:
        raise ValueError('Expected one video marker')
    ids = processor.tokenizer(template.replace(marker, expansion), add_special_tokens=False, return_tensors='pt')
    values = BatchFeature({**ids, **visual, 'mm_token_type_ids': (ids.input_ids == processor.video_token_id).long() * 2})
    rope_grids = torch.repeat_interleave(grid, grid[:, 0], 0)
    rope_grids[:, 0] = 1
    coordinates, _ = model.model.get_rope_index(**{**values, 'video_grid_thw': rope_grids})
    values['position_ids'] = torch.cat((torch.arange(ids.input_ids.shape[1])[None, None, :], coordinates), 0)
    return values


def joined_pair_inputs(processor, frames, question):
    """Rebuild what streaming appends, including suffix, without model forward."""
    builder = PairInputBuilder(processor, resize_mode='letterbox')
    ids = [builder.prefix_ids]
    count = builder.prefix_ids.shape[1]
    rope = count
    positions = [torch.arange(count).repeat(4, 1)[:, None, :]]
    pixels, grid = [], None
    for i in range(0, len(frames), 2):
        last = min(i + 1, len(frames) - 1)
        block, _ = builder.pair(frames[i], frames[last], count, rope, i == last)
        ids.append(block.input_ids)
        positions.append(block.position_ids)
        pixels.append(block.pixel_values_videos)
        if grid is None:
            grid = block.video_grid_thw.clone()
            grid[0, 0] = 0
        grid[0, 0] += block.video_grid_thw[0, 0]
        count += block.input_ids.shape[1]
        rope = int(block.position_ids[1:].max()) + 1
    suffix = builder.question_ids(question)
    size = suffix.shape[1]
    ids.append(suffix)
    positions.append(torch.stack((torch.arange(count, count+size), *[torch.arange(rope, rope+size) for _ in range(3)]))[:, None, :])
    tokens = torch.cat(ids, 1)
    return BatchFeature({'input_ids': tokens, 'attention_mask': torch.ones_like(tokens),
        'mm_token_type_ids': (tokens == processor.video_token_id).long() * 2,
        'pixel_values_videos': torch.cat(pixels, 0), 'video_grid_thw': grid,
        'position_ids': torch.cat(positions, 2)})


def identity(model, processor, frames, question):
    whole = whole_inputs(model, processor, frames, question)
    pair = joined_pair_inputs(processor, frames, question)
    equalities = {key: key in pair and torch.equal(value, pair[key]) for key, value in whole.items()}
    if set(whole) != set(pair) or not all(equalities.values()):
        raise AssertionError(f'Independent full/pair input mismatch: {equalities}')
    if whole.input_ids.shape[1] + 16 > 256:
        raise ValueError('Full prompt/answer exceeds registered total token budget')
    return whole, {'all_input_tensors_bitwise_equal': True, 'tensor_equalities': equalities,
                   'input_fingerprint': tensor_fingerprint(whole),
                   'input_tokens': whole.input_ids.shape[1], 'grid': whole.video_grid_thw.tolist()}
