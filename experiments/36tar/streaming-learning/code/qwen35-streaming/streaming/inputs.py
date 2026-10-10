"""Preserve Qwen3-VL's video token layout using actual delivered PTS."""
import hashlib

import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature


def tensor_fingerprint(inputs):
    digest = hashlib.sha256()
    for key in sorted(inputs):
        value = inputs[key]
        if isinstance(value, torch.Tensor):
            digest.update(key.encode())
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_window_inputs(processor, frames, prompt):
    if not frames:
        raise ValueError("Cannot question an empty window")
    pixels = np.stack([p.rgb for p in frames])
    video_inputs = processor.video_processor(videos=[pixels], do_sample_frames=False,
                                            return_metadata=False, return_tensors="pt")
    patch = processor.video_processor.temporal_patch_size
    times = [p.timestamp for p in frames]
    padded_ids = [p.frame_id for p in frames]
    pad = -len(times) % patch
    times += [times[-1]] * pad
    padded_ids += [padded_ids[-1]] * pad
    means = [sum(times[i:i+patch])/patch for i in range(0, len(times), patch)]
    grid = video_inputs["video_grid_thw"]
    if grid.shape[0] != 1 or int(grid[0,0]) != len(means):
        raise AssertionError("Temporal grid/known timestamps mismatch")
    per_group = int(grid[0,1:].prod()) // processor.video_processor.merge_size**2
    messages = [{"role":"user", "content":[{"type":"video"}, {"type":"text", "text":prompt}]}]
    template = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
    marker = processor.vision_start_token + processor.video_token + processor.vision_end_token
    if template.count(marker) != 1:
        raise AssertionError("Expected exactly one video marker")
    expanded = "".join(f"<{timestamp:.1f} seconds>" + processor.vision_start_token +
                       processor.video_token*per_group + processor.vision_end_token for timestamp in means)
    text = template.replace(marker, expanded, 1)
    tokens = processor.tokenizer([text], return_tensors="pt")
    types = torch.zeros_like(tokens["input_ids"])
    types[tokens["input_ids"] == processor.video_token_id] = 2
    tokens["mm_token_type_ids"] = types
    inputs = BatchFeature({**tokens, **video_inputs})
    if int((types == 2).sum()) != per_group*len(means):
        raise AssertionError("Visual placeholder count mismatch")
    if inputs.input_ids.shape[1] > 1024:
        raise ValueError("Window exceeds 1024-token smoke budget")
    return inputs, {"temporal_patch_means":means,
                    "rendered_timestamps":[f"{x:.1f}" for x in means],
                    "temporal_padding_frame_ids":padded_ids,
                    "visual_tokens":per_group*len(means), "video_grid_thw":grid.tolist()}
