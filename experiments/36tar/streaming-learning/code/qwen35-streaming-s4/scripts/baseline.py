"""Small, recorded original-model baseline. Never modifies model architecture."""
import argparse
import hashlib
import inspect
import json
import platform
import sys
import tempfile
import time
import traceback
import unittest
from pathlib import Path

import torch

PROJECT = Path(__file__).resolve().parents[3]
REPO = Path(__file__).resolve().parents[1]
REVISION = "2fc06364715b967f1860aea9cf38778875588b17"


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance():
    import importlib.metadata as metadata
    return {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "seed": 42,
        "packages": {name: metadata.version(name) for name in
                     ("torch", "torchvision", "transformers", "accelerate", "Pillow", "av")},
        "repository_snapshot": json.loads((REPO / "SNAPSHOT.json").read_text(encoding="utf-8-sig"))["commit"],
        "code_hashes": {str(path.relative_to(REPO)): file_hash(path) for path in
                        sorted(REPO.rglob("*.py")) if "__pycache__" not in path.parts},
    }


def cache_measure(cache):
    result = {}
    total = 0
    for field in ("key_cache", "value_cache", "conv_states", "recurrent_states"):
        tensors = getattr(cache, field)
        sizes = [None if t is None else list(t.shape) for t in tensors]
        byte_count = sum(t.numel() * t.element_size() for t in tensors if t is not None)
        result[field] = {"shapes": sizes, "bytes": byte_count}
        total += byte_count
    result["total_tensor_bytes"] = total
    result["seq_length"] = cache.get_seq_length()
    return result


def cpu_checks(out):
    from tests.test_local_model import LocalModelTests, tiny_config
    from qwen35_local import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    torch.set_num_threads(1)
    with (out / "unittest.log").open("w", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(LocalModelTests))
    report = {"original_tests": {"run": result.testsRun, "failures": len(result.failures),
                                 "errors": len(result.errors), "passed": result.wasSuccessful()},
              "diagnostic_scope": "random tiny text model, no pretrained capability evidence"}
    torch.manual_seed(42)
    model = Qwen3_5ForConditionalGeneration(
        Qwen3_5Config(**tiny_config(), attn_implementation="eager")).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        full = model(input_ids=tokens, use_cache=False).logits
        diagnostics = []
        for scheme in ((3, 1), (2, 2), (1, 3), (1, 1, 1, 1)):
            cache = None
            offset = 0
            chunks = []
            for size in scheme:
                output = model(input_ids=tokens[:, offset:offset + size],
                               past_key_values=cache, use_cache=True,
                               cache_position=torch.arange(offset, offset + size))
                cache = output.past_key_values
                chunks.append(output.logits)
                offset += size
            actual = torch.cat(chunks, dim=1)
            diagnostics.append({
                "scheme": list(scheme), "max_abs_logit_error": (actual-full).abs().max().item(),
                "matches_full_at_1e-5_1e-4": torch.allclose(actual, full, atol=1e-5, rtol=1e-4),
                "cache": cache_measure(cache),
            })
        report["chunk_diagnostics"] = diagnostics
    write_json(out / "cpu_summary.json", report)
    print(json.dumps({"original_tests": report["original_tests"], "chunk_matches":
                      {str(x["scheme"]): x["matches_full_at_1e-5_1e-4"] for x in diagnostics}}, indent=2))
    if not result.wasSuccessful():
        raise RuntimeError("Original repository tests failed; see unittest.log")


def prepare_assets(root):
    import av
    import numpy as np
    from PIL import Image, ImageDraw
    root.mkdir(parents=True, exist_ok=False)
    def picture(color):
        image = Image.new("RGB", (256, 256), "white")
        ImageDraw.Draw(image).rectangle((48, 48, 208, 208), fill=color)
        return image
    image_path = root / "red_square.png"
    picture("red").save(image_path)
    sequences = {
        "red_then_blue": ["red", "red", "blue", "blue"],
        "green_then_blue": ["green", "green", "blue", "blue"],
        "blue_then_red": ["blue", "blue", "red", "red"],
    }
    manifest = {"source": "self-generated synthetic, mechanism/IO diagnostic only",
                "fps": 1, "image": {"path": str(image_path), "sha256": file_hash(image_path)}, "videos": []}
    for name, colors in sequences.items():
        target = root / (name + ".mp4")
        with av.open(str(target), mode="w") as container:
            stream = container.add_stream("libx264", rate=1)
            stream.width = stream.height = 256
            stream.pix_fmt = "yuv420p"
            for color in colors:
                frame = av.VideoFrame.from_ndarray(np.asarray(picture(color)), format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        with av.open(str(target)) as container:
            decoded = list(container.decode(video=0))
            if len(decoded) != 4:
                raise RuntimeError("Synthetic video frame count mismatch")
        manifest["videos"].append({"name": name, "path": str(target), "colors": colors,
                                    "frames": 4, "sha256": file_hash(target)})
    write_json(root / "manifest.json", manifest)
    return manifest


def cases_for(assets):
    return [
        ("text", None, "What is 2 plus 2? Answer with one digit.", "4"),
        ("image", assets / "red_square.png", "What color is the square? Answer with one English color word.", "red"),
        ("red_then_blue", assets / "red_then_blue.mp4", "What color was the square at the beginning of the video? Answer with one English color word.", "red"),
        ("green_then_blue", assets / "green_then_blue.mp4", "What color was the square at the beginning of the video? Answer with one English color word.", "green"),
        ("blue_then_red", assets / "blue_then_red.mp4", "What color was the square at the beginning of the video? Answer with one English color word.", "blue"),
    ]


def make_inputs(processor, path, prompt):
    import av
    import numpy as np
    from PIL import Image
    content = []
    videos = None
    if path and path.suffix == ".png":
        with Image.open(path) as image:
            content.append({"type": "image", "image": image.convert("RGB")})
    elif path:
        with av.open(str(path)) as container:
            frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        videos = np.stack(frames)
        content.append({"type": "video"})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    if videos is None:
        return processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False, return_dict=True, return_tensors="pt")
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
    return processor(text=[text], videos=[videos], do_sample_frames=False,
        video_metadata=[{"fps": 1.0, "total_num_frames": len(frames),
                         "frames_indices": list(range(len(frames)))}], return_tensors="pt")


def add_video_positions(model, inputs):
    """Input-only compatibility for upstream PR 44474; never changes vision grids."""
    grid = inputs["video_grid_thw"]
    if inputs.input_ids.shape[0] != 1 or not bool(inputs.attention_mask.all()):
        raise ValueError("Baseline adapter supports one unpadded offline sample only")
    split_grid = torch.repeat_interleave(grid, grid[:, 0], dim=0)
    split_grid[:, 0] = 1
    # Only the position builder sees split temporal groups. The vision encoder
    # still receives the original [T,H,W] grid and all original pixels.
    position_kwargs = dict(inputs)
    position_kwargs["video_grid_thw"] = split_grid
    vision_positions, delta = model.model.get_rope_index(**position_kwargs)
    text_positions = torch.arange(inputs.input_ids.shape[1], device=grid.device).view(1, 1, -1)
    inputs["position_ids"] = torch.cat((text_positions, vision_positions), dim=0)
    model.model.rope_deltas = delta
    return {"adapter": "upstream_44474_input_positions", "position_grid": split_grid.tolist(),
            "vision_grid_unchanged": grid.tolist(), "rope_delta": delta.tolist(),
            "position_shape": list(inputs["position_ids"].shape)}


def check_position_oracle(model, inputs):
    """Independent coordinate oracle for this single-video, unpadded baseline."""
    import itertools
    grid = inputs.video_grid_thw
    if grid.shape[0] != 1 or bool((inputs.mm_token_type_ids == 1).any()):
        raise ValueError("Oracle requires one video without images")
    t, h, w = grid[0].tolist()
    merge = model.config.vision_config.spatial_merge_size
    h, w = h // merge, w // merge
    coordinates = []
    offset = groups = 0
    for modality, values in itertools.groupby(inputs.mm_token_type_ids[0].tolist()):
        length = len(list(values))
        if modality == 0:
            coordinates.extend([(offset+i,) * 3 for i in range(length)])
            offset += length
        elif modality == 2:
            if length != h*w:
                raise AssertionError("Video token run does not match one temporal grid")
            coordinates.extend([(offset, offset+y, offset+x) for y in range(h) for x in range(w)])
            offset += max(h, w)
            groups += 1
        else:
            raise AssertionError("Unexpected modality")
    expected = torch.tensor(coordinates, device=grid.device).T.unsqueeze(1)
    torch.testing.assert_close(inputs.position_ids[1:], expected, rtol=0, atol=0)
    assert groups == t
    expected_delta = int(expected.max()) + 1 - inputs.input_ids.shape[1]
    assert model.model.rope_deltas.item() == expected_delta
    assert inputs.position_ids.shape == (4, 1, inputs.input_ids.shape[1])
    return {"coordinate_oracle_matches": True, "temporal_groups": groups}


@torch.inference_mode()
def check_tiny_video_adapter():
    from tests.test_local_model import tiny_config, UpstreamConfig, UpstreamModel
    from qwen35_local import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from transformers.feature_extraction_utils import BatchFeature
    torch.manual_seed(42)
    local = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**tiny_config(), attn_implementation="eager")).eval()
    upstream = UpstreamModel(UpstreamConfig(**tiny_config(), attn_implementation="eager")).eval()
    upstream.load_state_dict(local.state_dict(), strict=True)
    inputs = BatchFeature(dict(input_ids=torch.tensor([[1,122,121,123,2,122,121,123,3]]),
        attention_mask=torch.ones(1,9,dtype=torch.long),
        mm_token_type_ids=torch.tensor([[0,0,2,0,0,0,2,0,0]]),
        video_grid_thw=torch.tensor([[2,2,2]]), pixel_values_videos=torch.randn(8,24)))
    original_grid = inputs.video_grid_thw.clone()
    diagnostic = add_video_positions(local, inputs)
    diagnostic.update(check_position_oracle(local, inputs))
    upstream.model.rope_deltas = local.model.rope_deltas.clone()
    actual = local(**inputs, use_cache=False).logits
    expected = upstream(**inputs, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(inputs.video_grid_thw, original_grid, rtol=0, atol=0)
    diagnostic["local_upstream_max_logit_error"] = (actual-expected).abs().max().item()
    # Exercise multiple generated tokens to check the explicit 4-axis path.
    generated = local.generate(**inputs, max_new_tokens=3, min_new_tokens=3, do_sample=False,
                               pad_token_id=0, eos_token_id=None)
    assert generated.shape[1] == inputs.input_ids.shape[1] + 3
    diagnostic["tiny_generation_tokens"] = 3
    return diagnostic, local


def preflight(out, model_dir, assets, repair_video_positions=False):
    from transformers import AutoProcessor
    manifest = json.loads((model_dir / "DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8-sig"))
    if manifest["revision"] != REVISION:
        raise RuntimeError("Unexpected weight revision")
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    config = json.loads((model_dir / "config.json").read_text())
    adapter_test, position_model = check_tiny_video_adapter() if repair_video_positions else (None, None)
    results = []
    for case_id, path, prompt, expected in cases_for(assets):
        inputs = make_inputs(processor, path, prompt)
        adapter = None
        if repair_video_positions and "video_grid_thw" in inputs:
            adapter = add_video_positions(position_model, inputs)
            adapter.update(check_position_oracle(position_model, inputs))
        entry = {"id":case_id, "shapes":{k:list(v.shape) for k,v in inputs.items() if isinstance(v,torch.Tensor)},
                 "input_tokens":inputs.input_ids.shape[1]}
        entry["position_adapter"] = adapter
        for token_name, grid_name in (("image_token_id","image_grid_thw"),("video_token_id","video_grid_thw")):
            if grid_name in inputs:
                count = int((inputs.input_ids == config[token_name]).sum())
                grid_tokens = int(inputs[grid_name].prod(-1).sum()) // config["vision_config"]["spatial_merge_size"]**2
                if count != grid_tokens:
                    raise RuntimeError("Visual placeholder/feature count mismatch")
                entry[grid_name] = inputs[grid_name].tolist()
                entry["visual_tokens"] = count
        if entry["input_tokens"] > 1024:
            raise RuntimeError("Smoke input exceeds declared small resource budget")
        results.append(entry)
    write_json(out / "preflight_summary.json", {"processor":type(processor).__name__,
               "tiny_video_adapter_test": adapter_test, "cases":results})
    print(json.dumps(results,indent=2))


def gpu_baseline(out, model_dir, assets, repair_video_positions=False, only_video=False):
    from transformers import AutoProcessor
    from qwen35_local import Qwen3_5ForConditionalGeneration
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; do not run official model on CPU silently")
    torch.manual_seed(42)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir, local_files_only=True, dtype=dtype, attn_implementation="sdpa").to("cuda").eval()
    sources = {}
    for name, module in (("model", model), ("vision", model.model.visual),
                         ("text", model.model.language_model)):
        source = Path(inspect.getfile(type(module))).resolve()
        if source.parent != REPO / "qwen35_local":
            raise RuntimeError(f"Unexpected implementation: {source}")
        sources[name] = str(source)
    report = {"model_directory": str(model_dir), "model_revision": REVISION,
              "dtype": str(dtype), "device": torch.cuda.get_device_name(), "sources": sources,
              "parameters": sum(p.numel() for p in model.parameters()),
              "repair_video_positions": repair_video_positions,
              "capability_scope": "tiny synthetic diagnostic, not a benchmark", "cases": []}
    for case_id, path, prompt, expected in cases_for(assets):
        if only_video and (path is None or path.suffix != ".mp4"):
            continue
        # This is independent offline baseline inference, not a streaming state protocol.
        model.model.rope_deltas = None
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        inputs = make_inputs(processor, path, prompt)
        inputs = inputs.to("cuda")
        adapter = None
        if repair_video_positions and "video_grid_thw" in inputs:
            adapter = add_video_positions(model, inputs)
            adapter.update(check_position_oracle(model, inputs))
        torch.cuda.synchronize()
        preparation = time.perf_counter() - start
        t0 = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=16, do_sample=False)
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter()-t0
        answer = processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
        case = {"id": case_id, "prompt": prompt, "expected": expected, "answer": answer,
                "position_adapter": adapter,
                "contains_expected_color_or_digit": expected in answer.lower(),
                "asset_sha256": None if path is None else file_hash(path),
                "input_shapes": {k:list(v.shape) for k,v in inputs.items() if isinstance(v,torch.Tensor)},
                "grid": {k: inputs[k].tolist() for k in ("image_grid_thw","video_grid_thw") if k in inputs},
                "input_tokens": inputs.input_ids.shape[1],
                "generated_tokens": output.shape[1]-inputs.input_ids.shape[1],
                "preparation_seconds": preparation, "generation_seconds": generation_seconds,
                "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved()/2**30}
        report["cases"].append(case)
        write_json(out / "gpu_summary.json", report)  # Preserve completed cases before a later failure.
        print(json.dumps({k:case[k] for k in ("id","answer","input_tokens","peak_allocated_gib")},ensure_ascii=False),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("cpu", "assets", "preflight", "gpu"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=PROJECT / "models" / "Qwen3.5-0.8B")
    parser.add_argument("--assets", type=Path, default=PROJECT / "datasets" / "baseline_synthetic")
    parser.add_argument("--repair-video-positions", action="store_true",
                        help="Input positions match upstream PR 44474; original model is untouched")
    parser.add_argument("--only-video", action="store_true", help="Reuse prior text/image results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    task_tmp = args.out / "tmp"
    task_tmp.mkdir()
    tempfile.tempdir = str(task_tmp.resolve())
    write_json(args.out / "provenance.json", provenance())
    write_json(args.out / "arguments.json", {key:str(value) if isinstance(value,Path) else value
               for key,value in vars(args).items()})
    try:
        if args.stage == "cpu":
            cpu_checks(args.out)
        elif args.stage == "assets":
            write_json(args.out / "assets_summary.json", prepare_assets(args.assets))
        elif args.stage == "preflight":
            preflight(args.out, args.model_dir, args.assets, args.repair_video_positions)
        else:
            gpu_baseline(args.out, args.model_dir, args.assets, args.repair_video_positions, args.only_video)
        write_json(args.out / "status.json", {"status":"completed", "stage":args.stage})
    except Exception:
        (args.out / "traceback.txt").write_text(traceback.format_exc(),encoding="utf-8")
        write_json(args.out / "status.json", {"status":"failed", "stage":args.stage})
        raise


if __name__ == "__main__":
    main()
