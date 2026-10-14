"""Matched raw source frames at two detail budgets; frozen original model only."""
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

HERE = Path(__file__).resolve().parent
CAP = HERE.parents[1]
spec = importlib.util.spec_from_file_location('detail_readout_reference', HERE.parent / 'protocol_readout_v1/probe.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
import av
import numpy as np
import psutil
from PIL import Image
import torch
from transformers import AutoProcessor
from streaming.preprocessing import fit_canvas
from streaming.online import PairInputBuilder, fp32_reference
from scripts.baseline import add_video_positions, check_position_oracle, file_hash, write_json
from scripts.pair_demo import verify_assets
from streaming.inputs import tensor_fingerprint

PROFILES = {'old_two_stage': (192, 256), 'direct_small': (192, 256), 'direct_large': (384, 512)}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sources():
    return {**reference.sources(), **{str(p.resolve()): file_hash(p) for p in HERE.iterdir()
                                    if p.suffix in ('.py', '.md', '.json')}}


def budget():
    return {'model_loads': 1, 'vision_max': 24, 'questions_max': 24, 'generated_max': 768,
            'language_max': 792, 'answer_tokens': 32, 'max_tokens': 512, 'wall_seconds': 360,
            'ram_min_gib': 3, 'gpu_free_min_gib': 5.5, 'conversion_free_min_gib': 2,
            'single_case_raw_rgb_max_bytes': 32 * 2**20, 'small_input_pool_max_bytes': 8 * 2**20,
            'concurrent_models': 1, 'precision': 'fp32_ieee_math'}


def request(args):
    return {'model_dir': str(args.model_dir), 'branch': args.branch,
            'reference_cpu': str(args.reference_cpu), 'reference_gpu': str(args.reference_gpu),
            'profiles': {name: list(canvas) for name, canvas in PROFILES.items()},
            'cases': [list(c) for c in reference.CASES], 'question_policy': 'unchanged_from_readout'}


def identity(args):
    return {'reference_cpu_sha256': file_hash(args.reference_cpu / 'proof.json'),
            'reference_gpu_sha256': file_hash(args.reference_gpu / 'summary.json'),
            'model_assets': verify_assets(args.model_dir)['files'],
            'model_manifest_sha256': file_hash(args.model_dir / 'DOWNLOAD_MANIFEST.json')}


def raw_pair(row, assets):
    source_id = row['identity'][0]['source_id']
    configs = reference.events.read(reference.events.HERE / 'cases.json')
    animal = reference.events.read(CAP / 'diagnostics/general_stream_v1/cases.json')['videos']
    registered = next(v for v in [*animal, *configs['natural_sources']] if v['id'] == source_id)
    path = Path(registered['path'])
    if assets[str(path)] != registered['sha256']:
        raise ValueError('Registered source identity mismatch')
    target = {r['source_frame_id']: r for r in row['identity']}
    arrays, metadata, origin = {}, {}, None
    began = time.perf_counter()
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if frame.pts is None or frame.time_base is None:
                raise ValueError('Real source PTS required')
            pts = float(frame.pts * frame.time_base)
            if origin is None:
                origin = pts
            if index in target:
                expected = target[index]
                if abs(pts - origin - expected['source_pts']) > 1e-9:
                    raise ValueError('Source frame/time identity changed')
                rgb = frame.to_ndarray(format='rgb24')
                if rgb.nbytes + sum(v.nbytes for v in arrays.values()) > 32 * 2**20:
                    raise RuntimeError('Raw pair RGB bound exceeded')
                thumbnail = Image.fromarray(rgb)
                thumbnail.thumbnail((256, 256))
                small = np.asarray(thumbnail)
                if hashlib.sha256(small.tobytes()).hexdigest() != expected['source_rgb_sha256']:
                    raise ValueError('Raw re-decode does not reproduce old thumbnail pixels')
                if hashlib.sha256(fit_canvas(small, (192, 256)).tobytes()).hexdigest() != expected['canvas_rgb_sha256']:
                    raise ValueError('Old canvas identity mismatch')
                arrays[index] = rgb
                metadata[index] = {'source_frame_id': index, 'source_pts': pts - origin,
                                   'raw_shape': list(rgb.shape), 'raw_rgb_bytes': rgb.nbytes,
                                   'raw_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                                   'old_thumbnail_sha256': expected['source_rgb_sha256'],
                                   'old_canvas_sha256': expected['canvas_rgb_sha256']}
            if len(arrays) == 2:
                break
    if len(arrays) != 2:
        raise ValueError('Requested raw pair absent')
    ordered = [r['source_frame_id'] for r in row['identity']]
    return [arrays[i] for i in ordered], [metadata[i] for i in ordered], time.perf_counter() - began


def pixels(raw, profile):
    canvas = PROFILES[profile]
    if profile == 'old_two_stage':
        images = []
        for rgb in raw:
            thumbnail = Image.fromarray(rgb)
            thumbnail.thumbnail((256, 256))
            images.append(fit_canvas(np.asarray(thumbnail), canvas))
    else:
        images = [fit_canvas(rgb, canvas) for rgb in raw]
    return np.stack(images)


def build(model, processor, builder, row, raw, profile):
    began = time.perf_counter()
    frames = pixels(raw, profile)
    times = [p.timestamp for p in row['pair']]
    ticks = [round(t * 1000) for t in times]
    inp = processor(text=[builder.template(row['question'])], videos=[frames],
                    video_metadata=[{'fps': 1000., 'frames_indices': ticks, 'total_num_frames': max(ticks)+1}],
                    do_resize=False, do_sample_frames=False, return_tensors='pt')
    add_video_positions(model, inp)
    check_position_oracle(model, inp)
    tokens = int(inp.video_grid_thw.prod()) // processor.video_processor.merge_size**2
    expected = 192 if profile == 'direct_large' else 48
    if tokens != expected or int((inp.input_ids == processor.video_token_id).sum()) != expected:
        raise ValueError('Visual budget/grid mismatch')
    if inp.input_ids.shape[1] + 32 > 512:
        raise ValueError('Total context bound exceeded')
    info = {'profile': profile, 'canvas': list(PROFILES[profile]), 'visual_tokens': tokens,
            'vision_patch_rows': inp.pixel_values_videos.shape[0], 'grid': inp.video_grid_thw.tolist(),
            'input_tokens': inp.input_ids.shape[1], 'input_fingerprint': tensor_fingerprint(inp),
            'pixel_sha256': reference.sha_tensor(inp.pixel_values_videos),
            'ids_sha256': reference.sha_tensor(inp.input_ids), 'position_sha256': reference.sha_tensor(inp.position_ids),
            'frame_rgb_sha256': [hashlib.sha256(f.tobytes()).hexdigest() for f in frames],
            'actual_pts': times, 'metadata_carrier_not_source_fps': True,
            'preprocess_seconds': time.perf_counter() - began}
    return inp, info


def stable_layout(info):
    return {k: v for k, v in info.items() if k != 'preprocess_seconds'}


@torch.inference_mode()
def preflight(args):
    from tests.test_online import small_model
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    builder = PairInputBuilder(processor, resize_mode='letterbox')
    current = small_model(processor)
    package = reference.original_package()
    original = package.Qwen3_5ForConditionalGeneration(package.Qwen3_5Config(**current.config.to_dict())).eval()
    original.set_attn_implementation('sdpa')
    original.load_state_dict(current.state_dict(), strict=True)
    rows, assets, small_bytes = reference.inputs()
    previous = read(args.reference_cpu / 'proof.json')
    entries = []
    for row, registered in zip(rows, previous['cases']):
        raw, metadata, decode_seconds = raw_pair(row, assets)
        profile_layouts, checks = {}, {}
        for profile in PROFILES:
            inp, info = build(original, processor, builder, row, raw, profile)
            if profile == 'old_two_stage' and info['input_fingerprint'] != registered['input_fingerprint']:
                raise ValueError('Old official input cannot be reproduced')
            if profile == 'direct_small':
                if any(info[key] != profile_layouts['old_two_stage'][key]
                       for key in ('ids_sha256', 'position_sha256', 'visual_tokens', 'grid', 'input_tokens')):
                    raise AssertionError('Same-budget input protocol changed')
            with fp32_reference():
                left = original(**inp, use_cache=False, logits_to_keep=1).logits
                right = current(**inp, use_cache=False, logits_to_keep=1).logits
            check = reference.closeness(left, right)
            if not check['allclose']:
                raise AssertionError('Original/current tiny fresh prefill mismatch')
            checks[profile] = check
            profile_layouts[profile] = stable_layout(info)
            del inp, left, right
        entries.append({'id': row['id'], 'question': row['question'], 'identity': row['identity'],
                        'raw_frames': metadata, 'layouts': profile_layouts, 'tiny_checks': checks,
                        'decode_seconds': decode_seconds})
        del raw
        print(json.dumps({'case': row['id'], 'raw_frames_verified': True, 'cpu_profiles': 3}), flush=True)
    write_json(args.out / 'proof.json', {'passed': True, 'cases': entries, 'source_hashes': sources(),
               'request': request(args), 'budget': budget(), 'identity': identity(args),
               'input_assets': assets, 'small_input_rgb_bytes': small_bytes,
               'scope': '24 real-input layouts and same-weight tiny fresh prefills, not recognition evidence'})


@torch.inference_mode()
def gpu(args):
    if args.preflight is None or read(args.preflight / 'status.json') != {'status': 'completed', 'stage': 'preflight'}:
        raise ValueError('Fresh CPU proof required')
    proof = read(args.preflight / 'proof.json')
    if not proof['passed'] or proof['source_hashes'] != sources() or proof['request'] != request(args):
        raise ValueError('Code/request proof drift')
    if proof['budget'] != budget() or proof['identity'] != identity(args):
        raise ValueError('Budget/model/reference drift')
    rows, assets, small_bytes = reference.inputs()
    if assets != proof['input_assets']:
        raise ValueError('Source media drift')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=CAP, text=True).strip()
    if branch != args.branch or args.branch in ('', 'main', 'master', 'HEAD'):
        raise ValueError('Configured personal branch required')
    free, total = torch.cuda.mem_get_info()
    ram = psutil.virtual_memory().available
    write_json(args.out / 'resource_precheck.json', {'ram_available_bytes': ram, 'gpu_free_bytes': free,
                                                    'gpu_total_bytes': total, 'branch': branch})
    if ram < 3 * 2**30 or free < 5.5 * 2**30:
        raise RuntimeError('Resource gate refused load')
    cls = reference.original_package().Qwen3_5ForConditionalGeneration
    began = time.perf_counter()
    model = cls.from_pretrained(args.model_dir, local_files_only=True, dtype=torch.bfloat16,
                               attn_implementation='sdpa').to('cuda').eval()
    if Path(inspect.getfile(type(model))).resolve() != reference.ORIGINAL / 'modeling_qwen3_5.py':
        raise AssertionError('Original implementation required')
    if torch.cuda.mem_get_info()[0] < 2 * 2**30:
        raise RuntimeError('No conversion room')
    model.float()
    torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    builder = PairInputBuilder(processor, resize_mode='letterbox')
    previous = {r['id']: r for r in read(args.reference_gpu / 'summary.json')['answers']}
    counts = {'vision': 0, 'language': 0, 'vision_patch_rows': 0}
    def guard(name):
        def hook(_module, inputs, kwargs):
            counts[name] += 1
            if counts[name] > budget()[name + '_max']:
                raise RuntimeError('Forward budget exceeded')
            if name == 'vision':
                tensor = kwargs.get('hidden_states')
                if tensor is None and inputs:
                    tensor = inputs[0]
                if not isinstance(tensor, torch.Tensor):
                    raise RuntimeError('Vision patch accounting failed')
                counts['vision_patch_rows'] += tensor.shape[0]
        return hook
    hooks = [model.model.visual.register_forward_pre_hook(guard('vision'), with_kwargs=True),
             model.model.language_model.register_forward_pre_hook(guard('language'), with_kwargs=True)]
    answers, generated, raw_peak = [], 0, 0
    try:
        with (args.out / 'answers.jsonl').open('x', encoding='utf-8') as log:
            for row, registered in zip(rows, proof['cases']):
                raw, metadata, decode_seconds = raw_pair(row, assets)
                if row['identity'] != registered['identity'] or metadata != registered['raw_frames']:
                    raise ValueError('CPU/GPU raw source identity drift')
                raw_peak = max(raw_peak, sum(rgb.nbytes for rgb in raw))
                for profile in PROFILES:
                    inp, info = build(model, processor, builder, row, raw, profile)
                    if stable_layout(info) != registered['layouts'][profile]:
                        raise ValueError('CPU/GPU preprocessing layout drift')
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    before = dict(counts)
                    answer_begin = time.perf_counter()
                    result, first_logits = reference.native_answer(model, processor, inp)
                    torch.cuda.synchronize()
                    record = {'id': row['id'], 'question': row['question'], 'profile': profile,
                              'identity': row['identity'], 'raw_frames': metadata, 'layout': info,
                              'decode_seconds': decode_seconds, 'answer_seconds': time.perf_counter() - answer_begin,
                              'result': result, 'counts': {key: counts[key]-before[key] for key in counts},
                              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                              'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30}
                    if profile == 'old_two_stage':
                        old = previous[row['id']]['native']
                        record['matches_previous_generated'] = old['generated_ids'] == result['generated_ids']
                        record['matches_previous_first_logits_sha'] = old['first_logits_sha256'] == result['first_logits_sha256']
                        if not record['matches_previous_generated'] or not record['matches_previous_first_logits_sha']:
                            raise AssertionError('Old A control output drift')
                    generated += len(result['generated_ids'])
                    if generated > budget()['generated_max']:
                        raise RuntimeError('Generation budget exceeded')
                    answers.append(record)
                    log.write(json.dumps(record, ensure_ascii=False) + '\n')
                    log.flush()
                    print(json.dumps({'case': row['id'], 'profile': profile, 'visual_tokens': info['visual_tokens'],
                                      'answer': result['answer']}, ensure_ascii=False), flush=True)
                    del inp, first_logits
                del raw
        write_json(args.out / 'summary.json', {'passed': True, 'answers': answers, 'counts': counts,
                   'generated_tokens': generated, 'raw_pair_peak_rgb_bytes': raw_peak,
                   'small_input_pool_rgb_bytes': small_bytes, 'source_hashes': sources(),
                   'wall_seconds_including_load': time.perf_counter()-began,
                   'scope': 'Familiar eight-pair detail ablation; C costs four times visual tokens, no generalization claim'})
    finally:
        for hook in hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preflight', 'gpu'), required=True)
    for name in ('model-dir', 'reference-cpu', 'reference-gpu', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--branch', required=True)
    args = parser.parse_args()
    for name in ('model_dir', 'reference_cpu', 'reference_gpu', 'out', 'preflight'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'tmp').mkdir()
    tempfile.tempdir = str(args.out / 'tmp')
    snapshot = args.out / 'source_snapshot'
    snapshot.mkdir()
    for path in HERE.iterdir():
        if path.suffix in ('.py', '.md', '.json'):
            shutil.copyfile(path, snapshot / path.name)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    write_json(args.out / 'source_hashes.json', sources())
    write_json(args.out / 'budget.json', budget())
    try:
        (preflight if args.stage == 'preflight' else gpu)(args)
        write_json(args.out / 'status.json', {'status': 'completed', 'stage': args.stage})
    except Exception as error:
        write_json(args.out / 'status.json', {'status': 'failed', 'stage': args.stage,
                   'error': type(error).__name__, 'message': str(error)})
        (args.out / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
