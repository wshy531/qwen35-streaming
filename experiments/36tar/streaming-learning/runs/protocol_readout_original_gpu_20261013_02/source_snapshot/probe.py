"""Matched official processor, original core and uncompressed feature readout."""
import argparse
import gc
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
ACTIVE = CAP / 'code/qwen35-streaming-s4'
sys.path.insert(0, str(ACTIVE))
sys.path.insert(0, str(HERE.parent / 'visual_memory_v1'))
sys.path.insert(0, str(HERE.parent / 'event_memory_v1'))
spec = importlib.util.spec_from_file_location('readout_event_inputs', HERE.parent / 'event_memory_v1/probe.py')
events = importlib.util.module_from_spec(spec)
spec.loader.exec_module(events)
import numpy as np
import psutil
import torch
from transformers import AutoProcessor
from features import context, raw_block
from scripts.baseline import file_hash, write_json, add_video_positions, check_position_oracle
from scripts.pair_demo import verify_assets
from scripts.protocol_capability_control import original_package, source_assets, ORIGINAL
from streaming.hybrid import HybridSequence
from streaming.online import PairInputBuilder, fp32_reference, greedy_decode
from streaming.inputs import tensor_fingerprint

ANIMAL = 'Name the main visible animal in these frames.'
ACTION = 'Briefly describe the visible action in these frames.'
CASES = [('animal_holdout', 'dog', 0), ('animal_holdout', 'pigeon', 0),
         ('animal_holdout', 'pigeon', 1), ('animal_holdout', 'cat', 0),
         ('animal_holdout', 'cat', 1), ('natural', 'clean_hands', 0),
         ('natural', 'clean_hands', 5), ('natural', 'hand_hygiene', 5)]


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def sources():
    paths = [*HERE.glob('*.py'), HERE / '方案.md',
             *ACTIVE.rglob('*.py'),
             * (HERE.parent / 'visual_memory_v1').glob('*.py'),
             * (HERE.parent / 'event_memory_v1').glob('*.py'),
             HERE.parent / 'event_memory_v1/cases.json',
             HERE.parent / 'general_stream_v1/cases.json']
    return {**{str(p.resolve()): file_hash(p) for p in sorted(paths)}, **source_assets()}


def budget(arm):
    current = arm == 'current'
    return {'model_loads': 1, 'questions_max': 16 if current else 8,
            'vision_max': 16 if current else 8, 'language_max': 560 if current else 280,
            'generated_max': 512 if current else 256, 'wall_seconds': 360,
            'answer_tokens': 32, 'max_tokens': 512, 'ram_min_gib': 3,
            'gpu_free_min_gib': 5.5, 'conversion_free_min_gib': 2,
            'input_rgb_max_bytes': 8 * 2**20, 'comparison_cpu_max_bytes': 16 * 2**20,
            'precision': 'fp32_ieee_math', 'concurrent_models': 1}


def request(args):
    return {'model_dir': str(args.model_dir), 'branch': args.branch,
            'cases': [list(c) for c in CASES], 'animal_question': ANIMAL, 'action_question': ACTION}


def inputs():
    data, identities, assets = events.inputs()
    rgb_bytes = sum(p.rgb.nbytes for split in data.values() for pairs in split.values()
                    for pair in pairs for p in pair)
    if rgb_bytes > 8 * 2**20:
        raise RuntimeError('Input RGB bound exceeded')
    selected = []
    for split, name, index in CASES:
        selected.append({'id': f'{name}_pair_{index}', 'pair': data[split][name][index],
                         'identity': identities[split][name][2*index:2*index+2],
                         'question': ANIMAL if split == 'animal_holdout' else ACTION})
    return selected, assets, rgb_bytes


def native_inputs(model, processor, builder, row):
    # An explicit timebase carrier for pre-sampled input; not source frame rate.
    timestamps = [p.timestamp for p in row['pair']]
    ticks = [round(t * 1000) for t in timestamps]
    metadata = {'fps': 1000., 'frames_indices': ticks, 'total_num_frames': max(ticks) + 1}
    result = processor(text=[builder.template(row['question'])],
                       videos=[np.stack([p.rgb for p in row['pair']])],
                       video_metadata=[metadata], do_resize=False, do_sample_frames=False,
                       return_tensors='pt')
    if result.video_grid_thw.tolist() != [[1, 12, 16]]:
        raise AssertionError('Bounded canvas/grid changed')
    add_video_positions(model, result)
    check_position_oracle(model, result)
    if result.input_ids.shape[1] + 32 > 512:
        raise ValueError('Context bound exceeded')
    block, record = builder.pair(*row['pair'], builder.prefix_ids.shape[1], builder.prefix_ids.shape[1])
    if not torch.equal(block.pixel_values_videos, result.pixel_values_videos):
        raise AssertionError('Official/pair preprocessed pixels differ')
    if f'<{sum(timestamps)/2:.1f} seconds>' != f'<{sum(ticks)/2000:.1f} seconds>':
        raise AssertionError('Millisecond carrier changed rendered timestamp')
    info = {'input_fingerprint': tensor_fingerprint(result), 'input_tokens': result.input_ids.shape[1],
            'rendered_text': processor.tokenizer.decode(result.input_ids[0], skip_special_tokens=False),
            'pixel_sha256': sha_tensor(result.pixel_values_videos), 'position_sha256': sha_tensor(result.position_ids),
            'input_ids': result.input_ids[0].tolist(), 'pair_grid': record['grid'],
            'actual_pts': timestamps, 'metadata_carrier': metadata,
            'metadata_carrier_not_source_fps': True, 'canvas': [192, 256]}
    return result, info


def closeness(left, right):
    if left.numel() * left.element_size() + right.numel() * right.element_size() > 16 * 2**20:
        raise RuntimeError('Comparison memory bound exceeded')
    return {'atol': 1e-4, 'rtol': 1e-4,
            'allclose': bool(torch.allclose(left, right, atol=1e-4, rtol=1e-4)),
            'bitwise_equal': bool(torch.equal(left, right)),
            'max_abs': float((left - right).abs().max()),
            'left_sha256': sha_tensor(left), 'right_sha256': sha_tensor(right)}


def stops(model, processor):
    eos = model.generation_config.eos_token_id
    result = set([] if eos is None else eos if isinstance(eos, list) else [eos])
    if processor.tokenizer.eos_token_id is not None:
        result.add(processor.tokenizer.eos_token_id)
    return result


@torch.inference_mode()
def native_answer(model, processor, inp):
    device = model.get_input_embeddings().weight.device
    saved = model.model.rope_deltas
    cache = None
    try:
        with fp32_reference():
            out = model(**inp.to(device), use_cache=True, logits_to_keep=1)
            logits, cache = out.logits, out.past_key_values
            first = logits.detach().cpu().clone()
            count = inp.input_ids.shape[1]
            rope = int(inp.position_ids[1:].max()) + 1
            del out
            generated, reason = [], 'budget'
            eos = stops(model, processor)
            for _ in range(32):
                token = int(logits[0, -1].argmax())
                generated.append(token)
                if token in eos:
                    reason = 'eos'
                    break
                out = model(input_ids=torch.tensor([[token]], device=device),
                            position_ids=torch.tensor([[[count]], [[rope]], [[rope]], [[rope]]], device=device),
                            attention_mask=torch.ones((1, count+1), dtype=torch.long, device=device),
                            cache_position=torch.tensor([count], device=device),
                            past_key_values=cache, use_cache=True, logits_to_keep=1)
                logits, cache = out.logits, out.past_key_values
                count += 1
                rope += 1
                del out
        return {'answer': processor.tokenizer.decode(generated, skip_special_tokens=True),
                'generated_ids': generated, 'stop_reason': reason,
                'first_logits_sha256': sha_tensor(first)}, first
    finally:
        cache = None
        model.model.rope_deltas = saved


@torch.inference_mode()
def feature_prefill(model, builder, row, native):
    device = model.get_input_embeddings().weight.device
    pair, _ = builder.pair(*row['pair'], 0, 0)
    with fp32_reference():
        vision = model.model.get_video_features(pair.pixel_values_videos.to(device),
                                              pair.video_grid_thw.to(device), return_dict=True)
        block = raw_block(torch.cat(vision.pooler_output), 0, row['pair'][0].timestamp, row['pair'][1].timestamp)
        del vision
        embeddings, positions, ids, meta = context(model, builder, [block], row['question'], 32)
    if not torch.equal(ids.cpu(), native.input_ids.cpu()) or not torch.equal(positions.cpu(), native.position_ids.cpu()):
        raise AssertionError('Official/feature token or position mismatch')
    return embeddings, positions, meta, sha_tensor(block.features)


@torch.inference_mode()
def feature_answer(model, builder, row, native):
    saved = model.model.rope_deltas
    sequence = HybridSequence(model, max_tokens=512, session_id='matched-single-pair')
    try:
        embeddings, positions, meta, feature_sha = feature_prefill(model, builder, row, native)
        with fp32_reference():
            logits = sequence.append(inputs_embeds=embeddings, position_ids=positions, logits_mode='last')
            first = logits.detach().cpu().clone()
            generated, reason = greedy_decode(sequence, logits, 32, stops(model, builder.processor))
        return {'answer': builder.processor.tokenizer.decode(generated, skip_special_tokens=True),
                'generated_ids': generated, 'stop_reason': reason, 'feature_sha256': feature_sha,
                'first_logits_sha256': sha_tensor(first), 'context': meta}, first
    finally:
        sequence.cache.reset()
        model.model.rope_deltas = saved


@torch.inference_mode()
def preflight(args):
    from tests.test_online import small_model
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    builder = PairInputBuilder(processor, resize_mode='letterbox')
    current = small_model(processor)
    package = original_package()
    original = package.Qwen3_5ForConditionalGeneration(package.Qwen3_5Config(**current.config.to_dict())).eval()
    original.set_attn_implementation('sdpa')
    original.load_state_dict(current.state_dict(), strict=True)
    rows, assets, rgb_bytes = inputs()
    entries = []
    for row in rows:
        native, info = native_inputs(current, processor, builder, row)
        saved = current.model.rope_deltas
        embeddings, positions, _meta, _sha = feature_prefill(current, builder, row, native)
        sequence = HybridSequence(current, max_tokens=512)
        try:
            with fp32_reference():
                left = current(**native, use_cache=False, logits_to_keep=1).logits
                right = sequence.append(inputs_embeds=embeddings, position_ids=positions, logits_mode='last')
                old = original(**native, use_cache=False, logits_to_keep=1).logits
            feature_check = closeness(left, right)
            original_check = closeness(left, old)
            if not feature_check['allclose'] or not original_check['allclose']:
                raise AssertionError('Tiny fresh-prefill readout differs')
            current_answer, _ = native_answer(current, processor, native)
            reused_answer, _ = feature_answer(current, builder, row, native)
            if current_answer['generated_ids'] != reused_answer['generated_ids']:
                raise AssertionError('Tiny full greedy readout differs')
            entries.append({'id': row['id'], 'identity': row['identity'], 'question': row['question'],
                            **info, 'tiny_feature_logits': feature_check, 'tiny_original_logits': original_check,
                            'tiny_greedy_ids_equal': True})
        finally:
            sequence.cache.reset()
            current.model.rope_deltas = saved
        print(json.dumps({'case': row['id'], 'cpu_same_inputs_positions_and_logits': True}), flush=True)
    proof = {'passed': True, 'cases': entries, 'source_hashes': sources(), 'request': request(args),
             'budgets': {arm: budget(arm) for arm in ('original', 'current')},
             'input_assets': assets, 'rgb_bytes': rgb_bytes,
             'model_assets': verify_assets(args.model_dir)['files'],
             'model_manifest_sha256': file_hash(args.model_dir / 'DOWNLOAD_MANIFEST.json'),
             'scope': 'Random tiny matched prefill/decode contracts; no recognition evidence'}
    write_json(args.out / 'proof.json', proof)


@torch.inference_mode()
def gpu(args):
    if args.preflight is None or read(args.preflight / 'status.json') != {'status': 'completed', 'stage': 'preflight', 'arm': 'both'}:
        raise ValueError('Fresh CPU proof required')
    proof = read(args.preflight / 'proof.json')
    if not proof['passed'] or proof['source_hashes'] != sources() or proof['request'] != request(args):
        raise ValueError('Source/request proof drift')
    if proof['budgets'][args.arm] != budget(args.arm) or proof['model_assets'] != verify_assets(args.model_dir)['files']:
        raise ValueError('Budget/model asset drift')
    if proof['model_manifest_sha256'] != file_hash(args.model_dir / 'DOWNLOAD_MANIFEST.json'):
        raise ValueError('Model manifest drift')
    rows, assets, rgb_bytes = inputs()
    if assets != proof['input_assets'] or [r['identity'] for r in rows] != [r['identity'] for r in proof['cases']]:
        raise ValueError('Input pixel/source drift')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=CAP, text=True).strip()
    if args.branch in ('', 'main', 'master', 'HEAD') or branch != args.branch:
        raise ValueError('Configured personal branch required')
    free, total = torch.cuda.mem_get_info()
    ram = psutil.virtual_memory().available
    write_json(args.out / 'resource_precheck.json', {'ram_available_bytes': ram, 'gpu_free_bytes': free,
                                                    'gpu_total_bytes': total, 'branch': branch})
    if free < 5.5 * 2**30 or ram < 3 * 2**30:
        raise RuntimeError('Resource gate refused load')
    if args.arm == 'original':
        cls = original_package().Qwen3_5ForConditionalGeneration
        expected = ORIGINAL / 'modeling_qwen3_5.py'
    else:
        from qwen35_local import Qwen3_5ForConditionalGeneration
        cls = Qwen3_5ForConditionalGeneration
        expected = ACTIVE / 'qwen35_local/modeling_qwen3_5.py'
    begin = time.perf_counter()
    model = cls.from_pretrained(args.model_dir, local_files_only=True, dtype=torch.bfloat16,
                               attn_implementation='sdpa').to('cuda').eval()
    if Path(inspect.getfile(type(model))).resolve() != expected:
        raise AssertionError('Wrong original/current implementation')
    if torch.cuda.mem_get_info()[0] < 2 * 2**30:
        raise RuntimeError('No conversion room')
    model.float()
    torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    builder = PairInputBuilder(processor, resize_mode='letterbox')
    counts = {'vision': 0, 'language': 0}
    def guard(name):
        def hook(_module, _args):
            counts[name] += 1
            if counts[name] > budget(args.arm)[name + '_max']:
                raise RuntimeError('Forward budget exceeded')
        return hook
    hooks = [model.model.visual.register_forward_pre_hook(guard('vision')),
             model.model.language_model.register_forward_pre_hook(guard('language'))]
    answers, generated = [], 0
    try:
        with (args.out / 'answers.jsonl').open('x', encoding='utf-8') as log:
            for row, registered in zip(rows, proof['cases']):
                native, info = native_inputs(model, processor, builder, row)
                if any(info[key] != registered[key] for key in info):
                    raise ValueError('GPU/CPU full official input drift')
                result, logits = native_answer(model, processor, native)
                record = {'id': row['id'], 'question': row['question'], 'identity': row['identity'],
                          'layout': info, 'native': result}
                generated += len(result['generated_ids'])
                if args.arm == 'current':
                    feature, second = feature_answer(model, builder, row, native)
                    generated += len(feature['generated_ids'])
                    record.update(feature=feature, matched_prefill=closeness(logits, second),
                                  generated_ids_equal=result['generated_ids'] == feature['generated_ids'])
                    # Preserve divergence as evidence instead of aborting/selecting a winner.
                    del second
                if generated > budget(args.arm)['generated_max']:
                    raise RuntimeError('Generated token budget exceeded')
                answers.append(record)
                log.write(json.dumps(record, ensure_ascii=False) + '\n')
                log.flush()
                print(json.dumps({'arm': args.arm, 'case': row['id'], 'native': result['answer'],
                                  'feature': record.get('feature', {}).get('answer'),
                                  'matched': record.get('generated_ids_equal')}, ensure_ascii=False), flush=True)
                del logits, native
        write_json(args.out / 'summary.json', {'passed': True, 'arm': args.arm, 'answers': answers,
                   'counts': counts, 'generated_tokens': generated, 'rgb_bytes': rgb_bytes,
                   'model_source': str(expected), 'source_hashes': sources(),
                   'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                   'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                   'wall_seconds_including_load': time.perf_counter() - begin,
                   'scope': 'Eight familiar pairs, matched bounded-canvas protocol diagnostic, not accuracy benchmark'})
    finally:
        for hook in hooks:
            hook.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preflight', 'gpu'), required=True)
    parser.add_argument('--arm', choices=('both', 'original', 'current'), required=True)
    for name in ('model-dir', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--branch', required=True)
    args = parser.parse_args()
    for name in ('model_dir', 'out', 'preflight'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    if (args.stage == 'preflight') != (args.arm == 'both'):
        parser.error('Preflight both, GPU one arm per process')
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
    write_json(args.out / 'budget.json', {arm: budget(arm) for arm in ('original', 'current')})
    try:
        (preflight if args.stage == 'preflight' else gpu)(args)
        write_json(args.out / 'status.json', {'status': 'completed', 'stage': args.stage, 'arm': args.arm})
    except Exception as error:
        write_json(args.out / 'status.json', {'status': 'failed', 'stage': args.stage, 'arm': args.arm,
                   'error': type(error).__name__, 'message': str(error)})
        (args.out / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
