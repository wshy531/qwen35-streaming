"""Registered image-slot/time-label interventions, with independent original/streaming arms."""
import argparse
import hashlib
import inspect
from pathlib import Path
import subprocess
import tempfile
import time
import traceback
import unittest

import psutil
import torch
from transformers import AutoProcessor

from scripts.baseline import PROJECT, REPO, file_hash, write_json
from scripts.fair_video import read, original_answer, stream_answer
from scripts.pair_demo import verify_assets
from scripts.protocol_capability_control import original_package, source_assets, ORIGINAL
from scripts.video_demo import branch_allowed
from streaming.action_inputs import load_case, relationships
from streaming.fair_inputs import identity
from streaming.online import fp32_reference

CONFIG = PROJECT / 'configs/action_ablation_cases.json'
PLAN = PROJECT / '洗脸误答单因素定位方案.md'
LAUNCHER = PROJECT / 'scripts/run_action_ablation.py'


def cases():
    config = read(CONFIG)
    specs = config['cases']
    if len(specs) != 12 or config['answer_tokens'] != 16 or config['question'] != 'What are the hands doing?':
        raise ValueError('Registered case configuration changed')
    relationships(specs)
    return specs, config['question']


def sources():
    return {**{p.relative_to(REPO).as_posix(): file_hash(p) for p in sorted(REPO.rglob('*.py'))},
        '@launcher': file_hash(LAUNCHER), '@plan': file_hash(PLAN), '@cases': file_hash(CONFIG),
        **{'@reference:' + name: digest for name, digest in source_assets().items()}}


def request(args):
    return {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()
            if key not in ('stage', 'arm', 'out', 'preflight')}


def budget(args):
    return {'model_loads_per_arm': 1, 'cases': 12, 'input_slots_max': 48, 'generated_tokens_per_arm_max': 192,
        'answer_tokens': 16, 'original_vision_max': 12, 'original_language_max': 204,
        'streaming_vision_max': 24, 'streaming_language_max': 240, 'wall_seconds': args.wall_seconds,
        'gpu_free_gib_min': 5.5, 'ram_available_gib_min': 3, 'precision': 'fp32_ieee_math',
        'tokens_max': 256, 'question_reserve': 96, 'rgb_per_case_max_bytes': 589824}


def assets(args):
    return {str(p.resolve()): file_hash(p) for p in (args.video, args.original_reference, args.streaming_reference)}


def case_inputs(model, processor, spec, args, question):
    frames, provenance = load_case(spec, args.video)
    whole, checked = identity(model, processor, frames, question)
    digests = {key: hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest() for key, value in whole.items()}
    return frames, whole, {**provenance, **checked, 'tensor_hashes': digests}


def check_relations(entries, specs):
    by_id = {entry['id']: entry for entry in entries}
    for relation in relationships(specs):
        entry, parent = by_id[relation['id']], by_id[relation['parent']]
        changed = [i for i in range(4) if entry['frames'][i]['canonical_rgb_sha256'] != parent['frames'][i]['canonical_rgb_sha256']]
        if changed != relation['changed_image_slots']:
            raise AssertionError('Actual image hashes do not match intervention')
        if not changed and entry['tensor_hashes']['pixel_values_videos'] != parent['tensor_hashes']['pixel_values_videos']:
            raise AssertionError('Time-only intervention changed visual processor tensor')


@torch.inference_mode()
def preflight(args):
    from tests.test_action_inputs import ActionInputTests
    from tests.test_online import small_model
    with (args.out / 'unittest.log').open('w', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ActionInputTests))
    write_json(args.out / 'cpu_tests.json', {'tests': result.testsRun, 'passed': result.wasSuccessful(), 'errors': len(result.errors), 'failures': len(result.failures)})
    if not result.wasSuccessful():
        raise AssertionError('Intervention contracts failed')
    model_assets = verify_assets(args.model_dir)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    current = small_model(processor)
    package = original_package()
    original = package.Qwen3_5ForConditionalGeneration(package.Qwen3_5Config(**current.config.to_dict())).eval()
    original.set_attn_implementation('sdpa')
    original.load_state_dict(current.state_dict(), strict=True)
    refs = [read(args.original_reference), read(args.streaming_reference)]
    specs, question = cases()
    entries = []
    for spec in specs:
        frames, whole, record = case_inputs(original, processor, spec, args, question)
        if spec['factor'] == 'actual_anchor':
            prior_id = 'hands_sparse_late_action' if spec['id'] == 'sparse_actual' else 'hands_dense_late_action'
            prior = [next(c for c in report['cases'] if c['id'] == prior_id) for report in refs]
            if any(c['input_fingerprint'] != record['input_fingerprint'] for c in prior):
                raise AssertionError('Actual anchor differs from third-stage input')
            record['anchor_prior_ids'] = prior[0]['generated_ids']
        with fp32_reference():
            a = original(**whole, use_cache=False, logits_to_keep=1).logits
            b = current(**whole, use_cache=False, logits_to_keep=1).logits
            if not torch.equal(a, b):
                raise AssertionError('Fresh tiny original/modified prefill differs')
        answer = stream_answer(current, processor, frames, question)
        assert answer['main_unchanged_bitwise'] and answer['main_counters_unchanged']
        entries.append(record)
        del frames, whole, a, b, answer
    check_relations(entries, specs)
    write_json(args.out / 'preflight_summary.json', {'cases': entries, 'relations': relationships(specs),
        'cpu_cases_passed': 12, 'scope': 'Random tiny contracts and fixed inputs, not semantic evidence'})
    write_json(args.out / 'proof.json', {'request': request(args), 'budget': budget(args), 'source_hashes': sources(),
        'inputs': assets(args), 'assets': model_assets['files'], 'model_manifest_sha256': file_hash(args.model_dir / 'DOWNLOAD_MANIFEST.json'),
        'cases': entries, 'summary_sha256': file_hash(args.out / 'preflight_summary.json'), 'cpu_tests_sha256': file_hash(args.out / 'cpu_tests.json')})
    print({'passed': True, 'interface_tests': result.testsRun, 'input_cases': 12}, flush=True)


def check_proof(args):
    if args.preflight is None or read(args.preflight / 'status.json') != {'status': 'completed', 'stage': 'preflight', 'arm': 'both'}:
        raise ValueError('Fresh completed CPU proof required')
    proof = read(args.preflight / 'proof.json')
    if proof['request'] != request(args) or proof['budget'] != budget(args) or proof['source_hashes'] != sources():
        raise ValueError('Source, request, configuration or budget changed')
    if proof['inputs'] != assets(args) or proof['assets'] != verify_assets(args.model_dir)['files'] or proof['model_manifest_sha256'] != file_hash(args.model_dir / 'DOWNLOAD_MANIFEST.json'):
        raise ValueError('Video/reference/model assets changed')
    if proof['summary_sha256'] != file_hash(args.preflight / 'preflight_summary.json') or proof['cpu_tests_sha256'] != file_hash(args.preflight / 'cpu_tests.json') or not read(args.preflight / 'cpu_tests.json')['passed']:
        raise ValueError('CPU evidence changed or failed')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
    if not branch_allowed(branch, args.allow_branch):
        raise ValueError('Allowed personal branch required')
    write_json(args.out / 'checked_branch.json', {'branch': branch})
    return proof


@torch.inference_mode()
def gpu(args):
    proof = check_proof(args)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    start = time.perf_counter()
    free, total = torch.cuda.mem_get_info()
    guard = {'gpu_free_bytes': free, 'gpu_total_bytes': total, 'ram_available_bytes': psutil.virtual_memory().available}
    write_json(args.out / 'resource_precheck.json', guard)
    if free < 5.5 * 2**30 or guard['ram_available_bytes'] < 3 * 2**30:
        raise RuntimeError('Resource guard refused model load')
    if args.arm == 'original':
        cls = original_package().Qwen3_5ForConditionalGeneration
    else:
        from qwen35_local import Qwen3_5ForConditionalGeneration
        cls = Qwen3_5ForConditionalGeneration
    model = cls.from_pretrained(args.model_dir, local_files_only=True, dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    if args.arm == 'original' and Path(inspect.getfile(type(model))).resolve() != ORIGINAL / 'modeling_qwen3_5.py':
        raise RuntimeError('Original arm source mismatch')
    if torch.cuda.mem_get_info()[0] < 2 * 2**30:
        raise RuntimeError('Insufficient room for FP32 conversion')
    model.float()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    counts = {'vision': 0, 'language': 0}
    limits = budget(args)
    def count(name):
        def hook(_module, _inputs):
            counts[name] += 1
            if counts[name] > limits[args.arm + '_' + name + '_max']:
                raise RuntimeError('Registered forward budget exceeded')
        return hook
    hooks = [model.model.visual.register_forward_pre_hook(count('vision')), model.model.language_model.register_forward_pre_hook(count('language'))]
    torch.cuda.reset_peak_memory_stats()
    report = {'arm': args.arm, 'budget': limits, 'model_source': inspect.getfile(type(model)), 'cases': [],
              'scope': 'Fixed diagnostic image/label interventions; synthetic labels explicitly marked'}
    specs, question = cases()
    generated = 0
    try:
        for spec, expected in zip(specs, proof['cases']):
            if time.perf_counter() - start >= args.wall_seconds:
                raise TimeoutError('Registered wall budget exceeded')
            frames, whole, record = case_inputs(model, processor, spec, args, question)
            if 'anchor_prior_ids' in expected:
                record['anchor_prior_ids'] = expected['anchor_prior_ids']
            if record != expected:
                raise AssertionError('Actual case differs from CPU proof')
            before = counts.copy()
            answer = original_answer(model, processor, whole) if args.arm == 'original' else stream_answer(model, processor, frames, question)
            torch.cuda.synchronize()
            generated += len(answer['generated_ids'])
            if generated > 192:
                raise RuntimeError('Generated-token budget exceeded')
            report['cases'].append({**record, **answer, 'forward_counts': {key: counts[key]-before[key] for key in counts}})
            write_json(args.out / 'summary.json', report)
            print({'arm': args.arm, 'id': spec['id'], 'answer': answer['answer']}, flush=True)
            del frames, whole, answer
        report.update(actual_forwards=counts, generated_tokens=generated, all_registered_cases_executed=len(report['cases']) == 12,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            wall_seconds_including_load=time.perf_counter() - start)
        write_json(args.out / 'summary.json', report)
    finally:
        for hook in hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preflight', 'gpu'), required=True)
    parser.add_argument('--arm', choices=('both', 'original', 'streaming'), default='both')
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--original-reference', type=Path, default=PROJECT / 'runs/fair_video_original_gpu_20261011_02/summary.json')
    parser.add_argument('--streaming-reference', type=Path, default=PROJECT / 'runs/fair_video_streaming_gpu_20261011_01/summary.json')
    parser.add_argument('--wall-seconds', type=int, default=240)
    parser.add_argument('--allow-branch', action='append', default=[])
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.wall_seconds <= 240 or (args.stage == 'preflight') != (args.arm == 'both'):
        parser.error('CPU arm=both; GPU arm=original/streaming; wall<=240')
    for name in ('model_dir', 'video', 'original_reference', 'streaming_reference', 'out'):
        setattr(args, name, getattr(args, name).resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'tmp').mkdir()
    tempfile.tempdir = str(args.out / 'tmp')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    write_json(args.out / 'arguments.json', {**request(args), 'stage': args.stage, 'arm': args.arm})
    write_json(args.out / 'budget.json', budget(args))
    write_json(args.out / 'source_hashes.json', sources())
    try:
        (preflight if args.stage == 'preflight' else gpu)(args)
        write_json(args.out / 'status.json', {'status': 'completed', 'stage': args.stage, 'arm': args.arm})
    except Exception:
        (args.out / 'traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        write_json(args.out / 'status.json', {'status': 'failed', 'stage': args.stage, 'arm': args.arm})
        raise


if __name__ == '__main__':
    main()
