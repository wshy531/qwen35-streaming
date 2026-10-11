"""Two independent FP32 arms over identical bounded current-epoch video inputs."""
import argparse
import inspect
import json
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
from scripts.pair_demo import verify_assets
from scripts.protocol_capability_control import original_package, source_assets, ORIGINAL
from scripts.video_demo import branch_allowed
from streaming.demo import PairDemo
from streaming.fair_inputs import load_case, identity
from streaming.online import OnlinePairSession, fp32_reference


CONFIG = PROJECT / 'configs/fair_video_cases.json'
PLAN = PROJECT / '多视频同输入公平对照方案.md'
LAUNCHER = PROJECT / 'scripts/run_fair_video.py'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sources():
    return {**{p.relative_to(REPO).as_posix(): file_hash(p) for p in sorted(REPO.rglob('*.py'))},
            '@launcher': file_hash(LAUNCHER), '@plan': file_hash(PLAN), '@cases': file_hash(CONFIG),
            **{'@reference:' + name: digest for name, digest in source_assets().items()}}


def request(args):
    return {key: str(value.resolve()) if isinstance(value, Path) else value for key, value in vars(args).items()
            if key not in ('stage', 'arm', 'out', 'preflight')}


def histories(args):
    return {'historical_coffee': args.coffee_events, 'historical_faces': args.overload_events}


def input_assets(args):
    return {str(p.resolve()): file_hash(p) for p in [args.video_dir / 'clean_hands.webm',
        args.video_dir / 'cat_play.webm', args.coffee_events, args.overload_events]}


def budget(args):
    return {'model_loads_per_arm': 1, 'cases': 14, 'real_frames_total_max': 56,
        'real_frames_per_case_max': 4, 'answer_tokens_per_case': 16, 'generated_tokens_per_arm_max': 224,
        'original_vision_max': 14, 'original_language_max': 238,
        'streaming_vision_max': 27, 'streaming_language_max': 280,
        'tokens_max': 256, 'question_reserve': 96, 'precision': 'fp32_ieee_math',
        'wall_seconds': args.wall_seconds, 'gpu_free_gib_min': 5.5, 'ram_available_gib_min': 3,
        'rgb_per_case_max_bytes': 589824, 'simultaneous_real_models': 1}


def cases():
    value = read(CONFIG)
    specs = value['cases']
    if len(specs) != 14 or len({c['id'] for c in specs}) != 14 or value['answer_tokens'] != 16:
        raise ValueError('Fixed registered configuration changed')
    return specs


@torch.inference_mode()
def stream_answer(model, processor, frames, question):
    events = []
    session = OnlinePairSession(model, processor, max_tokens=256, max_question_tokens=96,
                                max_answer_tokens=16, resize_mode='letterbox', video_logits='none')
    # Retain scalar answer/audit metadata only, not caches or image events.
    def sink(event):
        if event['kind'] == 'answer':
            events.append({'answer': event['answer'], 'main_unchanged_bitwise': event['main_unchanged_bitwise'],
                           'main_counters_unchanged': event['main_counters_unchanged']})
    demo = PairDemo(session, question, 16, sink, audit=True, first_qa_frames=len(frames), qa_every_frames=600)
    try:
        for packet in frames:
            demo.append(packet)
        demo.finish()
        if len(events) != 1 or demo.resets != 0:
            raise AssertionError('One fresh case must produce one QA without hidden history reset')
        return {**events[0]['answer'], 'main_unchanged_bitwise': events[0]['main_unchanged_bitwise'],
                'main_counters_unchanged': events[0]['main_counters_unchanged'],
                'memory_from_pts': demo.memory_from_pts, 'resets': demo.resets}
    finally:
        demo.close()


@torch.inference_mode()
def original_answer(model, processor, inputs):
    """Unmodified core whole prefill then single-token cached decode, same EOS budget."""
    model.model.rope_deltas = None
    device = model.get_input_embeddings().weight.device
    values = inputs.to(device)
    with fp32_reference():
        output = model(**values, use_cache=True, logits_to_keep=1)
        logits, cache = output.logits, output.past_key_values
        count = values.input_ids.shape[1]
        rope = int(values.position_ids[1:].max()) + 1
        eos = model.generation_config.eos_token_id
        eos_ids = set([] if eos is None else eos if isinstance(eos, list) else [eos])
        eos_ids.add(processor.tokenizer.eos_token_id)
        generated, reason = [], 'budget'
        del output, values
        for _ in range(16):
            token = int(logits[0, -1].argmax())
            generated.append(token)
            point = torch.tensor([[[count]], [[rope]], [[rope]], [[rope]]], device=device)
            output = model(input_ids=torch.tensor([[token]], device=device), position_ids=point,
                attention_mask=torch.ones((1, count+1), dtype=torch.long, device=device),
                cache_position=torch.tensor([count], device=device), past_key_values=cache,
                use_cache=True, logits_to_keep=1)
            logits, cache = output.logits, output.past_key_values
            count, rope = count+1, rope+1
            del output
            if token in eos_ids:
                reason = 'eos'
                break
    return {'generated_ids': generated, 'answer': processor.tokenizer.decode(generated, skip_special_tokens=True),
            'stop_reason': reason, 'reference_mode': 'fp32_ieee_math'}


@torch.inference_mode()
def preflight(args):
    from tests.test_online import small_model
    from tests.test_fair_inputs import FairInputTests
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    FairInputTests.processor = processor
    with (args.out / 'unittest.log').open('w', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(FairInputTests))
    write_json(args.out / 'cpu_tests.json', {'tests': result.testsRun, 'passed': result.wasSuccessful(),
                                          'failures': len(result.failures), 'errors': len(result.errors)})
    if not result.wasSuccessful():
        raise AssertionError('CPU fair-input tests failed')
    assets = verify_assets(args.model_dir)
    current = small_model(processor)
    package = original_package()
    original = package.Qwen3_5ForConditionalGeneration(package.Qwen3_5Config(**current.config.to_dict())).eval()
    original.set_attn_implementation('sdpa')
    original.load_state_dict(current.state_dict(), strict=True)
    entries = []
    for spec in cases():
        frames, provenance = load_case(spec, args.video_dir, histories(args))
        whole, checked = identity(original, processor, frames, spec['question'])
        with fp32_reference():
            left = original(**whole, use_cache=False, logits_to_keep=1).logits
            right = current(**whole, use_cache=False, logits_to_keep=1).logits
            if not torch.equal(left, right):
                raise AssertionError('Fresh tiny original/modified whole prefill differs')
        answer = stream_answer(current, processor, frames, spec['question'])
        assert answer['main_unchanged_bitwise'] and answer['main_counters_unchanged']
        entries.append({**provenance, **checked, 'tiny_fresh_prefill_bitwise_equal': True,
                        'tiny_streaming_main_unchanged': True})
        del frames, whole, left, right, answer
    write_json(args.out / 'preflight_summary.json', {'cases': entries, 'cpu_fair_cases_passed': len(entries),
        'scope': 'Random tiny weights: input, fresh prefill and fork contracts only; no semantic evidence'})
    write_json(args.out / 'proof.json', {'request': request(args), 'budget': budget(args),
        'source_hashes': sources(), 'inputs': input_assets(args), 'assets': assets['files'],
        'model_manifest_sha256': file_hash(args.model_dir / 'DOWNLOAD_MANIFEST.json'),
        'cases': entries, 'summary_sha256': file_hash(args.out / 'preflight_summary.json'),
        'cpu_tests_sha256': file_hash(args.out / 'cpu_tests.json')})
    print(json.dumps({'preflight_cases': len(entries), 'tests': result.testsRun, 'passed': True}), flush=True)


def check_proof(args):
    if args.preflight is None or read(args.preflight / 'status.json') != {'status': 'completed', 'stage': 'preflight', 'arm': 'both'}:
        raise ValueError('Completed fresh CPU proof required')
    proof = read(args.preflight / 'proof.json')
    if proof['request'] != request(args) or proof['budget'] != budget(args) or proof['source_hashes'] != sources():
        raise ValueError('Source, configuration, request or registered budget changed')
    if proof['inputs'] != input_assets(args) or proof['assets'] != verify_assets(args.model_dir)['files']:
        raise ValueError('Video/history/model assets changed')
    if proof['model_manifest_sha256'] != file_hash(args.model_dir / 'DOWNLOAD_MANIFEST.json'):
        raise ValueError('Model manifest changed')
    if proof['summary_sha256'] != file_hash(args.preflight / 'preflight_summary.json') or proof['cpu_tests_sha256'] != file_hash(args.preflight / 'cpu_tests.json'):
        raise ValueError('CPU evidence changed')
    if not read(args.preflight / 'cpu_tests.json')['passed'] or len(proof['cases']) != 14:
        raise ValueError('CPU cases failed or incomplete')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=REPO, text=True).strip()
    if not branch_allowed(branch, args.allow_branch):
        raise ValueError('GPU requires allowed personal branch')
    write_json(args.out / 'checked_branch.json', {'branch': branch})
    return proof


@torch.inference_mode()
def gpu(args):
    proof = check_proof(args)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    begin = time.perf_counter()
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
        raise RuntimeError('Original arm did not load frozen original source')
    if torch.cuda.mem_get_info()[0] < 2 * 2**30:
        raise RuntimeError('Insufficient room for FP32 conversion')
    model.float()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    counts = {'vision': 0, 'language': 0}
    registered = budget(args)
    def guard_hook(name):
        def count(_module, _inputs):
            counts[name] += 1
            if counts[name] > registered[args.arm + '_' + name + '_max']:
                raise RuntimeError('Registered forward budget exceeded')
        return count
    hooks = [model.model.visual.register_forward_pre_hook(guard_hook('vision')),
             model.model.language_model.register_forward_pre_hook(guard_hook('language'))]
    torch.cuda.reset_peak_memory_stats()
    report = {'arm': args.arm, 'budget': registered, 'model_source': inspect.getfile(type(model)),
              'cases': [], 'scope': 'Fixed two-video and labelled splice capability diagnosis; no training/speed claim'}
    generated = 0
    try:
        with (args.out / 'answers.jsonl').open('x', encoding='utf-8') as stream:
            for spec, expected in zip(cases(), proof['cases']):
                if time.perf_counter() - begin >= args.wall_seconds:
                    raise TimeoutError('Registered wall budget exceeded')
                frames, provenance = load_case(spec, args.video_dir, histories(args))
                whole, checked = identity(model, processor, frames, spec['question'])
                if {**provenance, **checked, 'tiny_fresh_prefill_bitwise_equal': True, 'tiny_streaming_main_unchanged': True} != expected:
                    raise AssertionError('Actual case input/source differs from CPU proof')
                before = counts.copy()
                torch.cuda.synchronize()
                start = time.perf_counter()
                answer = original_answer(model, processor, whole) if args.arm == 'original' else stream_answer(model, processor, frames, spec['question'])
                torch.cuda.synchronize()
                generated += len(answer['generated_ids'])
                if generated > registered['generated_tokens_per_arm_max']:
                    raise RuntimeError('Generated-token budget exceeded')
                record = {'id': spec['id'], 'question': spec['question'], 'kind': spec['kind'],
                    **provenance, **checked, **answer, 'expected': spec.get('expected'), 'rubric': spec.get('rubric'),
                    'strict_choice_correct': None if 'expected' not in spec else answer['answer'].strip() == spec['expected'],
                    'model_seconds': time.perf_counter() - start,
                    'forward_counts': {key: counts[key] - before[key] for key in counts}}
                report['cases'].append(record)
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                print(json.dumps({'arm': args.arm, 'id': spec['id'], 'answer': answer['answer']}, ensure_ascii=False), flush=True)
                write_json(args.out / 'summary.json', report)
                del frames, whole, answer
        report.update(actual_forwards=counts, generated_tokens=generated,
            all_registered_cases_executed=len(report['cases']) == 14,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            wall_seconds_including_load=time.perf_counter() - begin)
        write_json(args.out / 'summary.json', report)
    finally:
        for hook in hooks:
            hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preflight', 'gpu'), required=True)
    parser.add_argument('--arm', choices=('both', 'original', 'streaming'), default='both')
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--video-dir', type=Path, required=True)
    parser.add_argument('--coffee-events', type=Path, default=PROJECT / 'runs/generic_demo_gpu_20261011_01/events.jsonl')
    parser.add_argument('--overload-events', type=Path, default=PROJECT / 'runs/clock_overload_gpu_20261011_01/events.jsonl')
    parser.add_argument('--wall-seconds', type=int, default=240)
    parser.add_argument('--allow-branch', action='append', default=[])
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.wall_seconds <= 240 or (args.stage == 'preflight') != (args.arm == 'both'):
        parser.error('CPU arm=both; GPU arm=original or streaming; wall<=240')
    for name in ('model_dir', 'video_dir', 'coffee_events', 'overload_events', 'out'):
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
