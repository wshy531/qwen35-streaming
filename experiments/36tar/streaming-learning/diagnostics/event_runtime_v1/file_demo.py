"""Optional bounded event-memory video-file entry. Preflight before each GPU configuration."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
EVENT = HERE.parent / 'event_memory_v1'
RETRIEVAL = HERE.parent / 'event_retrieval_v1'
sys.path.insert(0, str(EVENT)); sys.path.insert(0, str(RETRIEVAL))
spec = importlib.util.spec_from_file_location('event_reference', EVENT / 'probe.py')
events = importlib.util.module_from_spec(spec); spec.loader.exec_module(events)
import torch
import psutil
import numpy as np
from transformers import AutoProcessor
from runtime import EventRuntime, bank_fingerprint
from routing import select
from streaming.online import OnlinePairSession
from streaming.window import FramePacket, read_packets
from streaming.preprocessing import fit_canvas
from scripts.baseline import write_json, file_hash, cache_measure
from scripts.pair_demo import verify_assets


def sources():
    return {**events.sources(), **{str(p.resolve()): file_hash(p) for p in RETRIEVAL.glob('*.py')},
            **{str(p.resolve()): file_hash(p) for p in HERE.glob('*.py')},
            str(RETRIEVAL / '方案.md'): file_hash(RETRIEVAL / '方案.md'),
            str(HERE / '方案.md'): file_hash(HERE / '方案.md')}


def request(args):
    return {name: str(getattr(args, name)) if isinstance(getattr(args, name), Path) else getattr(args, name)
            for name in ('model_dir', 'video', 'fps', 'duration', 'max_frames', 'max_tokens',
                         'query_every_pairs', 'question', 'history_question', 'threshold', 'branch', 'check_time')}


def budget(args):
    pairs = args.max_frames // 2
    questions = pairs // args.query_every_pairs + 1 + int(bool(args.history_question))
    return {'model_loads': 1, 'vision_max': pairs, 'questions_max': questions, 'generated_max': questions * 32,
            'language_max': pairs * 2 + questions * 33 + 9, 'wall_seconds': 360,
            'answer_tokens': 32, 'main_max_tokens': args.max_tokens, 'qa_max_tokens': 512,
            'bank_visual_tokens_max': 144, 'bank_tensor_bytes_max': 592704,
            'cpu_feature_temporaries_bound_bytes': 4 * 2**20, 'rgb_working_estimate_bound_bytes': 2**20,
            'ram_min_gib': 3, 'gpu_free_min_gib': 5.5, 'conversion_free_min_gib': 2,
            'file_bytes_max': 64 * 2**20}


def make(model, processor, args, threshold=None):
    session = OnlinePairSession(model, processor, max_tokens=args.max_tokens, max_question_tokens=64,
                                max_answer_tokens=32, resize_mode='letterbox', video_logits='none')
    return EventRuntime(session, args.threshold if threshold is None else threshold)


class Contracts(unittest.TestCase):
    def packet(self, i):
        return FramePacket(i, float(i), self.pixels[i % len(self.pixels)].copy(), i + 1)

    def test_100_frame_bounded_resets_and_query_then_continue(self):
        wrapper = make(self.model, self.processor, self.args, threshold=0.)
        try:
            for i in range(100):
                row = wrapper.feed(self.packet(i))
                if row['encoded']:
                    self.assertLessEqual(wrapper.bank.status()['tokens'], 144)
                    self.assertLessEqual(wrapper.bank.status()['tensor_bytes'], 592704)
                    self.assertLessEqual(wrapper.session.main.seen_tokens, wrapper.session.main_budget)
                    if i % 20 == 19:
                        before = (events.old_visual.cache_fingerprint(wrapper.session.main), bank_fingerprint(wrapper.bank))
                        wrapper.ask('What is visible?', max_new_tokens=2)
                        self.assertEqual(before, (events.old_visual.cache_fingerprint(wrapper.session.main), bank_fingerprint(wrapper.bank)))
            self.assertEqual(wrapper.encoded_pairs, 50)
            self.assertGreater(wrapper.capacity_resets, 0)
            self.assertEqual(wrapper.bank.last_id, 49)
        finally:
            wrapper.close(); self.assertEqual(wrapper.bank.status()['tensor_bytes'], 0)

    def test_explicit_reset_preserves_history_and_refuses_stale_current(self):
        wrapper = make(self.model, self.processor, self.args, threshold=0.)
        try:
            for i in range(6): wrapper.feed(self.packet(i))
            self.assertGreater(len(wrapper.bank.groups), 0)
            before = bank_fingerprint(wrapper.bank)
            wrapper.reset_main()
            self.assertEqual(bank_fingerprint(wrapper.bank), before)
            self.assertEqual(cache_measure(wrapper.session.main.cache)['total_tensor_bytes'], 0)
            self.assertEqual(wrapper.ask('What is visible?')['reason'], 'no_current_pair_in_epoch')
            self.assertTrue(wrapper.ask('What is visible?', 'group', 0, 2)['model_called'])
            self.assertEqual(bank_fingerprint(wrapper.bank), before)
            wrapper.feed(self.packet(6)); wrapper.feed(self.packet(7))
            self.assertTrue(wrapper.ask('What is visible?', max_new_tokens=2)['model_called'])
        finally: wrapper.close()

    def test_pending_and_odd_eof_are_not_synthetic_frames(self):
        wrapper = make(self.model, self.processor, self.args)
        try:
            for i in range(3): wrapper.feed(self.packet(i))
            self.assertEqual(wrapper.ask('What is visible?')['reason'], 'waiting_for_complete_pair')
            with self.assertRaises(ValueError): wrapper.reset_main()
            final = wrapper.finish()
            self.assertEqual(final['encoded_pairs'], 1)
            self.assertEqual(final['encoded_horizon'], 1.)
            self.assertEqual(final['last_received_pts'], 2.)
            self.assertEqual(final['dropped_odd_eof_frame']['frame_id'], 2)
            self.assertEqual(wrapper.ask('What is visible?', 'exact', 2.)['reason'], 'beyond_encoded_horizon')
            self.assertTrue(wrapper.ask('What is visible?', max_new_tokens=2)['model_called'])
            with self.assertRaises(ValueError): wrapper.feed(self.packet(3))
        finally: wrapper.close()

    def test_global_horizon_and_invalid_parameter_rejection(self):
        wrapper = make(self.model, self.processor, self.args)
        try:
            wrapper.feed(self.packet(0)); wrapper.feed(self.packet(1)); wrapper.reset_main()
            before = wrapper.status()
            with self.assertRaises(ValueError): wrapper.feed(self.packet(1))
            self.assertEqual(wrapper.status(), before)
            for kwargs in ({'max_new_tokens': 33}, {'mode': 'group', 'target': -1}, {'mode': 'exact', 'target': float('nan')}):
                with self.assertRaises(ValueError): wrapper.ask('What is visible?', **kwargs)
            self.assertEqual(wrapper.status(), before)
        finally: wrapper.close()
        with self.assertRaises(RuntimeError): wrapper.ask('What is visible?')

    def test_main_failure_does_not_commit_bank_and_hook_removed(self):
        wrapper = make(self.model, self.processor, self.args)
        try:
            wrapper.feed(self.packet(0)); hook_count = len(self.model.model.visual._forward_hooks)
            with patch.object(wrapper.session, 'commit_pair', side_effect=RuntimeError('deliberate write failure')):
                with self.assertRaises(RuntimeError): wrapper.feed(self.packet(1))
            self.assertEqual(wrapper.bank.last_id, -1)
            self.assertEqual(wrapper.bank.status()['tensor_bytes'], 0)
            self.assertEqual(len(self.model.model.visual._forward_hooks), hook_count)
            with self.assertRaises(RuntimeError): wrapper.ask('What is visible?')
        finally: wrapper.close()

    def test_pending_is_owned_and_source_aspect_changes_are_normalized(self):
        wrapper = make(self.model, self.processor, self.args)
        try:
            packet = self.packet(0); wrapper.feed(packet)
            saved = wrapper.pending.rgb.copy(); packet.rgb.fill(0)
            self.assertTrue(np.array_equal(wrapper.pending.rgb, saved))
            portrait = FramePacket(1, 1., self.pixels[1][:128, :64].copy(), 2)
            wrapper.feed(portrait)
            self.assertEqual(wrapper.session.frame_shape, (192, 256, 3))
            self.assertEqual(wrapper.encoded_pairs, 1)
        finally: wrapper.close()


@torch.inference_mode()
def replay(model, processor, args, expected_input=None):
    limits = budget(args); wrapper = make(model, processor, args)
    counts = {'vision': 0, 'language': 0}
    answers, inputs, ledgers, updates, checks = [], [], [], [], []
    generated = 0; last_asked_pair = -1
    feature_temporary_estimate_peak = rgb_estimate_peak = 0
    def guard(name, limit):
        def hook(_module, _args):
            counts[name] += 1
            if counts[name] > limit: raise RuntimeError('Forward budget exceeded')
        return hook
    hooks = [model.model.visual.register_forward_pre_hook(guard('vision', limits['vision_max'])),
             model.model.language_model.register_forward_pre_hook(guard('language', limits['language_max']))]
    log = (args.out / 'events.jsonl').open('x', encoding='utf-8')
    reader = read_packets(args.video, args.fps)

    def emit(row):
        log.write(json.dumps(row, ensure_ascii=False) + '\n'); log.flush()
        if row['kind'] in ('answer', 'reset', 'finish'):
            print(json.dumps(row, ensure_ascii=False), flush=True)

    def query(kind, mode='current', target=None, question=None):
        nonlocal generated, last_asked_pair
        before = (events.old_visual.cache_fingerprint(wrapper.session.main), wrapper.status(), bank_fingerprint(wrapper.bank), list(wrapper.session.ledger))
        result = wrapper.ask(question or args.question, mode, target)
        if (events.old_visual.cache_fingerprint(wrapper.session.main), wrapper.status(), bank_fingerprint(wrapper.bank), list(wrapper.session.ledger)) != before:
            raise AssertionError('Question polluted main/bank/status')
        called = bool(result.get('model_called', False))
        if called:
            generated += len(result['result']['generated_ids'])
            if generated > limits['generated_max'] or sum(int(a['result'].get('model_called', False)) for a in answers) + 1 > limits['questions_max']:
                raise RuntimeError('QA generation budget exceeded')
        row = {'kind': 'answer', 'trigger': kind, 'result': result,
               'status': wrapper.status(), 'main_and_bank_unchanged': True}
        answers.append(row); emit(row)
        if mode == 'current': last_asked_pair = wrapper.encoded_pairs

    try:
        for packet in reader:
            if packet.timestamp > args.duration: break
            pixels = fit_canvas(packet.rgb, (192, 256))
            identity = {'source_frame_id': packet.frame_id, 'source_pts': packet.timestamp,
                        'source_rgb_sha256': events.hashlib.sha256(packet.rgb.tobytes()).hexdigest(),
                        'canvas_sha256': events.hashlib.sha256(pixels.tobytes()).hexdigest()}
            result = wrapper.feed(packet); inputs.append(identity)
            # Estimate wrapper input/pending/canonical copies; decoder internals excluded.
            rgb_estimate_peak = max(rgb_estimate_peak, packet.rgb.nbytes + pixels.nbytes + 3 * 192 * 256 * 3)
            feature_temporary_estimate_peak = max(feature_temporary_estimate_peak,
                wrapper.bank.status()['tensor_bytes'] + 8 * 197568 + 8192)
            if feature_temporary_estimate_peak > 4 * 2**20 or rgb_estimate_peak > 2**20:
                raise RuntimeError('Small-tensor temporary estimate exceeds preregistered budget')
            if result['encoded']:
                ledgers.append(wrapper.session.ledger[-1])
                updates.append({'encoded_pairs': wrapper.encoded_pairs, 'source_pts': packet.timestamp,
                                'feature_sha256': result['feature_sha256'], 'group_update': result['group_update'],
                                'reset': result['reset']})
                if result['reset'] is not None: emit({'kind': 'reset', **result['reset'], 'status': wrapper.status()})
                if wrapper.encoded_pairs % args.query_every_pairs == 0: query('periodic')
            if wrapper.received >= args.max_frames: break
        emit({'kind': 'finish', 'status': wrapper.finish()})
        if wrapper.encoded_pairs != last_asked_pair: query('final')
        if args.history_question: query('first_closed_group', 'group', 0, args.history_question)
        for target in (args.check_time, (wrapper.bank.last_end or 0) + 100):
            metadata, rows = select(wrapper.bank, 'exact', target)
            checks.append({'target': target, 'selection': metadata, 'model_called': False})
            rows.clear()
        actual_input = {'frames': inputs, 'pair_ledgers': ledgers,
                        'capacity_reset_pair_ids': [u['encoded_pairs'] for u in updates if u['reset'] is not None],
                        'odd_eof_frame': wrapper.dropped_eof}
        if expected_input is not None and actual_input != expected_input:
            raise ValueError('CPU/GPU input pixels/template/position/reset identity differs')
        if counts['vision'] != wrapper.encoded_pairs: raise AssertionError('QA reencoded visual frames')
        return {'passed': True, 'counts': counts, 'model_called_queries': sum(int(a['result'].get('model_called', False)) for a in answers),
                'generated_tokens': generated, 'answers': answers, 'checks': checks,
                'input_identity': actual_input, 'updates': updates, 'final': wrapper.status(),
                'cpu_feature_temporary_estimate_peak_bytes': feature_temporary_estimate_peak,
                'rgb_working_estimate_peak_bytes': rgb_estimate_peak,
                'no_qa_visual_reencoding': True, 'default_integration': False,
                'scope': 'Finite sequential natural file entry, not wall-clock realtime or unlimited-history accuracy'}
    finally:
        reader.close(); log.close(); wrapper.close()
        for hook in hooks: hook.remove()


@torch.inference_mode()
def preflight(args):
    from tests.test_online import small_model
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = small_model(processor)
    # Synthetic fixtures make contracts independent of source length/content.
    # The separate replay below still verifies the actual source and all pixels.
    yy, xx = np.indices((192, 256))
    pixels = [np.stack(((xx + i * 37) % 256, (yy * 2 + i * 89) % 256,
                        (xx + yy + i * 17) % 256), axis=-1).astype(np.uint8) for i in range(6)]
    Contracts.model, Contracts.processor, Contracts.args, Contracts.pixels = model, processor, args, pixels
    with (args.out / 'unittest.log').open('x', encoding='utf-8') as log:
        tests = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not tests.wasSuccessful(): raise AssertionError('Runtime CPU contracts failed')
    report = replay(model, processor, args)
    report.update(cpu_tests=tests.testsRun, source_hashes=sources(), request=request(args), budget=budget(args),
                  source_video_sha256=file_hash(args.video), assets=verify_assets(args.model_dir)['files'],
                  model_manifest_sha256=file_hash(args.model_dir / 'DOWNLOAD_MANIFEST.json'),
                  scope='Tiny random CPU model: real source identity, runtime/reset/query contracts, no ability evidence')
    write_json(args.out / 'proof.json', report)
    print(json.dumps({'passed': True, 'cpu_tests': tests.testsRun, 'source_frames': len(report['input_identity']['frames']),
                      'replay_vision': report['counts']['vision'], 'gpu_loads': 0}), flush=True)


@torch.inference_mode()
def gpu(args):
    if args.preflight is None or events.read(args.preflight / 'status.json') != {'status': 'completed', 'stage': 'preflight'}:
        raise ValueError('Fresh CPU proof for this source/configuration required')
    proof = events.read(args.preflight / 'proof.json')
    if not proof['input_identity']['pair_ledgers']:
        raise ValueError('No complete source pair in proof; model load unnecessary')
    if (proof['source_hashes'] != sources() or proof['request'] != request(args) or proof['budget'] != budget(args) or
        proof['source_video_sha256'] != file_hash(args.video) or proof['assets'] != verify_assets(args.model_dir)['files'] or
        proof['model_manifest_sha256'] != file_hash(args.model_dir / 'DOWNLOAD_MANIFEST.json')):
        raise ValueError('Source/configuration/video/model proof drift')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=events.CAP, text=True).strip()
    if args.branch in ('', 'main', 'master', 'HEAD') or branch != args.branch: raise ValueError('Configured personal branch required')
    free, total = torch.cuda.mem_get_info(); ram = psutil.virtual_memory().available
    write_json(args.out / 'resource_precheck.json', {'ram_available_bytes': ram, 'gpu_free_bytes': free,
                                                    'gpu_total_bytes': total, 'branch': branch})
    if ram < 3 * 2**30 or free < 5.5 * 2**30: raise RuntimeError('Resource gate refused model load')
    from qwen35_local import Qwen3_5ForConditionalGeneration
    begin = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir, local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    if torch.cuda.mem_get_info()[0] < 2 * 2**30: raise RuntimeError('No FP32 conversion room')
    model.float(); torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    report = replay(model, processor, args, proof['input_identity'])
    report.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                  wall_seconds_including_load=time.perf_counter() - begin)
    write_json(args.out / 'summary.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preflight', 'gpu'), required=True)
    for name in ('model-dir', 'video', 'out'): parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--branch', required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--fps', type=float, default=1.)
    parser.add_argument('--duration', type=float, default=25.)
    parser.add_argument('--max-frames', type=int, default=40)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--query-every-pairs', type=int, default=4)
    parser.add_argument('--question', default='Briefly describe what is happening most recently.')
    parser.add_argument('--history-question', default='Briefly describe the visible action in these frames.')
    parser.add_argument('--check-time', type=float, default=.5)
    args = parser.parse_args()
    for name in ('model_dir', 'video', 'out', 'preflight'):
        value = getattr(args, name)
        if value is not None: setattr(args, name, value.resolve())
    if (not args.video.is_file() or args.video.stat().st_size > 64 * 2**20 or
        not math.isfinite(args.fps) or not 0 < args.fps <= 4 or not math.isfinite(args.duration) or not 0 < args.duration <= 600 or
        not 2 <= args.max_frames <= 120 or not 192 <= args.max_tokens <= 512 or not 1 <= args.query_every_pairs <= 60 or
        not math.isfinite(args.threshold) or not 0 <= args.threshold <= 2 or not math.isfinite(args.check_time) or args.check_time < 0):
        parser.error('Finite bounded source/configuration required')
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'tmp').mkdir(); tempfile.tempdir = str(args.out / 'tmp')
    torch.set_num_threads(1); torch.manual_seed(42)
    write_json(args.out / 'source_hashes.json', sources()); write_json(args.out / 'budget.json', budget(args))
    try:
        (preflight if args.stage == 'preflight' else gpu)(args)
        write_json(args.out / 'status.json', {'status': 'completed', 'stage': args.stage})
    except Exception as error:
        write_json(args.out / 'status.json', {'status': 'failed', 'stage': args.stage,
                   'error': type(error).__name__, 'message': str(error)})
        (args.out / 'error.log').write_text(traceback.format_exc(), encoding='utf-8'); raise


if __name__ == '__main__': main()
