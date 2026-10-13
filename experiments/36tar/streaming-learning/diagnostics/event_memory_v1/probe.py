"""Frozen same-budget event-representative controls, including natural file continuation."""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
import weakref

HERE = Path(__file__).resolve().parent
CAP = HERE.parents[1]
ACTIVE = CAP / 'code/qwen35-streaming-s4'
VISUAL = HERE.parent / 'visual_memory_v1'
sys.path.insert(0, str(VISUAL))
sys.path.insert(0, str(ACTIVE))
import psutil
import torch
from transformers import AutoProcessor
from scripts.baseline import file_hash, write_json
from scripts.pair_demo import verify_assets
from streaming.online import OnlinePairSession, PairInputBuilder, fp32_reference
from streaming.hybrid import HybridSequence
from streaming.window import FramePacket, read_packets
from streaming.preprocessing import fit_canvas
import experiment as old_visual
from features import raw_block, ShortLongMemory, context, answer
from memory import EventMemory, distance, fit_threshold


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sources():
    paths = [*ACTIVE.rglob('*.py'), *HERE.glob('*.py'), HERE / 'cases.json', HERE / '方案.md',
             VISUAL / 'features.py', VISUAL / 'experiment.py',
             CAP / 'diagnostics/general_stream_v1/cases.json',
             CAP / 'diagnostics/event_forest_v1/reference/memory_manager.py']
    return {str(p.resolve()): file_hash(p) for p in sorted(paths)}


def budget():
    return {'model_loads': 1, 'vision_max': 24, 'questions_max': 60, 'generated_max': 1920,
            'language_max': 2200, 'answer_tokens': 32, 'max_tokens': 512, 'wall_seconds': 480,
            'ram_min_gib': 3, 'gpu_free_min_gib': 5.5, 'conversion_free_min_gib': 2,
            'bank_visual_tokens_max': 144, 'bank_tensor_bytes_max': 596800,
            'owned_feature_controls_bytes_max': 16 * 2**20, 'metadata_bytes_per_bank_max': 4096,
            'precision': 'fp32_ieee_math', 'calibration': 'six fixed pairs, unlabeled largest gap',
            'spatial_compression': 'none'}


def request(args):
    return {'model_dir': str(args.model_dir), 'branch': args.branch}


def segment(row, start, count, natural):
    path = Path(row['path'])
    if file_hash(path) != row['sha256']:
        raise ValueError('Registered source SHA changed')
    reader = read_packets(path, 1.)
    packets, rows = [], []
    try:
        for _ in range(start):
            next(reader)
        for index in range(count):
            old = next(reader)
            pixels = fit_canvas(old.rgb, (192, 256))
            stamp = old.timestamp if natural else float(index)
            packets.append(FramePacket(index, stamp, pixels, index + 1, old.decode_seconds))
            rows.append({'session_frame_id': index, 'session_pts': stamp, 'source_id': row['id'],
                         'source_frame_id': old.frame_id, 'source_pts': old.timestamp,
                         'source_sha256': row['sha256'],
                         'source_rgb_sha256': hashlib.sha256(old.rgb.tobytes()).hexdigest(),
                         'canvas_rgb_sha256': hashlib.sha256(pixels.tobytes()).hexdigest(),
                         'session_time_is_source_time': natural})
    finally:
        reader.close()
    return [packets[i:i + 2] for i in range(0, count, 2)], rows


def inputs():
    cfg = read(HERE / 'cases.json')
    animal_sources = read(CAP / 'diagnostics/general_stream_v1/cases.json')['videos']
    result, identity, assets = {}, {}, {}
    for split in ('animal_calibration', 'animal_holdout'):
        result[split], identity[split] = {}, {}
        for name, start in cfg[split].items():
            row = next(r for r in animal_sources if r['id'] == name)
            pairs, rows = segment(row, start, 4, False)
            result[split][name], identity[split][name] = pairs, rows
            assets[row['path']] = row['sha256']
    result['natural'], identity['natural'] = {}, {}
    for row in cfg['natural_sources']:
        pairs, rows = segment(row, row['start_sample'], row['frames'], True)
        result['natural'][row['id']], identity['natural'][row['id']] = pairs, rows
        assets[row['path']] = row['sha256']
    # Assert the calibration and animal evaluation frames never overlap.
    for name in cfg['animal_calibration']:
        a = {r['source_frame_id'] for r in identity['animal_calibration'][name]}
        b = {r['source_frame_id'] for r in identity['animal_holdout'][name]}
        if a & b:
            raise ValueError('Calibration/evaluation source frames overlap')
    return result, identity, assets


def ordered_pairs(pool, order):
    rows = []
    for name in order:
        for pair in pool[name]:
            rows.append([FramePacket(2 * len(rows) + j, float(2 * len(rows) + j), p.rgb,
                                     2 * len(rows) + j + 1, p.decode_seconds) for j, p in enumerate(pair)])
    return rows


def reindex_blocks(pool, order):
    rows = []
    for name in order:
        for block in pool[name]:
            i = len(rows)
            rows.append(replace(block, first=i, last=i, begin=float(2 * i), end=float(2 * i + 1),
                                center=2 * i + .5))
    return rows


class Banks:
    def __init__(self, threshold, oracle=False):
        self.fifo = []
        self.mean = ShortLongMemory()
        self.event = EventMemory(threshold)
        self.oracle = EventMemory(threshold, (0, 2, 4)) if oracle else None

    def append(self, block):
        self.fifo.append(block.copy())
        if len(self.fifo) > 3:
            self.fifo.pop(0)
        self.mean.append(block)
        trace = self.event.append(block)
        if self.oracle is not None:
            self.oracle.append(block)
        return trace

    def export(self):
        result = {'fifo': [b.copy() for b in self.fifo], 'mean': self.mean.export(), 'event': self.event.export()}
        if self.oracle is not None:
            result['oracle_diagnostic'] = self.oracle.export()
        return result

    def status(self):
        result = {'fifo': {'tokens': sum(len(b.features) for b in self.fifo),
                           'tensor_bytes': sum(b.metadata()['tensor_bytes'] for b in self.fifo),
                           'blocks': [b.metadata() for b in self.fifo]},
                  'mean': self.mean.status(), 'event': self.event.status()}
        if self.oracle is not None:
            result['oracle_diagnostic'] = self.oracle.status()
        return result

    def clear(self):
        self.fifo.clear()
        self.mean.clear()
        self.event.clear()
        if self.oracle is not None:
            self.oracle.clear()


def memory_fingerprint(banks):
    exports = banks.export()
    return {'status': banks.status(), 'tensors': {name: [[old_visual.tensor_sha(b.features),
                old_visual.tensor_sha(b.mass), old_visual.tensor_sha(b.xy)] for b in blocks]
                for name, blocks in exports.items()}}


def audit_budget(banks, exports=(), pool=()):
    for row in banks.status().values():
        if row['tokens'] > 144 or row['tensor_bytes'] > 596800 or row.get('metadata_json_bytes', 0) > 4096:
            raise RuntimeError('Bank budget exceeded')
    # Explicit conservative accounting counts each owned export and pool copy.
    resident = sum(row['tensor_bytes'] for row in banks.status().values())
    copies = sum(b.metadata()['tensor_bytes'] for group in exports for b in group)
    control = sum(b.metadata()['tensor_bytes'] for b in pool)
    total = resident + copies + control
    if total > 16 * 2**20:
        raise RuntimeError('Owned CPU feature controls exceeded budget')
    return total


def animal_words(text):
    return ['bird' if word in ('bird', 'birds', 'pigeon', 'pigeons') else
            'dog' if word in ('dog', 'dogs') else 'cat'
            for word in re.findall(r'\b(?:dog|dogs|bird|birds|pigeon|pigeons|cat|cats)\b', text.lower())]


class Contracts(unittest.TestCase):
    @staticmethod
    def block(index, axis=0, dim=16):
        x = torch.zeros((48, dim))
        x[:, axis] = 1
        return raw_block(x, index, 2 * index, 2 * index + 1)

    def test_calibration_algorithm_cannot_use_labels(self):
        blocks = [self.block(i, i // 2) for i in range(6)]
        fit = fit_threshold(blocks)
        self.assertEqual(fit['threshold'], .5)
        self.assertEqual(fit['adjacent_distances'], [0., 1., 0., 1., 0.])
        self.assertTrue(fit_threshold([self.block(i) for i in range(6)])['fallback_used'])

    def test_changes_keep_real_unblended_representatives(self):
        bank = EventMemory(.5)
        for i in range(6):
            bank.append(self.block(i, i // 2))
        self.assertEqual([b.first for b in bank.export()], [1, 3, 5])
        self.assertEqual([(g.first, g.last) for g in bank.groups], [(0, 1), (2, 3)])
        for b in bank.export():
            torch.testing.assert_close(b.features, self.block(b.first, b.first // 2).features, rtol=0, atol=0)
        self.assertEqual(bank.exact_available(.5)['reason'], 'group_span_only_no_exact_pair')
        self.assertEqual(bank.exact_available(2.5)['reason'], 'retained_real_pair_available')
        self.assertEqual(bank.exact_available(99)['reason'], 'beyond_encoded_horizon')

    def test_no_change_does_not_fake_dense_history(self):
        bank = EventMemory(.5)
        for i in range(8):
            bank.append(self.block(i))
        self.assertEqual(bank.status()['tokens'], 48)
        self.assertEqual(bank.exact_available(10.5)['reason'], 'group_span_only_no_exact_pair')
        self.assertEqual(bank.exact_available(14.5)['reason'], 'retained_real_pair_available')

    def test_512_pair_budget_and_eviction(self):
        bank = EventMemory(.5)
        for i in range(512):
            bank.append(self.block(i, i % 3))
            state = bank.status()
            self.assertLessEqual(state['tokens'], 144)
            self.assertLessEqual(state['tensor_bytes'], 144 * (16 * 4 + 20) + 16 * 4)
            self.assertLessEqual(state['metadata_json_bytes'], 4096)
        self.assertEqual(bank.evicted_groups, 509)
        self.assertEqual([b.first for b in bank.export()], [509, 510, 511])
        bank.clear()
        self.assertEqual(bank.status()['tensor_bytes'], 0)

    def test_owned_exports_and_evicted_features_release(self):
        bank = EventMemory(.5)
        original = self.block(0)
        bank.append(original)
        original.features.zero_()
        bank.append(self.block(1, 1))
        saved = bank.export()[0].features.clone()
        export = bank.export()
        export[0].features.fill_(9)
        torch.testing.assert_close(bank.export()[0].features, saved, rtol=0, atol=0)
        doomed = weakref.ref(bank.current.features)
        del export
        bank.append(self.block(2, 1))
        gc.collect()
        self.assertIsNone(doomed())

    def test_invalid_append_is_atomic_and_future_suffix_causal(self):
        a, b = EventMemory(.5), EventMemory(.5)
        for i in range(3):
            a.append(self.block(i)); b.append(self.block(i))
        before = a.status()
        for item in (self.block(2), self.block(4), replace(self.block(3), begin=1),
                     replace(self.block(3), features=torch.full((48, 16), float('nan')))):
            with self.assertRaises(ValueError):
                a.append(item)
            self.assertEqual(a.status(), before)
        self.assertEqual(a.status(), b.status())
        a.append(self.block(3, 1)); b.append(self.block(3, 2))
        self.assertEqual(a.groups[0].metadata(), b.groups[0].metadata())

    def test_zero_vectors_finite_and_oracle_explicit(self):
        z = raw_block(torch.zeros((48, 16)), 0, 0, 1)
        self.assertEqual(distance(z, z), 1.)
        oracle = EventMemory(.5, (0, 2, 4))
        for i in range(6):
            oracle.append(self.block(i))
        self.assertTrue(oracle.status()['oracle_diagnostic'])
        self.assertEqual([b.first for b in oracle.export()], [1, 3, 5])

    def test_layout_uses_real_representative_times_and_no_fake_span(self):
        bank = EventMemory(.5)
        for b in self.blocks:
            bank.append(b)
        rows = bank.export()
        _, pos, ids, metadata = context(self.model, self.builder, rows, 'What appears?', 2)
        decoded = self.builder.processor.tokenizer.decode(ids[0])
        self.assertNotIn('compressed memory', decoded)
        torch.testing.assert_close(pos[0, 0].cpu(), torch.arange(ids.shape[1]), rtol=0, atol=0)
        self.assertLessEqual(metadata['visual_tokens'], 144)


@torch.inference_mode()
def replay(model, processor, args, data, identity, expected_layout=None):
    cfg = read(HERE / 'cases.json')
    builder = PairInputBuilder(processor, resize_mode='letterbox')
    counts = {'vision': 0, 'language': 0}
    queries, layouts, natural_ledgers, bank_traces = [], {}, {}, {}
    generated = 0
    control_peak = 0
    def guard(name, limit):
        def hook(_module, _args):
            counts[name] += 1
            if counts[name] > limit:
                raise RuntimeError('Forward budget exceeded')
        return hook
    hooks = [model.model.visual.register_forward_pre_hook(guard('vision', 24)),
             model.model.language_model.register_forward_pre_hook(guard('language', 2200))]
    anchor = HybridSequence(model, max_tokens=512, session_id='event-memory-protected')
    with fp32_reference():
        anchor.append(input_ids=builder.prefix_ids.to(model.get_input_embeddings().weight.device), logits_mode='none')
    anchor_before = old_visual.cache_fingerprint(anchor)
    log = (args.out / 'answers.jsonl').open('x', encoding='utf-8')

    def query(case, phase, banks, questions, main=None, expected_order=None):
        nonlocal generated, control_peak
        exports = banks.export()
        control_peak = max(control_peak, audit_budget(banks, exports.values(), calibration_blocks + heldout_flat))
        before = memory_fingerprint(banks)
        main_before = None if main is None else (old_visual.cache_fingerprint(main.main), main.status(), list(main.ledger))
        for method, blocks in exports.items():
            for question in questions:
                result = answer(model, builder, blocks, question['text'], 32)
                generated += len(result['generated_ids'])
                if generated > 1920 or len(queries) >= 60:
                    raise RuntimeError('Question/generation budget exceeded')
                row = {'case': case, 'phase': phase, 'method': method, 'question_id': question['id'],
                       'result': result, 'bank_status': banks.status()[method],
                       'memory_unchanged': True, 'main_unchanged': True}
                if expected_order is not None:
                    words = animal_words(result['answer'])
                    expected = [expected_order[-1]] if question['id'] == 'current' else [expected_order[0]] if question['id'] == 'early' else expected_order
                    row['controlled_scoring'] = {'extracted_animal_words': words, 'expected': expected,
                                                'correct': words == expected, 'scope': 'fixed montage diagnostic only'}
                if memory_fingerprint(banks) != before or old_visual.cache_fingerprint(anchor) != anchor_before:
                    raise AssertionError('Question polluted bank or protected main')
                if main is not None and (old_visual.cache_fingerprint(main.main), main.status(), list(main.ledger)) != main_before:
                    raise AssertionError('Question polluted live natural stream')
                queries.append(row)
                log.write(json.dumps(row, ensure_ascii=False) + '\n'); log.flush()
                print(json.dumps({'case': case, 'phase': phase, 'method': method,
                                  'question': question['id'], 'answer': result['answer']}, ensure_ascii=False), flush=True)
        exports.clear()

    try:
        calibration_pairs = ordered_pairs(data['animal_calibration'], ['dog', 'pigeon', 'cat'])
        calibration_blocks, layouts['calibration'] = old_visual.encode(model, builder, calibration_pairs)
        fit = fit_threshold(calibration_blocks)
        print(json.dumps({'calibration_frozen': fit, 'evaluation_answers_seen': 0}), flush=True)
        heldout_pool, heldout_flat = {}, []
        for name, pairs in data['animal_holdout'].items():
            heldout_pool[name], layouts['animal_' + name] = old_visual.encode(model, builder, pairs)
            heldout_flat.extend(heldout_pool[name])
        for order in cfg['animal_orders']:
            case = 'controlled_' + '_'.join(order)
            banks = Banks(fit['threshold'], oracle=True)
            blocks = reindex_blocks(heldout_pool, order)
            trace = []
            for block in blocks:
                trace.append(banks.append(block))
            exports = banks.export()
            # Source labels are a post-hoc audit only, never input to grouping or QA.
            label = {i: ('bird' if order[i // 2] == 'pigeon' else order[i // 2]) for i in range(6)}
            coverage = {name: [{'retained_pair': b.first if b.leaves == 1 else None,
                               'source_subjects_in_span': sorted({label[i] for i in range(b.first, b.last + 1)}),
                               'temporal_mean': b.leaves > 1} for b in rows] for name, rows in exports.items()}
            bank_traces[case] = {'updates': trace, 'coverage': coverage, 'final': banks.status(),
                                'exact_old': banks.event.exact_available(.5),
                                'future': banks.event.exact_available(99.)}
            exports.clear()
            expected_order = ['bird' if n == 'pigeon' else n for n in order]
            query(case, 'final', banks, cfg['animal_questions'], expected_order=expected_order)
            banks.clear()

        for name, pairs in data['natural'].items():
            session = OnlinePairSession(model, processor, max_tokens=512, max_question_tokens=64,
                                        max_answer_tokens=32, resize_mode='letterbox', video_logits='none')
            banks = Banks(fit['threshold'])
            trace, hashes = [], []
            try:
                for index, pair in enumerate(pairs):
                    session.append(pair[0])
                    captured = []
                    def capture(_module, _args, output):
                        value = output.pooler_output
                        if captured or not isinstance(value, torch.Tensor) or value.shape[0] != 48:
                            raise ValueError('Exactly one projected pair required')
                        captured.append(value.detach().to('cpu', dtype=torch.float32).clone())
                    capture_hook = model.model.visual.register_forward_hook(capture)
                    try:
                        session.append(pair[1])
                    finally:
                        capture_hook.remove()
                    if len(captured) != 1 or session.last_encoded_id != pair[1].frame_id:
                        raise AssertionError('No successful main pair to commit')
                    block = raw_block(captured.pop(), index, pair[0].timestamp, pair[1].timestamp)
                    hashes.append(old_visual.tensor_sha(block.features))
                    trace.append(banks.append(block))
                    control_peak = max(control_peak, audit_budget(banks, (), calibration_blocks + heldout_flat + [block]))
                    if index + 1 == cfg['natural_mid_query_after_pairs']:
                        query(name, 'mid_after_four_pairs', banks, cfg['natural_questions'][:1], main=session)
                query(name, 'final', banks, cfg['natural_questions'], main=session)
                natural_ledgers[name] = list(session.ledger)
                bank_traces[name] = {'updates': trace, 'final': banks.status(), 'feature_sha256': hashes,
                                    'encoded_end_pts': session.last_encoded_pts,
                                    'continued_after_mid_query': True,
                                    'future': banks.event.exact_available(session.last_encoded_pts + 100),
                                    'missing': banks.event.exact_available(0.)}
            finally:
                session.close(); banks.clear()
        if counts['vision'] != 24 or len(queries) != 60:
            raise AssertionError('Registered execution schedule drift')
        if expected_layout is not None and {'direct': layouts, 'natural_ledgers': natural_ledgers} != expected_layout:
            raise ValueError('GPU pixel/template/position layout differs from CPU proof')
        return {'passed': True, 'calibration': fit, 'queries': queries, 'counts': counts,
                'model_called_queries': len(queries), 'generated_tokens': generated,
                'banks': bank_traces, 'layout_identity': {'direct': layouts, 'natural_ledgers': natural_ledgers},
                'owned_feature_control_bytes_peak': control_peak,
                'calibration_feature_sha256': [old_visual.tensor_sha(b.features) for b in calibration_blocks],
                'heldout_feature_sha256': {n: [old_visual.tensor_sha(b.features) for b in rows] for n, rows in heldout_pool.items()},
                'input_frames': identity, 'threshold_not_fit_on_evaluation': True,
                'no_qa_visual_reencoding': True, 'default_integration': False,
                'scope': 'Fixed finite same-budget diagnostics; animal orderings reuse sources, no benchmark or speed claim'}
    finally:
        log.close(); anchor.cache.reset()
        for hook in hooks:
            hook.remove()


@torch.inference_mode()
def preflight(args):
    from tests.test_online import small_model
    data, identity, assets = inputs()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = small_model(processor)
    builder = PairInputBuilder(processor, resize_mode='letterbox')
    blocks, _ = old_visual.encode(model, builder, ordered_pairs(data['animal_calibration'], ['dog', 'pigeon', 'cat']))
    Contracts.model, Contracts.builder, Contracts.blocks = model, builder, blocks
    with (args.out / 'unittest.log').open('x', encoding='utf-8') as log:
        tests = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not tests.wasSuccessful():
        raise AssertionError('Event memory CPU contracts failed')
    report = replay(model, processor, args, data, identity)
    report.update(cpu_tests=tests.testsRun, request=request(args), budget=budget(), source_hashes=sources(),
                  source_assets=assets, assets=verify_assets(args.model_dir)['files'],
                  model_manifest_sha256=file_hash(args.model_dir / 'DOWNLOAD_MANIFEST.json'),
                  scope='CPU tiny random weights, processor/pixels and interfaces only; answers are not ability evidence')
    write_json(args.out / 'proof.json', report)
    print(json.dumps({'passed': True, 'cpu_tests': tests.testsRun, 'queries': 60, 'vision': 24, 'gpu_loads': 0}), flush=True)


@torch.inference_mode()
def gpu(args):
    if args.preflight is None or read(args.preflight / 'status.json') != {'status': 'completed', 'stage': 'preflight'}:
        raise ValueError('Fresh successful CPU proof required')
    proof = read(args.preflight / 'proof.json')
    if (proof['request'] != request(args) or proof['source_hashes'] != sources() or proof['budget'] != budget() or
        proof['assets'] != verify_assets(args.model_dir)['files'] or
        proof['model_manifest_sha256'] != file_hash(args.model_dir / 'DOWNLOAD_MANIFEST.json')):
        raise ValueError('Proof/source/request/model drift')
    data, identity, assets = inputs()
    if identity != proof['input_frames'] or assets != proof['source_assets']:
        raise ValueError('Frozen evaluation input drift')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=CAP, text=True).strip()
    if args.branch in ('', 'main', 'master', 'HEAD') or branch != args.branch:
        raise ValueError('Configured non-main branch required')
    free, total = torch.cuda.mem_get_info()
    ram = psutil.virtual_memory().available
    write_json(args.out / 'resource_precheck.json', {'ram_available_bytes': ram, 'gpu_free_bytes': free,
                                                    'gpu_total_bytes': total, 'branch': branch})
    if free < 5.5 * 2**30 or ram < 3 * 2**30:
        raise RuntimeError('Resource guard refused model load')
    from qwen35_local import Qwen3_5ForConditionalGeneration
    begin = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir, local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    if torch.cuda.mem_get_info()[0] < 2 * 2**30:
        raise RuntimeError('No FP32 conversion room')
    model.float(); torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    report = replay(model, processor, args, data, identity, proof['layout_identity'])
    report.update(source_assets=assets, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                  wall_seconds_including_load=time.perf_counter() - begin)
    write_json(args.out / 'summary.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preflight', 'gpu'), required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--branch', required=True)
    args = parser.parse_args()
    for name in ('model_dir', 'out', 'preflight'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'tmp').mkdir(); tempfile.tempdir = str(args.out / 'tmp')
    torch.set_num_threads(1); torch.manual_seed(42)
    write_json(args.out / 'budget.json', budget()); write_json(args.out / 'source_hashes.json', sources())
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
