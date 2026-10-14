"""Select real event representatives and isolate recognition from mixed-history reading."""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import unittest

HERE = Path(__file__).resolve().parent
EVENT = HERE.parent / 'event_memory_v1'
sys.path.insert(0, str(EVENT))
spec = importlib.util.spec_from_file_location('event_reference', EVENT / 'probe.py')
events = importlib.util.module_from_spec(spec); spec.loader.exec_module(events)
import torch
import psutil
from transformers import AutoProcessor
from features import answer, raw_block
from memory import EventMemory
from routing import select
from streaming.hybrid import HybridSequence
from streaming.online import PairInputBuilder, fp32_reference
from scripts.baseline import write_json, file_hash
from scripts.pair_demo import verify_assets

ANIMAL = 'Name the main visible animal in these frames.'
ACTION = 'Briefly describe the visible action in these frames.'


def read(path):
    return events.read(path)


def sources():
    return {**events.sources(), **{str(p.resolve()): file_hash(p) for p in HERE.glob('*.py')},
            str(HERE / '方案.md'): file_hash(HERE / '方案.md')}


def budget():
    return {'model_loads': 1, 'vision_max': 18, 'questions_max': 32, 'generated_max': 1024,
            'language_max': 1200, 'answer_tokens': 32, 'max_tokens': 512, 'wall_seconds': 360,
            'ram_min_gib': 3, 'gpu_free_min_gib': 5.5, 'conversion_free_min_gib': 2,
            'feature_controls_bytes_max': 16 * 2**20,
            'animal_question': ANIMAL, 'action_question': ACTION,
            'threshold_source': 'frozen_gpu77_calibration_only'}


def request(args):
    return {'model_dir': str(args.model_dir), 'reference_run': str(args.reference_run), 'branch': args.branch}


class Contracts(unittest.TestCase):
    @staticmethod
    def bank():
        bank = EventMemory(.5)
        for i in range(6):
            x = torch.zeros((48, 16)); x[:, i // 2] = 1
            bank.append(raw_block(x, i, 2 * i, 2 * i + 1))
        return bank

    def test_selected_pair_time_differs_from_group_span(self):
        bank = self.bank()
        meta, blocks = select(bank, 'group', 0)
        self.assertEqual(meta['actual_visual_interval'], [2., 3.])
        self.assertEqual(meta['group_interval'], [0., 3.])
        self.assertTrue(meta['whole_group_content_not_guaranteed'])
        self.assertEqual([b.first for b in blocks], [1])
        self.assertEqual(select(bank, 'current')[1][0].first, 5)

    def test_future_missing_and_span_only_never_select(self):
        bank = self.bank()
        for mode, target, reason in [('exact', 99., 'beyond_encoded_horizon'),
                                     ('exact', .5, 'group_span_only_no_exact_pair'),
                                     ('group', 2, 'no_retained_closed_group')]:
            meta, rows = select(bank, mode, target)
            self.assertFalse(meta['model_called']); self.assertEqual(meta['reason'], reason)
            self.assertEqual(rows, [])
        for mode, target in [('group', -1), ('exact', float('nan')), ('not_a_mode', None)]:
            with self.assertRaises(ValueError): select(bank, mode, target)
        self.assertEqual(select(EventMemory(.5), 'current')[0]['reason'], 'no_current_evidence')

    def test_selection_returns_owned_features_and_does_not_rewrite_bank(self):
        bank = self.bank(); before = bank.status(); original = bank.groups[0].representative.features.clone()
        selected = select(bank, 'exact', 2.5)[1][0]
        selected.features.zero_()
        self.assertEqual(bank.status(), before)
        torch.testing.assert_close(bank.groups[0].representative.features, original, rtol=0, atol=0)
        bank.clear(); self.assertEqual(bank.status()['tensor_bytes'], 0)


@torch.inference_mode()
def replay(model, processor, args, data, identity, compare_features=False, expected_layout=None):
    cfg = read(EVENT / 'cases.json')
    original = read(args.reference_run / 'summary.json')
    threshold = original['calibration']['threshold']
    builder = PairInputBuilder(processor, resize_mode='letterbox')
    counts = {'vision': 0, 'language': 0}
    layouts, feature_identity, queries, checks, schedules = {}, {}, [], [], {}
    generated = 0
    def guard(name, limit):
        def hook(_module, _args):
            counts[name] += 1
            if counts[name] > limit: raise RuntimeError('Forward budget exceeded')
        return hook
    hooks = [model.model.visual.register_forward_pre_hook(guard('vision', 18)),
             model.model.language_model.register_forward_pre_hook(guard('language', 1200))]
    anchor = HybridSequence(model, max_tokens=512)
    with fp32_reference():
        anchor.append(input_ids=builder.prefix_ids.to(model.get_input_embeddings().weight.device), logits_mode='none')
    anchor_before = events.old_visual.cache_fingerprint(anchor)
    log = (args.out / 'answers.jsonl').open('x', encoding='utf-8')
    all_blocks = []

    def query(case, kind, blocks, question, metadata=None, bank=None, expected=None):
        nonlocal generated
        before = None if bank is None else events.memory_fingerprint(type('OneBank', (), {
            'export': lambda _self: {'event': bank.export()}, 'status': lambda _self: {'event': bank.status()}})())
        if not blocks:
            row = {'case': case, 'kind': kind, 'model_called': False, 'selection': metadata}
        else:
            result = answer(model, builder, blocks, question, 32)
            generated += len(result['generated_ids'])
            row = {'case': case, 'kind': kind, 'model_called': True, 'selection': metadata,
                   'result': result, 'main_unchanged': True, 'bank_unchanged': True}
            if expected is not None:
                words = events.animal_words(result['answer'])
                row['controlled_scoring'] = {'expected': expected, 'extracted': words, 'correct': words == [expected]}
        if generated > 1024 or sum(int(r['model_called']) for r in queries) + int(row['model_called']) > 32:
            raise RuntimeError('Generation/question budget exceeded')
        if events.old_visual.cache_fingerprint(anchor) != anchor_before:
            raise AssertionError('Protected main polluted')
        if bank is not None:
            after = events.memory_fingerprint(type('OneBank', (), {
                'export': lambda _self: {'event': bank.export()}, 'status': lambda _self: {'event': bank.status()}})())
            if after != before: raise AssertionError('Query changed event bank')
        queries.append(row); log.write(json.dumps(row, ensure_ascii=False) + '\n'); log.flush()
        print(json.dumps({'case': case, 'kind': kind, 'called': row['model_called'],
                          'answer': row.get('result', {}).get('answer'), 'selection': metadata}, ensure_ascii=False), flush=True)

    try:
        pool = {}
        for name, pairs in data['animal_holdout'].items():
            pool[name], layouts[name] = events.old_visual.encode(model, builder, pairs)
            all_blocks.extend(pool[name])
            feature_identity[name] = [events.old_visual.tensor_sha(b.features) for b in pool[name]]
            if compare_features and feature_identity[name] != original['heldout_feature_sha256'][name]:
                raise ValueError('Animal projections differ from frozen GPU77')
            label = 'bird' if name == 'pigeon' else name
            for index, block in enumerate(pool[name]):
                query(name, 'single_pair_' + str(index), [block.copy()], ANIMAL, expected=label)
            query(name, 'two_pairs', [b.copy() for b in pool[name]], ANIMAL, expected=label)
        for order in cfg['animal_orders']:
            case = 'controlled_' + '_'.join(order)
            bank = EventMemory(threshold)
            for block in events.reindex_blocks(pool, order): bank.append(block)
            schedules[case] = bank.status()
            for index in (0, 1):
                metadata, blocks = select(bank, 'group', index)
                label = 'bird' if order[index] == 'pigeon' else order[index]
                query(case, 'routed_group_' + str(index), blocks, ANIMAL, metadata, bank, expected=label)
            metadata, blocks = select(bank, 'current')
            label = 'bird' if order[-1] == 'pigeon' else order[-1]
            query(case, 'routed_current', blocks, ANIMAL, metadata, bank, expected=label)
            for mode, target in [('exact', .5), ('exact', 99.)]:
                checks.append({'case': case, 'mode': mode, 'target': target, 'selection': select(bank, mode, target)[0]})
            bank.clear()
        for name, pairs in data['natural'].items():
            blocks, layouts[name] = events.old_visual.encode(model, builder, pairs)
            all_blocks.extend(blocks)
            feature_identity[name] = [events.old_visual.tensor_sha(b.features) for b in blocks]
            if compare_features and feature_identity[name] != original['banks'][name]['feature_sha256']:
                raise ValueError('Natural projections differ from frozen GPU77 capture')
            bank = EventMemory(threshold)
            for block in blocks: bank.append(block)
            schedules[name] = bank.status()
            for index in (0, 3, 4, 5):
                query(name, 'single_pair_' + str(index), [blocks[index].copy()], ACTION)
            query(name, 'same_current_all_events', bank.export(), cfg['natural_questions'][0]['text'], bank=bank)
            metadata, selected = select(bank, 'current')
            query(name, 'same_current_selected', selected, cfg['natural_questions'][0]['text'], metadata, bank)
            if name == 'clean_hands':
                query(name, 'same_early_all_events', bank.export(), cfg['natural_questions'][1]['text'], bank=bank)
                metadata, selected = select(bank, 'group', 0)
                query(name, 'same_early_selected', selected, cfg['natural_questions'][1]['text'], metadata, bank)
            else:
                checks.append({'case': name, 'mode': 'group', 'target': 0, 'selection': select(bank, 'group', 0)[0]})
            checks.append({'case': name, 'mode': 'exact', 'target': bank.last_end + 100,
                           'selection': select(bank, 'exact', bank.last_end + 100)[0]})
            bank.clear()
        if counts['vision'] != 18: raise AssertionError('Vision schedule drift')
        total = sum(b.metadata()['tensor_bytes'] for b in all_blocks)
        # Up to three bank blocks + one full export + two selection copies + detector temporaries.
        conservative_peak = total + 8 * 197568 + 8192
        if conservative_peak > 16 * 2**20: raise RuntimeError('Feature control budget exceeded')
        if expected_layout is not None and layouts != expected_layout:
            raise ValueError('CPU/GPU preprocessed layout differs')
        return {'passed': True, 'queries': queries, 'checks': checks, 'counts': counts,
                'model_called_queries': sum(int(r['model_called']) for r in queries),
                'generated_tokens': generated, 'layout': layouts, 'feature_sha256': feature_identity,
                'input_frames': identity, 'threshold': threshold, 'threshold_changed': False,
                'bank_schedules': schedules, 'feature_controls_conservative_peak_bytes': conservative_peak,
                'all_projection_shas_match_gpu77': compare_features,
                'scope': 'Explicit evidence selection diagnostics; neutral identification and fixed-question contrasts reported separately'}
    finally:
        log.close(); anchor.cache.reset(); all_blocks.clear()
        for hook in hooks: hook.remove()


@torch.inference_mode()
def preflight(args):
    from tests.test_online import small_model
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = small_model(processor)
    with (args.out / 'unittest.log').open('x', encoding='utf-8') as log:
        tests = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not tests.wasSuccessful(): raise AssertionError('Selection contract failed')
    data, identity, assets = events.inputs()
    report = replay(model, processor, args, data, identity)
    report.update(cpu_tests=tests.testsRun, source_hashes=sources(), request=request(args), budget=budget(),
                  source_assets=assets, assets=verify_assets(args.model_dir)['files'],
                  model_manifest_sha256=file_hash(args.model_dir / 'DOWNLOAD_MANIFEST.json'),
                  reference_summary_sha256=file_hash(args.reference_run / 'summary.json'),
                  scope='Tiny random CPU weights, no recognition evidence')
    write_json(args.out / 'proof.json', report)
    print(json.dumps({'passed': True, 'cpu_tests': tests.testsRun, 'queries': report['model_called_queries'], 'gpu_loads': 0}), flush=True)


@torch.inference_mode()
def gpu(args):
    if args.preflight is None or read(args.preflight / 'status.json') != {'status': 'completed', 'stage': 'preflight'}:
        raise ValueError('Fresh CPU proof required')
    proof = read(args.preflight / 'proof.json')
    if (proof['source_hashes'] != sources() or proof['request'] != request(args) or proof['budget'] != budget() or
        proof['assets'] != verify_assets(args.model_dir)['files'] or
        proof['model_manifest_sha256'] != file_hash(args.model_dir / 'DOWNLOAD_MANIFEST.json') or
        proof['reference_summary_sha256'] != file_hash(args.reference_run / 'summary.json')):
        raise ValueError('Source/input/reference proof drift')
    data, identity, assets = events.inputs()
    if identity != proof['input_frames'] or assets != proof['source_assets']: raise ValueError('Input drift')
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=events.CAP, text=True).strip()
    if args.branch in ('', 'main', 'master', 'HEAD') or branch != args.branch: raise ValueError('Non-main configured branch required')
    free, total = torch.cuda.mem_get_info(); ram = psutil.virtual_memory().available
    write_json(args.out / 'resource_precheck.json', {'ram_available_bytes': ram, 'gpu_free_bytes': free,
                                                    'gpu_total_bytes': total, 'branch': branch})
    if free < 5.5 * 2**30 or ram < 3 * 2**30: raise RuntimeError('Resource gate refused load')
    from qwen35_local import Qwen3_5ForConditionalGeneration
    begin = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir, local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    if torch.cuda.mem_get_info()[0] < 2 * 2**30: raise RuntimeError('No conversion room')
    model.float(); torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    report = replay(model, processor, args, data, identity, True, proof['layout'])
    report.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
                  wall_seconds_including_load=time.perf_counter() - begin)
    write_json(args.out / 'summary.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preflight', 'gpu'), required=True)
    for name in ('model-dir', 'reference-run', 'out'): parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--branch', required=True)
    args = parser.parse_args()
    for name in ('model_dir', 'reference_run', 'out', 'preflight'):
        value = getattr(args, name)
        if value is not None: setattr(args, name, value.resolve())
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'tmp').mkdir(); tempfile.tempdir = str(args.out / 'tmp')
    torch.set_num_threads(1); torch.manual_seed(42)
    write_json(args.out / 'source_hashes.json', sources()); write_json(args.out / 'budget.json', budget())
    try:
        (preflight if args.stage == 'preflight' else gpu)(args)
        write_json(args.out / 'status.json', {'status': 'completed', 'stage': args.stage})
    except Exception as error:
        write_json(args.out / 'status.json', {'status': 'failed', 'stage': args.stage,
                   'error': type(error).__name__, 'message': str(error)})
        (args.out / 'error.log').write_text(traceback.format_exc(), encoding='utf-8'); raise


if __name__ == '__main__': main()
