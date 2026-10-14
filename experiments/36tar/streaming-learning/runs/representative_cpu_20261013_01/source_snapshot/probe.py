"""Same grouping/queries, independently budgeted causal representative policies."""
import argparse
from dataclasses import replace
import gc
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
import weakref

HERE = Path(__file__).resolve().parent
EVENT = HERE.parent / 'event_memory_v1'
spec = importlib.util.spec_from_file_location('representative_event_reference', EVENT / 'probe.py')
events = importlib.util.module_from_spec(spec)
spec.loader.exec_module(events)
import psutil
import torch
from transformers import AutoProcessor
from bank import RepresentativeBank
from features import answer, raw_block
from memory import distance
from scripts.baseline import file_hash, write_json
from scripts.pair_demo import verify_assets
from streaming.online import PairInputBuilder

POLICIES = ('first', 'last', 'centroid')
ANIMAL = 'Name the main visible animal in these frames.'
ACTION = 'Briefly describe the visible action in these frames.'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sources():
    return {**events.sources(), **{str(p.resolve()): file_hash(p) for p in HERE.iterdir()
                                   if p.suffix in ('.py', '.md', '.json')}}


def budget():
    return {'model_loads': 1, 'vision_max': 18, 'questions_max': 64, 'generated_max': 2048,
            'language_max': 2160, 'answer_tokens': 32, 'max_tokens': 512, 'wall_seconds': 360,
            'ram_min_gib': 3, 'gpu_free_min_gib': 5.5, 'conversion_free_min_gib': 2,
            'bank_visual_tokens_max': 192, 'bank_tensor_bytes_max': 794368,
            'bank_append_working_bytes_conservative_max': int(1.1 * 2**20),
            'feature_controls_conservative_max': 16 * 2**20, 'input_rgb_max_bytes': 8 * 2**20,
            'same_limit_not_same_actual_use': True}


def request(args):
    return {'model_dir': str(args.model_dir), 'reference_run': str(args.reference_run), 'branch': args.branch,
            'policies': list(POLICIES), 'animal_question': ANIMAL, 'action_question': ACTION}


def fingerprint(bank):
    return {'status': bank.status(),
            'visual': [{'pair': b.first, 'sha': [events.old_visual.tensor_sha(t)
                       for t in (b.features, b.mass, b.xy)]} for b in bank.resident_blocks()],
            'mean': None if bank.mean is None else events.old_visual.tensor_sha(bank.mean)}


class Contracts(unittest.TestCase):
    @staticmethod
    def block(index, vector=None):
        value = torch.zeros(16)
        value[0] = 1
        if vector is not None:
            value[:len(vector)] = torch.tensor(vector)
        return raw_block(value.repeat(48, 1), index, 2*index, 2*index+1)

    def test_first_last_centroid_and_causal_tie(self):
        banks = {p: RepresentativeBank(p) for p in POLICIES}
        for index, vector in enumerate(([1., 0.], [0., 1.], [1., 1.], [1., 1.])):
            for bank in banks.values():
                bank.append(self.block(index, vector))
            if index == 1:
                self.assertEqual(banks['centroid'].representative.first, 0)
        self.assertEqual([banks[p].representative.first for p in POLICIES], [0, 3, 2])
        self.assertEqual([banks[p].latest.first for p in POLICIES], [3, 3, 3])
        for bank in banks.values():
            before = fingerprint(bank)
            self.assertEqual(bank.select('exact', 99.)[0]['reason'], 'beyond_encoded_horizon')
            self.assertEqual(fingerprint(bank), before)

    def test_shared_storage_and_owned_input_and_selection(self):
        bank = RepresentativeBank('first')
        original = self.block(0)
        bank.append(original)
        self.assertIs(bank.latest, bank.representative)
        self.assertEqual(bank.status()['tokens'], 48)
        original.features.zero_()
        self.assertEqual(float(bank.latest.features[:, 0].sum()), 48.)
        bank.append(self.block(1, [0., 1.]))
        self.assertEqual(bank.status()['tokens'], 96)
        before = fingerprint(bank)
        selected = bank.select('open')[1][0]
        selected.features.zero_()
        self.assertEqual(fingerprint(bank), before)

    def test_512_pairs_resident_budget_and_group_eviction(self):
        for policy in POLICIES:
            bank = RepresentativeBank(policy)
            for index in range(512):
                bank.append(self.block(index, [1., float(index % 5)]), boundary=index > 0 and index % 7 == 0)
                status = bank.status()
                self.assertLessEqual(status['tokens'], 192)
                self.assertLessEqual(status['tensor_bytes'], 192*(16*4+20)+16*4)
                self.assertLessEqual(len(bank.groups), 2)
                self.assertEqual(bank.latest.first, index)
            self.assertGreater(bank.evicted, 0)
            bank.clear()
            self.assertEqual(bank.status()['tensor_bytes'], 0)

    def test_scope_exact_time_and_no_future(self):
        bank = RepresentativeBank('first')
        for index in range(4):
            bank.append(self.block(index), boundary=index == 2)
        meta, blocks = bank.select('group', 0)
        self.assertEqual(meta['actual_pair_interval'], [0., 1.])
        self.assertEqual(meta['group_interval'], [0., 3.])
        self.assertTrue(meta['one_pair_not_whole_group'])
        self.assertEqual(bank.select('exact', 2.5)[0]['reason'], 'no_exact_retained_pair')
        self.assertEqual(bank.select('exact', 4.5)[1][0].first, 2)
        self.assertEqual(bank.select('current')[1][0].first, 3)
        self.assertEqual(bank.select('group', 99)[0]['reason'], 'no_closed_group')
        for mode, target in (('exact', float('nan')), ('group', -1), ('invalid', None)):
            with self.assertRaises(ValueError):
                bank.select(mode, target)

    def test_rejected_append_is_atomic(self):
        bank = RepresentativeBank('centroid')
        bank.append(self.block(0))
        before = fingerprint(bank)
        for block, flag in ((self.block(0), False), (self.block(2), False),
                            (replace(self.block(1), begin=.5), False),
                            (self.block(1), 1),
                            (replace(self.block(1), features=torch.full((48, 16), float('nan'))), False)):
            with self.assertRaises(ValueError):
                bank.append(block, flag)
            self.assertEqual(fingerprint(bank), before)

    def test_expired_latest_and_cleared_storage_release(self):
        bank = RepresentativeBank('first')
        bank.append(self.block(0))
        bank.append(self.block(1))
        expired = weakref.ref(bank.latest.features)
        bank.append(self.block(2), True)
        gc.collect()
        self.assertIsNone(expired())
        references = [weakref.ref(b.features) for b in bank.resident_blocks()]
        bank.clear()
        gc.collect()
        self.assertTrue(all(ref() is None for ref in references))


@torch.inference_mode()
def replay(model, processor, args, compare_features=False, expected_layout=None):
    data, identity, assets = events.inputs()
    cfg = read(EVENT / 'cases.json')
    original = read(args.reference_run / 'summary.json')
    threshold = original['calibration']['threshold']
    if threshold != 0.047536224126815796:
        raise ValueError('Frozen threshold changed')
    rgb_bytes = sum(p.rgb.nbytes for split in data.values() for pairs in split.values() for pair in pairs for p in pair)
    if rgb_bytes > 8 * 2**20:
        raise RuntimeError('Input RGB bound exceeded')
    builder = PairInputBuilder(processor, resize_mode='letterbox')
    counts = {'vision': 0, 'language': 0}
    def guard(name):
        def hook(_module, _args):
            counts[name] += 1
            if counts[name] > budget()[name + '_max']:
                raise RuntimeError('Forward budget exceeded')
        return hook
    hooks = [model.model.visual.register_forward_pre_hook(guard('vision')),
             model.model.language_model.register_forward_pre_hook(guard('language'))]
    pool, layouts, hashes, all_blocks = {}, {}, {}, []
    answers, traces, statuses, generated, peak_bank_bytes = [], {}, {}, 0, 0
    stream = (args.out / 'answers.jsonl').open('x', encoding='utf-8')
    try:
        for name, pairs in data['animal_holdout'].items():
            pool[name], layouts[name] = events.old_visual.encode(model, builder, pairs)
            all_blocks.extend(pool[name])
            hashes[name] = [events.old_visual.tensor_sha(b.features) for b in pool[name]]
            if compare_features and hashes[name] != original['heldout_feature_sha256'][name]:
                raise ValueError('Animal projections differ from GPU77')
        for name, pairs in data['natural'].items():
            pool[name], layouts[name] = events.old_visual.encode(model, builder, pairs)
            all_blocks.extend(pool[name])
            hashes[name] = [events.old_visual.tensor_sha(b.features) for b in pool[name]]
            if compare_features and hashes[name] != original['banks'][name]['feature_sha256']:
                raise ValueError('Natural projections differ from GPU77')
        cases = [(f'controlled_{"_".join(order)}', events.reindex_blocks(pool, order), ANIMAL, order)
                 for order in cfg['animal_orders']]
        cases += [(name, pool[name], ACTION, None) for name in data['natural']]
        for case, blocks, question, order in cases:
            banks = {p: RepresentativeBank(p) for p in POLICIES}
            traces[case] = []
            previous = None
            for block in blocks:
                change = None if previous is None else distance(previous, block)
                boundary = change is not None and change > threshold
                steps = {p: bank.append(block, boundary) for p, bank in banks.items()}
                # Only supplied grouping varies with time, never with representative policy.
                shapes = [[(g.first, g.last, g.begin, g.end) for g in b.groups] +
                          [(b.open_first, b.latest.last, b.open_begin, b.last_end)] for b in banks.values()]
                if any(row != shapes[0] for row in shapes):
                    raise AssertionError('Policies changed shared grouping')
                for bank in banks.values():
                    state = bank.status()
                    peak_bank_bytes = max(peak_bank_bytes, state['tensor_bytes'])
                    if state['tokens'] > 192 or state['tensor_bytes'] > 794368:
                        raise RuntimeError('Bank resident bound exceeded')
                traces[case].append({'pair': block.first, 'change': change, 'shared_boundary': boundary,
                                     'policies': steps})
                previous = block
            statuses[case] = {p: bank.status() for p, bank in banks.items()}
            for policy, bank in banks.items():
                queries = [('group', i) for i in range(len(bank.groups))] + [('open', None), ('current', None)]
                for mode, target in queries:
                    before = fingerprint(bank)
                    selection, chosen = bank.select(mode, target)
                    if not chosen:
                        raise AssertionError('Scheduled representative absent')
                    result = answer(model, builder, chosen, question, 32)
                    if fingerprint(bank) != before:
                        raise AssertionError('Question changed latest/representative/mean')
                    generated += len(result['generated_ids'])
                    row = {'case': case, 'policy': policy, 'mode': mode, 'target': target,
                           'selection': selection, 'bank_status': bank.status(), 'result': result,
                           'bank_unchanged': True, 'threshold_changed': False}
                    if compare_features and order is not None:
                        name = order[chosen[0].first // 2]
                        expected = 'bird' if name == 'pigeon' else name
                        words = events.animal_words(result['answer'])
                        row['animal_subject_diagnostic'] = {'expected': expected, 'extracted': words, 'correct': words == [expected]}
                    answers.append(row)
                    if len(answers) > 64 or generated > 2048:
                        raise RuntimeError('Question/generation bound exceeded')
                    stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                    stream.flush()
                    print(json.dumps({'case': case, 'policy': policy, 'mode': mode,
                                      'pair': selection['pair_id'], 'answer': result['answer']}, ensure_ascii=False), flush=True)
                    del chosen
                before = fingerprint(bank)
                if bank.select('exact', bank.last_end + 100)[0]['reason'] != 'beyond_encoded_horizon':
                    raise AssertionError('Future request permitted')
                if fingerprint(bank) != before:
                    raise AssertionError('Rejected request changed state')
            for bank in banks.values():
                bank.clear()
        if counts['vision'] != 18:
            raise AssertionError('Vision schedule changed')
        if expected_layout is not None and layouts != expected_layout:
            raise ValueError('CPU/GPU visual preprocessing drift')
        conservative = sum(b.metadata()['tensor_bytes'] for b in all_blocks) + 3*794368 + 197568 + int(1.1*2**20)
        if conservative > 16 * 2**20:
            raise RuntimeError('Feature control bound exceeded')
        return {'passed': True, 'answers': answers, 'counts': counts, 'generated_tokens': generated,
                'layout': layouts, 'feature_sha256': hashes, 'input_frames': identity, 'input_assets': assets,
                'rgb_bytes': rgb_bytes, 'traces': traces, 'bank_statuses': statuses,
                'bank_peak_tensor_bytes': peak_bank_bytes, 'feature_controls_conservative_bytes': conservative,
                'threshold': threshold, 'all_projection_shas_match_gpu77': compare_features,
                'scope': 'Causal representative-only candidate with shared group decisions; familiar controlled/natural inputs'}
    finally:
        stream.close()
        for hook in hooks:
            hook.remove()


@torch.inference_mode()
def preflight(args):
    from tests.test_online import small_model
    with (args.out / 'unittest.log').open('x', encoding='utf-8') as stream:
        tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not tests.wasSuccessful():
        raise AssertionError('Representative contract failed')
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = small_model(processor)
    report = replay(model, processor, args)
    report.update(cpu_tests=tests.testsRun, source_hashes=sources(), request=request(args), budget=budget(),
                  assets=verify_assets(args.model_dir)['files'],
                  manifest_sha256=file_hash(args.model_dir / 'DOWNLOAD_MANIFEST.json'),
                  reference_sha256=file_hash(args.reference_run / 'summary.json'),
                  scope='Tiny random CPU contracts, not recognition evidence')
    write_json(args.out / 'proof.json', report)


@torch.inference_mode()
def gpu(args):
    if args.preflight is None or read(args.preflight / 'status.json') != {'status': 'completed', 'stage': 'preflight'}:
        raise ValueError('Fresh CPU proof required')
    proof = read(args.preflight / 'proof.json')
    if (proof['source_hashes'] != sources() or proof['request'] != request(args) or proof['budget'] != budget()
            or proof['assets'] != verify_assets(args.model_dir)['files']
            or proof['manifest_sha256'] != file_hash(args.model_dir / 'DOWNLOAD_MANIFEST.json')
            or proof['reference_sha256'] != file_hash(args.reference_run / 'summary.json')):
        raise ValueError('Code/reference/model/budget proof drift')
    _data, identities, assets = events.inputs()
    if identities != proof['input_frames'] or assets != proof['input_assets']:
        raise ValueError('Media/frame drift')
    del _data
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=events.CAP, text=True).strip()
    if args.branch in ('', 'main', 'master', 'HEAD') or branch != args.branch:
        raise ValueError('Configured personal branch required')
    free, total = torch.cuda.mem_get_info()
    ram = psutil.virtual_memory().available
    write_json(args.out / 'resource_precheck.json', {'ram_available_bytes': ram, 'gpu_free_bytes': free,
                                                    'gpu_total_bytes': total, 'branch': branch})
    if ram < 3*2**30 or free < 5.5*2**30:
        raise RuntimeError('Resource gate refused load')
    from qwen35_local import Qwen3_5ForConditionalGeneration
    began = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir, local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    if torch.cuda.mem_get_info()[0] < 2*2**30:
        raise RuntimeError('No conversion room')
    model.float()
    torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    report = replay(model, processor, args, True, proof['layout'])
    report.update(peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  wall_seconds_including_load=time.perf_counter()-began)
    write_json(args.out / 'summary.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preflight', 'gpu'), required=True)
    for name in ('model-dir', 'reference-run', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--branch', required=True)
    args = parser.parse_args()
    for name in ('model_dir', 'reference_run', 'out', 'preflight'):
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
