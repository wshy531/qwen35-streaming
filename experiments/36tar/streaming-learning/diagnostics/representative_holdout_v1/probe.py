"""Disjoint sampled-frame evaluation with real state writes and midstream reads."""
import argparse
import hashlib
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
REP = HERE.parent / 'representative_v1'
RUNTIME = HERE.parent / 'event_runtime_v1'
RETRIEVAL = HERE.parent / 'event_retrieval_v1'
sys.path.insert(0, str(REP))
spec = importlib.util.spec_from_file_location('holdout_representative_reference', REP / 'probe.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(RETRIEVAL))
import numpy as np
import psutil
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor
from bank import RepresentativeBank
from features import answer
from runtime import EventRuntime, bank_fingerprint
from scripts.baseline import file_hash, write_json
from scripts.pair_demo import verify_assets
from streaming.online import OnlinePairSession
from streaming.window import FramePacket

POLICIES = ('first', 'centroid')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def config():
    return read(HERE / 'cases.json')


def budget():
    return {'model_loads': 1, 'vision_max': 22, 'questions_max': 72, 'generated_max': 2304,
            'language_max': 2448, 'answer_tokens': 32, 'main_max_tokens': 256, 'qa_max_tokens': 512,
            'wall_seconds': 360, 'ram_min_gib': 3, 'gpu_free_min_gib': 5.5, 'conversion_free_min_gib': 2,
            'input_rgb_max_bytes': 8*2**20, 'external_feature_controls_max_bytes': 8*2**20,
            'main_cache_and_fingerprint_temporaries_accounted_separately': True}


def request(args):
    return {'model_dir': str(args.model_dir), 'branch': args.branch,
            'reference_run': str(args.reference_run), 'fixed_cases': config()}


def sources():
    paths = [*HERE.iterdir(), *(RUNTIME / 'runtime.py', RETRIEVAL / 'routing.py')]
    return {**reference.sources(), **{str(p.resolve()): file_hash(p) for p in paths
                                    if p.is_file() and p.suffix in ('.py', '.md', '.json')}}


def inputs(args):
    events = reference.events
    registered = events.read(events.CAP / 'diagnostics/general_stream_v1/cases.json')['videos']
    registered += events.read(events.HERE / 'cases.json')['natural_sources']
    previous = read(args.reference_run / 'summary.json')['input_frames']
    old_frames = {}
    for split in previous.values():
        for rows in split.values():
            for row in rows:
                old_frames.setdefault(row['source_sha256'], set()).add(row['source_frame_id'])
    cases, assets, used = [], {}, {}
    for case in config()['cases']:
        source = next(r for r in registered if r['id'] == case['source_id'])
        pairs, identity = events.segment(source, case['start'], case['frames'], True)
        overlap = {r['source_frame_id'] for r in identity} & old_frames.get(source['sha256'], set())
        if overlap:
            raise ValueError('Candidate held-out source frames overlap calibration/evaluation')
        already = used.setdefault(source['sha256'], set())
        ids = {r['source_frame_id'] for r in identity}
        if already & ids:
            raise ValueError('New case segments overlap')
        already.update(ids)
        assets[source['path']] = source['sha256']
        question = reference.ANIMAL if case['question_kind'] == 'animal' else reference.ACTION
        cases.append({'config': case, 'pairs': pairs, 'identity': identity, 'question': question})
    rgb = sum(p.rgb.nbytes for case in cases for pair in case['pairs'] for p in pair)
    if rgb > 8*2**20 or sum(len(c['pairs']) for c in cases) != 22:
        raise RuntimeError('Registered input bound changed')
    return cases, assets, rgb


def make(model, processor):
    session = OnlinePairSession(model, processor, max_tokens=256, max_question_tokens=64,
                                max_answer_tokens=32, resize_mode='letterbox', video_logits='none')
    return EventRuntime(session, config()['threshold'])


def protected(wrapper, banks):
    global_rope = wrapper.session.model.model.rope_deltas
    return {'main': reference.events.old_visual.cache_fingerprint(wrapper.session.main),
            'global_rope': None if global_rope is None else reference.events.old_visual.tensor_sha(global_rope),
            'internal_bank': bank_fingerprint(wrapper.bank),
            'candidates': {p: reference.fingerprint(bank) for p, bank in banks.items()},
            'epoch': wrapper.epoch, 'encoded_horizon': wrapper.bank.last_end}


def main_storage(wrapper):
    tensors = [t for field in ('key_cache', 'value_cache', 'conv_states', 'recurrent_states')
               for t in getattr(wrapper.session.main.cache, field) if t is not None]
    sizes = [t.numel()*t.element_size() for t in tensors]
    return {'main_cache_tensor_bytes': sum(sizes),
            'fingerprint_largest_tensor_bytes': max(sizes, default=0),
            'fingerprint_cpu_temporary_conservative_bytes': 2*max(sizes, default=0)}


def feed_pair(wrapper, banks, pair):
    before = {p: reference.fingerprint(b) for p, b in banks.items()}
    first = wrapper.feed(pair[0])
    if first['encoded']:
        raise AssertionError('First member must wait for pair')
    row = wrapper.feed(pair[1])
    if not row['encoded']:
        raise AssertionError('Complete pair not encoded')
    if {p: reference.fingerprint(b) for p, b in banks.items()} != before:
        raise AssertionError('Main write/reset changed candidate bank before new update')
    split = row['group_update']['closed_old_group']
    updates = {p: b.append(wrapper.bank.current, split) for p, b in banks.items()}
    ranges = [[(g.first, g.last, g.begin, g.end) for g in b.groups] for b in banks.values()]
    internal = [(g.first, g.last, g.begin, g.end) for g in wrapper.bank.groups]
    if any(r != internal for r in ranges):
        raise AssertionError('Representative policy changed grouping')
    return row, updates


class Contracts(unittest.TestCase):
    def packets(self):
        return [FramePacket(i, float(i), np.full((32, 32, 3), 30+20*i, np.uint8), i+1) for i in range(6)]

    def test_disjoint_and_bounded_real_inputs(self):
        cases, _assets, rgb = inputs(self.args)
        self.assertEqual(sum(len(c['pairs']) for c in cases), 22)
        self.assertLessEqual(rgb, 8*2**20)
        self.assertEqual(len(cases), 5)

    def test_pending_future_reject_and_mid_query_then_continue(self):
        wrapper = make(self.model, self.processor)
        banks = {p: RepresentativeBank(p) for p in POLICIES}
        frames = self.packets()
        try:
            wrapper.feed(frames[0])
            self.assertEqual(wrapper.encoded_pairs, 0)
            for bank in banks.values():
                self.assertEqual(bank.select('current')[0]['reason'], 'no_evidence')
            wrapper.feed(frames[1])
            for bank in banks.values():
                bank.append(wrapper.bank.current)
                self.assertEqual(bank.select('exact', 2.5)[0]['reason'], 'beyond_encoded_horizon')
            before = protected(wrapper, banks)
            for bank in banks.values():
                answer(self.model, wrapper.session.builder, bank.select('open')[1], reference.ANIMAL, 2)
            self.assertEqual(protected(wrapper, banks), before)
            feed_pair(wrapper, banks, frames[2:4])
            self.assertEqual(wrapper.encoded_pairs, 2)
            self.assertTrue(all(b.latest.first == 1 for b in banks.values()))
        finally:
            wrapper.close()
            for bank in banks.values():
                bank.clear()

    def test_capacity_reset_keeps_candidate_evidence_before_update(self):
        wrapper = make(self.model, self.processor)
        banks = {p: RepresentativeBank(p) for p in POLICIES}
        try:
            frames = self.packets()
            rows = [feed_pair(wrapper, banks, frames[i:i+2])[0] for i in range(0, 6, 2)]
            self.assertIsNotNone(rows[-1]['reset'])
            self.assertEqual(wrapper.capacity_resets, 1)
            self.assertEqual(wrapper.encoded_pairs, 3)
            self.assertTrue(all(b.latest.first == 2 for b in banks.values()))
        finally:
            wrapper.close()
            for bank in banks.values():
                bank.clear()


def contact_sheets(cases, out):
    for case in cases:
        sheet = Image.new('RGB', (512, len(case['pairs'])*220), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, pair in enumerate(case['pairs']):
            draw.text((4, i*220+3), f'pair {i} PTS={[p.timestamp for p in pair]}', fill='black')
            for side, p in enumerate(pair):
                sheet.paste(Image.fromarray(p.rgb), (256*side, i*220+25))
        sheet.save(out / (case['config']['id'] + '_contact.jpg'), quality=85)


@torch.inference_mode()
def replay(model, processor, args, expected_layout=None, save_photos=False):
    cases, assets, rgb = inputs(args)
    if save_photos:
        contact_sheets(cases, args.out)
    counts = {'vision': 0, 'language': 0}
    def guard(name):
        def hook(_module, _args):
            counts[name] += 1
            if counts[name] > budget()[name + '_max']:
                raise RuntimeError('Forward budget exceeded')
        return hook
    hooks = [model.model.visual.register_forward_pre_hook(guard('vision')),
             model.model.language_model.register_forward_pre_hook(guard('language'))]
    answers, timelines, layouts, generated = [], {}, {}, 0
    peak_main, peak_fingerprint, peak_external = 0, 0, 0
    stream = (args.out / 'answers.jsonl').open('x', encoding='utf-8')
    try:
        for case in cases:
            name = case['config']['id']
            wrapper = make(model, processor)
            banks = {p: RepresentativeBank(p) for p in POLICIES}
            layouts[name], timelines[name] = [], []
            try:
                for index, pair in enumerate(case['pairs']):
                    row, updates = feed_pair(wrapper, banks, pair)
                    record = wrapper.session.ledger[-1]
                    layouts[name].append(record)
                    storage = main_storage(wrapper)
                    peak_main = max(peak_main, storage['main_cache_tensor_bytes'])
                    peak_fingerprint = max(peak_fingerprint, storage['fingerprint_cpu_temporary_conservative_bytes'])
                    external = wrapper.bank.status()['tensor_bytes'] + sum(b.status()['tensor_bytes'] for b in banks.values())
                    peak_external = max(peak_external, external)
                    if external > 592704 + 2*794368:
                        raise RuntimeError('Bank resident bound exceeded')
                    point = {'pair': index, 'vision_calls_so_far': counts['vision'],
                             'runtime': wrapper.status(), 'candidate_updates': updates, 'storage': storage,
                             'capacity_reset_candidates_unchanged_before_append': row['reset'] is not None,
                             'feature_sha256': row['feature_sha256'], 'queries': []}
                    if index+1 in case['config']['checkpoints']:
                        horizon = wrapper.bank.last_end
                        before_queries = protected(wrapper, banks)
                        for policy, bank in banks.items():
                            queries = [('open', None), ('current', None)]
                            if bank.groups:
                                queries.append(('group', 0))
                            for mode, target in queries:
                                metadata, blocks = bank.select(mode, target)
                                if not blocks or any(b.end > horizon for b in blocks):
                                    raise AssertionError('Unarrived evidence selected')
                                vision_before = counts['vision']
                                result = answer(model, wrapper.session.builder, blocks, case['question'], 32)
                                if counts['vision'] != vision_before or protected(wrapper, banks) != before_queries:
                                    raise AssertionError('Question encoded new vision or changed protected state')
                                generated += len(result['generated_ids'])
                                observed = {'case': name, 'after_pairs': index+1, 'policy': policy, 'mode': mode,
                                            'target': target, 'selection': metadata, 'result': result,
                                            'received_horizon': wrapper.last_received_pts, 'encoded_horizon': horizon,
                                            'vision_calls_at_query': counts['vision'], 'main_epoch': wrapper.epoch,
                                            'protected_unchanged': True, 'no_query_vision': True,
                                            'candidate_status': bank.status()}
                                if case['config']['question_kind'] == 'animal':
                                    expected = 'bird' if case['config']['source_id'] == 'pigeon' else 'dog'
                                    words = reference.events.animal_words(result['answer'])
                                    observed['animal_clip_subject_check'] = {'expected_clip_subject': expected,
                                        'extracted': words, 'matches_clip_subject': words == [expected],
                                        'not_prevalidated_per_frame_ground_truth': True}
                                answers.append(observed)
                                point['queries'].append(len(answers)-1)
                                if len(answers) > 72 or generated > 2304:
                                    raise RuntimeError('Question/generation bound exceeded')
                                stream.write(json.dumps(observed, ensure_ascii=False)+'\n')
                                stream.flush()
                                print(json.dumps({'case': name, 'pairs': index+1, 'policy': policy,
                                    'mode': mode, 'selected': metadata['pair_id'], 'answer': result['answer']}, ensure_ascii=False), flush=True)
                            if bank.select('exact', horizon+100)[0]['reason'] != 'beyond_encoded_horizon':
                                raise AssertionError('Future access permitted')
                        if protected(wrapper, banks) != before_queries:
                            raise AssertionError('Rejected query changed protected state')
                    timelines[name].append(point)
                wrapper.finish()
            finally:
                wrapper.close()
                for bank in banks.values():
                    bank.clear()
        if counts['vision'] != 22:
            raise AssertionError('Vision not once per arrived pair')
        if expected_layout is not None and layouts != expected_layout:
            raise ValueError('CPU/GPU video layout or capacity schedule differs')
        conservative = 592704 + 2*794368 + 4*197568 + 3*4096
        if conservative > 8*2**20:
            raise RuntimeError('External feature working bound exceeded')
        return {'passed': True, 'answers': answers, 'timelines': timelines, 'layout': layouts,
                'counts': counts, 'generated_tokens': generated, 'assets': assets, 'rgb_bytes': rgb,
                'input_frames': {c['config']['id']: c['identity'] for c in cases},
                'peak_main_cache_tensor_bytes': peak_main, 'peak_fingerprint_cpu_temporary_bytes': peak_fingerprint,
                'peak_external_bank_tensor_bytes': peak_external, 'external_working_conservative_bytes': conservative,
                'scope': 'Disjoint candidate/calibration frames, same old source videos, real pair writes and midstream evidence reads'}
    finally:
        stream.close()
        for hook in hooks:
            hook.remove()


@torch.inference_mode()
def preflight(args):
    from tests.test_online import small_model
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = small_model(processor)
    Contracts.model, Contracts.processor, Contracts.args = model, processor, args
    with (args.out / 'unittest.log').open('x', encoding='utf-8') as stream:
        tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not tests.wasSuccessful():
        raise AssertionError('Holdout/runtime composition contract failed')
    report = replay(model, processor, args, save_photos=True)
    report.update(cpu_tests=tests.testsRun, source_hashes=sources(), request=request(args), budget=budget(),
                  model_assets=verify_assets(args.model_dir)['files'],
                  model_manifest_sha256=file_hash(args.model_dir / 'DOWNLOAD_MANIFEST.json'),
                  reference_sha256=file_hash(args.reference_run / 'summary.json'),
                  scope='Random tiny CPU state/causal contracts, no recognition evidence')
    write_json(args.out / 'proof.json', report)


@torch.inference_mode()
def gpu(args):
    if args.preflight is None or read(args.preflight / 'status.json') != {'status': 'completed', 'stage': 'preflight'}:
        raise ValueError('Fresh CPU proof required')
    proof = read(args.preflight / 'proof.json')
    if (proof['source_hashes'] != sources() or proof['request'] != request(args) or proof['budget'] != budget()
            or proof['model_assets'] != verify_assets(args.model_dir)['files']
            or proof['model_manifest_sha256'] != file_hash(args.model_dir / 'DOWNLOAD_MANIFEST.json')
            or proof['reference_sha256'] != file_hash(args.reference_run / 'summary.json')):
        raise ValueError('Source/request/model/reference proof drift')
    cases, assets, _rgb = inputs(args)
    if assets != proof['assets'] or {c['config']['id']: c['identity'] for c in cases} != proof['input_frames']:
        raise ValueError('Held-out pixel identity drift')
    del cases
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=reference.events.CAP, text=True).strip()
    if args.branch in ('', 'main', 'master', 'HEAD') or branch != args.branch:
        raise ValueError('Configured personal branch required')
    free, total = torch.cuda.mem_get_info()
    ram = psutil.virtual_memory().available
    write_json(args.out / 'resource_precheck.json', {'ram_available_bytes': ram, 'gpu_free_bytes': free,
                                                    'gpu_total_bytes': total, 'branch': branch})
    if ram < 3*2**30 or free < 5.5*2**30:
        raise RuntimeError('Resource gate refused load')
    from qwen35_local import Qwen3_5ForConditionalGeneration
    begin = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir, local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    if torch.cuda.mem_get_info()[0] < 2*2**30:
        raise RuntimeError('No conversion room')
    model.float()
    torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    result = replay(model, processor, args, proof['layout'])
    result.update(source_hashes=sources(), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  wall_seconds_including_load=time.perf_counter()-begin)
    write_json(args.out / 'summary.json', result)


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
    if {p.name for p in args.out.iterdir()} != {'source_snapshot', 'bootstrap.json'}:
        raise ValueError('Fresh watchdog bootstrap required')
    expected = {p.name: file_hash(p) for p in HERE.iterdir() if p.suffix in ('.py', '.md', '.json')}
    if read(args.out / 'bootstrap.json') != {'prepared_by': 'representative_holdout_v1_watchdog', 'files': expected}:
        raise ValueError('Bootstrap source identity drift')
    (args.out / 'tmp').mkdir()
    tempfile.tempdir = str(args.out / 'tmp')
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
