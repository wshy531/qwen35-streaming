"""Real main-state writes and bounded event-forest reads; no default integration."""
import argparse
from dataclasses import replace
import gc
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
import weakref

HERE = Path(__file__).resolve().parent
CAP = HERE.parents[1]
HOLDOUT = HERE.parent/'representative_holdout_v1'
spec = importlib.util.spec_from_file_location('forest_existing_runtime', HOLDOUT/'probe.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
events = prior.reference.events
import torch
import psutil
from transformers import AutoProcessor
from bank import RepresentativeBank
from features import raw_block, context as old_context, answer as old_answer
from forest import ForestBank, RecentBank, raw, merge, penalties, fingerprint, context, answer
from scripts.baseline import file_hash, write_json
from scripts.pair_demo import verify_assets
from runtime import bank_fingerprint
from streaming.window import FramePacket

POLICIES = ('recent4', 'event_first', 'forest')


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def config():
    return read(HERE/'cases.json')


def budget():
    return {'model_loads': 1, 'vision_max': 18, 'questions_max': 72, 'generated_max': 2304,
            'language_max': 2500, 'answer_tokens': 32, 'wall_seconds': 360, 'main_max_tokens': 256,
            'qa_max_tokens': 512, 'bank_visual_tokens_max': 192, 'ram_min_gib': 3,
            'gpu_free_min_gib': 5.5, 'conversion_free_min_gib': 2, 'rgb_max_bytes': 8*2**20,
            'external_working_conservative_max_bytes': 12*2**20, 'precision': 'fp32_ieee_math'}


def request(args):
    return {'model_dir': str(args.model_dir), 'branch': args.branch,
            'reference_run': str(args.reference_run), 'cases': config()}


def sources():
    paths = [*HERE.iterdir(), HERE.parent/'event_forest_v1/prototype.py']
    return {**prior.sources(), **{str(p.resolve()): file_hash(p) for p in paths
            if p.is_file() and p.suffix in ('.py', '.md', '.json')}}


def inputs():
    registered = events.read(CAP/'diagnostics/general_stream_v1/cases.json')['videos']
    pairs, identities, assets = [], [], {}
    for name, start, count in config()['controlled_order']:
        row = next(r for r in registered if r['id'] == name)
        part, identity = events.segment(row, start, count, False)
        assets[row['path']] = row['sha256']
        for pair, entries in zip(part, [identity[i:i+2] for i in range(0, len(identity), 2)]):
            base = 2*len(pairs)
            pairs.append([FramePacket(base+j, float(base+j), p.rgb, base+j+1, p.decode_seconds) for j, p in enumerate(pair)])
            identities.extend([{**entry, 'session_frame_id': base+j, 'session_pts': float(base+j),
                                'session_time_is_source_time': False, 'repeated_source_control': True}
                               for j, entry in enumerate(entries)])
    natural = next(r for r in events.read(events.HERE/'cases.json')['natural_sources'] if r['id'] == config()['natural_source'])
    natural_pairs, natural_identity = events.segment(natural, natural['start_sample'], natural['frames'], True)
    assets[natural['path']] = natural['sha256']
    rgb = sum(p.rgb.nbytes for sequence in (pairs, natural_pairs) for pair in sequence for p in pair)
    if len(pairs) != 12 or len(natural_pairs) != 6 or rgb > budget()['rgb_max_bytes']:
        raise ValueError('Fixed input budget changed')
    cases = [{'name': 'controlled_repeated', 'pairs': pairs, 'identity': identities,
              'checkpoints': config()['controlled_checkpoints'], 'reset_after': config()['controlled_reset_after'],
              'questions': config()['controlled_questions']},
             {'name': 'natural_clean', 'pairs': natural_pairs, 'identity': natural_identity,
              'checkpoints': config()['natural_checkpoints'], 'reset_after': [6],
              'questions': config()['natural_questions']}]
    return cases, assets, rgb


def banks():
    return {'recent4': RecentBank(), 'event_first': RepresentativeBank('first'), 'forest': ForestBank()}


def nodes(bank):
    if isinstance(bank, RepresentativeBank):
        return [raw(b) for b in sorted(bank.resident_blocks(), key=lambda b: b.first)]
    return bank.export()


def fingerprints(candidate):
    return {p: prior.reference.fingerprint(b) if isinstance(b, RepresentativeBank) else fingerprint(b)
            for p, b in candidate.items()}


def protected(wrapper, candidate):
    rope = wrapper.session.model.model.rope_deltas
    return {'main': events.old_visual.cache_fingerprint(wrapper.session.main),
            'rope': None if rope is None else events.old_visual.tensor_sha(rope),
            'internal': bank_fingerprint(wrapper.bank), 'candidates': fingerprints(candidate),
            'epoch': wrapper.epoch, 'encoded_pairs': wrapper.encoded_pairs}


def feed(wrapper, candidate, pair):
    before = fingerprints(candidate)
    if wrapper.feed(pair[0])['encoded']:
        raise AssertionError('Pending first frame encoded')
    row = wrapper.feed(pair[1])
    if not row['encoded'] or fingerprints(candidate) != before:
        raise AssertionError('Main write/reset changed external forest before append')
    split = row['group_update']['closed_old_group']
    updates = {p: b.append(wrapper.bank.current, split) for p, b in candidate.items()}
    if any(b.status()['tokens'] > 192 for b in candidate.values()):
        raise RuntimeError('Shared token cap exceeded')
    return row, updates


class Contracts(unittest.TestCase):
    @staticmethod
    def block(i, vector=None):
        x = torch.randn((48, 16), generator=torch.Generator().manual_seed(800+i))
        if vector is not None:
            x = torch.full((48, 16), float(vector))
        return raw_block(x, i, 2*i, 2*i+1)

    def test_merge_and_penalties_against_frozen_forest(self):
        path = HERE.parent/'event_forest_v1/prototype.py'
        spec = importlib.util.spec_from_file_location('forest_frozen_numerics', path)
        old = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = old
        spec.loader.exec_module(old)
        for i in range(3):
            a, b = raw(self.block(2*i)), raw(self.block(2*i+1))
            merged = merge(a, b)
            x, mass = torch.cat((a.features, b.features)), torch.cat((a.mass, b.mass))
            expected, weights, _ = old.soft_merge(x, mass, 4)
            torch.testing.assert_close(merged.features, expected, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(merged.mass, weights, atol=0, rtol=0)
            self.assertTrue(torch.equal(merged.coords, torch.cat((a.coords, b.coords))[1::2]))
        roots = [raw(self.block(i)) for i in range(3)]
        reference = old.EventForest()
        old_nodes = [old.Node(n.features, n.mass, n.first, n.last, n.begin, n.end, n.center) for n in roots]
        actual, expected = penalties(roots, 7), reference.penalties(old_nodes, 7)
        for a, b in zip(actual, expected):
            for k in a:
                self.assertAlmostEqual(a[k], b[k], places=6)

    def test_multi_level_mass_positions_pressure_and_release(self):
        bank = ForestBank(); total = torch.zeros(16); old_ref = None
        for i in range(512):
            block = self.block(i); total += block.features.sum(0)
            bank.append(block, boundary=(i > 0 and i % 2 == 0))
            state = bank.status()
            self.assertLessEqual(state['tokens'], 192)
            self.assertLessEqual(state['metadata_json_bytes'], 16384)
            self.assertEqual(state['mass'], (i+1)*48)
            self.assertEqual([n.first for n in bank.nodes()], sorted(n.first for n in bank.nodes()))
            if i == 0:
                old_ref = weakref.ref(bank.latest.features)
            if i == 5:
                gc.collect(); self.assertIsNone(old_ref())
        actual = sum((n.features*n.mass).sum(0) for n in bank.nodes())
        torch.testing.assert_close(actual/(512*48), total/(512*48), atol=1e-5, rtol=1e-5)
        self.assertGreater(bank.long_merges, 0)
        self.assertGreater(max(n.depth for n in bank.nodes()), 2)
        self.assertEqual(bank.exact(99999), 'beyond_encoded_horizon')
        self.assertEqual(bank.exact(.5), 'compressed_interval_not_exact_frame')
        exported = bank.export(); before = fingerprint(bank)
        exported[0].features.zero_(); self.assertEqual(fingerprint(bank), before)
        with self.assertRaises(ValueError):
            bank.append(self.block(513))
        self.assertEqual(fingerprint(bank), before)

    def test_loss_and_zero_are_explicit(self):
        merged = merge(raw(self.block(0, 1)), raw(self.block(1, -1)))
        self.assertEqual(float(merged.mass.sum()), 96)
        self.assertFalse(merged.metadata()['exact_pair'])
        # Opposite constant blocks can match within themselves; do not assume
        # they are averaged together by a similarity-based matching algorithm.
        a, b = self.block(0, 1), self.block(1, 1)
        a.features[1::2] = -1
        b.features[1::2] = -1
        lost = merge(raw(a), raw(b))
        self.assertFalse(bool(torch.any(torch.all(lost.features == 1, dim=1))))
        self.assertEqual(float(lost.mass.sum()), 96)
        torch.testing.assert_close((lost.features*lost.mass).sum(0), torch.zeros(16), atol=1e-5, rtol=0)
        zeros = merge(raw(self.block(0, 0)), raw(self.block(1, 0)))
        self.assertTrue(torch.isfinite(zeros.features).all())

    def test_raw_adapter_matches_existing_context(self):
        width = self.model.get_input_embeddings().embedding_dim
        block = raw_block(torch.randn((48, width)), 0, 0, 1)
        question = 'Name the visible subject.'
        a = context(self.model, self.builder, [raw(block)], question, 2)
        b = old_context(self.model, self.builder, [block], question, 2)
        for x, y in zip(a[:3], b[:3]):
            self.assertTrue(torch.equal(x, y))
        from streaming.hybrid import HybridSequence
        from streaming.online import fp32_reference
        with fp32_reference():
            left = HybridSequence(self.model, max_tokens=512)
            right = HybridSequence(self.model, max_tokens=512)
            x = left.append(inputs_embeds=a[0], position_ids=a[1], logits_mode='last')
            y = right.append(inputs_embeds=b[0], position_ids=b[1], logits_mode='last')
            self.assertTrue(torch.equal(x, y))
            left.reset(); right.reset()

    def test_real_composition_queries_reset_continue(self):
        import numpy as np
        wrapper = prior.make(self.model, self.processor)
        candidate = banks()
        packets = [FramePacket(i, float(i), np.full((32, 32, 3), 30+10*i, np.uint8), i+1) for i in range(8)]
        try:
            feed(wrapper, candidate, packets[:2])
            before = protected(wrapper, candidate)
            answer(self.model, wrapper.session.builder, nodes(candidate['forest']), 'Name the visible subject.', 2)
            self.assertEqual(protected(wrapper, candidate), before)
            frozen = fingerprints(candidate)
            wrapper.reset_main()
            self.assertEqual(fingerprints(candidate), frozen)
            self.assertEqual(wrapper.session.main.seen_tokens, 0)
            self.assertEqual(wrapper.ask('Name the visible subject.', 'current')['reason'], 'no_current_pair_in_epoch')
            for pair in (packets[2:4], packets[4:6], packets[6:8]):
                feed(wrapper, candidate, pair)
            self.assertEqual(wrapper.encoded_pairs, 4)
            self.assertTrue(wrapper.current_in_epoch)
        finally:
            wrapper.close()
            for b in candidate.values(): b.clear()


def photos(cases, out):
    from PIL import Image, ImageDraw
    for case in cases:
        sheet = Image.new('RGB', (512, len(case['pairs'])*220), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, pair in enumerate(case['pairs']):
            draw.text((4, i*220+3), f'pair {i}, session PTS {[p.timestamp for p in pair]}', fill='black')
            for j, p in enumerate(pair): sheet.paste(Image.fromarray(p.rgb), (256*j, i*220+25))
        sheet.save(out/(case['name']+'_contact.jpg'), quality=85)


@torch.inference_mode()
def replay(model, processor, args, expected=None, save_photos=False):
    cases, assets, rgb = inputs()
    if save_photos: photos(cases, args.out)
    counts = {'vision': 0, 'language': 0}
    def guard(name):
        def hook(_m, _a):
            counts[name] += 1
            if counts[name] > budget()[name+'_max']: raise RuntimeError('Forward budget exceeded')
        return hook
    handles = [model.model.visual.register_forward_pre_hook(guard('vision')),
               model.model.language_model.register_forward_pre_hook(guard('language'))]
    answers, layout, timelines = [], {}, {}
    generated = peak_main = peak_temp = peak_external = 0
    stream = (args.out/'answers.jsonl').open('x', encoding='utf-8')
    def query(wrapper, candidate, case, index, policy, question, phase, selected=None):
        nonlocal generated
        before = protected(wrapper, candidate); vision_before = counts['vision']
        selected = nodes(candidate[policy]) if selected is None else selected
        if any(n.end > wrapper.bank.last_end for n in selected): raise AssertionError('Future root read')
        result = answer(model, wrapper.session.builder, selected, question, 32)
        if protected(wrapper, candidate) != before or counts['vision'] != vision_before:
            raise AssertionError('Read contaminated state or encoded extra vision')
        generated += len(result['generated_ids'])
        row = {'case': case['name'], 'after_pairs': index+1, 'policy': policy, 'phase': phase,
               'result': result, 'encoded_horizon': wrapper.bank.last_end,
               'main_epoch': wrapper.epoch, 'main_seen': wrapper.session.main.seen_tokens,
               'protected_unchanged': True, 'no_query_vision': True}
        answers.append(row)
        if len(answers) > 72 or generated > 2304: raise RuntimeError('QA budget exceeded')
        stream.write(json.dumps(row, ensure_ascii=False)+'\n'); stream.flush()
        print(json.dumps({'case': case['name'], 'after': index+1, 'policy': policy, 'phase': phase,
                          'question': question, 'answer': result['answer']}, ensure_ascii=False), flush=True)
        return len(answers)-1
    try:
        for case in cases:
            wrapper = prior.make(model, processor); candidate = banks()
            layout[case['name']], timelines[case['name']] = [], []
            try:
                for index, pair in enumerate(case['pairs']):
                    row, updates = feed(wrapper, candidate, pair)
                    layout[case['name']].append(wrapper.session.ledger[-1])
                    storage = prior.main_storage(wrapper)
                    peak_main = max(peak_main, storage['main_cache_tensor_bytes'])
                    peak_temp = max(peak_temp, storage['fingerprint_cpu_temporary_conservative_bytes'])
                    resident = wrapper.bank.status()['tensor_bytes']+sum(b.status()['tensor_bytes'] for b in candidate.values())
                    peak_external = max(peak_external, resident)
                    if resident > 4*192*(1024*4+4+24)+4096:
                        raise RuntimeError('External resident bound exceeded')
                    point = {'pair': index, 'feature_sha256': row['feature_sha256'], 'boundary': row['group_update']['closed_old_group'],
                             'capacity_reset': row['reset'], 'updates': updates, 'states': {p: b.status() for p, b in candidate.items()},
                             'vision_calls': counts['vision'], 'storage': storage, 'queries': []}
                    if case['name'] == 'controlled_repeated' and index in (0,2,3):
                        raw_node = raw(wrapper.bank.current)
                        point['queries'].append(query(wrapper, candidate, case, index, 'forest',
                            'Name the main visible animal in these frames.', 'single_pair_control', [raw_node]))
                        # Same exact visual input/protocol as the older feature reader.
                        before = protected(wrapper, candidate)
                        old = old_answer(model, wrapper.session.builder, [wrapper.bank.current],
                                         'Name the main visible animal in these frames.', 32)
                        generated += len(old['generated_ids'])
                        current = answers[point['queries'][-1]]['result']
                        if old['generated_ids'] != current['generated_ids'] or protected(wrapper, candidate) != before:
                            raise AssertionError('Uncompressed real adapter behavior differs')
                        point['raw_adapter_old_answer_ids_equal'] = True
                        point['old_adapter_control_result'] = old
                    if index+1 in case['checkpoints']:
                        for policy in POLICIES:
                            for question in case['questions']:
                                point['queries'].append(query(wrapper, candidate, case, index, policy, question, 'midstream'))
                        if candidate['forest'].exact(wrapper.bank.last_end+100) != 'beyond_encoded_horizon':
                            raise AssertionError('Future query not rejected')
                    if index+1 in case['reset_after']:
                        before = fingerprints(candidate)
                        point['explicit_reset'] = wrapper.reset_main()
                        if fingerprints(candidate) != before or wrapper.session.main.seen_tokens != 0:
                            raise AssertionError('Explicit reset changed forest')
                        point['current_after_reset_rejected'] = wrapper.ask(case['questions'][0], 'current')['reason'] == 'no_current_pair_in_epoch'
                        if not point['current_after_reset_rejected']: raise AssertionError('Stale current allowed')
                        for policy in POLICIES:
                            point['queries'].append(query(wrapper, candidate, case, index, policy, case['questions'][0], 'after_reset'))
                        point['bank_hashes_after_reset'] = fingerprints(candidate)
                    timelines[case['name']].append(point)
                wrapper.finish()
                if args.stage == 'gpu' and case['name'] == 'controlled_repeated' and candidate['forest'].long_merges == 0:
                    raise AssertionError('Registered real stream did not exercise long forest merges')
            finally:
                wrapper.close()
                for b in candidate.values(): b.clear()
        if counts['vision'] != 18: raise AssertionError('Not once per real pair')
        if expected is not None and layout != expected: raise AssertionError('CPU/GPU video layout differs')
        conservative = 4*192*(1024*4+4+24)+4096 + 3*192*(1024*4+4+24) + 4*96*1024*4 + 48*48*4*3
        if conservative > 12*2**20: raise RuntimeError('Conservative feature bound exceeded')
        return {'passed': True, 'answers': answers, 'timelines': timelines, 'layout': layout,
                'counts': counts, 'generated_tokens_including_old_adapter_controls': generated,
                'unlogged_old_adapter_control_questions': 3, 'model_questions': len(answers)+3,
                'input_frames': {c['name']: c['identity'] for c in cases}, 'assets': assets, 'rgb_bytes': rgb,
                'peak_main_cache_tensor_bytes': peak_main, 'peak_fingerprint_cpu_temporary_bytes': peak_temp,
                'peak_external_resident_tensor_bytes': peak_external, 'external_working_conservative_bytes': conservative,
                'scope': 'Finite known/repeated sources, same upper cap not same actual use; real forest, no training or official full replication'}
    finally:
        stream.close()
        for h in handles: h.remove()


@torch.inference_mode()
def preflight(args):
    from tests.test_online import small_model
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    model = small_model(processor)
    wrapper = prior.make(model, processor)
    Contracts.model, Contracts.processor, Contracts.builder = model, processor, wrapper.session.builder
    wrapper.close()
    with (args.out/'unittest.log').open('x', encoding='utf-8') as stream:
        tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    if not tests.wasSuccessful(): raise AssertionError('Forest adapter contracts failed')
    report = replay(model, processor, args, save_photos=True)
    report.update(cpu_tests=tests.testsRun, source_hashes=sources(), request=request(args), budget=budget(),
                  model_assets=verify_assets(args.model_dir)['files'],
                  model_manifest_sha256=file_hash(args.model_dir/'DOWNLOAD_MANIFEST.json'),
                  reference_sha256=file_hash(args.reference_run/'summary.json'),
                  scope='Random tiny CPU mechanism and numeric controls; no semantic scores')
    write_json(args.out/'proof.json', report)


@torch.inference_mode()
def gpu(args):
    if args.preflight is None or read(args.preflight/'status.json') != {'status':'completed','stage':'preflight'}:
        raise ValueError('Fresh successful CPU proof required')
    proof = read(args.preflight/'proof.json')
    if (proof['source_hashes'] != sources() or proof['request'] != request(args) or proof['budget'] != budget()
            or proof['model_assets'] != verify_assets(args.model_dir)['files']
            or proof['model_manifest_sha256'] != file_hash(args.model_dir/'DOWNLOAD_MANIFEST.json')
            or proof['reference_sha256'] != file_hash(args.reference_run/'summary.json')):
        raise ValueError('Proof drift')
    if read(args.reference_run/'summary.json')['calibration']['threshold'] != config()['threshold']:
        raise ValueError('Frozen grouping threshold drift')
    cases, assets, _ = inputs()
    if assets != proof['assets'] or {c['name']:c['identity'] for c in cases} != proof['input_frames']:
        raise ValueError('Input pixel identity drift')
    del cases
    branch = subprocess.check_output(['git','branch','--show-current'], cwd=CAP, text=True).strip()
    if args.branch in ('', 'main', 'master', 'HEAD') or branch != args.branch:
        raise ValueError('Configured personal branch required')
    free, total = torch.cuda.mem_get_info(); ram = psutil.virtual_memory().available
    write_json(args.out/'resource_precheck.json', {'ram_available_bytes':ram,'gpu_free_bytes':free,'gpu_total_bytes':total,'branch':branch})
    if ram < 3*2**30 or free < 5.5*2**30: raise RuntimeError('Resource gate refused load')
    from qwen35_local import Qwen3_5ForConditionalGeneration
    started = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    if torch.cuda.mem_get_info()[0] < 2*2**30: raise RuntimeError('No FP32 conversion room')
    model.float(); torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    result = replay(model, processor, args, proof['layout'])
    result.update(source_hashes=sources(), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  wall_seconds_including_load=time.perf_counter()-started)
    write_json(args.out/'summary.json', result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('preflight','gpu'), required=True)
    for name in ('model-dir','reference-run','out'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--preflight', type=Path)
    parser.add_argument('--branch', required=True)
    args = parser.parse_args()
    for name in ('model_dir','reference_run','out','preflight'):
        if getattr(args,name) is not None: setattr(args,name,getattr(args,name).resolve())
    if {p.name for p in args.out.iterdir()} != {'source_snapshot','bootstrap.json'}:
        raise ValueError('Fresh watchdog bootstrap required')
    expected = {p.name:file_hash(p) for p in HERE.iterdir() if p.suffix in ('.py','.md','.json')}
    if read(args.out/'bootstrap.json') != {'prepared_by':'forest_runtime_v1_watchdog','files':expected}:
        raise ValueError('Bootstrap source identity drift')
    (args.out/'tmp').mkdir(); tempfile.tempdir = str(args.out/'tmp')
    torch.set_num_threads(1); torch.manual_seed(42)
    write_json(args.out/'source_hashes.json',sources()); write_json(args.out/'budget.json',budget())
    try:
        (preflight if args.stage == 'preflight' else gpu)(args)
        write_json(args.out/'status.json',{'status':'completed','stage':args.stage})
    except Exception as error:
        write_json(args.out/'status.json',{'status':'failed','stage':args.stage,'error':type(error).__name__,'message':str(error)})
        (args.out/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise


if __name__ == '__main__': main()
