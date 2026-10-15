"""Audit archived 0.8B closeout receipts without loading any model or media."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT.parents[2]
CORE = ROOT / 'code/qwen35-streaming-s4'
RUNS = ROOT / 'runs'

def read(relative):
    return json.loads((RUNS / relative).read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_sources(mapping, base):
    assert mapping
    for rel, expected in mapping.items():
        aliases = {'@launcher': ROOT/'scripts/run_video_demo.py',
                   '@live_launcher': ROOT/'scripts/run_live_demo.py',
                   '@live_plan': ROOT/'实时输入接口验收方案.md',
                   '@file_plan': ROOT/'文件实时来源验收方案.md'}
        path = aliases[rel] if rel.startswith('@') else base / rel.replace('\\', '/')
        assert path.resolve().is_relative_to(REPO.resolve()), rel
        assert sha(path) == expected.lower(), str(path)
    return len(mapping)

def main(out):
    if out.exists():
        raise SystemExit('Use a fresh output directory')
    cpu = read('closeout_08b_cpu_20261015_02/summary.json')
    assert cpu['all_passed'] and cpu['test_count'] == 195 and not cpu['syntax_errors']
    source_counts = {}
    for suite in cpu['suites']:
        assert suite['passed'] and not any(suite[k] for k in ('failures', 'errors', 'skipped'))
        name = suite['suite']
        source_counts[name] = verify_sources(read(f'closeout_08b_cpu_20261015_02/{name}/source_hashes.json'), REPO)
    temporal = read('closeout_08b_real_20261015_01/temporal/summary.json')
    time_file = read('closeout_08b_time_file_cpu_20261015_02/summary.json')
    for receipt, count in ((temporal, 7), (time_file, 8)):
        assert receipt['passed'] and receipt['tests'] == count
        assert not any(receipt[k] for k in ('failures', 'errors', 'skipped'))
    capture = read('closeout_08b_capture_20261015_01/summary.json')
    assert capture['all_passed'] and capture['paired_cases'] == 6 and capture['child_runs'] == 12
    assert capture['before_completed_cases'] == 5 and capture['after_completed_cases'] == 6
    assert all(r['passed'] and all(r['checks'].values()) for r in capture['results'])
    source_counts['capture'] = verify_sources(capture['source_hashes'], CORE)
    quality = read('closeout_08b_quality_gpu_20261015_01/summary.json')
    proof = read('closeout_08b_quality_cpu_20261015_01/proof.json')
    assert proof['passed'] and proof['sources'] == quality['source_hashes']
    assert proof['inputs'] == quality['input_identity']
    source_counts['quality'] = verify_sources(quality['source_hashes'], ROOT)
    assert len(quality['comparisons']) == 6 and all(r['same_generated_ids'] for r in quality['comparisons'])
    original, stream = quality['reports']
    assert original['arm'] == 'original' and stream['arm'] == 'streaming'
    assert len(original['answers']) == len(stream['answers']) == 6
    for a, b in zip(original['answers'], stream['answers']):
        assert (a['file'], a['checkpoint'], a['generated_ids']) == (b['file'], b['checkpoint'], b['generated_ids'])
        assert a['stop_reason'] == b['stop_reason'] == 'eos'
        assert b['main_unchanged_bitwise'] is True
    summaries = {}
    for name, cpu_id, gpu_id, frames, resets in (
        ('generic', '01', '01', 8, 1), ('sparse_file', '02', '03', 2, 0)):
        cpu_dir = f'closeout_08b_{name}_cpu_20261015_{cpu_id}'
        gpu_dir = f'closeout_08b_{name}_gpu_20261015_{gpu_id}'
        gate = read(cpu_dir + '/proof.json')
        sources = read(gpu_dir + '/source_hashes.json')
        assert gate['source_hashes'] == sources
        source_counts[name] = verify_sources(sources, CORE)
        assert read(gpu_dir + '/status.json')['status'] == 'completed'
        summary = read(gpu_dir + '/summary.json')
        final = summary['final']; playback = summary['playback']
        assert final['encoded_real_frames'] == frames and final['questions'] == 1 and final['resets'] == resets
        assert final['finished'] and not final['invalid']
        assert playback['delivered_samples'] == frames and playback['dropped_samples'] == 0
        assert playback['producer_alive'] is False and playback['max_buffered_samples'] <= 1
        assert summary['source_fingerprint'] == gate['source_fingerprint']
        if name == 'sparse_file':
            assert playback['worker_alive'] is False and playback['capture_progress_messages'] > 0
            assert summary['max_encoded_pair_gap_seconds'] == 5.0
        events = [json.loads(line) for line in (RUNS/gpu_dir/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        answer_events = [row for row in events if isinstance(row.get('answer'), dict)]
        assert all(row['main_unchanged_bitwise'] and row['main_counters_unchanged'] for row in answer_events)
        answers = [row['answer'] for row in answer_events]
        assert len(answers) == 1 and answers[0]['stop_reason'] == 'eos'
        assert answers[0]['answer'] == 'A man is shaving his face.'
        summaries[name] = {'run': gpu_dir, 'frames': frames, 'questions': 1, 'resets': resets,
                           'generated_tokens': final['generated_tokens'], 'peak_allocated_gib': summary['peak_allocated_gib'],
                           'dropped_samples': 0, 'producer_alive': False, 'answer': answers[0]['answer']}
    assert read('closeout_08b_sparse_file_gpu_20261015_02/status.json')['status'] == 'failed'
    assert not (RUNS/'closeout_08b_sparse_file_gpu_20261015_02/summary.json').exists()
    failures = {
        'audit_capture_path': 'closeout_08b_delivery_20261015_01',
        'audit_launcher_alias': 'closeout_08b_delivery_20261015_02',
        'full_cpu_initial': 'closeout_08b_cpu_20261015_01',
        'time_file_missing_entry_alias': 'closeout_08b_real_20261015_01/time_file',
        'sparse_cli_minimum_frame_rejection': 'closeout_08b_real_20261015_01/sparse_file_cpu.console.log',
        'sparse_gpu_resource_refusal_before_weight_load': 'closeout_08b_sparse_file_gpu_20261015_02',
    }
    for rel in failures.values():
        assert (RUNS/rel).exists(), rel
    paths = subprocess.check_output(['git', 'ls-files', '--cached', '-z', '--', '*.py'], cwd=REPO).split(b'\0')
    paths = {REPO/p.decode('utf-8') for p in paths if p}
    paths.update(Path(__file__).parent.glob('*.py'))
    for path in paths:
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    manifest = {}
    # HTML viewers contain local frame pixels and stay on the machine.
    for folder in sorted(RUNS.glob('closeout_08b_*')):
        if folder == out or not folder.is_dir():
            continue
        for path in sorted(folder.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.log', '.txt', '.py', '.md'):
                manifest[path.relative_to(ROOT).as_posix()] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    tokens = sum(r['generated_tokens'] for r in quality['reports']) + sum(r['generated_tokens'] for r in summaries.values())
    report = {'passed': True, 'full_cpu_tests': 195, 'extra_contract_tests': 15, 'total_successful_test_executions': 210,
              'source_maps_verified': source_counts, 'python_ast_parsed': len(paths),
              'capture_pairs_passed': 6, 'quality_token_pairs_equal': 6, 'quality_main_cache_checks_unchanged': 6,
              'real_model_loads': 4, 'real_questions': 14, 'generated_tokens': tokens,
              'demo_results': summaries, 'failures_preserved': failures,
              'scope': 'Finite 0.8B engineering closeout; historical optional experiments retain their original evidence; no accuracy, speedup, camera or 2B claim',
              'text_evidence_files': len(manifest), 'text_evidence_bytes': sum(r['bytes'] for r in manifest.values())}
    assert tokens <= 416
    out.mkdir(parents=True)
    for name, value in (('verification.json', report), ('evidence_manifest.json', manifest)):
        (out/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))

if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
