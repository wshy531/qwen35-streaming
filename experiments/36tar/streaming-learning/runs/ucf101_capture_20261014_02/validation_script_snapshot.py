"""Bounded before/after capture validation on original UCF101 mirror clips."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT/'code/qwen35-streaming-s4'
sys.path.insert(0,str(CORE))

import av
from streaming.live_source import LiveSource
from streaming.playback import SourceDigest
from streaming.preprocessing import fit_canvas
from streaming.window import FramePacket,read_packets

BASE = '450e61c'
PREFIX = 'experiments/36tar/streaming-learning/code/qwen35-streaming-s4/'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def expected(path,fps):
    digest, identities = SourceDigest(),[]
    reader = read_packets(path,fps)
    try:
        for packet in reader:
            if packet.timestamp >= 12: break
            normalized = FramePacket(packet.frame_id,packet.timestamp,
                fit_canvas(packet.rgb,(192,256)),packet.decoded_count)
            digest.add(normalized)
            identities.append([packet.frame_id,packet.timestamp,packet.decoded_count])
            if len(identities)==3: break
    finally: reader.close()
    return digest.snapshot(),identities


def execute(source_class,path,fps,run,stem,baseline_worker=None):
    source = source_class('file',str(path),fps,12,3,idle_seconds=3.,stderr_path=run/(stem+'.stderr.log'))
    identities, failure = [],None
    original_popen = subprocess.Popen
    def legacy_spawn(command,**kwargs):
        kwargs['cwd']=CORE
        kwargs['env']=dict(kwargs['env'])
        kwargs['env']['PYTHONPATH']=str(CORE)+os.pathsep+kwargs['env'].get('PYTHONPATH','')
        return original_popen([command[0],str(baseline_worker),*command[3:]],**kwargs)
    context = mock.patch.object(subprocess,'Popen',side_effect=legacy_spawn) if baseline_worker else mock.patch.dict({}, {})
    begin = time.monotonic()
    try:
        with context:
            for packet in source.start():
                identities.append([packet.frame_id,packet.timestamp,packet.decoded_count])
    except Exception as error:
        failure = type(error).__name__+': '+str(error)
    finally: source.close()
    state = source.status()
    return {'ok':failure is None,'error':failure,'reason':source.reason,'wall_seconds':time.monotonic()-begin,
            'frames':identities,'fingerprint':source.digest.snapshot(),'status':state,
            'cleaned_up':not state['worker_alive'] and not state['producer_alive']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    args.data_dir,args.out=args.data_dir.resolve(),args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.data_dir/'manifest.json').read_text(encoding='utf-8-sig'))
    (args.out/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    for name in ('scripts/capture_worker.py','streaming/live_source.py'):
        data=subprocess.check_output(['git','show',BASE+':'+PREFIX+name],cwd=CORE)
        (args.out/('baseline_'+Path(name).name)).write_bytes(data)
    baseline_path=args.out/'baseline_live_source.py'
    spec=importlib.util.spec_from_file_location('dataset_baseline_source',baseline_path)
    baseline=importlib.util.module_from_spec(spec);sys.modules[spec.name]=baseline;spec.loader.exec_module(baseline)
    rows=[]
    for clip in manifest['clips']:
        path=args.data_dir/clip['file']
        if path.stat().st_size!=clip['bytes'] or sha(path)!=clip['sha256']:
            raise ValueError('Dataset bytes differ: '+clip['file'])
        with av.open(str(path)) as container:
            stream=container.streams.video[0]
            duration=float(stream.duration*stream.time_base)
        for fps in (.2,1.):
            fingerprint,frames=expected(path,fps)
            prefix=path.stem+'-'+str(fps)
            before=execute(baseline.LiveSource,path,fps,args.out,prefix+'-before',args.out/'baseline_capture_worker.py')
            after=execute(LiveSource,path,fps,args.out,prefix+'-after')
            failure_expected=len(frames)>1 and frames[1][1]-frames[0][1]>3.
            checks={'after_completed':after['ok'],'after_frames_match':after['frames']==frames,
                    'after_pixels_match':after['fingerprint']==fingerprint,
                    'after_no_drops':after['status']['dropped_samples']==0,
                    'bounded_pending_rgb':after['status']['peak_buffered_rgb_bytes']<=192*256*3,
                    'after_torch_not_loaded':after['status']['worker_torch_loaded'] is False,
                    'both_cleaned_up':before['cleaned_up'] and after['cleaned_up'],
                    'baseline_expected_outcome':(not before['ok'] and before['reason']=='source_idle_timeout') if failure_expected
                        else before['ok'] and before['frames']==frames and before['fingerprint']==fingerprint}
            row={'file':clip['file'],'duration_seconds':duration,'fps':fps,'max_samples':3,'duration_limit':12,
                 'idle_seconds':3,'expected_frames':frames,'baseline_failure_expected':failure_expected,
                 'before':before,'after':after,'checks':checks,'passed':all(checks.values())}
            rows.append(row)
            (args.out/'partial.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            print(json.dumps({'file':clip['file'],'fps':fps,'before':before['reason'],
                              'before_frames':len(before['frames']),'after_frames':len(after['frames']),
                              'passed':row['passed']}),flush=True)
    report={'dataset':'UCF101','clips':len(manifest['clips']),'paired_cases':len(rows),
            'child_runs':2*len(rows),'all_passed':all(r['passed'] for r in rows),
            'before_completed_cases':sum(r['before']['ok'] for r in rows),
            'after_completed_cases':sum(r['after']['ok'] for r in rows),'results':rows,
            'scope':'Capture transport only: no Qwen weights, generated answers, action accuracy or GPU inference.',
            'source_hashes':{str(p.relative_to(CORE)):sha(p) for p in
                (CORE/'scripts/capture_worker.py',CORE/'streaming/live_source.py')},
            'script_sha256':sha(Path(__file__)), 'baseline_commit':BASE}
    (args.out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['all_passed']:raise AssertionError('Dataset capture comparison failed; evidence retained')


if __name__=='__main__':main()
