"""Bounded file/camera entry: fresh CPU proof before one GPU stream."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile
import time
import traceback
import unittest

import psutil
import torch
from transformers import AutoProcessor

from scripts.baseline import REPO,file_hash,write_json
from scripts.pair_demo import verify_assets
from scripts.video_demo import replay,budget,request,source_hashes as generic_hashes,branch_allowed
from streaming.live_source import LiveSource


def source_hashes():
    value = generic_hashes()
    value['@live_launcher'] = file_hash(REPO.parents[1]/'scripts/run_live_demo.py')
    value['@live_plan'] = file_hash(REPO.parents[1]/'实时输入接口验收方案.md')
    value['@file_plan'] = file_hash(REPO.parents[1]/'文件实时来源验收方案.md')
    return value


def binding(args):
    return {k:v for k,v in request(args).items() if k!='capture_check'}


def factory(args,kind):
    return lambda callback:LiveSource(kind,str(args.video) if kind=='file' else args.device,args.fps,args.duration,args.max_frames,
        on_frame=callback,idle_seconds=args.idle_seconds,size=args.capture_size,capture_fps=args.capture_fps,
        stderr_path=args.out/'capture_worker.log')


def check_preflight(args):
    if args.preflight is None:
        raise ValueError('Fresh CPU --preflight required')
    read = lambda name:json.loads((args.preflight/name).read_text(encoding='utf-8-sig'))
    proof = read('proof.json')
    if read('status.json')!={'status':'completed','stage':'preflight'} or not read('cpu_tests.json')['passed']:
        raise ValueError('CPU preflight did not pass')
    if proof['request']!=binding(args) or proof['source_hashes']!=source_hashes() or proof['budget']!=budget(args):
        raise ValueError('CPU proof source/request/budget differs')
    if file_hash(args.preflight/'summary.json')!=proof['summary_sha256'] or file_hash(args.preflight/'cpu_tests.json')!=proof['cpu_tests_sha256']:
        raise ValueError('CPU evidence differs')
    if file_hash(args.model_dir/'DOWNLOAD_MANIFEST.json')!=proof['manifest_sha256'] or verify_assets(args.model_dir)['files']!=proof['assets']:
        raise ValueError('Model assets differ')
    return proof


def preflight(args):
    from tests.test_live_source import LiveSourceTests
    from tests.test_online import small_model
    with (args.out/'unittest.log').open('w',encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(LiveSourceTests))
    write_json(args.out/'cpu_tests.json',{'tests':result.testsRun,'passed':result.wasSuccessful(),
        'errors':len(result.errors),'failures':len(result.failures),'real_camera_opened':False})
    if not result.wasSuccessful():
        raise AssertionError('Live transport contracts failed')
    assets = verify_assets(args.model_dir)
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    kind = 'file' if args.input_kind=='file' else 'synthetic'
    report = replay(small_model(processor),processor,args,source_factory=factory(args,kind))
    if report['final']['questions']<1 or report['playback']['worker_alive'] or report['playback']['producer_alive']:
        raise AssertionError('Synthetic model path incomplete or worker did not exit')
    write_json(args.out/'proof.json',{'request':binding(args),'budget':budget(args),'source_hashes':source_hashes(),
        'summary_sha256':file_hash(args.out/'summary.json'),'cpu_tests_sha256':file_hash(args.out/'cpu_tests.json'),
        'manifest_sha256':file_hash(args.model_dir/'DOWNLOAD_MANIFEST.json'),'assets':assets['files'],
        'scope':('Real file frames / tiny random CPU interface, not pretrained semantics' if kind=='file'
                 else 'Synthetic tiny CPU transport/model interface, not camera or pretrained semantics'),
        'real_camera_opened':False,'camera_frames_replayable':False,
        'input_kind':args.input_kind,'video_sha256':file_hash(args.video) if args.input_kind=='file' else None,
        'source_fingerprint':report['source_fingerprint'],
        'gpu_source_comparison':('exact complete selected file-source fingerprint; consumer delivery may differ' if kind=='file'
                                 else 'live device input changes; audit actual frames, no synthetic fingerprint equality claim')})


def capture_check(args):
    if args.input_kind!='camera':
        raise ValueError('File mode never opens a camera; no capture-check needed')
    check_preflight(args)
    # A separate explicit action: three actual device frames, no model or RGB saved.
    source = LiveSource('dshow',args.device,args.fps,min(10.,args.duration),3,
        idle_seconds=args.idle_seconds,size=args.capture_size,capture_fps=args.capture_fps,
        stderr_path=args.out/'capture_worker.log')
    rows = []
    try:
        for p in source.start():
            rows.append({'frame_id':p.frame_id,'timestamp':p.timestamp,'device_pts':p.device_pts,
                'shape':list(p.rgb.shape),'decoded_count':p.decoded_count,'timestamp_basis':p.timestamp_basis})
    finally:
        source.close()
    if len(rows)!=3:
        raise ValueError('Device probe requires exactly three delivered frames')
    write_json(args.out/'capture_check.json',{'request':binding(args),'source_hashes':source_hashes(),
        'frames':rows,'source':source.status(),'real_camera_opened':True,'pixels_saved':False,
        'model_loaded':False,'scope':'Device input contract only; not GPU video understanding'})


def gpu(args):
    proof = check_preflight(args)
    if args.input_kind=='file' and file_hash(args.video)!=proof['video_sha256']:
        raise ValueError('Video bytes differ from CPU proof')
    if args.input_kind=='camera' and args.capture_check is None:
        raise ValueError('Fresh --capture-check required before camera GPU load')
    read = lambda name:json.loads((args.capture_check/name).read_text(encoding='utf-8-sig'))
    if args.input_kind=='camera':
        capture = read('capture_check.json')
        if read('status.json')!={'status':'completed','stage':'capture-check'} or capture['request']!=binding(args) or capture['source_hashes']!=source_hashes() or not capture['real_camera_opened']:
            raise ValueError('Camera contract/config differs or did not complete')
    branch = subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    if not branch_allowed(branch,args.allow_branch):
        raise ValueError('Camera GPU requires allowed non-main branch')
    write_json(args.out/'checked_branch.json',{'branch':branch})
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    free,total = torch.cuda.mem_get_info()
    resource = {'gpu_free_bytes':free,'gpu_total_bytes':total,'ram_available_bytes':psutil.virtual_memory().available}
    write_json(args.out/'resource_precheck.json',resource)
    if free<5.5*2**30 or resource['ram_available_bytes']<3*2**30:
        raise RuntimeError('Resource guard refused camera GPU load')
    from qwen35_local import Qwen3_5ForConditionalGeneration
    begin = time.perf_counter()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(args.model_dir,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    if torch.cuda.mem_get_info()[0]<2*2**30:
        raise RuntimeError('Insufficient FP32 conversion room')
    model.float()
    torch.cuda.reset_peak_memory_stats()
    processor = AutoProcessor.from_pretrained(args.model_dir,local_files_only=True)
    kind = 'file' if args.input_kind=='file' else 'dshow'
    report = replay(model,processor,args,expected_source=proof['source_fingerprint'] if kind=='file' else None,
                    source_factory=factory(args,kind))
    report['wall_seconds_including_load'] = time.perf_counter()-begin
    report['sensor_to_answer_latency_measured'] = False
    if kind=='dshow':
        report['scope'] += ' · 真实设备/接收后滞后，非曝光到回答延迟'
    report['real_camera_opened'] = kind=='dshow'
    report['input_kind'] = args.input_kind
    write_json(args.out/'summary.json',report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['preflight','capture-check','gpu'],required=True)
    parser.add_argument('--model-dir',type=Path,required=True)
    parser.add_argument('--input-kind',choices=['file','camera'],default='camera')
    parser.add_argument('--video',type=Path)
    parser.add_argument('--device',default='Integrated Camera')
    parser.add_argument('--capture-size',default='640x480')
    parser.add_argument('--capture-fps',type=int,default=15)
    parser.add_argument('--idle-seconds',type=float,default=3.)
    parser.add_argument('--fps',type=float,default=1.)
    parser.add_argument('--duration',type=float,default=12.)
    parser.add_argument('--max-frames',type=int,default=12)
    parser.add_argument('--question',default='Describe what is happening now.')
    parser.add_argument('--answer-tokens',type=int,default=16)
    parser.add_argument('--max-tokens',type=int,default=288)
    parser.add_argument('--qa-every',type=int,default=2)
    parser.add_argument('--first-qa-frames',type=int,default=4)
    parser.add_argument('--wall-seconds',type=int,default=90)
    parser.add_argument('--allow-branch',action='append',default=[])
    parser.add_argument('--ui-port',type=int,default=-1)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--preflight',type=Path)
    parser.add_argument('--capture-check',type=Path)
    args = parser.parse_args()
    if (args.input_kind=='file' and (args.video is None or args.stage=='capture-check')) or (args.input_kind=='camera' and args.video is not None):
        parser.error('File input requires --video and never capture-check; camera input has no file')
    try:
        w,h = [int(x) for x in args.capture_size.split('x')]
        valid = 1<=w<=1280 and 1<=h<=720
    except ValueError:
        valid = False
    if (not valid or not 1<=args.capture_fps<=30 or not math.isfinite(args.fps) or not 0<args.fps<=10 or
        not math.isfinite(args.duration) or not 0<args.duration<=30 or not 4<=args.max_frames<=60 or
        not math.isfinite(args.idle_seconds) or not .2<=args.idle_seconds<=10 or not 1<=args.answer_tokens<=32 or
        not 200<=args.max_tokens<=512 or not 1<=args.qa_every<=60 or not 1<=args.first_qa_frames<=60 or
        not 1<=args.wall_seconds<=180 or not -1<=args.ui_port<=65535):
        parser.error('Live request exceeds bounded camera limits')
    args.out,args.model_dir = args.out.resolve(),args.model_dir.resolve()
    # Common replay policy; source reader belongs exclusively to capture worker.
    args.video = args.video.resolve() if args.video else None
    args.playback,args.repeat_count,args.repeat_period,args.consumer_delay_seconds = 'clock',1,None,0.
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'tmp').mkdir()
    tempfile.tempdir = str(args.out/'tmp')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    write_json(args.out/'arguments.json',binding(args))
    write_json(args.out/'budget.json',budget(args))
    write_json(args.out/'source_hashes.json',source_hashes())
    try:
        {'preflight':preflight,'capture-check':capture_check,'gpu':gpu}[args.stage](args)
        write_json(args.out/'status.json',{'status':'completed','stage':args.stage})
    except Exception:
        (args.out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        write_json(args.out/'status.json',{'status':'failed','stage':args.stage})
        raise


if __name__=='__main__':
    main()
