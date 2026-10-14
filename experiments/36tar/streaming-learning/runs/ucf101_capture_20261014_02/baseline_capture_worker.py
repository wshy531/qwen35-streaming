"""Pure capture child: imports no torch/model, transmits bounded RGB packets."""
import argparse
import ctypes
import json
import math
import os
import struct
import sys
import threading
import time


def send(meta,rgb=None):
    data = json.dumps(meta,separators=(',',':')).encode('utf-8')
    if len(data)>4096:
        raise ValueError('Metadata exceeds transport budget')
    sys.stdout.buffer.write(struct.pack('!I',len(data))+data)
    if rgb is not None:
        sys.stdout.buffer.write(rgb.tobytes())
    sys.stdout.buffer.flush()


def owner_guard(owner_pid,deadline,stop):
    # An OS process handle stays attached to the original parent if PID is reused.
    if os.name=='nt':
        kernel = ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.argtypes = [ctypes.c_ulong,ctypes.c_int,ctypes.c_ulong]
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p,ctypes.c_ulong]
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = kernel.OpenProcess(0x00100000,0,owner_pid)
        if not handle:
            os._exit(72)
        alive = lambda: kernel.WaitForSingleObject(handle,0)==258
    else:
        handle = None
        alive = lambda: os.getppid()==owner_pid
    try:
        while not stop.wait(.05):
            if not alive() or time.monotonic()>deadline:
                os._exit(73)  # Stops even a blocked native camera read.
    finally:
        if handle:
            kernel.CloseHandle(handle)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind',choices=['synthetic','dshow','file'],required=True)
    p.add_argument('--device',default='Integrated Camera')
    p.add_argument('--fps',type=float,required=True)
    p.add_argument('--duration',type=float,required=True)
    p.add_argument('--max-frames',type=int,required=True)
    p.add_argument('--origin',type=float,required=True)
    p.add_argument('--owner-pid',type=int,required=True)
    p.add_argument('--size',default='640x480')
    p.add_argument('--capture-fps',type=int,default=15)
    p.add_argument('--fixture',choices=['normal','stall','error','burst'],default='normal')
    a = p.parse_args()
    w,h = [int(x) for x in a.size.split('x')]
    if not (1<=w<=1280 and 1<=h<=720 and 1<=a.capture_fps<=30 and math.isfinite(a.fps) and 0<a.fps<=10 and math.isfinite(a.duration) and 0<a.duration<=120 and 1<=a.max_frames<=120):
        raise ValueError('Capture budget invalid')
    if a.kind!='synthetic' and a.fixture!='normal':
        raise ValueError('No device fault fixtures')
    stop = threading.Event()
    threading.Thread(target=owner_guard,args=(a.owner_pid,time.monotonic()+a.duration+5,stop),daemon=True).start()
    try:
        import numpy as np
        if a.fixture=='stall':
            while True:
                time.sleep(.1)
        if a.fixture=='error':
            raise RuntimeError('Explicit synthetic disconnect')
        count, last_timestamp = 0, None
        def frame(rgb,frame_id,decoded,device_pts,received=None):
            nonlocal count, last_timestamp
            now = time.monotonic()-a.origin
            if received is None:
                while last_timestamp is not None and now<=last_timestamp:
                    time.sleep(.001)
                    now = time.monotonic()-a.origin
            if now>=a.duration:
                return False
            send({'kind':'frame','frame_id':frame_id,'decoded_count':decoded,'timestamp':now if received is None else received,
                  'shape':list(rgb.shape),'device_pts':device_pts,'source_kind':a.kind,
                  'timestamp_basis':'receive_monotonic'},rgb)
            count += 1
            last_timestamp = now if received is None else received
            return count<a.max_frames
        if a.kind=='synthetic':
            for i in range(a.max_frames):
                if i and a.fixture!='burst':
                    time.sleep(1/a.fps)
                rgb = np.full((64,96,3),[32+16*(i%8),64,128],np.uint8)
                if not frame(rgb,i,i+1,None):
                    break
        elif a.kind=='file':
            from streaming.window import read_packets
            from streaming.preprocessing import fit_canvas
            with_reader = read_packets(a.device,a.fps)
            reason = 'source_eof'
            try:
                for packet in with_reader:
                    if packet.timestamp>=a.duration:
                        reason = 'live_duration_limit'
                        break
                    while True:
                        remaining = a.origin+packet.timestamp-time.monotonic()
                        if remaining<=0:
                            break
                        time.sleep(min(.02,remaining))
                    rgb = fit_canvas(packet.rgb,(192,256))
                    send({'kind':'frame','frame_id':packet.frame_id,'decoded_count':packet.decoded_count,
                          'timestamp':packet.timestamp,'device_pts':packet.timestamp,'shape':list(rgb.shape),
                          'source_kind':'file','timestamp_basis':'source_pts'},rgb)
                    count += 1
                    if count>=a.max_frames:
                        reason = 'frame_budget_limit'
                        break
            finally:
                with_reader.close()
            send({'kind':'end','reason':reason,'frames':count,'torch_loaded':'torch' in sys.modules})
            return 0
        else:
            import av
            from PIL import Image
            from streaming.window import CausalSampler
            from streaming.preprocessing import fit_canvas
            sampler = CausalSampler(a.fps)
            if 'dshow' not in av.formats_available:
                raise RuntimeError('Installed PyAV does not support dshow')
            # Video only, no microphone; cap native real-time buffer to 8 MiB.
            with av.open('video='+a.device,format='dshow',options={'video_size':a.size,
                'framerate':str(a.capture_fps),'rtbufsize':str(8*2**20)},timeout=(3.,3.)) as container:
                for i,image in enumerate(container.decode(video=0)):
                    now = time.monotonic()-a.origin
                    if now>=a.duration:
                        break
                    if image.width>w or image.height>h:
                        raise ValueError('Negotiated camera dimensions exceed requested budget')
                    if sampler.accept(now):
                        rgb = Image.fromarray(image.to_ndarray(format='rgb24'))
                        rgb.thumbnail((256,256))
                        rgb = fit_canvas(np.asarray(rgb),(192,256))
                        device_pts = None if image.pts is None or image.time_base is None else float(image.pts*image.time_base)
                        if not frame(rgb,i,i+1,device_pts,now):
                            break
        send({'kind':'end','reason':'frame_budget_limit' if count>=a.max_frames else 'live_duration_limit','frames':count,
              'torch_loaded':'torch' in sys.modules})
        return 0
    except Exception as error:
        send({'kind':'error','message':type(error).__name__+': '+str(error)})
        return 1
    finally:
        stop.set()


if __name__=='__main__':
    sys.exit(main())
