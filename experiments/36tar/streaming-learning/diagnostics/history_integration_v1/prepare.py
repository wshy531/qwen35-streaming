"""Create a six-frame lossless controlled clip from existing attributed assets."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'code/qwen35-streaming-s4'))
import av
import numpy as np
from PIL import Image,ImageDraw
from streaming.preprocessing import fit_canvas
from streaming.window import read_packets


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    args.out.mkdir(parents=True,exist_ok=False)
    config = json.loads((HERE.with_name('general_stream_v1')/'cases.json').read_text(encoding='utf-8'))
    frames,provenance = [],[]
    starts = {'dog':12,'pigeon':0,'cat':0}
    for name in ('dog','pigeon','cat'):
        row = next(r for r in config['videos'] if r['id']==name)
        path = Path(row['path'])
        if sha(path) != row['sha256']: raise AssertionError('Source asset changed')
        reader = read_packets(path,1.)
        try:
            for _ in range(starts[name]): next(reader)
            for _ in range(2):
                packet = next(reader)
                pixels = fit_canvas(packet.rgb,(192,256))
                index = len(frames)
                provenance.append({'session_frame_id':index,'session_pts':float(index),
                    'source_id':name,'source_path':str(path),'source_sha256':row['sha256'],
                    'source_frame_id':packet.frame_id,'source_pts':packet.timestamp,
                    'canvas_rgb_sha256':hashlib.sha256(pixels.tobytes()).hexdigest()})
                frames.append(pixels)
        finally: reader.close()
    path = args.out/'controlled_six_frames.mkv'
    with av.open(str(path),'w') as container:
        stream = container.add_stream('ffv1',rate=1)
        stream.width,stream.height,stream.pix_fmt = 256,192,'bgr0'
        for i,pixels in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(pixels,format='rgb24')
            frame.pts,frame.time_base = i,Fraction(1,1)
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    actual = list(read_packets(path,1.))
    if len(actual)!=6: raise AssertionError('Lossless clip frame count changed')
    for pixels,packet,row in zip(frames,actual,provenance):
        if packet.timestamp != row['session_pts'] or not np.array_equal(pixels,packet.rgb):
            raise AssertionError('Lossless decoded pixels/PTS differ')
    report = {'scope':'Controlled dog→pigeon→cat montage; synthetic 0..5 session timeline, not natural event sequence',
              'video':str(path),'video_sha256':sha(path),'video_bytes':path.stat().st_size,
              'lossless_pixel_and_pts_verified':True,'sample_start_indices':starts,'frames':provenance}
    (args.out/'provenance.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    canvas = Image.new('RGB',(768,440),'white')
    draw = ImageDraw.Draw(canvas)
    for i,pixels in enumerate(frames):
        x,y = (i%3)*256,(i//3)*220
        canvas.paste(Image.fromarray(pixels),(x,y))
        draw.text((x+6,y+194),str(i)+'s '+provenance[i]['source_id'],fill='black')
    canvas.save(args.out/'contact_sheet.png')
    print(json.dumps({k:v for k,v in report.items() if k!='frames'},ensure_ascii=False))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    prepare(parser.parse_args())
