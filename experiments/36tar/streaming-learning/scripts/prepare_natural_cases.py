"""Freeze labels and generate one declared VFR derivative before any model answers."""
import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import sys

import av
from PIL import Image,ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code/qwen35-streaming"))
from streaming.window import read_packets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",type=Path,default=ROOT/"datasets/natural_commons_v1")
    parser.add_argument("--inspection-out",type=Path)
    args = parser.parse_args()
    assets = args.assets.resolve()
    config = assets/"cases.json"
    if config.exists():
        raise FileExistsError("Frozen case labels already exist")
    manifest = json.loads((assets/"manifest.json").read_text(encoding="utf-8"))
    video = assets/"hands_vfr.mp4"
    if video.exists():
        raise FileExistsError("VFR asset already exists")
    if args.inspection_out is not None:
        inspection = args.inspection_out.resolve()
        inspection.mkdir(parents=True,exist_ok=False)
    else:
        inspection = ROOT/"runs/natural_inspection_20261009_01"
        inspection.mkdir(parents=True,exist_ok=True)
    if (inspection/"hands_vfr_contact.jpg").exists():
        raise FileExistsError("Existing inspection evidence cannot be overwritten")
    packets = list(read_packets(assets/"clean_hands.webm",1)) # Preparation/annotation, not GPU replay.
    source = packets[10:18]
    ticks = [0,400,1500,1600,3100,4700,6000,6500]
    with av.open(str(video),"w") as container:
        stream = container.add_stream("libx264",rate=30,options={"bf":"0","g":"1","crf":"0","rc-lookahead":"0","sync-lookahead":"0"})
        stream.width,stream.height = source[0].rgb.shape[1],source[0].rgb.shape[0]
        stream.pix_fmt = "yuv444p"
        stream.time_base = Fraction(1,1000)
        stream.codec_context.time_base = Fraction(1,1000)
        for packet,tick in zip(source,ticks):
            frame = av.VideoFrame.from_ndarray(packet.rgb,format="rgb24")
            frame.pts,frame.time_base = tick,Fraction(1,1000)
            for encoded in stream.encode(frame):
                container.mux(encoded)
        for encoded in stream.encode():
            container.mux(encoded)
    derivative = {"id":"hands_vfr","file":video.name,"author":"local derivative of CDC footage",
        "license":manifest["videos"][1]["license"],"source_file":"clean_hands.webm",
        "source_frames":[{"frame_id":x.frame_id,"timestamp":x.timestamp} for x in source],
        "declared_display_pts_ms":ticks,"bytes":video.stat().st_size,
        "sha256":hashlib.sha256(video.read_bytes()).hexdigest(),
        "local_changes":"8 already-sampled resized RGB frames re-encoded H264 yuv444p CRF0; controlled irregular display PTS; no audio"}
    manifest["derived_videos"] = [derivative]
    (assets/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    prompt = "What is the main visible subject in the LAST frame of this video window? Answer only CAT, HANDS, or OTHER."
    identities = [{"video":"cat_play","sample_index":i,"prompt":prompt,"expected":"CAT","kind":"native_last_subject"} for i in [1,3,5,7]]
    identities += [{"video":"clean_hands","sample_index":i,"prompt":prompt,"expected":"HANDS","kind":"native_last_subject"} for i in [3,7,13,15,19]]
    identities += [{"video":"hands_vfr","sample_index":i,"prompt":prompt,"expected":"HANDS","kind":"controlled_vfr_last_subject"} for i in [1,3]]
    orders = []
    for variant in ["original","reverse_pixels"]:
        for options in ["faucet_first","towel_first"]:
            wording = ("hands under the faucet or taking a paper towel? Answer only FAUCET or TOWEL."
                       if options == "faucet_first" else "taking a paper towel or hands under the faucet? Answer only TOWEL or FAUCET.")
            orders.append({"video":"clean_hands","sample_index":15,"prompt":"Which happens FIRST in these frames: "+wording,
                "expected":"FAUCET" if variant == "original" else "TOWEL","kind":"order_diagnostic",
                "variant":variant,"options":options})
    report = {"frozen_before_model_answers":True,"annotation":"assistant visually inspected actual 1FPS decoded thumbnails, not independent human benchmark",
              "window":4,"fps":1,"questions":15,"max_new_tokens":16,"wall_seconds":600,"cases":identities+orders,
              "vfr_expected_frame_ids":[0,2,4,5,6],"vfr_expected_timestamps":[0,1.5,3.1,4.7,6],
              "score":"strip+upper strict; secondary whole-word with optional trailing period/exclamation only"}
    config.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    selected = list(read_packets(video,1))
    canvas = Image.new("RGB",(258*5,218),"#eeeeee")
    draw = ImageDraw.Draw(canvas)
    for i,p in enumerate(selected):
        canvas.paste(Image.fromarray(p.rgb),(258*i,24))
        draw.text((258*i+4,4),f'{p.timestamp:.3f}s | frame {p.frame_id}',fill="black")
    canvas.save(inspection/"hands_vfr_contact.jpg",quality=93)
    print(json.dumps({"fixed_questions":15,"vfr_selected":[[x.frame_id,x.timestamp] for x in selected]}))


if __name__ == "__main__":
    main()
