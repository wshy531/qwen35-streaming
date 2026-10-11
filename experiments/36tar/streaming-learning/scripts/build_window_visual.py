"""Build a recorded-results explainer using actual decoded synthetic thumbnails."""
import argparse
import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0,str(root/"code"/"qwen35-streaming"))
    from streaming.window import read_packets
    from PIL import Image
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--run",type=Path,default=root/"runs"/"window_gpu_20261009_01")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output exists; choose a new visual path")
    report = json.loads((args.run/"gpu_summary.json").read_text(encoding="utf-8"))
    arguments = json.loads((args.run/"arguments.json").read_text(encoding="utf-8"))
    manifest = json.loads((Path(arguments["assets"])/"manifest.json").read_text(encoding="utf-8"))
    videos = []
    for source in manifest["videos"]:
        images = {}
        hashes = {}
        for packet in read_packets(Path(arguments["assets"])/source["file"]):
            hashes[packet.frame_id] = hashlib.sha256(packet.rgb.tobytes()).hexdigest()
            image = Image.fromarray(packet.rgb)
            image.thumbnail((96,96))
            memory = BytesIO()
            image.save(memory,format="PNG")
            images[packet.frame_id] = "data:image/png;base64,"+base64.b64encode(memory.getvalue()).decode()
        for record in report["cases"]:
            if record["video"] == source["id"]:
                if [hashes[i] for i in record["frame_ids"]] != record["frame_sha256"]:
                    raise ValueError("Decoded visual pixels differ from recorded model inputs")
        windows = [{"cutoff":c["cutoff_seconds"],"ids":c["frame_ids"],"answer":c["answer"].strip(),
                    "times":c["rendered_timestamps"]} for c in report["cases"] if c["video"] == source["id"]]
        if len(windows) != 6:
            raise ValueError("This explainer expects the fixed six-trigger S2 result")
        videos.append({"images":images,"windows":windows})
    template = (root/"scripts"/"window_replay.fragment.html").read_text(encoding="utf-8")
    text = template.replace("__WINDOW_DATA__",json.dumps({"videos":videos},ensure_ascii=False))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(text,encoding="utf-8")
    if args.out.stat().st_size > 1_000_000:
        raise ValueError("Visual exceeds inline budget")
    print(f"Recorded visual: {args.out.stat().st_size} bytes")


if __name__ == "__main__":
    main()
