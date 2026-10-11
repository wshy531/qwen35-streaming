"""Download two fixed, attributed small Commons originals for local inference."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]/"datasets"/"natural_commons_v1"
SOURCES = [
    {"id":"cat_play","file":"cat_play.webm","title":"Cat Plays.webm","author":"Riverarvi",
     "page":"https://commons.wikimedia.org/wiki/File:Cat_Plays.webm",
     "page_revision":"https://commons.wikimedia.org/w/index.php?title=File:Cat_Plays.webm&oldid=932108700",
     "url":"https://upload.wikimedia.org/wikipedia/commons/f/f9/Cat_Plays.webm",
     "license":"CC BY-SA 4.0","license_url":"https://creativecommons.org/licenses/by-sa/4.0/",
     "bytes":2279724,"expected_sha1":"9d3f2a0bd25a03b8938543eca96e9ce4a8d735d2"},
    {"id":"clean_hands","file":"clean_hands.webm","title":"Clean hands short.webm",
     "author":"Centers for Disease Control and Prevention",
     "page":"https://commons.wikimedia.org/wiki/File:Clean_hands_short.webm",
     "page_revision":"https://commons.wikimedia.org/w/index.php?title=File:Clean_hands_short.webm&oldid=1260648365",
     "url":"https://upload.wikimedia.org/wikipedia/commons/8/89/Clean_hands_short.webm",
     "license":"Public domain (US federal work), Commons PDM 1.0",
     "license_url":"https://creativecommons.org/publicdomain/mark/1.0/",
     "bytes":4172129,"expected_sha1":"086c61006e3b38bb0aed44d8f5e8b25ba6989bf5"}
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets",type=Path,default=ROOT)
    asset_root = parser.parse_args().assets.resolve()
    asset_root.mkdir(parents=True,exist_ok=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    records = []
    for source in SOURCES:
        target = asset_root/source["file"]
        if not target.exists():
            request = urllib.request.Request(source["url"],headers={"User-Agent":"Streaming-Video-Learning/1.0 (local research sample)"})
            with opener.open(request,timeout=45) as response:
                content = response.read(source["bytes"]+1)
            if len(content) != source["bytes"] or hashlib.sha1(content).hexdigest() != source["expected_sha1"]:
                raise ValueError("Downloaded original differs from recorded Commons size/SHA1")
            with target.open("xb") as stream:
                stream.write(content)
        content = target.read_bytes()
        assert len(content) == source["bytes"] and hashlib.sha1(content).hexdigest() == source["expected_sha1"]
        records.append({**source,"sha256":hashlib.sha256(content).hexdigest(),
                        "local_changes":"original bytes unchanged; inference resizes RGB; audio ignored"})
        print("VERIFIED",source["id"],len(content),flush=True)
    # A resume verifies originals without erasing already-declared derivatives.
    manifest_path = asset_root/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {
        "retrieved_date":"2026-10-09","source":"Wikimedia Commons original real footage","not_a_benchmark":True}
    manifest["videos"] = records
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
