"""Restore exact text into a fresh directory, without running any experiment."""
import argparse
import json
from pathlib import Path
from verify import verified_files

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
target=args.out.resolve()
if target.exists():raise ValueError('Fresh directory required; existing data will not be overwritten')
files,receipt=verified_files()
prepared=[]
for name,data in files.items():
    path=(target/name).resolve()
    if not path.is_relative_to(target):raise ValueError('Outside destination')
    prepared.append((path,data))
target.mkdir(parents=True)
for path,data in prepared:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(data)
print(json.dumps(dict(status='restored',**receipt,executed=False,assets_included=False,output=str(target))))
