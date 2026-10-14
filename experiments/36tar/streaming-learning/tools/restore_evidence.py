"""Restore a small gzip/base64 evidence envelope, checking original byte SHA."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('envelope',type=Path)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
envelope=json.loads(args.envelope.read_text(encoding='utf-8'))
if envelope['encoding']!='gzip+base64':raise ValueError('Unsupported evidence encoding')
data=gzip.decompress(base64.b64decode(envelope['data'],validate=True))
if len(data)!=envelope['original_bytes'] or hashlib.sha256(data).hexdigest()!=envelope['original_sha256']:
    raise ValueError('Evidence byte identity failed')
with args.out.open('xb') as stream:stream.write(data)
print(json.dumps({'restored':str(args.out.resolve()),'bytes':len(data),'sha256':envelope['original_sha256']}))
