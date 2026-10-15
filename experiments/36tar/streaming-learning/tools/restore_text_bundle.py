"""Restore verified UTF-8 experiment text into a fresh directory, never execute."""
import argparse,base64,gzip,hashlib,io,json
from pathlib import Path,PurePosixPath
p=argparse.ArgumentParser(description=__doc__);p.add_argument('envelope',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
destination=a.out.resolve()
if destination.exists():raise ValueError('Fresh output directory required')
if a.envelope.stat().st_size>8*2**20:raise ValueError('Envelope byte budget')
env=json.loads(a.envelope.read_text(encoding='utf-8-sig'))
if env['encoding']!='gzip+base64' or type(env['original_bytes']) is not int or not 0<env['original_bytes']<=4*2**20:raise ValueError('Bounded envelope required')
with gzip.GzipFile(fileobj=io.BytesIO(base64.b64decode(env['data'],validate=True))) as stream:
    raw=stream.read(4*2**20+1)
if len(raw)!=env['original_bytes'] or hashlib.sha256(raw).hexdigest()!=env['original_sha256']:raise ValueError('Envelope identity mismatch')
bundle=json.loads(raw)
if bundle['format']!='exact_utf8_text_bundle_v1' or not 1<=len(bundle['files'])<=128:raise ValueError('Unexpected text bundle')
prepared=[]
for name,e in bundle['files'].items():
    relative=PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name or relative.suffix not in ['.py','.md','.json','.jsonl','.log','.txt']:
        raise ValueError('Unexpected text path')
    data=e['text'].encode('utf-8')
    if len(data)!=e['bytes'] or hashlib.sha256(data).hexdigest()!=e['sha256']:raise ValueError('Original byte identity failed')
    target=(destination/Path(*relative.parts)).resolve()
    if not target.is_relative_to(destination):raise ValueError('Outside destination')
    if any(target==old for old,_ in prepared):raise ValueError('Duplicate text path')
    prepared.append((target,data))
if sum(len(data) for _,data in prepared)>4*2**20:raise ValueError('Text byte budget')
targets={target for target,_ in prepared}
if any(parent in targets for target in targets for parent in target.parents):raise ValueError('Conflicting file and directory paths')
destination.mkdir(parents=True)
for target,data in prepared:
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:stream.write(data)
print(json.dumps({'passed':True,'files':len(prepared),'bytes':sum(len(data) for _,data in prepared),'executed':False}))
