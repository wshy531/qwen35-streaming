"""Restore bundled historical source bytes only into a fresh directory; never execute them."""
import argparse,base64,gzip,hashlib,json
from pathlib import Path,PurePosixPath

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('envelope',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();destination=args.out.resolve()
    if destination.exists():raise ValueError('A fresh output directory is required')
    env=json.loads(args.envelope.read_text(encoding='utf-8-sig'))
    if env['original_bytes']>2**20:raise ValueError('Bounded source bundle required')
    raw=gzip.decompress(base64.b64decode(env['data'],validate=True))
    if len(raw)!=env['original_bytes'] or hashlib.sha256(raw).hexdigest()!=env['original_sha256']:raise ValueError('Envelope identity mismatch')
    bundle=json.loads(raw)
    if bundle['format']!='exact_source_snapshots_v2' or len(bundle['files'])>64 or len(bundle['blobs'])>64:raise ValueError('Unexpected source bundle')
    prepared=[]
    for name,entry in bundle['files'].items():
        relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name or relative.suffix not in ('.py','.md','.json'):
            raise ValueError('Unexpected relative source path')
        path=destination/Path(*relative.parts)
        if not path.is_relative_to(destination):raise ValueError('Outside destination')
        data=base64.b64decode(bundle['blobs'][entry['sha256']],validate=True)
        if len(data)!=entry['bytes'] or hashlib.sha256(data).hexdigest()!=entry['sha256']:raise ValueError('Source identity mismatch')
        data.decode('utf-8-sig');prepared.append((path,data))
    if sum(len(data) for _,data in prepared)>2**20:raise ValueError('Source byte limit')
    destination.mkdir(parents=True)
    for path,data in prepared:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(data)
    print(json.dumps({'passed':True,'files':len(prepared),'bytes':sum(len(data) for _,data in prepared),'executed':False}))

if __name__=='__main__':main()
