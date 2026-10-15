"""Read-only audit of PR5 text capsules and preservation of both PR source trees."""
import ast,base64,gzip,hashlib,io,json,subprocess,sys
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT.parents[2]
PEER='80b12924b79f3ea75176c54b48a917287104836e'
CLOSEOUT='0d864af5fbd33201f54fac3a613a550c14fb828f'
EXPECTED_FAILED_SNAPSHOT='runs/visual_channel_cpu_20261015_01/source_snapshot/probe.py'

def sha(data):return hashlib.sha256(data).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main(out):
    if out.exists():raise ValueError('Fresh output required')
    subprocess.run(['git','merge-base','--is-ancestor',PEER,'HEAD'],cwd=REPO,check=True)
    subprocess.run(['git','merge-base','--is-ancestor',CLOSEOUT,'HEAD'],cwd=REPO,check=True)
    subprocess.run(['git','diff','--exit-code',CLOSEOUT,'--',str(ROOT/'code')],cwd=REPO,check=True,stdout=subprocess.PIPE)
    files=subprocess.check_output(['git','diff','--name-only','-z','6ed58af...'+PEER],cwd=REPO).split(b'\0')
    mutable={ROOT/'.gitattributes',ROOT/'README.md',ROOT/'实验总纲.md',ROOT/'tools/restore_text_bundle.py'}
    preserved=0
    for name in files:
        if not name:continue
        rel=name.decode('utf-8');path=REPO/rel
        if path in mutable:continue
        assert path.read_bytes()==subprocess.check_output(['git','show',PEER+':'+rel],cwd=REPO),rel
        preserved+=1
    rows=[];failed=[];python_count=0;json_count=0
    for p in sorted((ROOT/'runs').glob('*/full_text.json.gz-base64.json')):
        env=read(p);assert env['encoding']=='gzip+base64' and 0<env['original_bytes']<=4*2**20
        with gzip.GzipFile(fileobj=io.BytesIO(base64.b64decode(env['data'],validate=True))) as stream:raw=stream.read(4*2**20+1)
        assert len(raw)==env['original_bytes'] and sha(raw)==env['original_sha256']
        bundle=json.loads(raw);assert bundle['format']=='exact_utf8_text_bundle_v1'
        receipt=read(p.parent/'verification.json');assert receipt['passed'] and receipt.get('default_promoted',receipt.get('next_candidate_activated')) is False
        hashes=set();count=0
        for name,value in bundle['files'].items():
            path=PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts and '\\' not in name and ':' not in name
            data=value['text'].encode('utf-8');assert len(data)==value['bytes'] and sha(data)==value['sha256']
            hashes.add(sha(data));count+=1
            if path.suffix=='.py':
                try:ast.parse(data.decode('utf-8-sig'),filename=name);python_count+=1
                except SyntaxError:
                    assert name==EXPECTED_FAILED_SNAPSHOT
                    failures=read(p.parent/'failed_attempts.json')
                    assert any(v['run']=='visual_channel_cpu_20261015_01' and v['status']['status']=='failed' for v in failures)
                    failed.append(name)
            if path.suffix=='.json':json.loads(data.decode('utf-8-sig'));json_count+=1
            if path.suffix=='.jsonl':
                for line in data.decode('utf-8-sig').splitlines():json.loads(line)
        checked=[]
        for key in ('cpu_proof_sha256','gpu_summary_sha256'):
            if key in receipt:assert receipt[key] in hashes;checked.append(key)
        if 'summary_sha256' in receipt:
            for digest in receipt['summary_sha256'].values():assert digest in hashes
            checked.append('summary_sha256')
        rows.append(dict(run=p.parent.name,files=count,envelope_bytes=len(raw),receipt_hashes_checked=checked))
    assert len(rows)==10 and failed==[EXPECTED_FAILED_SNAPSHOT]
    cpu=read(ROOT/'runs/pr5_integration_cpu_20261016_01/summary.json');assert cpu['all_passed'] and cpu['test_count']==195
    for name,n in (('temporal',7),('time_file',8)):
        s=read(ROOT/f'runs/pr5_integration_contracts_20261016_01/{name}/summary.json');assert s['passed'] and s['tests']==n and s['skipped']==0
    tools=read(ROOT/'runs/pr5_integration_tools_20261016_01/summary.json');assert tools['passed'] and tools['tests']==8
    for name,digest in tools['sources'].items():assert sha((REPO/name.replace('\\','/')).read_bytes())==digest
    result=dict(passed=True,peer_commit=PEER,closeout_commit=CLOSEOUT,both_histories_preserved=True,
                peer_files_byte_identical_except_documented_edits=preserved,default_code_byte_identical_to_closeout=True,
                capsules=rows,archived_files_verified=sum(v['files'] for v in rows),archived_python_ast_passed=python_count,
                archived_json_parsed=json_count,expected_historical_syntax_failure_preserved=failed,
                fresh_cpu_tests=218,new_gpu_loads=0,old_quality_is_archived_evidence_not_new_run=True)
    out.mkdir(parents=True);(out/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main(Path(sys.argv[1]).resolve())
