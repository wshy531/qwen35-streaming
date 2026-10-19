"""Bounded archive validation and fixed-answer audit; no Torch/model/network."""
import argparse
import base64
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path,PurePosixPath
import re
import sys

if not __debug__:
    raise RuntimeError('Run archive verification without Python optimization; checks must remain enabled')

CAP=Path(__file__).resolve().parents[1]
MAX_BUNDLE=4*2**20
def sha(data):return hashlib.sha256(data).hexdigest()
def unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate JSON key')
        result[key]=value
    return result
def decode(raw):return json.loads(raw.decode('utf-8-sig'),object_pairs_hook=unique)
def safe(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or not p.parts or p.as_posix()!=name or p.suffix not in ['.py','.md','.json','.jsonl','.log','.txt']:
        raise ValueError('invalid original text path')
    return p


def verified_files():
    assert (CAP/'ARCHIVE_INDEX.json').stat().st_size<=4*2**20
    index=decode((CAP/'ARCHIVE_INDEX.json').read_bytes());assert index['format']=='bounded_exact_text_collection_v1'
    assert len(index['bundles'])<=64 and len(index['files'])<=1024
    files={};folded=set();total=0
    for item in index['bundles']:
        name=item['path'];relative=PurePosixPath(name)
        assert relative.parts[0]=='archive' and relative.suffix=='.json' and '..' not in relative.parts and ':' not in name and '\\' not in name
        path=(CAP/Path(*relative.parts)).resolve();assert path.is_relative_to(CAP.resolve()) and path.stat().st_size<=8*2**20
        content=path.read_bytes();assert len(content)==item['bytes'] and sha(content)==item['sha256']
        envelope=decode(content);assert envelope['encoding']=='gzip+base64'
        assert type(envelope['original_bytes']) is int and 0<envelope['original_bytes']<=MAX_BUNDLE
        with gzip.GzipFile(fileobj=io.BytesIO(base64.b64decode(envelope['data'],validate=True))) as stream:raw=stream.read(MAX_BUNDLE+1)
        assert len(raw)==envelope['original_bytes'] and sha(raw)==envelope['original_sha256']
        bundle=decode(raw);assert bundle['format']=='exact_utf8_text_bundle_v1' and 0<len(bundle['files'])<=128
        assert len(bundle['files'])==item['files'] and len(raw)==item['original_bytes']
        for name,entry in bundle['files'].items():
            safe(name);assert name.casefold() not in folded;folded.add(name.casefold())
            data=entry['text'].encode('utf-8');expected=index['files'][name]
            assert len(data)==entry['bytes']==expected['bytes'] and sha(data)==entry['sha256']==expected['sha256']
            total+=len(data);assert total<=48*2**20
            files[name]=data
    assert set(files)==set(index['files'])
    assert (CAP/'SOURCE_MAP.json').stat().st_size<=65536
    for original,item in decode((CAP/'SOURCE_MAP.json').read_bytes()).items():
        safe(original);assert original.casefold() not in folded;folded.add(original.casefold())
        relative=safe(item['path']);assert relative.parts[0]=='src' and relative.suffix=='.py'
        source=(CAP/Path(*relative.parts)).resolve();assert source.is_relative_to(CAP.resolve()) and source.stat().st_size<=2**20
        data=source.read_bytes();assert sha(data)==item['sha256'];files[original]=data;total+=len(data)
    assert len(files)<=1024 and total<=48*2**20
    paths={PurePosixPath(name) for name in files}
    assert all(not any(parent in paths for parent in p.parents) for p in paths)
    return files,dict(bundles=len(index['bundles']),original_files=len(files),text_bytes=total)


def audit_results(files):
    sys.path.insert(0,str(CAP/'src'));from score_memory_eval import parse_choice
    def j(name):return decode(files[name])
    def lines(name):return [decode(s.encode('utf-8')) for s in files[name].decode('utf-8').splitlines()]
    root='runs/08b_question_readout_new24_20261019_01/'
    questions={q['id']:q for q in j(root+'scorer_rows.json')};assert len(questions)==50
    predictions=lines(root+'native/predictions.jsonl')+lines(root+'memory/predictions.jsonl')
    lookup={(r['id'],r['arm']):r for r in predictions};assert len(lookup)==len(predictions)==400
    arms=['native_original','full_pairwise','M1_all8','recent4','uniform4','Q4','annotation4','text_only']
    assert set(lookup)=={(key,arm) for key in questions for arm in arms}
    scores={};primary={}
    for arm in arms:
        scores[arm]=sum(parse_choice(lookup[key,arm]['text'],len(q['options']))==q['answer_index'] for key,q in questions.items())
        primary[arm]=sum(parse_choice(lookup[key,arm]['text'],len(q['options']))==q['answer_index'] for key,q in questions.items() if q['curation']['stratum'] in ['F','T'])
    assert list(scores.values())==[15,15,22,27,25,22,22,19]
    assert list(primary.values())==[12,12,19,23,21,18,19,17]
    assert all(lookup[key,'native_original']['generated_ids']==lookup[key,'full_pairwise']['generated_ids'] for key in questions)
    assert sum(len(r['generated_ids']) for r in predictions)==938
    assert all(r['visual_tokens']<=192 for r in predictions if r['arm'] in ['recent4','uniform4','Q4','annotation4'])
    main='runs/08b_question_readout_20261019_01/';main_records=0
    for stage,n in [('dev24',24),('public74',74),('open47',47)]:
        rows=lines(main+stage+'/predictions.jsonl');assert len(rows)==n*7;main_records+=len(rows)
        assert j(main+stage+'/completed.json')['query_bank_isolation']
    old=lines(main+'runtime_engineering_attempt01/dev24/predictions.jsonl')
    new={(r['id'],r['arm']):r for r in lines(main+'dev24/predictions.jsonl')}
    assert len(old)==154 and all(r['generated_ids']==new[r['id'],r['arm']]['generated_ids'] for r in old)
    reviews=lines('runs/memory_target_new24_20261019_01/question_reviews.jsonl');assert len(reviews)==72
    assert Counter(r['content_verdict'] for r in reviews)==dict(supported=50,ambiguous=22)
    assert all(not r['model_outputs_used'] and not r['old_evidence_necessity_proven'] for r in reviews)
    return dict(new50=scores,primary45=primary,main_answers=main_records+len(predictions),failed_attempt_answers_separate=154,
        preserved_old_attempt_generated_ids=True,new_model_loads=0,GPU_used=False,media_verified_in_this_check=False,
        limitations='Archived RGB/PTS audits are historical receipts; this publication check has no media or pretrained execution.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path);args=parser.parse_args()
    files,receipt=verified_files();receipt.update(audit_results(files));receipt['status']='passed'
    if args.out:
        assert not args.out.exists(),'fresh report path required';args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
