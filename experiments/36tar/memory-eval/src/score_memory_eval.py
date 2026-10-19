"""Strict MCQ scoring, paired attribution, no model or dataset downloads."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def parse_choice(text, n_options):
    if not isinstance(text, str):
        return None
    m = re.fullmatch(r'\s*([A-Ea-e])\s*[.!]?\s*', text)
    index = ord(m[1].upper())-ord('A') if m else -1
    return index if 0 <= index < n_options else None


def prompt(row):
    return row['question']+'\n'+'\n'.join(
        f'{chr(65+i)}. {v}' for i, v in enumerate(row['options'])) + \
        '\nReturn only the single option letter. Do not explain.'


def score(rows, predictions):
    expected = {r['id']: r for r in rows}
    if len(expected) != len(rows) or not rows:
        raise ValueError('nonempty unique evaluation IDs required')
    if any(not 2 <= len(r['options']) <= 5 or not isinstance(r['answer_index'],int)
           or not 0 <= r['answer_index'] < len(r['options']) for r in rows):
        raise ValueError('invalid options/answer index')
    observed = {}
    for p in predictions:
        if p['id'] not in expected or p['id'] in observed:
            raise ValueError('unknown or duplicate prediction ID')
        observed[p['id']] = p
    details = []
    for key, row in expected.items():
        p = observed.get(key)
        if p is None:
            status, choice = 'missing', None
        elif p.get('status', 'ok') != 'ok':
            status, choice = p['status'], None
        else:
            choice = parse_choice(p.get('text'), len(row['options']))
            status = 'ok' if choice is not None else 'format_error'
        details.append(dict(id=key, video=row.get('source_group_id',f"{row['dataset']}:{row['source_id']}"),
            task=f"{row['dataset']}:{row['task']}", status=status,
            predicted_index=choice, answer_index=row['answer_index'],
            correct=status == 'ok' and choice == row['answer_index']))
    def aggregate(items):
        videos = defaultdict(list)
        for item in items:
            videos[item['video']].append(int(item['correct']))
        correct = sum(x['correct'] for x in items)
        answered = sum(x['status']=='ok' for x in items)
        return dict(questions=len(items), videos=len(videos), correct=correct,
            micro_accuracy=correct/len(items),
            video_macro_accuracy=sum(sum(v)/len(v) for v in videos.values())/len(videos),
            answered=answered, answered_only_accuracy=correct/answered if answered else None,
            statuses=dict(Counter(x['status'] for x in items)))
    return dict(overall=aggregate(details),
        by_task={t: aggregate([x for x in details if x['task']==t])
                 for t in sorted({x['task'] for x in details})}, details=details)


def paired(reference, candidate):
    r = {x['id']:x for x in reference['details']}
    c = {x['id']:x for x in candidate['details']}
    if r.keys() != c.keys():
        raise ValueError('paired IDs differ')
    names = {(True,True):'retained_correct', (True,False):'new_loss',
             (False,True):'correction', (False,False):'both_wrong'}
    return dict(Counter(names[(bool(r[k]['correct']), bool(c[k]['correct']))] for k in r))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', required=True,
        help='Frozen {schema, rows, input_protocol} bundle; not the candidate manifest')
    p.add_argument('--predictions', required=True,
        help='{evaluation_sha256, input_trace_sha256, model_revision, rows}')
    p.add_argument('--reference')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    from evaluation_gate import require_frozen, require_prediction_identity
    bundle = json.loads(Path(args.bundle).read_text(encoding='utf-8'))
    require_frozen(bundle)
    prediction = json.loads(Path(args.predictions).read_text(encoding='utf-8'))
    if any(not prediction.get(field) for field in
           ('evaluation_sha256','input_trace_sha256','model_revision')):
        raise ValueError('missing evaluation/input/model identity')
    require_prediction_identity(bundle,prediction)
    result = score(bundle['rows'], prediction['rows'])
    if args.reference:
        ref = json.loads(Path(args.reference).read_text(encoding='utf-8'))
        require_prediction_identity(bundle,ref)
        for field in ('evaluation_sha256','input_trace_sha256','model_revision'):
            if not prediction.get(field) or prediction.get(field) != ref.get(field):
                raise ValueError(f'paired identity mismatch: {field}')
        result['paired'] = paired(score(bundle['rows'], ref['rows']), result)
    result['evaluation_sha256'] = digest(bundle)
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['overall']))


if __name__ == '__main__':
    main()
