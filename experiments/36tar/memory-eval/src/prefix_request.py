"""Pure CPU, answer-blind draft input requests; not a model execution adapter.

Only question/options and actual prefix frames can reach the model-facing
payload. Dataset/task labels, GT, author intervals and reviewer witnesses stay
out. An odd final frame is padded in the query snapshot only, never committed
as a fabricated second real frame in the ongoing stream.
"""
import math
from pathlib import PurePosixPath

from score_memory_eval import digest


def make_request(row,media,trace):
    query=row['query_time_s']
    if not isinstance(query,(int,float)) or not math.isfinite(query) or query<0:
        raise ValueError('query must be finite, nonnegative media-relative seconds')
    if row['source_path']!=media['source_path'] or row['source_path']!=trace['source_path']:
        raise ValueError('source identities differ')
    if trace['source_sha256']!=media['sha256']:
        raise ValueError('sample trace belongs to another media asset')
    path=PurePosixPath(media['media_path'])
    if path.is_absolute() or '..' in path.parts or ':' in str(path) or '\\' in str(path):
        raise ValueError('project-relative media path required')
    if not row['question'].strip() or not row['options'] or not all(isinstance(s,str) and s.strip() for s in row['options']):
        raise ValueError('nonempty question and options required')
    frames=trace['frames']
    for i,f in enumerate(frames):
        if not math.isfinite(f['pts_s']) or f['pts_s']<0 or not isinstance(f['frame_id'],int) or f['frame_id']<0:
            raise ValueError('invalid real frame identity')
        if len(f['rgb_sha256'])!=64 or any(c not in '0123456789abcdef' for c in f['rgb_sha256']):
            raise ValueError('actual RGB SHA256 required')
        if i and (frames[i-1]['pts_s']>=f['pts_s'] or frames[i-1]['frame_id']>=f['frame_id']):
            raise ValueError('real frames must strictly increase')
    # A source trace may extend to later questions. Each request has its own
    # cutoff; annotation frames are neither consulted nor inserted here.
    prefix=[dict(frame_id=f['frame_id'],pts_s=f['pts_s'],rgb_sha256=f['rgb_sha256'])
        for f in frames if f['pts_s']<=query]
    if not prefix:raise ValueError('no real frame before query')
    pairs=[dict(frame_ids=[prefix[i]['frame_id'],prefix[i+1]['frame_id']],
        pts_s=[prefix[i]['pts_s'],prefix[i+1]['pts_s']]) for i in range(0,len(prefix)-1,2)]
    tail=None
    if len(prefix)%2:
        last=prefix[-1]
        tail=dict(frame_ids=[last['frame_id'],last['frame_id']],pts_s=[last['pts_s'],last['pts_s']],
            synthetic_second_frame=True,commit_to_main_stream=False)
    model_visible=dict(question=row['question'],options=list(row['options']),query_time_s=query,
        frames=prefix,completed_real_pairs=pairs,snapshot_tail_pair=tail)
    return dict(id=row['id'],media_path=str(path),media_sha256=media['sha256'],
        input_protocol='draft_.5fps_192x256_no_true_2b_interface_receipt',
        model_visible=model_visible,input_request_sha256=digest(model_visible))

