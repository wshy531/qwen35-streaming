"""Question likelihood selection inside a fixed M1 bank; no labels/options API."""
import math
import time
import torch


def choose(bank,scores=None,*,kind='uniform',intervals=()):
    if not bank:return []
    if len(bank)<=4:return list(bank)
    if kind=='recent':return list(bank[-4:])
    older=list(bank[:-2]);recent=list(bank[-2:])
    if kind=='uniform':picks=[older[0],older[-1]]
    elif kind=='question':
        if scores is None or set(scores)!={n.first for n in older}:raise ValueError('score every eligible stored node')
        if not all(math.isfinite(x) for x in scores.values()):raise ValueError('finite question scores')
        picks=sorted(older,key=lambda n:(-scores[n.first],n.first))[:2]
    elif kind=='annotation':
        intervals=intervals or ()
        def hit(n):return sum(a<=t<=b for a,b in intervals for t in [n.begin,n.end])
        picks=sorted(older,key=lambda n:(-hit(n),n.first))[:2]
    else:raise ValueError('unknown fixed selector')
    result=sorted([*picks,*recent],key=lambda n:n.first)
    if len(result)!=4 or len({n.first for n in result})!=4:raise ValueError('four distinct stored pairs')
    return result


def question_span(builder,ids,question):
    target=builder.tokenize(question)[0].tolist();allids=ids[0].tolist()
    starts=[i for i in range(len(allids)-len(target)+1) if allids[i:i+len(target)]==target]
    if len(starts)!=1 or starts[0]<1 or len(target)>128:raise ValueError('unique untruncated original question token span required')
    return starts[0],starts[0]+len(target)


@torch.inference_mode()
def score_node(env,model,builder,node,question):
    # Original question only. No choice strings, answer, annotated interval or other nodes.
    saved=model.model.rope_deltas;session=None;start=time.perf_counter()
    emb,pos,ids,meta=env['context'](model,builder,[node],question,
        limits=env['Limits'](8192,6144,128),max_new_tokens=8)
    first,end=question_span(builder,ids,question)
    try:
        session=env['HybridSequence'](model,max_tokens=8192,session_id='question-compatibility')
        initial=session.append(inputs_embeds=emb[:,:first],position_ids=pos[:,:,:first],logits_mode='last')
        following=session.append(inputs_embeds=emb[:,first:end],position_ids=pos[:,:,first:end],logits_mode='full')
        logits=torch.cat((initial,following[:,:-1]),1).float()
        labels=ids[:,first:end].to(logits.device)
        token_logp=torch.log_softmax(logits,-1).gather(-1,labels[:,:,None]).squeeze(-1)
        if not torch.isfinite(token_logp).all():raise FloatingPointError('question likelihood nonfinite')
        return float(token_logp.mean()),dict(pair=node.first,question_tokens=end-first,
            question_ids=labels[0].tolist(),score_mean_logp=float(token_logp.mean()),
            scorer_input_tokens=end,seconds=time.perf_counter()-start)
    finally:
        if session:session.cache.reset()
        model.model.rope_deltas=saved


@torch.inference_mode()
def question_select(env,model,builder,bank,question):
    if len(bank)<=4:return list(bank),[]
    records=[];scores={}
    for node in bank[:-2]:
        score,record=score_node(env,model,builder,node,question)
        scores[node.first]=score;records.append(record)
    return choose(bank,scores,kind='question'),records
