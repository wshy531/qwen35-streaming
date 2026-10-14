"""Independent projected visual features and explicit representative positions."""
from dataclasses import dataclass, replace
import itertools
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Block:
    features: torch.Tensor
    mass: torch.Tensor
    xy: torch.Tensor
    grid: tuple
    first: int
    last: int
    begin: float
    end: float
    center: float
    leaves: int = 1
    depth: int = 0

    def metadata(self):
        return {'first':self.first,'last':self.last,'begin':self.begin,'end':self.end,
                'center':self.center,'leaves':self.leaves,'depth':self.depth,
                'grid':list(self.grid),'tokens':len(self.features),
                'tensor_bytes':sum(t.numel()*t.element_size() for t in (self.features,self.mass,self.xy)),
                'exact_pair':self.leaves==1}

    def copy(self):
        return replace(self,features=self.features.clone(),mass=self.mass.clone(),xy=self.xy.clone())


def validate(block):
    if not isinstance(block,Block):raise ValueError('Visual Block required')
    tensors=(block.features,block.mass,block.xy)
    if any(not isinstance(t,torch.Tensor) or t.device.type!='cpu' or t.requires_grad for t in tensors):
        raise ValueError('Owned CPU inference tensors required')
    x,m,xy=tensors
    if (x.dtype!=torch.float32 or x.ndim!=2 or not 1<=len(x)<=48 or not 1<=x.shape[1]<=1024 or
        m.shape!=(len(x),1) or m.dtype!=torch.float32 or xy.shape!=(len(x),2) or xy.dtype!=torch.long or
        not torch.isfinite(x).all() or not torch.isfinite(m).all() or torch.any(m<=0)):
        raise ValueError('Bounded finite projected features, masses and coordinates required')
    if (block.grid!=(6,8) or torch.any(xy<0) or torch.any(xy[:,0]>=6) or torch.any(xy[:,1]>=8) or
        len(torch.unique(xy,dim=0))!=len(xy)):
        raise ValueError('Unique coordinates in the original 6x8 grid required')
    if (any(type(i) is not int or not 0<=i<=2**31-1 for i in (block.first,block.last)) or
        block.first>block.last or block.leaves!=block.last-block.first+1 or block.depth<0 or
        any(isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or
            not 0<=t<=2**31 for t in (block.begin,block.end,block.center)) or
        not block.begin<=block.center<=block.end):
        raise ValueError('Bounded event provenance required')


def raw_block(features,event_id,begin,end):
    xy=torch.tensor([(y,x) for y in range(6) for x in range(8)],dtype=torch.long)
    block=Block(features.detach().to(device='cpu',dtype=torch.float32).clone(),
                torch.ones((len(features),1)),xy,(6,8),event_id,event_id,float(begin),float(end),(begin+end)/2)
    validate(block)
    if len(block.features)!=48:raise ValueError('One complete 6x8 projected pair required')
    return block


@torch.no_grad()
def reduce(block,method):
    validate(block)
    if method=='full':return block.copy()
    if method not in ('odd','tome') or len(block.features)!=48:
        raise ValueError('Registered 48→24 odd/ToMe reduction required')
    if method=='odd':
        return replace(block,features=block.features[1::2].clone(),mass=block.mass[1::2].clone(),xy=block.xy[1::2].clone())
    metric=F.normalize(block.features,dim=-1,eps=1e-12)
    destination=(metric[::2] @ metric[1::2].T).argmax(1)
    weighted=block.features[1::2]*block.mass[1::2]
    weighted=weighted.scatter_add(0,destination[:,None].expand(-1,weighted.shape[1]),block.features[::2]*block.mass[::2])
    mass=block.mass[1::2].scatter_add(0,destination[:,None],block.mass[::2])
    result=replace(block,features=weighted/mass,mass=mass,xy=block.xy[1::2].clone())
    validate(result)
    return result


class ShortLongMemory:
    """Recent full pair + at most two compressed roots, never child tensors.

    Aligned temporal weighted pooling replaces official cross-event ToMe here.
    Events are given complete frame pairs, not recognized semantic events.
    Exports are copies. Single caller only; no rollback/concurrency guarantee.
    """
    def __init__(self):
        self.clear()

    def clear(self):
        self._roots=[]
        self._recent=None
        self.last_id=-1
        self.last_end=None
        self.merges=0
        self.peak_bytes=0

    @torch.no_grad()
    def append(self,block):
        validate(block)
        if (block.first!=self.last_id+1 or block.first!=block.last or len(block.features)!=48 or
            not torch.equal(block.xy,torch.tensor([(y,x) for y in range(6) for x in range(8)])) or
            not torch.equal(block.mass,torch.ones((48,1))) or
            (self.last_end is not None and block.begin<=self.last_end)):
            raise ValueError('Ordered complete real event pairs required')
        incoming=block.copy()
        roots=[*self._roots]
        if self._recent is not None:roots.append(self._recent)
        records=[]
        while len(roots)>2:
            counts=[(a.leaves+b.leaves)/2 for a,b in zip(roots,roots[1:])]
            scores=[]
            for i,(a,b) in enumerate(zip(roots,roots[1:])):
                similarity=float(F.cosine_similarity(a.features.mean(0),b.features.mean(0),dim=0,eps=1e-12))
                count=counts[i]/max(counts)+1e-6
                temporal=(a.center+b.center)/2/max(incoming.end,1e-12)
                scores.append({'similarity_penalty':1-similarity,'count_penalty':count,'time_penalty':temporal,
                               'total':.4*(1-similarity)+.4*count+.2*temporal})
            index=min(range(len(scores)),key=lambda i:scores[i]['total'])
            a,b=roots[index:index+2]
            if not torch.equal(a.xy,b.xy):raise ValueError('Aligned temporal pooling requires identical cells')
            mass=a.mass+b.mass
            features=(a.features*a.mass+b.features*b.mass)/mass
            leaves=a.leaves+b.leaves
            parent=Block(features,mass,a.xy.clone(),a.grid,a.first,b.last,a.begin,b.end,
                         (a.center*a.leaves+b.center*b.leaves)/leaves,leaves,max(a.depth,b.depth)+1)
            validate(parent)
            records.append({'left':a.metadata(),'right':b.metadata(),'parent':parent.metadata(),'penalty':scores[index]})
            roots[index:index+2]=[parent]
        self._roots,self._recent=roots,incoming
        self.last_id,self.last_end=block.last,block.end
        self.merges+=len(records)
        self.peak_bytes=max(self.peak_bytes,self.status()['tensor_bytes'])
        return records

    def export(self):
        return [b.copy() for b in [*self._roots,*([self._recent] if self._recent is not None else [])]]

    def status(self):
        rows=[b.metadata() for b in [*self._roots,*([self._recent] if self._recent is not None else [])]]
        return {'events':self.last_id+1,'blocks':rows,'long_roots':len(self._roots),
                'recent_pairs':int(self._recent is not None),'tokens':sum(b['tokens'] for b in rows),
                'tensor_bytes':sum(b['tensor_bytes'] for b in rows),'merges':self.merges,
                'peak_tensor_bytes':self.peak_bytes,'provenance_is_not_exact_frame':True}


@torch.inference_mode()
def context(model,builder,blocks,question,max_new_tokens=32):
    """Irregular surviving coordinates with original grid span; no fake grid."""
    from streaming.hybrid import HybridSequence
    if not blocks:raise ValueError('Past visual evidence required')
    if any(a.last>=b.first or a.end>=b.begin for a,b in zip(blocks,blocks[1:])):
        raise ValueError('Ordered nonoverlapping memory blocks required')
    suffix=builder.question_ids(question)
    if suffix.shape[1]>64 or not 1<=max_new_tokens<=32:raise ValueError('Registered QA budget exceeded')
    offset=builder.prefix_ids.shape[1]
    all_ids=[builder.prefix_ids]
    coordinates=[torch.arange(offset).expand(3,-1)]
    insertions=[]
    for block in blocks:
        validate(block)
        if block.features.shape[1]!=model.get_input_embeddings().embedding_dim:raise ValueError('Projection dimension mismatch')
        if block.leaves==1:
            label=f'<{block.center:.1f} seconds>'
        else:
            label=f'<{block.begin:.1f}-{block.end:.1f} seconds, compressed memory>'
        ids=builder.tokenize(label+builder.processor.vision_start_token+
                             builder.processor.video_token*len(block.features)+builder.processor.vision_end_token)
        types=(ids[0]==builder.processor.video_token_id).long()
        positions=[];groups=0
        for modality,values in itertools.groupby(types.tolist()):
            size=len(list(values))
            if not modality:
                positions.extend((offset+i,)*3 for i in range(size));offset+=size
            else:
                if size!=len(block.features):raise ValueError('Placeholder mismatch')
                positions.extend((offset,offset+int(y),offset+int(x)) for y,x in block.xy)
                offset+=max(block.grid);groups+=1
        if groups!=1:raise ValueError('Exactly one visual run required')
        all_ids.append(ids);coordinates.append(torch.tensor(positions).T)
        insertions.append(block)
    all_ids.append(suffix)
    coordinates.append(torch.arange(offset,offset+suffix.shape[1]).expand(3,-1))
    ids=torch.cat(all_ids,1)
    if ids.shape[1]+max_new_tokens>512:raise ValueError('Total language budget exceeded')
    positions=torch.cat((torch.arange(ids.shape[1])[None,:],torch.cat(coordinates,1)),0)[:,None,:]
    device=model.get_input_embeddings().weight.device
    embeddings=model.get_input_embeddings()(ids.to(device))
    features=torch.cat([b.features for b in insertions]).to(embeddings)
    mask=(ids.to(device)==builder.processor.video_token_id).unsqueeze(-1).expand_as(embeddings)
    if int(mask.sum())!=features.numel():raise ValueError('Embedding insertion mismatch')
    embeddings=embeddings.masked_scatter(mask,features)
    return embeddings,positions.to(device),ids,{'input_tokens':ids.shape[1],
        'visual_tokens':len(features),'blocks':[b.metadata() for b in blocks],
        'positions_use_survivor_coordinates':True,'grid_span_preserved':True,
        'sequence_axis_contiguous':True,'projected_mass_not_used_in_attention':True}


@torch.inference_mode()
def answer(model,builder,blocks,question,max_new_tokens=32):
    from streaming.hybrid import HybridSequence
    from streaming.online import fp32_reference,greedy_decode
    seq=HybridSequence(model,max_tokens=512,session_id='visual-memory-query')
    saved=model.model.rope_deltas
    try:
        with fp32_reference():
            embeddings,positions,ids,metadata=context(model,builder,blocks,question,max_new_tokens)
            logits=seq.append(inputs_embeds=embeddings,position_ids=positions,logits_mode='last')
            eos=model.generation_config.eos_token_id
            stops=set([] if eos is None else eos if isinstance(eos,list) else [eos])
            if builder.processor.tokenizer.eos_token_id is not None:stops.add(builder.processor.tokenizer.eos_token_id)
            generated,reason=greedy_decode(seq,logits,max_new_tokens,stops)
        return {'question':question,'answer':builder.processor.tokenizer.decode(generated,skip_special_tokens=True),
                'generated_ids':generated,'stop_reason':reason,**metadata}
    finally:
        seq.cache.reset()
        model.model.rope_deltas=saved


def variants(blocks):
    if len(blocks)!=6:raise ValueError('Registered six complete event pairs required')
    bank=ShortLongMemory();trace=[]
    for block in blocks:trace.extend(bank.append(block))
    memory=bank.status();exports=bank.export();bank.clear()
    return {'full_all':[b.copy() for b in blocks],
            'odd_all':[reduce(b,'odd') for b in blocks],
            'tome_all':[reduce(b,'tome') for b in blocks],
            'recent_three':[b.copy() for b in blocks[-3:]],
            'first_recent_two':[b.copy() for b in [blocks[0],*blocks[-2:]]],
            'short_long_aligned':exports},memory,trace
