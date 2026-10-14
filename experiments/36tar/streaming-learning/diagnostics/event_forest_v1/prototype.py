"""Bounded CPU event-token consolidation inspired by StreamForest; not a VLM."""
from dataclasses import dataclass
import json
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Node:
    features: torch.Tensor
    mass: torch.Tensor
    begin_id: int
    end_id: int
    begin_time: float
    end_time: float
    centroid: float
    leaves: int = 1
    merge_steps: int = 0
    depth: int = 0

    def metadata(self):
        return {'begin_id':self.begin_id,'end_id':self.end_id,'begin_time':self.begin_time,
                'end_time':self.end_time,'centroid':self.centroid,'leaves':self.leaves,
                'merge_steps':self.merge_steps,'depth':self.depth,'tokens':len(self.features)}


def normalized_metric(x,heads):
    return F.normalize(x.reshape(len(x),heads,-1).mean(1),dim=-1,eps=1e-12)


@torch.no_grad()
def soft_merge(x,mass,heads):
    """Balanced ToMe-style matching, carrying original token mass across levels."""
    metric=normalized_metric(x,heads)
    scores=metric[::2] @ metric[1::2].T
    best,destination=scores.max(dim=1)
    remove=len(x)//2
    order=torch.argsort(best,descending=True,stable=True)
    chosen,unmerged=order[:remove],order[remove:]
    src,dst=x[::2],x[1::2]
    src_mass,dst_mass=mass[::2],mass[1::2]
    weighted=dst*dst_mass
    weighted=weighted.scatter_add(0,destination[chosen,None].expand(-1,x.shape[1]),src[chosen]*src_mass[chosen])
    weights=dst_mass.scatter_add(0,destination[chosen,None],src_mass[chosen])
    result=torch.cat((src[unmerged],weighted/weights),dim=0)
    counts=torch.cat((src_mass[unmerged],weights),dim=0)
    assert len(result)==math.ceil(len(x)/2) and torch.isfinite(result).all()
    return result,counts,float(best.topk(remove).values.mean())


class EventForest:
    """Only compressed roots retained; provenance spans are not exact frames.

    Input events are supplied by the caller, not automatically recognized.
    No child tensors, history list or generated text lives in this object.
    Every append consolidates; exports are copies. Not concurrent/reentrant.
    """
    def __init__(self,dim=16,heads=4,token_budget=64,max_event_tokens=16,
                 weights=(.4,.4,.2),metadata_budget=16384,sink=None):
        if any(type(x) is not int or x<1 for x in (dim,heads,token_budget,max_event_tokens,metadata_budget)):
            raise ValueError('Positive integer budgets required')
        if dim%heads or max_event_tokens>token_budget or metadata_budget<512:
            raise ValueError('Aligned heads, one-event room and >=512 metadata bytes required')
        if len(weights)!=3 or any(not math.isfinite(w) or w<0 for w in weights) or sum(weights)<=0:
            raise ValueError('Three finite nonnegative weights required')
        self.dim,self.heads=dim,heads
        self.token_budget,self.max_event_tokens=token_budget,max_event_tokens
        self.weights=tuple(float(w) for w in weights)
        self.metadata_budget,self.sink=metadata_budget,sink
        self.clear()

    def clear(self):
        self._roots=[]
        self.last_id=-1
        self.last_time=None
        self.merges=0
        self.peak_tokens=0
        self.peak_tensor_bytes=0
        self.peak_pair_matrix_bytes=0

    @staticmethod
    def tokens(nodes):return sum(len(n.features) for n in nodes)
    @staticmethod
    def metadata_bytes(nodes):return len(json.dumps([n.metadata() for n in nodes],separators=(',',':')).encode())

    def penalties(self,nodes,now):
        raw=[]
        for a,b in zip(nodes,nodes[1:]):
            x=torch.cat((a.features,b.features));metric=normalized_metric(x,self.heads)
            scores=metric[::2] @ metric[1::2].T
            r=len(x)//2
            similarity=float(scores.max(dim=1).values.topk(r).values.mean())
            self.peak_pair_matrix_bytes=max(self.peak_pair_matrix_bytes,scores.numel()*scores.element_size())
            raw.append((1-similarity,(a.leaves+b.leaves)/2,(a.centroid+b.centroid)/2/max(now,1e-12)))
        max_count=max(row[1] for row in raw)
        return [{'similarity':s,'merge_count':c/max_count+1e-6,'time':t,
                 'total':self.weights[0]*s+self.weights[1]*(c/max_count+1e-6)+self.weights[2]*t}
                for s,c,t in raw]

    @torch.no_grad()
    def append(self,event_id,begin,end,features):
        if type(event_id) is not int or event_id!=self.last_id+1 or event_id>2**31-1:
            raise ValueError('Consecutive bounded event IDs required')
        if (any(isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) or t<0 for t in (begin,end)) or
            begin>end or (self.last_time is not None and begin<=self.last_time)):
            raise ValueError('Ordered nonoverlapping finite event times required')
        if (not isinstance(features,torch.Tensor) or features.device.type!='cpu' or features.dtype!=torch.float32 or
            features.ndim!=2 or features.shape[1]!=self.dim or not 1<=len(features)<=self.max_event_tokens or
            features.requires_grad or not torch.isfinite(features).all()):
            raise ValueError('Bounded finite CPU float32 event features required')
        x=features.detach().clone()
        nodes=[*self._roots,Node(x,torch.ones((len(x),1),dtype=torch.float32),event_id,event_id,float(begin),float(end),(begin+end)/2)]
        records=[]
        while self.tokens(nodes)>self.token_budget or self.metadata_bytes(nodes)>self.metadata_budget:
            if len(nodes)<2:raise RuntimeError('Budget cannot fit one root')
            scores=self.penalties(nodes,float(end))
            index=min(range(len(scores)),key=lambda i:scores[i]['total'])
            a,b=nodes[index:index+2]
            values,mass,_=soft_merge(torch.cat((a.features,b.features)),torch.cat((a.mass,b.mass)),self.heads)
            leaves=a.leaves+b.leaves
            parent=Node(values,mass,a.begin_id,b.end_id,a.begin_time,b.end_time,
                        (a.centroid*a.leaves+b.centroid*b.leaves)/leaves,leaves,
                        a.merge_steps+b.merge_steps+1,max(a.depth,b.depth)+1)
            records.append({'kind':'merge','left':a.metadata(),'right':b.metadata(),
                            'selected_index':index,'penalty':scores[index],'parent':parent.metadata(),
                            'child_tensors_retained':False})
            nodes[index:index+2]=[parent]
        self._roots=nodes
        self.last_id,self.last_time=event_id,float(end)
        self.merges+=len(records)
        self.peak_tokens=max(self.peak_tokens,self.tokens(nodes))
        self.peak_tensor_bytes=max(self.peak_tensor_bytes,self.tensor_bytes())
        if self.sink:
            for record in records:self.sink(record)
            self.sink({'kind':'append','event_id':event_id,**self.status()})
        return self.status()

    def tensor_bytes(self):
        return sum(t.numel()*t.element_size() for n in self._roots for t in (n.features,n.mass))

    def export(self):
        return [(n.metadata(),n.features.clone(),n.mass.clone()) for n in self._roots]

    def status(self):
        return {'events':self.last_id+1,'roots':[n.metadata() for n in self._roots],
                'tokens':self.tokens(self._roots),'tensor_bytes':self.tensor_bytes(),
                'metadata_serialized_bytes':self.metadata_bytes(self._roots),'merges':self.merges,
                'peak_tokens_after_append':self.peak_tokens,'peak_tensor_bytes_after_append':self.peak_tensor_bytes,
                'peak_pair_matrix_bytes':self.peak_pair_matrix_bytes,
                'span_means_compressed_provenance_not_exact_frame':True}
