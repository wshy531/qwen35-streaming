"""Three bounded grid-block slots divided into recent and compressed history."""
from dataclasses import replace

import torch
import torch.nn.functional as F
from features import validate,raw_block,answer


class SplitMemory:
    def __init__(self,recent):
        if recent not in (1,2,3):raise ValueError('Registered 1/2/3 recent block budget')
        self.recent_capacity=recent
        self.long_capacity=3-recent
        self.clear()

    def clear(self):
        self.recent=[];self.roots=[];self.last_id=-1;self.last_end=None;self.merges=0;self.discarded=0

    @torch.no_grad()
    def append(self,block):
        validate(block)
        if (block.first!=block.last or block.first!=self.last_id+1 or len(block.features)!=48 or
            not torch.equal(block.mass,torch.ones((48,1))) or
            (self.last_end is not None and block.begin<=self.last_end)):
            raise ValueError('Consecutive full pair events required')
        item=block.copy();self.recent.append(item)
        if len(self.recent)>self.recent_capacity:
            expired=self.recent.pop(0)
            if self.long_capacity:self.roots.append(expired)
            else:self.discarded+=1
        while len(self.roots)>self.long_capacity:
            counts=[(a.leaves+b.leaves)/2 for a,b in zip(self.roots,self.roots[1:])]
            penalties=[]
            for i,(a,b) in enumerate(zip(self.roots,self.roots[1:])):
                similarity=float(F.cosine_similarity(a.features.mean(0),b.features.mean(0),dim=0,eps=1e-12))
                penalties.append(.4*(1-similarity)+.4*(counts[i]/max(counts)+1e-6)+.2*((a.center+b.center)/2/max(item.end,1e-12)))
            index=min(range(len(penalties)),key=lambda i:penalties[i])
            a,b=self.roots[index:index+2]
            if not torch.equal(a.xy,b.xy):raise ValueError('Aligned spatial cells required')
            mass=a.mass+b.mass;leaves=a.leaves+b.leaves
            parent=replace(a,features=(a.features*a.mass+b.features*b.mass)/mass,mass=mass,
                last=b.last,end=b.end,center=(a.center*a.leaves+b.center*b.leaves)/leaves,
                leaves=leaves,depth=max(a.depth,b.depth)+1,xy=a.xy.clone())
            validate(parent);self.roots[index:index+2]=[parent];self.merges+=1
        self.last_id,self.last_end=block.last,block.end

    def status(self):
        rows=[b.metadata() for b in [*self.roots,*self.recent]]
        return {'recent_capacity':self.recent_capacity,'long_capacity':self.long_capacity,
                'events':self.last_id+1,'blocks':rows,'tokens':sum(r['tokens'] for r in rows),
                'tensor_bytes':sum(r['tensor_bytes'] for r in rows),'merges':self.merges,'discarded_pairs':self.discarded}


class TieredSession:
    def __init__(self,session):
        self.session=session;self.banks={i:SplitMemory(i) for i in (1,2,3)}
        self.hashes=[];self.since_reset=False;self.invalid=False

    @torch.inference_mode()
    def append(self,packet):
        if self.invalid:raise RuntimeError('Invalid wrapper')
        if self.banks[1].last_id>=7:raise ValueError('Registered eight-pair limit')
        first=self.session.pending;captured=[]
        def hook(_m,_a,output):
            value=output.pooler_output
            if not isinstance(value,torch.Tensor) or value.shape[0]!=48 or captured:
                raise ValueError('Single 48-token projected pair required')
            captured.append(value.detach().to('cpu',dtype=torch.float32).clone())
        handle=self.session.model.model.visual.register_forward_hook(hook)
        try:
            result=self.session.append(packet)
            if captured:
                if first is None or self.session.last_encoded_id!=packet.frame_id:raise ValueError('Successful main pair required')
                block=raw_block(captured[0],self.banks[1].last_id+1,first.timestamp,packet.timestamp)
                for bank in self.banks.values():bank.append(block)
                from experiment import tensor_sha
                self.hashes.append(tensor_sha(block.features));self.since_reset=True
            return result
        except Exception:
            self.invalid=True;raise
        finally:handle.remove();captured.clear()

    def snapshot(self):
        from experiment import tensor_sha
        return {i:{'status':bank.status(),'hashes':[[tensor_sha(b.features),tensor_sha(b.mass),tensor_sha(b.xy)]
                         for b in [*bank.roots,*bank.recent]]} for i,bank in self.banks.items()}

    def reset_main(self):
        before=self.snapshot();self.session.reset();self.since_reset=False
        if self.snapshot()!=before:raise AssertionError('Main reset changed external feature banks')

    @staticmethod
    def deny(reason):return {'model_called':False,'reason':reason,'answer':None}

    def current(self,recent,question):
        if self.invalid:raise RuntimeError('Invalid wrapper')
        if self.session.pending is not None:return self.deny('waiting_for_complete_pair')
        if not self.since_reset:return self.deny('no_current_pair_in_epoch')
        rows=[b.copy() for b in self.banks[recent].recent]
        result=answer(self.session.model,self.session.builder,rows,question,16)
        return {**result,'model_called':True,'recent_capacity':recent,
                'evidence_interval':[rows[0].begin,rows[-1].end],'evidence_scope':'unmerged_recent_pairs'}

    def coarse(self,recent,question):
        if self.invalid:raise RuntimeError('Invalid wrapper')
        bank=self.banks[recent]
        if not bank.roots:return self.deny('no_long_term_root')
        block=bank.roots[0].copy()
        result=answer(self.session.model,self.session.builder,[block],question,16)
        return {**result,'model_called':True,'evidence_interval':[block.begin,block.end],
                'evidence_scope':'whole_root_interval','compressed':block.leaves>1}

    def exact_available(self,timestamp):
        bank=self.banks[2]
        if bank.last_end is None or timestamp>bank.last_end:return self.deny('beyond_encoded_horizon')
        block=next((b for b in [*bank.roots,*bank.recent] if b.begin<=timestamp<=b.end),None)
        if block is None:return self.deny('no_retained_pair')
        if block.leaves>1:return self.deny('compressed_only_no_exact_pair')
        return {'model_called':False,'reason':'unmerged_pair_available','answer':None}

    def close(self):
        self.session.close()
        for bank in self.banks.values():bank.clear()
        self.hashes.clear()
