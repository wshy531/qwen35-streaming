"""Read features from successful main writes, query explicit evidence scope."""
import math

import torch
from features import ShortLongMemory,raw_block,answer


class FeatureWindowSession:
    """Independent wrapper, one caller; no atomic rollback/concurrency promise."""
    def __init__(self,session):
        self.session=session
        self.memory=ShortLongMemory()
        self.hashes=[]  # Registered finite experiment: <=6 checksums, no tensors.
        self.invalid=False

    @torch.inference_mode()
    def append(self,packet):
        if self.invalid:raise RuntimeError('Wrapper invalid; close required')
        if self.memory.last_id>=5:raise ValueError('Registered six-pair experiment limit')
        first=self.session.pending
        captured=[]
        def hook(_module,_inputs,output):
            features=output.pooler_output
            if not isinstance(features,torch.Tensor) or features.ndim!=2 or features.shape[0]!=48:
                raise ValueError('Expected one complete projected pair')
            if captured:raise ValueError('One vision call per complete pair')
            captured.append(features.detach().to('cpu',dtype=torch.float32).clone())
        handle=self.session.model.model.visual.register_forward_hook(hook)
        try:
            result=self.session.append(packet)
            if captured:
                if first is None or self.session.last_encoded_id!=packet.frame_id:
                    raise ValueError('Capture must correspond to successful completed main pair')
                block=raw_block(captured[0],self.memory.last_id+1,first.timestamp,packet.timestamp)
                from experiment import tensor_sha
                self.hashes.append(tensor_sha(block.features))
                self.memory.append(block)
            return result
        except Exception:
            self.invalid=True
            raise
        finally:
            handle.remove()
            captured.clear()

    def _deny(self,reason,**extra):
        return {'model_called':False,'reason':reason,'answer':None,**extra}

    def _ready(self):
        if self.invalid or self.session.closed or self.session.invalid:
            raise RuntimeError('Ready feature window required')

    def _ask(self,block,question,scope):
        result=answer(self.session.model,self.session.builder,[block],question,16)
        return {**result,'model_called':True,'evidence_scope':scope,
                'evidence_interval':[block.begin,block.end],
                'compressed':block.leaves>1,'exact_pair_evidence':block.leaves==1}

    def current(self,question):
        self._ready()
        if self.session.pending is not None:return self._deny('waiting_for_complete_pair')
        blocks=self.memory.export()
        if not blocks:return self._deny('no_encoded_evidence')
        return self._ask(blocks[-1],question,'latest_complete_pair')

    def at(self,timestamp,question):
        self._ready()
        if (isinstance(timestamp,bool) or not isinstance(timestamp,(int,float)) or
            not math.isfinite(timestamp) or timestamp<0):raise ValueError('Finite nonnegative target time required')
        if self.session.last_encoded_pts is None or timestamp>self.session.last_encoded_pts:
            return self._deny('beyond_encoded_horizon',target_time=timestamp)
        block=next((b for b in self.memory.export() if b.begin<=timestamp<=b.end),None)
        if block is None:return self._deny('no_retained_pair',target_time=timestamp)
        if block.leaves>1:
            return self._deny('compressed_only_no_exact_pair',target_time=timestamp,
                              available_compressed_interval=[block.begin,block.end])
        return {**self._ask(block,question,'retained_complete_pair'),'target_time':timestamp}

    def coarse(self,index,question):
        self._ready()
        if type(index) is not int or not 0<=index<len(self.memory._roots):
            return self._deny('no_long_term_root')
        return self._ask(self.memory.export()[index],question,'whole_long_term_interval')

    def close(self):
        self.session.close()
        self.memory.clear()
        self.hashes.clear()
