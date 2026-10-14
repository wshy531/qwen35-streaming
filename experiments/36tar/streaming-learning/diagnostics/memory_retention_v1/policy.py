"""Experimental first-anchor + recent-pairs retention, outside the default core."""
from collections import deque
from streaming.frame_memory import TemporalFrameMemory


class AnchorRecentMemory(TemporalFrameMemory):
    """Keep the first sampled pair and use remaining space for recent evidence.

    If the byte cap cannot fit both anchor and newest pair, evict the anchor.
    This is a two-age policy, not semantic selection or a memory forest.
    """
    def __init__(self,max_pairs=3,every_pairs=1,max_bytes=884736):
        if type(max_pairs) is not int or max_pairs<2:
            raise ValueError('Anchor/recent policy requires at least two pair slots')
        super().__init__(max_pairs,every_pairs,max_bytes)

    def clear(self):
        super().clear()
        self.anchor_seed_id=None
        self.anchor_released=0

    def add_pair(self,first,second):
        self.validate_pair(first,second)
        if self.last_id is not None and (first.frame_id<=self.last_id or first.timestamp<=self.last_pts):
            raise ValueError('Memory observations must be globally monotone')
        selected=self.observed_pairs%self.every_pairs==0
        if selected:
            size=first.rgb.nbytes+second.rgb.nbytes
            if size>self.max_bytes:raise ValueError('One pair exceeds memory budget')
            pair=(self.copy(first),self.copy(second))
            candidates=list(self._pairs)+[pair]
            seed=first.frame_id if self.anchor_seed_id is None else self.anchor_seed_id
            def payload():return sum(p.rgb.nbytes for group in candidates for p in group)
            while len(candidates)>self.max_pairs or payload()>self.max_bytes:
                protected=candidates[0][0].frame_id==seed
                index=1 if protected and len(candidates)>2 else 0
                removed=candidates.pop(index)
                self.evicted_pairs+=1
                if removed[0].frame_id==seed:self.anchor_released+=1
            self._pairs=deque(candidates)
            self.anchor_seed_id=seed
        self.observed_pairs+=1
        self.last_id,self.last_pts=second.frame_id,second.timestamp
        self.peak_bytes=max(self.peak_bytes,self.bytes)
        assert self.bytes<=self.max_bytes and len(self._pairs)<=self.max_pairs
        return selected

    def evidence(self):
        value=super().evidence()
        value.update(policy='first_sampled_anchor_plus_recent_pairs',anchor_seed_id=self.anchor_seed_id,
                     anchor_retained=any(p[0].frame_id==self.anchor_seed_id for p in self._pairs),
                     anchor_released=self.anchor_released)
        return value
