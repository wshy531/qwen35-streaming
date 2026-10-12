"""Bounded query-independent real-frame anchors; no model state or summaries."""
from collections import deque
import hashlib
import math
import numpy as np
from streaming.window import FramePacket


class TemporalFrameMemory:
    """Keep every Nth completed pair, evict oldest, query only retained intervals.

    Capacity describes owned persistent RGB payload, excluding decoder/model
    working memory. Returned arrays are copies so callers cannot mutate storage.
    This module neither preserves nor edits any KV/recursive model state.
    """
    def __init__(self,max_pairs=3,every_pairs=4,max_bytes=884736):
        if any(not isinstance(x,int) or isinstance(x,bool) or x<1 for x in (max_pairs,every_pairs,max_bytes)):
            raise ValueError('Positive memory limits required')
        self.max_pairs,self.every_pairs,self.max_bytes=max_pairs,every_pairs,max_bytes
        self.clear()

    def clear(self):
        self._pairs=deque();self.observed_pairs=self.evicted_pairs=0
        self.last_id=self.last_pts=None;self.peak_bytes=0

    @staticmethod
    def validate_pair(first,second):
        for p in (first,second):
            if not isinstance(p,FramePacket) or not isinstance(p.rgb,np.ndarray) or p.rgb.dtype!=np.uint8 or p.rgb.ndim!=3 or p.rgb.shape[2]!=3:
                raise ValueError('Real uint8 RGB packets required')
            if min(p.rgb.shape[:2])<1 or max(p.rgb.shape[:2])>256 or not math.isfinite(p.timestamp) or p.timestamp<0 or p.frame_id<0 or p.decoded_count<=p.frame_id:
                raise ValueError('Invalid bounded frame identity/time/shape')
        if first.frame_id>=second.frame_id or first.timestamp>=second.timestamp or first.rgb.shape!=second.rgb.shape:
            raise ValueError('Two distinct ordered real frames required; no EOF duplication')

    @staticmethod
    def copy(p):
        rgb=p.rgb.copy();rgb.flags.writeable=False
        return FramePacket(p.frame_id,p.timestamp,rgb,p.decoded_count,p.decode_seconds)

    def add_pair(self,first,second):
        self.validate_pair(first,second)
        if self.last_id is not None and (first.frame_id<=self.last_id or first.timestamp<=self.last_pts):
            raise ValueError('Archive observation must be globally monotone')
        selected=self.observed_pairs%self.every_pairs==0
        if selected:
            pair_bytes=first.rgb.nbytes+second.rgb.nbytes
            if pair_bytes>self.max_bytes:raise ValueError('Single anchor exceeds archive byte limit')
            # Copy first; a rejected allocation must not mutate archive counters.
            pair=(self.copy(first),self.copy(second))
            while self._pairs and (len(self._pairs)>=self.max_pairs or self.bytes+pair_bytes>self.max_bytes):
                self._pairs.popleft();self.evicted_pairs+=1
            self._pairs.append(pair)
        self.observed_pairs+=1;self.last_id=second.frame_id;self.last_pts=second.timestamp
        self.peak_bytes=max(self.peak_bytes,self.bytes)
        assert self.bytes<=self.max_bytes and len(self._pairs)<=self.max_pairs
        return selected

    @property
    def bytes(self):return sum(p.rgb.nbytes for pair in self._pairs for p in pair)

    def retrieve_at(self,timestamp):
        if not isinstance(timestamp,(int,float)) or isinstance(timestamp,bool) or not math.isfinite(timestamp) or timestamp<0:
            raise ValueError('Finite nonnegative requested time required')
        if self.last_pts is None or timestamp>self.last_pts:
            return None,'not_observed'
        for pair in self._pairs:
            if pair[0].timestamp<=timestamp<=pair[1].timestamp:
                return tuple(self.copy(p) for p in pair),'retained_interval'
        return None,'not_retained'

    def evidence(self):
        return {'policy':'every_nth_real_pair_fifo','max_pairs':self.max_pairs,'every_pairs':self.every_pairs,
                'max_bytes':self.max_bytes,'persistent_rgb_bytes':self.bytes,'peak_persistent_rgb_bytes':self.peak_bytes,
                'observed_pairs':self.observed_pairs,'evicted_pairs':self.evicted_pairs,
                'intervals':[{'frame_ids':[p.frame_id for p in pair],'timestamps':[p.timestamp for p in pair],
                              'rgb_sha256':[hashlib.sha256(p.rgb.tobytes()).hexdigest() for p in pair]} for pair in self._pairs]}
