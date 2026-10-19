"""CPU policy adapter only: whole pair payloads, no model/cache modification.

M1 is a deterministic temporal hierarchy control, NOT a VideoChat-Online
reproduction or a semantic event detector. A node span is not retained evidence.
"""
from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Pair:
    index: int
    start_s: float
    end_s: float
    payload: object = None


@dataclass(frozen=True)
class Node:
    representative: Pair
    first_index: int
    last_index: int
    level: int = 0


class RecentFIFO:
    def __init__(self, capacity=8):
        if not isinstance(capacity,int) or isinstance(capacity,bool) or capacity < 1:
            raise ValueError('positive capacity required')
        self.capacity = capacity
        self.recent = deque(maxlen=capacity)
        self.last = None

    def append(self, pair):
        if not isinstance(pair.index,int) or pair.index < 0 or not all(
                math.isfinite(t) and t >= 0 for t in (pair.start_s,pair.end_s)) or \
                pair.start_s > pair.end_s or (self.last is not None and
                (pair.index <= self.last.index or pair.start_s < self.last.end_s)):
            raise ValueError('pairs must arrive in chronological order')
        self.recent.append(pair)
        # Only a scalar boundary, never a hidden reference to an evicted payload.
        self.last = Pair(pair.index, pair.start_s, pair.end_s)

    def read(self):
        return tuple(self.recent)


class RecentLongHierarchy(RecentFIFO):
    def __init__(self, short_capacity=4, long_capacity=4):
        super().__init__(short_capacity)
        if not isinstance(long_capacity,int) or isinstance(long_capacity,bool) or long_capacity < 1:
            raise ValueError('positive long capacity required')
        self.long_capacity = long_capacity
        self.long = []

    def append(self, pair):
        # Validate before any mutation, including promotion.
        if not isinstance(pair.index,int) or pair.index < 0 or not all(
                math.isfinite(t) and t >= 0 for t in (pair.start_s,pair.end_s)) or \
                pair.start_s > pair.end_s or (self.last is not None and
                (pair.index <= self.last.index or pair.start_s < self.last.end_s)):
            raise ValueError('pairs must arrive in chronological order')
        if len(self.recent) == self.capacity:
            old = self.recent[0]
            self.long.append(Node(old, old.index, old.index))
            if len(self.long) > self.long_capacity:
                adjacent = [i for i in range(len(self.long)-1)
                            if self.long[i].level == self.long[i+1].level]
                i = adjacent[0] if adjacent else 0
                left, right = self.long[i:i+2]
                # Keep actual unmodified earlier features, discard the other.
                self.long[i:i+2] = [Node(left.representative, left.first_index,
                    right.last_index, max(left.level, right.level)+1)]
        super().append(pair)

    def read(self):
        return tuple(n.representative for n in self.long) + tuple(self.recent)

    def provenance(self):
        return tuple((n.first_index, n.last_index, n.representative.index, n.level)
                     for n in self.long)
