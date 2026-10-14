"""Latest observation and causal event representative have distinct lifetimes."""
from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from features import validate


def descriptor(block):
    value = block.features.mean(0)
    return F.normalize(value, dim=0, eps=1e-12)


def score(value, center):
    if float(value.norm()) == 0 or float(center.norm()) == 0:
        return 1.0
    return max(0., min(2., 1. - float(F.cosine_similarity(value, center, dim=0, eps=1e-12))))


@dataclass
class Group:
    representative: object
    first: int
    last: int
    begin: float
    end: float

    def metadata(self):
        return {'first': self.first, 'last': self.last, 'begin': self.begin, 'end': self.end,
                'representative_pair': self.representative.first,
                'actual_pair_interval': [self.representative.begin, self.representative.end],
                'one_pair_not_whole_group': True}


class RepresentativeBank:
    def __init__(self, policy):
        if policy not in ('first', 'last', 'centroid'):
            raise ValueError('Fixed first/last/centroid policy required')
        self.policy = policy
        self.clear()

    def clear(self):
        self.groups = []
        self.latest = self.representative = self.mean = None
        self.count = 0
        self.open_first = self.open_begin = None
        self.last_id = -1
        self.last_end = None
        self.evicted = 0

    def resident_blocks(self):
        candidates = [g.representative for g in self.groups]
        candidates += [b for b in (self.representative, self.latest) if b is not None]
        # The latest pair and the representative share one owned Block when equal.
        return list({b.first: b for b in candidates}.values())

    @torch.no_grad()
    def append(self, block, boundary=False):
        validate(block)
        if (type(boundary) is not bool or block.first != block.last or block.first != self.last_id + 1
                or len(block.features) != 48 or not torch.equal(block.mass, torch.ones_like(block.mass))
                or (self.last_end is not None and block.begin <= self.last_end)):
            raise ValueError('Consecutive ordered full real pair and explicit shared boundary required')
        if self.latest is not None and block.features.shape != self.latest.features.shape:
            raise ValueError('Projection width must stay fixed')
        incoming = block.copy()
        if self.latest is not None and boundary:
            self.groups.append(Group(self.representative, self.open_first, self.latest.last,
                                     self.open_begin, self.latest.end))
            if len(self.groups) > 2:
                self.groups.pop(0)
                self.evicted += 1
            self.representative = self.mean = None
            self.count = 0
        value = descriptor(incoming)
        replace = self.representative is None
        if replace:
            self.open_first, self.open_begin = incoming.first, incoming.begin
            self.mean = value
        else:
            self.mean = (self.mean * self.count + value) / (self.count + 1)
            if self.policy == 'last':
                replace = True
            elif self.policy == 'centroid':
                replace = score(value, self.mean) + 1e-6 < score(descriptor(self.representative), self.mean)
        if replace:
            self.representative = incoming
        self.latest = incoming
        self.count += 1
        self.last_id, self.last_end = incoming.first, incoming.end
        return {'boundary': boundary, 'representative_replaced': replace, 'status': self.status()}

    def status(self):
        blocks = self.resident_blocks()
        visual_bytes = sum(b.metadata()['tensor_bytes'] for b in blocks)
        mean_bytes = 0 if self.mean is None else self.mean.numel() * self.mean.element_size()
        return {'policy': self.policy, 'groups': [g.metadata() for g in self.groups],
                'open_first': self.open_first, 'open_begin': self.open_begin, 'open_count': self.count,
                'open_representative': None if self.representative is None else self.representative.first,
                'latest_pair': None if self.latest is None else self.latest.first,
                'tokens': sum(len(b.features) for b in blocks), 'visual_tensor_bytes': visual_bytes,
                'mean_tensor_bytes': mean_bytes, 'tensor_bytes': visual_bytes + mean_bytes,
                'last_id': self.last_id, 'last_end': self.last_end, 'evicted_groups': self.evicted,
                'shared_latest_and_representative': self.latest is not None and self.latest is self.representative,
                'whole_history_not_retained': True, 'centroid_is_online_candidate_not_exact_medoid': True}

    def select(self, mode, target=None):
        span = None
        if mode == 'current':
            block = self.latest
        elif mode == 'open':
            block = self.representative
            if block is not None:
                span = [self.open_begin, self.last_end]
        elif mode == 'group':
            if type(target) is not int or target < 0:
                raise ValueError('Nonnegative retained group index required')
            if target >= len(self.groups):
                return {'model_called': False, 'reason': 'no_closed_group'}, []
            group = self.groups[target]
            block, span = group.representative, [group.begin, group.end]
        elif mode == 'exact':
            if not isinstance(target, (int, float)) or not math.isfinite(target) or target < 0:
                raise ValueError('Finite nonnegative time required')
            if self.last_end is None or target > self.last_end:
                return {'model_called': False, 'reason': 'beyond_encoded_horizon'}, []
            block = next((b for b in self.resident_blocks() if b.begin <= target <= b.end), None)
            if block is None:
                return {'model_called': False, 'reason': 'no_exact_retained_pair'}, []
        else:
            raise ValueError('Explicit current/open/group/exact mode required')
        if block is None:
            return {'model_called': False, 'reason': 'no_evidence'}, []
        return {'reason': 'selected', 'mode': mode, 'pair_id': block.first,
                'actual_pair_interval': [block.begin, block.end], 'group_interval': span,
                'one_pair_not_whole_group': span is not None}, [block.copy()]
