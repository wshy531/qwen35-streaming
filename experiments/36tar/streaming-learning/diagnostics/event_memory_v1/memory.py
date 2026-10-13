"""Causal change grouping with retained real representatives, no temporal blending."""
from dataclasses import dataclass
import json
import math

import torch
import torch.nn.functional as F
from features import raw_block, validate


def distance(left, right):
    a = left.features.mean(0) if hasattr(left, 'features') else left
    b = right.features.mean(0) if hasattr(right, 'features') else right
    # A zero vector gives a conservative maximal change instead of NaN.
    if float(a.norm()) == 0 or float(b.norm()) == 0:
        return 1.0
    return max(0.0, min(2.0, 1.0 - float(F.cosine_similarity(a, b, dim=0, eps=1e-12))))


def fit_threshold(blocks):
    """Largest gap of calibration distances only; no source labels or QA input."""
    if len(blocks) != 6:
        raise ValueError('Registered six-pair unlabeled calibration required')
    values = [distance(a, b) for a, b in zip(blocks, blocks[1:])]
    ordered = sorted(values)
    gaps = [b - a for a, b in zip(ordered, ordered[1:])]
    index = max(range(len(gaps)), key=lambda i: gaps[i])
    threshold = (ordered[index] + ordered[index + 1]) / 2 if gaps[index] > 1e-6 else .1
    return {'threshold': threshold, 'adjacent_distances': values,
            'rule': 'largest_sorted_gap_midpoint_no_labels', 'fallback_used': gaps[index] <= 1e-6}


@dataclass
class Group:
    representative: object
    first: int
    last: int
    begin: float
    end: float

    def metadata(self):
        return {'group_first': self.first, 'group_last': self.last,
                'group_begin': self.begin, 'group_end': self.end,
                'representative': self.representative.metadata(),
                'scope': 'one_real_pair_sample_not_complete_group_evidence'}


class EventMemory:
    """Latest pair plus at most two closed change groups, last real pair per group.

    Changes compare the two already arrived adjacent pairs. No labels,
    child feature tensors, whole history, or generated summaries are retained.
    Closed oldest groups are evicted when the fixed three-slot budget is full.
    """
    def __init__(self, threshold, oracle_boundaries=None):
        if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 2:
            raise ValueError('Finite cosine-distance threshold in [0,2] required')
        self.threshold = float(threshold)
        self.oracle_boundaries = None if oracle_boundaries is None else frozenset(oracle_boundaries)
        self.clear()

    def clear(self):
        self.groups = []
        self.current = None
        self.open_first = None
        self.open_begin = None
        self.last_id = -1
        self.last_end = None
        self.boundaries = 0
        self.evicted_groups = 0
        self.evicted_pairs = 0

    @torch.no_grad()
    def append(self, block):
        validate(block)
        if (block.first != block.last or block.first != self.last_id + 1 or len(block.features) != 48 or
            not torch.equal(block.mass, torch.ones_like(block.mass)) or
            (self.last_end is not None and block.begin <= self.last_end)):
            raise ValueError('Consecutive ordered full pairs required')
        incoming = block.copy()
        change = None
        split = False
        if self.current is not None:
            expired = self.current
            change = distance(expired, incoming)
            split = (incoming.first in self.oracle_boundaries if self.oracle_boundaries is not None
                     else change > self.threshold)
            if split:
                self.groups.append(Group(expired, self.open_first, expired.last, self.open_begin, expired.end))
                self.boundaries += 1
                self.open_first, self.open_begin = incoming.first, incoming.begin
            # Within an ongoing group only the latest pair is retained.
            if len(self.groups) > 2:
                old = self.groups.pop(0)
                self.evicted_groups += 1
                self.evicted_pairs += old.last - old.first + 1
        else:
            self.open_first, self.open_begin = incoming.first, incoming.begin
        self.current = incoming
        self.last_id, self.last_end = block.last, block.end
        return {'processed_pair': block.first, 'adjacent_pair_distance': change, 'closed_old_group': split,
                'status': self.status()}

    def export(self):
        return [g.representative.copy() for g in self.groups] + ([] if self.current is None else [self.current.copy()])

    def status(self):
        blocks = [g.representative for g in self.groups] + ([] if self.current is None else [self.current])
        feature_bytes = sum(b.metadata()['tensor_bytes'] for b in blocks)
        auxiliary = 0
        groups = [g.metadata() for g in self.groups]
        return {'tokens': sum(len(b.features) for b in blocks), 'tensor_bytes': feature_bytes + auxiliary,
                'visual_tensor_bytes': feature_bytes, 'detector_tensor_bytes': auxiliary,
                'groups': groups, 'current': None if self.current is None else self.current.metadata(),
                'open_group_first': self.open_first, 'open_group_begin': self.open_begin,
                'events_received': self.last_id + 1, 'groups_started': self.boundaries,
                'evicted_groups': self.evicted_groups, 'evicted_pairs': self.evicted_pairs,
                'metadata_json_bytes': len(json.dumps(groups, sort_keys=True).encode()),
                'group_span_does_not_imply_exact_evidence': True,
                'oracle_diagnostic': self.oracle_boundaries is not None}

    def exact_available(self, timestamp):
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Finite nonnegative target time required')
        if self.last_end is None or timestamp > self.last_end:
            return {'model_called': False, 'reason': 'beyond_encoded_horizon'}
        blocks = [g.representative for g in self.groups] + ([] if self.current is None else [self.current])
        if any(b.begin <= timestamp <= b.end for b in blocks):
            return {'model_called': False, 'reason': 'retained_real_pair_available'}
        if any(g.begin <= timestamp <= g.end for g in self.groups):
            return {'model_called': False, 'reason': 'group_span_only_no_exact_pair'}
        if self.open_begin is not None and self.open_begin <= timestamp <= self.last_end:
            return {'model_called': False, 'reason': 'group_span_only_no_exact_pair'}
        return {'model_called': False, 'reason': 'no_retained_pair'}
