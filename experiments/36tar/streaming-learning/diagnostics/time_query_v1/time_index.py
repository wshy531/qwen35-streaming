"""Restricted question grammar -> bounded raw-pair time lookup, no semantic labels."""
import hashlib
import math
import re
from forest import validate


class RawTimeIndex:
    patterns = (
        r'Name the animal visible at ([0-9]+(?:\.[0-9]+)?) seconds\.',
        r"Is the animal's coat mostly light-colored or mostly dark-colored at ([0-9]+(?:\.[0-9]+)?) seconds\? Answer light or dark\.",
    )

    def __init__(self, max_pairs=3):
        if type(max_pairs) is not int or not 1 <= max_pairs <= 3:
            raise ValueError('One to three raw pairs required')
        self.max_pairs = max_pairs
        self.nodes = []
        self.horizon = None

    def append(self, node):
        validate(node)
        if not node.metadata()['exact_pair']:
            raise ValueError('Compressed provenance is not exact visual-time evidence')
        if self.nodes and (node.first <= self.nodes[-1].last or node.begin <= self.horizon):
            raise ValueError('Strict causal append required')
        self.nodes.append(node.copy())
        self.horizon = node.end
        if len(self.nodes) > self.max_pairs:
            self.nodes.pop(0)

    def query(self, question):
        matches = [re.fullmatch(p, question) for p in self.patterns] if isinstance(question, str) else []
        match = next((m for m in matches if m is not None), None)
        if match is None:
            return {'ok':False, 'reason':'unsupported_or_ambiguous_question'}
        stamp = float(match.group(1))
        if not math.isfinite(stamp):
            return {'ok':False, 'reason':'nonfinite_time'}
        if self.horizon is None or stamp > self.horizon:
            return {'ok':False, 'reason':'future_or_empty', 'time':stamp}
        found = [n for n in self.nodes if n.begin <= stamp <= n.end]
        if len(found) != 1:
            return {'ok':False, 'reason':'no_retained_exact_pair', 'time':stamp}
        node = found[0]
        return {'ok':True, 'time':stamp, 'selected_pair':node.first,
                'selected_interval':[node.begin,node.end], 'selection_uses':'parsed_time_and_intervals_only',
                'node':node.copy()}

    def fingerprint(self):
        return [(n.first,n.last,n.begin,n.end,*[
            hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
            for t in (n.features,n.mass,n.coords)]) for n in self.nodes]

    def clear(self):
        self.nodes.clear()
        self.horizon = None
