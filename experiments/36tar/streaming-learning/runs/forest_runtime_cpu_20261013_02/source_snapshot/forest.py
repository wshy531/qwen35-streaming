"""Causal event forest with real token provenance; compressed positions are heuristic."""
from dataclasses import dataclass
import hashlib
import itertools
import json
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Node:
    features: torch.Tensor
    mass: torch.Tensor
    coords: torch.Tensor  # original pair ID, row, column of the surviving destination
    first: int
    last: int
    begin: float
    end: float
    center: float
    groups: int = 1
    depth: int = 0

    def copy(self):
        return Node(self.features.clone(), self.mass.clone(), self.coords.clone(), self.first,
                    self.last, self.begin, self.end, self.center, self.groups, self.depth)

    def metadata(self):
        return {'first': self.first, 'last': self.last, 'begin': self.begin, 'end': self.end,
                'center': self.center, 'groups': self.groups, 'depth': self.depth,
                'tokens': len(self.features), 'original_pairs': self.last-self.first+1,
                'mass': float(self.mass.sum()), 'surviving_source_pair_ids': sorted(set(self.coords[:, 0].tolist())),
                'tensor_bytes': sum(t.numel()*t.element_size() for t in (self.features, self.mass, self.coords)),
                'exact_pair': self.first == self.last and len(self.features) == 48 and bool(torch.all(self.mass == 1)),
                'span_is_provenance_not_full_detail': True}


def validate(node):
    if not isinstance(node, Node):
        raise ValueError('Position-aware node required')
    x, m, c = node.features, node.mass, node.coords
    if any(t.device.type != 'cpu' or t.requires_grad for t in (x, m, c)):
        raise ValueError('Owned CPU inference tensors required')
    if (x.dtype != torch.float32 or x.ndim != 2 or not 1 <= len(x) <= 48 or not 1 <= x.shape[1] <= 1024
            or m.dtype != torch.float32 or m.shape != (len(x), 1) or c.dtype != torch.long
            or c.shape != (len(x), 3) or not torch.isfinite(x).all() or not torch.isfinite(m).all()
            or bool(torch.any(m <= 0)) or len(torch.unique(c, dim=0)) != len(c)):
        raise ValueError('Finite bounded features, mass and unique source coordinates required')
    if (type(node.first) is not int or type(node.last) is not int or not 0 <= node.first <= node.last <= 2**31-1
            or bool(torch.any(c[:, 0] < node.first)) or bool(torch.any(c[:, 0] > node.last))
            or bool(torch.any(c[:, 1:] < 0)) or bool(torch.any(c[:, 1] >= 6)) or bool(torch.any(c[:, 2] >= 8))
            or not 1 <= node.groups <= node.last-node.first+1 or node.depth < 0
            or any(not math.isfinite(t) or t < 0 for t in (node.begin, node.end, node.center))
            or not node.begin <= node.center <= node.end):
        raise ValueError('Ordered bounded node provenance required')


def raw(block):
    from features import validate as validate_block
    validate_block(block)
    if block.first != block.last or len(block.features) != 48 or not bool(torch.all(block.mass == 1)):
        raise ValueError('Complete real pair required')
    coords = torch.cat((torch.full((48, 1), block.first, dtype=torch.long), block.xy), 1)
    node = Node(block.features.clone(), block.mass.clone(), coords, block.first, block.last,
                block.begin, block.end, block.center)
    validate(node)
    return node


def metric(x, heads=4):
    if x.shape[1] % heads:
        raise ValueError('Projection width aligned to fixed heads required')
    return F.normalize(x.reshape(len(x), heads, -1).mean(1), dim=-1, eps=1e-12)


@torch.no_grad()
def merge(left, right, within=False):
    validate(left); validate(right)
    if left.last+1 != right.first or left.end >= right.begin or left.features.shape[1] != right.features.shape[1]:
        raise ValueError('Adjacent nonoverlapping nodes required')
    x, mass, coords = [torch.cat((getattr(left, k), getattr(right, k))) for k in ('features', 'mass', 'coords')]
    values = metric(x)
    scores = values[::2] @ values[1::2].T
    best, destination = scores.max(1)
    remove = len(x)//2
    order = torch.argsort(best, descending=True, stable=True)
    chosen, unmerged = order[:remove], order[remove:]
    weighted = (x[1::2]*mass[1::2]).scatter_add(0, destination[chosen, None].expand(-1, x.shape[1]), x[::2][chosen]*mass[::2][chosen])
    weights = mass[1::2].scatter_add(0, destination[chosen, None], mass[::2][chosen])
    output = torch.cat((x[::2][unmerged], weighted/weights))
    output_mass = torch.cat((mass[::2][unmerged], weights))
    output_coords = torch.cat((coords[::2][unmerged], coords[1::2]))
    # Preserve numeric merge, but emit causal source order for the language sequence.
    permutation = torch.argsort(output_coords[:, 0]*48+output_coords[:, 1]*8+output_coords[:, 2], stable=True)
    nleft, nright = left.last-left.first+1, right.last-right.first+1
    parent = Node(output[permutation], output_mass[permutation], output_coords[permutation],
                  left.first, right.last, left.begin, right.end,
                  (left.center*nleft+right.center*nright)/(nleft+nright),
                  max(left.groups, right.groups) if within else left.groups+right.groups,
                  max(left.depth, right.depth)+1)
    validate(parent)
    return parent


def penalties(nodes, now):
    rows = []
    for a, b in zip(nodes, nodes[1:]):
        value = metric(torch.cat((a.features, b.features)))
        scores = value[::2] @ value[1::2].T
        n = len(value)//2
        similarity = float(scores.max(1).values.topk(n).values.mean())
        rows.append((1-similarity, (a.groups+b.groups)/2, (a.center+b.center)/2/max(now, 1e-12)))
    maximum = max(r[1] for r in rows)
    return [{'similarity': s, 'merge_count': c/maximum+1e-6, 'time': t,
             'total': .4*s+.4*(c/maximum+1e-6)+.2*t} for s, c, t in rows]


class ForestBank:
    """Recent one pair + ongoing event root + 96-token closed forest; no children."""
    def __init__(self):
        self.clear()

    def clear(self):
        self.roots = []
        self.open = self.latest = None
        self.last_id = -1
        self.last_end = None
        self.within_merges = self.long_merges = self.closed_events = 0

    def nodes(self):
        return [*self.roots, *([] if self.open is None else [self.open]), *([] if self.latest is None else [self.latest])]

    @torch.no_grad()
    def append(self, block, boundary=False):
        incoming = raw(block)
        if (type(boundary) is not bool or incoming.first != self.last_id+1
                or (self.last_end is not None and incoming.begin <= self.last_end)
                or (self.latest is not None and self.latest.features.shape[1] != incoming.features.shape[1])):
            raise ValueError('Consecutive causal pair and shared boundary required')
        records = []
        if self.latest is not None:
            if self.open is None:
                self.open = self.latest
            else:
                old = self.open
                self.open = merge(old, self.latest, within=True)
                self.within_merges += 1
                records.append({'kind': 'within_event', 'left': old.metadata(), 'right': self.latest.metadata(), 'parent': self.open.metadata()})
            if boundary:
                self.roots.append(self.open)
                self.open = None
                self.closed_events += 1
            while sum(len(n.features) for n in self.roots) > 96:
                costs = penalties(self.roots, incoming.end)
                selected = min(range(len(costs)), key=lambda i: costs[i]['total'])
                left, right = self.roots[selected:selected+2]
                parent = merge(left, right)
                self.roots[selected:selected+2] = [parent]
                self.long_merges += 1
                records.append({'kind': 'long_forest', 'selected': selected, 'penalties': costs,
                                'left': left.metadata(), 'right': right.metadata(), 'parent': parent.metadata()})
        self.latest = incoming
        self.last_id, self.last_end = incoming.last, incoming.end
        if self.status()['tokens'] > 192 or self.status()['metadata_json_bytes'] > 16384:
            raise RuntimeError('Forest resident budget exceeded')
        return records

    def export(self):
        return [n.copy() for n in self.nodes()]

    def status(self):
        nodes = self.nodes()
        rows = [n.metadata() for n in nodes]
        return {'tokens': sum(len(n.features) for n in nodes), 'tensor_bytes': sum(r['tensor_bytes'] for r in rows),
                'nodes': rows, 'within_merges': self.within_merges, 'long_merges': self.long_merges,
                'closed_events': self.closed_events, 'last_id': self.last_id, 'last_end': self.last_end,
                'mass': sum(float(n.mass.sum()) for n in nodes),
                'metadata_json_bytes': len(json.dumps(rows).encode()), 'all_children_discarded': True}

    def exact(self, timestamp):
        if self.last_end is None or timestamp > self.last_end:
            return 'beyond_encoded_horizon'
        for n in self.nodes():
            if n.begin <= timestamp <= n.end:
                return 'retained_real_pair' if n.metadata()['exact_pair'] else 'compressed_interval_not_exact_frame'
        return 'no_retained_pair'


class RecentBank:
    def __init__(self):
        self.items = []

    def append(self, block, boundary=False):
        self.items.append(raw(block))
        self.items = self.items[-4:]
        return []

    def nodes(self):
        return self.items

    def export(self):
        return [n.copy() for n in self.items]

    def status(self):
        rows = [n.metadata() for n in self.items]
        return {'tokens': sum(r['tokens'] for r in rows), 'tensor_bytes': sum(r['tensor_bytes'] for r in rows), 'nodes': rows}

    def clear(self):
        self.items = []


def fingerprint(bank):
    return {'status': bank.status(), 'tensors': [[hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
             for t in (n.features, n.mass, n.coords)] for n in bank.nodes()]}


@torch.inference_mode()
def context(model, builder, nodes, question, max_new_tokens=32):
    if not nodes or len(nodes) > 4 or sum(len(n.features) for n in nodes) > 192:
        raise ValueError('Bounded nonempty memory required')
    if any(a.last >= b.first or a.end >= b.begin for a, b in zip(nodes, nodes[1:])):
        raise ValueError('Causal nonoverlapping root order required')
    offset = builder.prefix_ids.shape[1]
    all_ids = [builder.prefix_ids]
    positions = [torch.arange(offset).expand(3, -1)]
    for node in nodes:
        validate(node)
        if node.features.shape[1] != model.get_input_embeddings().embedding_dim:
            raise ValueError('Projection dimension mismatch')
        label = f'<{node.center:.1f} seconds>' if node.metadata()['exact_pair'] else f'<{node.begin:.1f}-{node.end:.1f} seconds, compressed memory>'
        ids = builder.tokenize(label+builder.processor.vision_start_token+builder.processor.video_token*len(node.features)+builder.processor.vision_end_token)
        types = (ids[0] == builder.processor.video_token_id).tolist()
        coords = []
        for modality, values in itertools.groupby(types):
            size = len(list(values))
            if modality:
                if size != len(node.features):
                    raise ValueError('Placeholder/feature mismatch')
                coords.extend((offset+int(source)-node.first, offset+int(y), offset+int(x)) for source, y, x in node.coords)
                offset += max(8, node.last-node.first+1)
            else:
                coords.extend((offset+i,)*3 for i in range(size))
                offset += size
        all_ids.append(ids)
        positions.append(torch.tensor(coords).T)
    suffix = builder.question_ids(question)
    if suffix.shape[1] > 64 or not 1 <= max_new_tokens <= 32:
        raise ValueError('QA reservation exceeded')
    all_ids.append(suffix)
    positions.append(torch.arange(offset, offset+suffix.shape[1]).expand(3, -1))
    ids = torch.cat(all_ids, 1)
    if ids.shape[1]+max_new_tokens > 512:
        raise ValueError('Language context budget exceeded')
    four = torch.cat((torch.arange(ids.shape[1])[None, :], torch.cat(positions, 1)), 0)[:, None, :]
    device = model.get_input_embeddings().weight.device
    embeddings = model.get_input_embeddings()(ids.to(device))
    visual = torch.cat([n.features for n in nodes]).to(embeddings)
    mask = (ids.to(device) == builder.processor.video_token_id).unsqueeze(-1).expand_as(embeddings)
    if int(mask.sum()) != visual.numel():
        raise ValueError('Visual embedding insertion mismatch')
    return embeddings.masked_scatter(mask, visual), four.to(device), ids, {
        'input_tokens': ids.shape[1], 'visual_tokens': len(visual), 'nodes': [n.metadata() for n in nodes],
        'destination_source_coordinates_heuristic': True, 'source_time_rank_preserved': True,
        'original_grid_span': [6, 8], 'language_axis_contiguous': True, 'mass_not_applied_to_attention': True}


@torch.inference_mode()
def answer(model, builder, nodes, question, max_new_tokens=32):
    from streaming.hybrid import HybridSequence
    from streaming.online import fp32_reference, greedy_decode
    session = HybridSequence(model, max_tokens=512, session_id='forest-read')
    saved = model.model.rope_deltas
    try:
        with fp32_reference():
            embeddings, positions, ids, meta = context(model, builder, nodes, question, max_new_tokens)
            logits = session.append(inputs_embeds=embeddings, position_ids=positions, logits_mode='last')
            eos = model.generation_config.eos_token_id
            stops = set([] if eos is None else eos if isinstance(eos, list) else [eos])
            if builder.processor.tokenizer.eos_token_id is not None:
                stops.add(builder.processor.tokenizer.eos_token_id)
            generated, reason = greedy_decode(session, logits, max_new_tokens, stops)
        return {'question': question, 'answer': builder.processor.tokenizer.decode(generated, skip_special_tokens=True),
                'generated_ids': generated, 'stop_reason': reason, **meta}
    finally:
        session.cache.reset()
        model.model.rope_deltas = saved
