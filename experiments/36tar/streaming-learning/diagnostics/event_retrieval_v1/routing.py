"""Explicit evidence selection from a frozen event bank, never a semantic oracle."""
import math


def select(bank, mode, target=None):
    if mode == 'current':
        blocks = [] if bank.current is None else [bank.current.copy()]
        scope = 'latest_real_pair'
    elif mode == 'group':
        if type(target) is not int or target < 0:
            raise ValueError('Nonnegative retained group index required')
        if target >= len(bank.groups):
            return {'model_called': False, 'reason': 'no_retained_closed_group'}, []
        group = bank.groups[target]
        blocks, scope = [group.representative.copy()], 'representative_of_closed_group'
    elif mode == 'exact':
        result = bank.exact_available(target)
        if result['reason'] != 'retained_real_pair_available':
            return result, []
        rows = [g.representative for g in bank.groups] + ([] if bank.current is None else [bank.current])
        blocks = [b.copy() for b in rows if b.begin <= target <= b.end]
        scope = 'retained_real_pair_at_target'
    else:
        raise ValueError('Explicit current/group/exact selection required')
    if not blocks:
        return {'model_called': False, 'reason': 'no_current_evidence'}, []
    block = blocks[0]
    metadata = {'reason': 'selected', 'evidence_scope': scope,
                'actual_visual_interval': [block.begin, block.end], 'representative_pair_id': block.first,
                'visual_tokens': len(block.features)}
    if mode == 'group':
        metadata['group_interval'] = [bank.groups[target].begin, bank.groups[target].end]
        metadata['whole_group_content_not_guaranteed'] = True
    return metadata, blocks
