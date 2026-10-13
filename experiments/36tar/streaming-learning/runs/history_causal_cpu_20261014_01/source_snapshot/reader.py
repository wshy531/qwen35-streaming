"""Native bounded reference and packed same-source-span reading candidate."""
import hashlib
import itertools
import torch
from forest import validate
@torch.inference_mode()
def native_context(model, builder, nodes, question, max_new_tokens=32):
    if not nodes or len(nodes) > 8 or sum(len(n.features) for n in nodes) > 384:
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
    if suffix.shape[1] > 64 or not 1 <= max_new_tokens <= 24:
        raise ValueError('QA reservation exceeded')
    all_ids.append(suffix)
    positions.append(torch.arange(offset, offset+suffix.shape[1]).expand(3, -1))
    ids = torch.cat(all_ids, 1)
    if ids.shape[1]+max_new_tokens > 768:
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
def native_answer(model, builder, nodes, question, max_new_tokens=32):
    from streaming.hybrid import HybridSequence
    from streaming.online import fp32_reference, greedy_decode
    session = HybridSequence(model, max_tokens=768, session_id='forest-read')
    saved = model.model.rope_deltas
    try:
        with fp32_reference():
            embeddings, positions, ids, meta = native_context(model, builder, nodes, question, max_new_tokens)
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


_native_context = native_context


def trace(embeddings, positions, ids, meta):
    return embeddings, positions, ids, {
        **meta, 'input_ids': ids[0].tolist(), 'four_axis_positions': positions[:, 0].cpu().tolist(),
        'position_sha256': hashlib.sha256(positions.cpu().contiguous().numpy().tobytes()).hexdigest(),
        'ids_sha256': hashlib.sha256(ids.contiguous().numpy().tobytes()).hexdigest(),
    }


def native_context(model, builder, nodes, question, max_new_tokens=24):
    return trace(*_native_context(model, builder, nodes, question, max_new_tokens))


@torch.inference_mode()
def packed_context(model, builder, nodes, question, max_new_tokens=24):
    if not nodes or len(nodes) > 8 or sum(len(n.features) for n in nodes) > 384:
        raise ValueError('Bounded evidence required')
    if any(a.last >= b.first or a.end >= b.begin for a, b in zip(nodes, nodes[1:])):
        raise ValueError('Nonoverlapping causal nodes required')
    for node in nodes: validate(node)
    width = model.get_input_embeddings().embedding_dim
    if any(n.features.shape[1] != width for n in nodes): raise ValueError('Projection width mismatch')
    first, last, begin, end = nodes[0].first, nodes[-1].last, nodes[0].begin, nodes[-1].end
    total = sum(len(n.features) for n in nodes)
    offset = builder.prefix_ids.shape[1]
    pos = [torch.arange(offset).expand(3, -1)]
    label = f'<{begin:.1f}-{end:.1f} seconds>'
    visual_ids = builder.tokenize(label+builder.processor.vision_start_token+
                                 builder.processor.video_token*total+builder.processor.vision_end_token)
    types = (visual_ids[0] == builder.processor.video_token_id).tolist()
    coordinates = torch.cat([n.coords for n in nodes]).tolist()
    positions = []
    for modality, group in itertools.groupby(types):
        size = len(list(group))
        if modality:
            if size != total: raise ValueError('Visual placeholder mismatch')
            positions.extend((offset+source-first, offset+y, offset+x) for source,y,x in coordinates)
            offset += max(8,last-first+1)
        else:
            positions.extend((offset+i,)*3 for i in range(size)); offset += size
    pos.append(torch.tensor(positions).T)
    suffix = builder.question_ids(question)
    if suffix.shape[1] > 64 or not 1 <= max_new_tokens <= 24: raise ValueError('QA reservation')
    pos.append(torch.arange(offset,offset+suffix.shape[1]).expand(3,-1))
    ids = torch.cat([builder.prefix_ids, visual_ids, suffix],1)
    if ids.shape[1]+max_new_tokens > 768: raise ValueError('QA context bound')
    positions = torch.cat((torch.arange(ids.shape[1])[None,:],torch.cat(pos,1)),0)[:,None,:]
    device = model.get_input_embeddings().weight.device
    embeddings = model.get_input_embeddings()(ids.to(device))
    visual = torch.cat([n.features for n in nodes]).to(embeddings)
    mask = (ids.to(device) == builder.processor.video_token_id).unsqueeze(-1).expand_as(embeddings)
    if int(mask.sum()) != visual.numel(): raise ValueError('Visual insertion mismatch')
    meta = {'input_tokens':ids.shape[1], 'visual_tokens':total, 'nodes':[n.metadata() for n in nodes],
            'layout':'packed', 'label':label, 'source_span':[first,last],
            'mass_not_applied_to_attention':True, 'coordinates_are_heuristic':True,
            'question_three_axis_positions':pos[-1].tolist()}
    return trace(embeddings.masked_scatter(mask,visual),positions.to(device),ids,meta)


@torch.inference_mode()
def packed_answer(model, builder, nodes, question, max_new_tokens=24):
    from streaming.hybrid import HybridSequence
    from streaming.online import fp32_reference, greedy_decode
    session = HybridSequence(model,max_tokens=768,session_id='packed-evidence-read')
    saved = model.model.rope_deltas
    try:
        with fp32_reference():
            embeddings,positions,ids,meta = packed_context(model,builder,nodes,question,max_new_tokens)
            logits = session.append(inputs_embeds=embeddings,position_ids=positions,logits_mode='last')
            eos = model.generation_config.eos_token_id
            stops = set([] if eos is None else eos if isinstance(eos,list) else [eos])
            if builder.processor.tokenizer.eos_token_id is not None: stops.add(builder.processor.tokenizer.eos_token_id)
            generated,reason = greedy_decode(session,logits,max_new_tokens,stops)
        return {'question':question,'answer':builder.processor.tokenizer.decode(generated,skip_special_tokens=True),
                'generated_ids':generated,'stop_reason':reason,**meta}
    finally:
        session.cache.reset(); model.model.rope_deltas = saved
