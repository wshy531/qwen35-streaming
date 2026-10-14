"""A value/position2x2; metadata provenance is never rewritten by intervention."""
from dataclasses import replace
import hashlib
import torch
from forest import merge,validate
from reader import packed_context,trace


def variants(nodes):
    if len(nodes)!=2 or nodes[0].last+1!=nodes[1].first:raise ValueError('Two complete adjacent pairs required')
    fused=merge(nodes[0],nodes[1],within=True)
    values=torch.cat([n.features for n in nodes])[1::2].clone()
    coordinates=torch.cat([n.coords for n in nodes])[1::2]
    if not torch.equal(fused.coords,coordinates):raise AssertionError('Frozen destination selection changed')
    untouched=replace(fused,features=values,mass=torch.ones((48,1)),depth=0)
    validate(untouched)
    return {'raw96':(nodes,'destination'),'first48':(nodes[:1],'destination'),
            'last48':(nodes[1:],'destination'),'merged_destination':([fused],'destination'),
            'unmixed_destination':([untouched],'destination'),
            'merged_first_grid':([fused],'first_grid'),'unmixed_first_grid':([untouched],'first_grid')}


@torch.inference_mode()
def context(model,builder,nodes,question,position_mode='destination'):
    embeddings,positions,ids,meta=packed_context(model,builder,nodes,question,24)
    if ids.shape[1]+24>512:raise ValueError('Small-read512 bound')
    if position_mode=='first_grid':
        if len(nodes)!=1 or len(nodes[0].features)!=48:raise ValueError('Position intervention only48 features')
        visual=torch.nonzero(ids[0]==builder.processor.video_token_id,as_tuple=False).flatten().to(positions.device)
        base=int(positions[1,0,visual[0]])-int(nodes[0].coords[0,0])+nodes[0].first
        xy=torch.cartesian_prod(torch.arange(6),torch.arange(8)).to(positions.device)
        positions=positions.clone()
        positions[1,0,visual]=base
        positions[2,0,visual]=base+xy[:,0]
        positions[3,0,visual]=base+xy[:,1]
    elif position_mode!='destination':raise ValueError('Unknown fixed position intervention')
    meta={**meta,'position_mode':position_mode,'applied_grid_is_artificial_intervention':position_mode=='first_grid',
          'node_coordinates_remain_original_destination_provenance':True,
          'visual_values_sha256':hashlib.sha256(torch.cat([n.features for n in nodes]).contiguous().numpy().tobytes()).hexdigest(),
          'intervention_note':'First-grid changes applied RoPE only; it does not claim those coordinates are real fused-token origins.'}
    return trace(embeddings,positions,ids,meta)


@torch.inference_mode()
def answer(model,builder,nodes,question,position_mode='destination'):
    from streaming.hybrid import HybridSequence
    from streaming.online import fp32_reference,greedy_decode
    session=HybridSequence(model,max_tokens=512,session_id='first-fusion-control')
    saved=model.model.rope_deltas
    try:
        with fp32_reference():
            embeddings,positions,ids,meta=context(model,builder,nodes,question,position_mode)
            logits=session.append(inputs_embeds=embeddings,position_ids=positions,logits_mode='last')
            eos=model.generation_config.eos_token_id
            stops=set([] if eos is None else eos if isinstance(eos,list) else [eos])
            if builder.processor.tokenizer.eos_token_id is not None:stops.add(builder.processor.tokenizer.eos_token_id)
            generated,reason=greedy_decode(session,logits,24,stops)
        return {'question':question,'answer':builder.processor.tokenizer.decode(generated,skip_special_tokens=True),
                'generated_ids':generated,'stop_reason':reason,**meta}
    finally:
        session.cache.reset();model.model.rope_deltas=saved
