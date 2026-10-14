"""Geometry-only48-token selections with explicit common-window readout."""
from dataclasses import replace
import hashlib
import itertools
import torch
from forest import merge,validate
from reader import trace
from intervention import context as frozen_context


def spatial_sort(node):
    order=torch.argsort(node.coords[:,1]*16+node.coords[:,2]*2+node.coords[:,0],stable=True)
    return replace(node,features=node.features[order].clone(),mass=node.mass[order].clone(),coords=node.coords[order].clone())


def variants(nodes):
    if len(nodes)!=2 or [n.first for n in nodes]!=[0,1]:raise ValueError('Two fixed complete pairs required')
    features=torch.cat([n.features for n in nodes]);coords=torch.cat([n.coords for n in nodes])
    prototype=merge(nodes[0],nodes[1],within=True)
    selections={}
    masks={'first':coords[:,0]==0,'last':coords[:,0]==1,'odd':coords[:,2]%2==1,
           'even':coords[:,2]%2==0,'checker':(coords[:,1]+coords[:,2]+coords[:,0])%2==0}
    for name,mask in masks.items():
        selected=coords[mask].clone();first=int(selected[:,0].min());last=int(selected[:,0].max())
        node=replace(prototype,features=features[mask].clone(),mass=torch.ones((48,1)),coords=selected,
                     first=first,last=last,begin=nodes[first].begin,end=nodes[last].end,
                     center=float(torch.tensor([nodes[int(i)].center for i in selected[:,0]]).mean()),depth=0)
        validate(node);selections[name]=spatial_sort(node)
    selections['merged']=spatial_sort(prototype)
    legacy_odd=replace(prototype,features=features[1::2].clone(),mass=torch.ones((48,1)),depth=0)
    result={'raw96':(nodes,'source'),'first_actual':(nodes[:1],'actual')}
    for name,node in selections.items():
        if len(node.features)!=48:raise AssertionError('Every geometry arm must have48 tokens')
        for mode in ('source','grid'):result[name+'_'+mode]=([node],mode)
    result.update(legacy_merged=([prototype],'legacy'),legacy_odd=([legacy_odd],'legacy'))
    return result


@torch.inference_mode()
def context(model,builder,nodes,question,mode='source'):
    if mode in ('actual','legacy'):
        embeddings,positions,ids,meta=frozen_context(model,builder,nodes,question,'destination')
        return embeddings,positions,ids,{**meta,'selection_mode':mode,'common_window_is_diagnostic':False}
    if mode not in ('source','grid') or not nodes or sum(len(n.features) for n in nodes)>96:
        raise ValueError('Bounded matched-window mode required')
    for n in nodes:validate(n)
    total=sum(len(n.features) for n in nodes)
    if mode=='grid' and total!=48:raise ValueError('Canonical grid only for48 values')
    coords=torch.cat([n.coords for n in nodes])
    if bool(torch.any(coords[:,0]<0)) or bool(torch.any(coords[:,0]>1)):raise ValueError('Fixed source window0..1')
    prefix=builder.prefix_ids;offset=prefix.shape[1]
    middle=builder.tokenize('<0.0-3.0 seconds>'+builder.processor.vision_start_token+
                            builder.processor.video_token*total+builder.processor.vision_end_token)
    positions=[]
    for visual,group in itertools.groupby((middle[0]==builder.processor.video_token_id).tolist()):
        size=len(list(group))
        if visual:
            if size!=total:raise ValueError('Visual placeholder mismatch')
            applied=coords if mode=='source' else torch.cat((torch.zeros((48,1),dtype=torch.long),torch.cartesian_prod(torch.arange(6),torch.arange(8))),1)
            positions.extend(tuple(offset+int(v) for v in coordinate) for coordinate in applied)
            offset+=8
        else:
            positions.extend((offset+i,)*3 for i in range(size));offset+=size
    suffix=builder.question_ids(question)
    if suffix.shape[1]>64:raise ValueError('Question reserve')
    ids=torch.cat([prefix,middle,suffix],1)
    if ids.shape[1]+24>512:raise ValueError('Small context bound')
    question_positions=torch.arange(offset,offset+suffix.shape[1]).expand(3,-1)
    three=torch.cat([torch.arange(prefix.shape[1]).expand(3,-1),torch.tensor(positions).T,question_positions],1)
    four=torch.cat([torch.arange(ids.shape[1])[None,:],three],0)[:,None,:]
    device=model.get_input_embeddings().weight.device
    embeddings=model.get_input_embeddings()(ids.to(device));visual=torch.cat([n.features for n in nodes]).to(embeddings)
    mask=(ids.to(device)==builder.processor.video_token_id).unsqueeze(-1).expand_as(embeddings)
    if int(mask.sum())!=visual.numel():raise ValueError('Visual insertion mismatch')
    meta={'input_tokens':ids.shape[1],'visual_tokens':total,'nodes':[n.metadata() for n in nodes],
          'selection_mode':mode,'common_window_is_diagnostic':True,'label':'<0.0-3.0 seconds>',
          'common_window_is_not_claim_of_full_retained_coverage':True,
          'actual_selected_pair_ids':sorted(set(coords[:,0].tolist())),
          'unique_spatial_cells':len(torch.unique(coords[:,1:],dim=0)),
          'question_three_axis_positions':question_positions.tolist(),
          'original_provenance_unchanged':True,'grid_is_artificial_applied_position':mode=='grid',
          'mass_not_applied_to_attention':True,
          'visual_values_sha256':hashlib.sha256(torch.cat([n.features for n in nodes]).contiguous().numpy().tobytes()).hexdigest()}
    return trace(embeddings.masked_scatter(mask,visual),four.to(device),ids,meta)


@torch.inference_mode()
def answer(model,builder,nodes,question,mode='source'):
    from streaming.hybrid import HybridSequence
    from streaming.online import fp32_reference,greedy_decode
    session=HybridSequence(model,max_tokens=512,session_id='geometry-layout-control')
    saved=model.model.rope_deltas
    try:
        with fp32_reference():
            embeddings,positions,ids,meta=context(model,builder,nodes,question,mode)
            logits=session.append(inputs_embeds=embeddings,position_ids=positions,logits_mode='last')
            eos=model.generation_config.eos_token_id
            stops=set([] if eos is None else eos if isinstance(eos,list) else [eos])
            if builder.processor.tokenizer.eos_token_id is not None:stops.add(builder.processor.tokenizer.eos_token_id)
            generated,reason=greedy_decode(session,logits,24,stops)
        return {'question':question,'answer':builder.processor.tokenizer.decode(generated,skip_special_tokens=True),
                'generated_ids':generated,'stop_reason':reason,**meta}
    finally:
        session.cache.reset();model.model.rope_deltas=saved
