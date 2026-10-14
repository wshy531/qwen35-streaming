"""Bounded six-frame current-epoch reference; no whole-video history retention."""
import json
import numpy as np
import torch
from transformers.feature_extraction_utils import BatchFeature
from streaming.fair_inputs import joined_pair_inputs,rgb_sha
from streaming.inputs import tensor_fingerprint
from streaming.preprocessing import fit_canvas
from streaming.window import FramePacket,read_packets


def contexts(path):
    rows,current=[],[]
    for line in path.read_text(encoding='utf-8').splitlines():
        event=json.loads(line)
        if event['kind']=='capacity_reset':current=[]
        elif event['kind']=='encoded':
            record=event['record']
            current.extend([{'frame_id':fid,'timestamp':pts} for fid,pts in zip(record['frame_ids'],record['timestamps'])][:1 if record['synthetic_padding'] else 2])
            if len(current)>6:raise ValueError('Registered epoch exceeds six real frames')
        elif event['kind']=='answer':
            if not current or current[-1]['frame_id']!=event['answer']['frame_id'] or current[0]['timestamp']!=event['status']['memory_from_pts']:
                raise ValueError('Answer not backed by this current epoch')
            rows.append({'frames':[dict(x) for x in current],'question':event['answer']['question'],
                'timestamp':event['answer']['timestamp'],'memory_from_pts':event['status']['memory_from_pts'],
                'generated_ids':event['answer']['generated_ids'],'answer':event['answer']['answer'],
                'main_unchanged_bitwise':event['main_unchanged_bitwise']})
    return rows


def select(video,metadata):
    ids=[x['frame_id'] for x in metadata]
    if not 1<=len(ids)<=6 or ids!=sorted(set(ids)):raise ValueError('One to six ordered real frames required')
    frames=[];provenance=[]
    reader=read_packets(video,1)
    try:
        for packet in reader:
            if packet.frame_id in ids:
                spec=metadata[len(frames)]
                if abs(packet.timestamp-spec['timestamp'])>1e-9:raise ValueError('Source PTS differs from actual epoch')
                canvas=fit_canvas(packet.rgb,(192,256));canvas.flags.writeable=False
                frames.append(FramePacket(packet.frame_id,packet.timestamp,canvas,packet.decoded_count))
                provenance.append({'frame_id':packet.frame_id,'timestamp':packet.timestamp,'rgb_sha256':rgb_sha(canvas)})
            if packet.frame_id>=ids[-1]:break
    finally:reader.close()
    if len(frames)!=len(ids):raise ValueError('Missing selected epoch frames')
    return tuple(frames),provenance


def identity(model,processor,frames,question):
    """Independent processor whole input and original RoPE, up to six frames."""
    if not 1<=len(frames)<=6:raise ValueError('Bounded six-frame nonempty prefix required')
    if any(a.frame_id>=b.frame_id or a.timestamp>=b.timestamp for a,b in zip(frames,frames[1:])):raise ValueError('Frame identity/time must increase')
    padded=tuple(frames) if len(frames)%2==0 else tuple(frames)+(frames[-1],)
    visual=processor.video_processor(videos=[np.stack([fit_canvas(p.rgb,(192,256)) for p in padded])],
        do_resize=False,do_sample_frames=False,return_metadata=False,return_tensors='pt')
    grid=visual.video_grid_thw
    unit=int(grid[0,1:].prod())//processor.video_processor.merge_size**2
    expansion=''.join(f'<{(padded[i].timestamp+padded[i+1].timestamp)/2:.1f} seconds>'+processor.vision_start_token+
        processor.video_token*unit+processor.vision_end_token for i in range(0,len(padded),2))
    template=processor.apply_chat_template([{'role':'user','content':[{'type':'video'},{'type':'text','text':question}]}],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    marker=processor.vision_start_token+processor.video_token+processor.vision_end_token
    if template.count(marker)!=1:raise ValueError('Expected one video marker')
    ids=processor.tokenizer(template.replace(marker,expansion),add_special_tokens=False,return_tensors='pt')
    whole=BatchFeature({**ids,**visual,'mm_token_type_ids':(ids.input_ids==processor.video_token_id).long()*2})
    rope_grids=torch.repeat_interleave(grid,grid[:,0],0);rope_grids[:,0]=1
    coordinates,_=model.model.get_rope_index(**{**whole,'video_grid_thw':rope_grids})
    whole['position_ids']=torch.cat((torch.arange(ids.input_ids.shape[1])[None,None,:],coordinates),0)
    pair=joined_pair_inputs(processor,frames,question)
    equality={k:k in pair and torch.equal(v,pair[k]) for k,v in whole.items()}
    if set(whole)!=set(pair) or not all(equality.values()):raise AssertionError('Independent whole/pair six-frame input differs')
    if whole.input_ids.shape[1]+16>288:raise ValueError('Total input/answer exceeds 288 Token budget')
    return whole,{'all_input_tensors_bitwise_equal':True,'tensor_equalities':equality,
        'input_fingerprint':tensor_fingerprint(whole),'input_tokens':whole.input_ids.shape[1],
        'grid':whole.video_grid_thw.tolist()}
