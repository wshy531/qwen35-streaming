"""Bounded synchronous pair streaming and isolated QA in explicit FP32 reference mode."""
from contextlib import contextmanager
import itertools
import math
import time

import numpy as np
from PIL import Image
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from transformers.feature_extraction_utils import BatchFeature

from streaming.hybrid import HybridSequence
from streaming.window import FramePacket


@contextmanager
def fp32_reference():
    matmul = torch.backends.cuda.matmul.fp32_precision
    conv = torch.backends.cudnn.conv.fp32_precision
    try:
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        torch.backends.cuda.matmul.fp32_precision = matmul
        torch.backends.cudnn.conv.fp32_precision = conv


class PairInputBuilder:
    def __init__(self,processor,canvas=(192,256)):
        self.processor = processor
        factor = processor.video_processor.patch_size*processor.video_processor.merge_size
        if (len(canvas) != 2 or any(not isinstance(x,int) or isinstance(x,bool) or x < factor or x > 256 or x%factor for x in canvas)
                or processor.video_processor.temporal_patch_size != 2):
            raise ValueError("Fixed pair canvas must have aligned sides <=256 and temporal patch=2")
        self.canvas = tuple(canvas)
        marker = processor.vision_start_token+processor.video_token+processor.vision_end_token
        self.marker = marker
        prefix,_ = self.template("probe").split(marker)
        self.prefix_text = prefix
        self.prefix_ids = self.tokenize(prefix)

    def template(self,question):
        return self.processor.apply_chat_template([{"role":"user","content":[{"type":"video"},{"type":"text","text":question}]}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)

    def tokenize(self,text):
        return self.processor.tokenizer(text,add_special_tokens=False,return_tensors="pt").input_ids

    def question_ids(self,question):
        if (not isinstance(question,str) or not question.strip() or
                any(token in question for token in self.processor.tokenizer.all_special_tokens)):
            raise ValueError("Nonempty question without template control tokens required")
        prefix,suffix = self.template(question).split(self.marker)
        assert prefix == self.prefix_text
        return self.tokenize(suffix)

    def positions(self,ids,grid,seen_tokens,next_rope):
        types = (ids[0] == self.processor.video_token_id).long()*2
        height,width = [int(x)//self.processor.video_processor.merge_size for x in grid[0,1:]]
        offset = next_rope
        coordinates,groups = [],0
        for modality,values in itertools.groupby(types.tolist()):
            size = len(list(values))
            if modality == 0:
                coordinates += [(offset+i,)*3 for i in range(size)]
                offset += size
            else:
                assert size == height*width
                coordinates += [(offset,offset+y,offset+x) for y in range(height) for x in range(width)]
                offset += max(height,width)
                groups += 1
        assert groups == 1
        text = torch.arange(seen_tokens,seen_tokens+ids.shape[1])[None,None,:]
        return torch.cat((text,torch.tensor(coordinates).T[None,:,:].transpose(0,1)),0)

    def pair(self,first,second,seen_tokens,next_rope,synthetic_padding=False):
        images = [np.asarray(Image.fromarray(p.rgb).resize((self.canvas[1],self.canvas[0]),Image.Resampling.BICUBIC))
                  for p in (first,second)]
        pixels = self.processor.video_processor(videos=[np.stack(images)],do_resize=False,do_sample_frames=False,
            return_metadata=False,return_tensors="pt")
        grid = pixels.video_grid_thw
        assert grid.shape == (1,3) and int(grid[0,0]) == 1
        count = int(grid[0,1:].prod())//self.processor.video_processor.merge_size**2
        mean = (first.timestamp+second.timestamp)/2
        block = f"<{mean:.1f} seconds>"+self.processor.vision_start_token+self.processor.video_token*count+self.processor.vision_end_token
        ids = self.tokenize(block)
        positions = self.positions(ids,grid,seen_tokens,next_rope)
        inputs = BatchFeature({"input_ids":ids,"attention_mask":torch.ones_like(ids),
            "mm_token_type_ids":(ids == self.processor.video_token_id).long()*2,"position_ids":positions,**pixels})
        return inputs,{"frame_ids":[first.frame_id,second.frame_id],"timestamps":[first.timestamp,second.timestamp],
            "temporal_mean":mean,"rendered_timestamp":f"{mean:.1f}","synthetic_padding":synthetic_padding,
            "grid":grid.tolist(),"token_ids":ids[0].tolist(),"block_text":block}


@torch.inference_mode()
def greedy_decode(branch,logits,max_new_tokens,eos_ids):
    generated = []
    reason = "budget"
    for _ in range(max_new_tokens):
        token = int(logits[0,-1].argmax())
        generated.append(token)
        logits = branch.append(input_ids=torch.tensor([[token]],device=logits.device))
        if token in eos_ids:
            reason = "eos"
            break
    return generated,reason


class OnlinePairSession:
    """One session, no concurrency/rollback guarantee; fixed canvas and FP32 only."""
    def __init__(self,model,processor,canvas=(192,256),max_tokens=256,max_question_tokens=64,max_answer_tokens=16):
        if model.training or model.get_input_embeddings().weight.dtype != torch.float32:
            raise ValueError("Explicit FP32 eval model required")
        if any(not isinstance(x,int) or isinstance(x,bool) or x < 1 for x in (max_tokens,max_question_tokens,max_answer_tokens)):
            raise ValueError("Positive integer token budgets required")
        self.builder = PairInputBuilder(processor,canvas)
        self.model,self.processor = model,processor
        self.max_tokens,self.max_question_tokens,self.max_answer_tokens = max_tokens,max_question_tokens,max_answer_tokens
        self.main_budget = max_tokens-max_question_tokens-max_answer_tokens
        if self.main_budget <= self.builder.prefix_ids.shape[1]:
            raise ValueError("No video budget after reserving question/answer")
        self.main = HybridSequence(model,max_tokens=max_tokens)
        self.reset()

    def reset(self):
        self.main.reset()
        self.pending = None
        self.last_received_id = self.last_received_pts = self.last_encoded_id = self.last_encoded_pts = None
        self.frame_shape = None
        self.accepted_frames = self.encoded_frames = self.vision_encodings = 0
        self.ledger = []
        self.invalid = self.finished = self.closed = False

    def close(self):
        self.reset()
        self.closed = True

    def check_ready(self,for_input=False):
        if self.closed or self.invalid:
            raise RuntimeError("Session closed or invalid; reset required")
        if for_input and self.finished:
            raise ValueError("EOF already submitted; reset before new frames")

    def status(self):
        return {"accepted_frames":self.accepted_frames,"encoded_real_frames":self.encoded_frames,
            "pending_frame_id":None if self.pending is None else self.pending.frame_id,
            "last_received_id":self.last_received_id,"last_encoded_id":self.last_encoded_id,"last_encoded_pts":self.last_encoded_pts,
            "seen_tokens":self.main.seen_tokens,"next_rope_position":self.main.next_rope_position,
            "vision_encodings":self.vision_encodings,"pending_rgb_bytes":0 if self.pending is None else self.pending.rgb.nbytes,
            "finished":self.finished,"invalid":self.invalid,"closed":self.closed,"reference_mode":"fp32_ieee_math"}

    def validate(self,packet):
        if (not isinstance(packet,FramePacket) or not isinstance(packet.frame_id,int) or isinstance(packet.frame_id,bool)
            or packet.frame_id < 0 or packet.decoded_count <= packet.frame_id or not math.isfinite(packet.timestamp) or packet.timestamp < 0):
            raise ValueError("Valid frame ID/count/PTS required")
        if self.last_received_id is not None and (packet.frame_id <= self.last_received_id or packet.timestamp <= self.last_received_pts):
            raise ValueError("Frame IDs and actual PTS must strictly increase")
        if (not isinstance(packet.rgb,np.ndarray) or packet.rgb.dtype != np.uint8 or packet.rgb.ndim != 3 or
                packet.rgb.shape[-1] != 3 or min(packet.rgb.shape[:2]) < 1 or max(packet.rgb.shape[:2]) > 256):
            raise ValueError("Bounded HxWx3 uint8 RGB required")
        if self.frame_shape is not None and packet.rgb.shape != self.frame_shape:
            raise ValueError("Source frame shape changed; start a new session")

    @torch.inference_mode()
    def commit_pair(self,first,second,synthetic_padding=False):
        prefix_size = self.builder.prefix_ids.shape[1] if self.main.seen_tokens == 0 else 0
        inputs,record = self.builder.pair(first,second,self.main.seen_tokens+prefix_size,
            self.main.next_rope_position+prefix_size,synthetic_padding)
        if self.main.seen_tokens+prefix_size+inputs.input_ids.shape[1] > self.main_budget:
            raise ValueError("Main token budget exceeded before encoding; reset required")
        device = self.model.get_input_embeddings().weight.device
        inputs = inputs.to(device)
        try:
            with fp32_reference():
                embeddings = self.model.get_input_embeddings()(inputs.input_ids)
                self.vision_encodings += 1
                vision = self.model.model.get_video_features(inputs.pixel_values_videos,inputs.video_grid_thw,return_dict=True)
                features = torch.cat(vision.pooler_output,0).to(embeddings)
                mask = (inputs.input_ids == self.processor.video_token_id).unsqueeze(-1).expand_as(embeddings)
                assert mask.sum() == features.numel()
                embeddings = embeddings.masked_scatter(mask,features)
                if prefix_size:
                    self.main.append(input_ids=self.builder.prefix_ids.to(device))
                logits = self.main.append(inputs_embeds=embeddings,position_ids=inputs.position_ids)
        except Exception:
            self.invalid = True
            raise
        self.ledger.append(record)
        self.encoded_frames += 1 if synthetic_padding else 2
        self.last_encoded_id,self.last_encoded_pts = second.frame_id,second.timestamp
        return logits

    def append(self,packet):
        self.check_ready(for_input=True)
        self.validate(packet)
        if self.pending is None:
            pixels = packet.rgb.copy()
            pixels.flags.writeable = False
            self.pending = FramePacket(packet.frame_id,packet.timestamp,pixels,packet.decoded_count,packet.decode_seconds)
            self.frame_shape = packet.rgb.shape
            logits = None
        else:
            logits = self.commit_pair(self.pending,packet)
            self.pending = None
        self.last_received_id,self.last_received_pts = packet.frame_id,packet.timestamp
        self.accepted_frames += 1
        return {"status":self.status(),"logits":logits}

    def finish(self):
        self.check_ready(for_input=True)
        logits = None
        if self.pending is not None:
            logits = self.commit_pair(self.pending,self.pending,True)
            self.pending = None
        self.finished = True
        return {"status":self.status(),"logits":logits}

    @torch.inference_mode()
    def ask(self,question,max_new_tokens=16,return_evidence=False):
        self.check_ready()
        if self.pending is not None or not self.ledger:
            raise ValueError("Waiting for a complete frame pair; no current answer")
        if (not isinstance(max_new_tokens,int) or isinstance(max_new_tokens,bool) or
                not 1 <= max_new_tokens <= self.max_answer_tokens):
            raise ValueError("Answer token budget exceeded")
        suffix = self.builder.question_ids(question)
        if suffix.shape[1] > self.max_question_tokens or self.main.seen_tokens+suffix.shape[1]+max_new_tokens > self.max_tokens:
            raise ValueError("Question/answer exceeds reserved token budget")
        counts = self.main.seen_tokens,self.main.next_rope_position
        device = self.model.get_input_embeddings().weight.device
        begin = time.perf_counter()
        try:
            with fp32_reference():
                branch = self.main.fork()
                logits = branch.append(input_ids=suffix.to(device))
                initial = logits if return_evidence else None
                eos = self.model.generation_config.eos_token_id
                eos_ids = set([] if eos is None else eos if isinstance(eos,list) else [eos])
                generated,reason = greedy_decode(branch,logits,max_new_tokens,eos_ids)
        except Exception:
            self.invalid = True
            raise
        assert counts == (self.main.seen_tokens,self.main.next_rope_position)
        result = {"question":question,"answer":self.processor.tokenizer.decode(generated,skip_special_tokens=True),
            "generated_ids":generated,"stop_reason":reason,"frame_id":self.last_encoded_id,"timestamp":self.last_encoded_pts,
            "wall_seconds":time.perf_counter()-begin,"reference_mode":"fp32_ieee_math"}
        if return_evidence:
            result["evidence"] = {"suffix_ids":suffix,"initial_logits":initial,"branch_cache":branch.cache}
        return result
