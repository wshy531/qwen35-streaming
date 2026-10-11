"""Single-sample inference state with explicit token/RoPE positions and owned cache."""
import torch

from qwen35_local.modeling_qwen3_5 import Qwen3_5DynamicCache


class HybridSequence:
    """Language sequence only: callers must supply pre-encoded visual embeddings."""
    def __init__(self,model,max_tokens=256,session_id="main"):
        if not isinstance(max_tokens,int) or isinstance(max_tokens,bool) or max_tokens < 1:
            raise ValueError("Positive integer token budget required")
        if not isinstance(session_id,str) or not session_id:
            raise ValueError("Nonempty session ID required")
        self.model,self.max_tokens,self.session_id = model,max_tokens,session_id
        self.cache = Qwen3_5DynamicCache(model.config.text_config)
        self.seen_tokens = 0
        self.next_rope_position = 0

    @torch.inference_mode()
    def append(self,input_ids=None,inputs_embeds=None,position_ids=None):
        if self.model.training:
            raise ValueError("This bounded interface is inference-only; call eval first")
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Supply exactly one token or embedding input")
        tensor = input_ids if input_ids is not None else inputs_embeds
        expected_rank = 2 if input_ids is not None else 3
        if tensor.ndim != expected_rank or tensor.shape[0] != 1 or tensor.shape[1] == 0:
            raise ValueError("Nonempty single-sample unpadded sequence required")
        length = tensor.shape[1]
        if self.seen_tokens+length > self.max_tokens:
            raise ValueError("Token budget exceeded; reset or choose an explicit new budget")
        embedding = self.model.get_input_embeddings()
        if tensor.device != embedding.weight.device:
            raise ValueError("Inputs must already be on the model device")
        if input_ids is not None:
            if tensor.dtype != torch.long or tensor.min() < 0 or tensor.max() >= embedding.num_embeddings:
                raise ValueError("Invalid token IDs")
        elif (tensor.shape[-1] != embedding.embedding_dim or tensor.dtype != embedding.weight.dtype
              or not torch.isfinite(tensor).all()):
            raise ValueError("Embedding shape/dtype/values mismatch")
        text_positions = torch.arange(self.seen_tokens,self.seen_tokens+length,device=tensor.device)
        if position_ids is None:
            if inputs_embeds is not None:
                raise ValueError("Explicit four-axis positions required for supplied embeddings")
            rope = torch.arange(self.next_rope_position,self.next_rope_position+length,device=tensor.device)
            position_ids = torch.stack((text_positions,rope,rope,rope))[:,None,:]
        if (position_ids.shape != (4,1,length) or position_ids.dtype != torch.long or
            position_ids.device != tensor.device or not torch.equal(position_ids[0,0],text_positions) or
            torch.any(position_ids < 0)):
            raise ValueError("Invalid four-axis positions or text positions restart at the wrong offset")
        assert self.cache.get_seq_length() == self.seen_tokens
        embeddings = embedding(input_ids) if input_ids is not None else inputs_embeds
        result = self.model.model.language_model(inputs_embeds=embeddings,position_ids=position_ids,
            cache_position=text_positions,attention_mask=torch.ones((1,self.seen_tokens+length),dtype=torch.long,device=tensor.device),
            past_key_values=self.cache,use_cache=True)
        self.cache = result.past_key_values
        self.seen_tokens += length
        self.next_rope_position = max(self.next_rope_position,int(position_ids[1:].max())+1)
        assert self.cache.get_seq_length() == self.seen_tokens
        return self.model.lm_head(result.last_hidden_state)

    def fork(self,session_id="question"):
        branch = type(self)(self.model,self.max_tokens,session_id)
        branch.cache = self.cache.clone()
        branch.seen_tokens = self.seen_tokens
        branch.next_rope_position = self.next_rope_position
        return branch

    def reset(self,session_id="main"):
        if not isinstance(session_id,str) or not session_id:
            raise ValueError("Nonempty session ID required")
        self.cache.reset()
        self.seen_tokens = self.next_rope_position = 0
        self.session_id = session_id
        self.model.model.rope_deltas = None
