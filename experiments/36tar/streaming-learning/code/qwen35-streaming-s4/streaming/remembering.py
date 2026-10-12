"""Optional bounded history evidence beside an unchanged primary PairDemo."""
import torch

from streaming.demo import FIELDS, PairDemo
from streaming.frame_memory import TemporalFrameMemory
from streaming.online import OnlinePairSession


class RememberingPairDemo(PairDemo):
    """Explicit-time retrieval from real observed pairs; no semantic search.

    Same single consumer owns observation and queries. No camera or concurrency
    guarantee. Main capacity reset does not clear the independent CPU frame bank.
    """
    def __init__(self, *args, memory_every_pairs=4, **kwargs):
        self.frame_memory = TemporalFrameMemory(every_pairs=memory_every_pairs)
        self.history_queries = self.history_model_calls = 0
        super().__init__(*args, **kwargs)

    def status(self):
        state = super().status()
        state['external_frame_memory'] = self.frame_memory.evidence()
        state['history_queries'] = self.history_queries
        state['history_model_calls'] = self.history_model_calls
        # This field refers only to model mixed state, never the separate bank.
        state['history_across_reset'] = False
        return state

    def encode(self, first, second, padding):
        super().encode(first,second,padding)
        if not padding:
            selected = self.frame_memory.add_pair(first,second)
            self.emit('memory_updated',selected=selected,frame_ids=[first.frame_id,second.frame_id])

    @torch.inference_mode()
    def ask_at(self, timestamp, question, max_new_tokens=None):
        if self.closed or self.invalid: raise RuntimeError('Demo closed or invalid')
        self.session.check_ready()
        tokens = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if type(tokens) is not int or not 1 <= tokens <= self.session.max_answer_tokens:
            raise ValueError('Invalid history answer budget')
        if self.session.builder.question_ids(question).shape[1] > self.session.max_question_tokens:
            raise ValueError('History question exceeds reserved budget')
        frames,reason = self.frame_memory.retrieve_at(timestamp)
        if frames is None and reason == 'not_observed' and self.last_pts is not None and timestamp <= self.last_pts:
            # A pending/EOF-only frame arrived, but is not a retained real pair.
            reason = 'not_retained'
        answer = {'question':question,'target_time':timestamp,'route':'external_frame_memory',
                  'retrieval_reason':reason,'model_called':False,'generated_ids':[],
                  'stop_reason':'no_evidence','timestamp':timestamp,'evidence_interval':None,
                  'evidence_frame_ids':[],'answer':'没有保留该时间的画面证据，无法据此回答。',
                  'main_counters_unchanged':True,'main_unchanged_bitwise':None}
        if frames is not None:
            main = self.session
            before = main.status()
            cache = main.main.cache.clone() if self.audit else None
            rope = main.model.model.rope_deltas
            saved_rope = None if rope is None else rope.clone()
            branch = None
            try:
                branch = OnlinePairSession(main.model,main.processor,canvas=main.builder.canvas,
                    max_tokens=main.max_tokens,max_question_tokens=main.max_question_tokens,
                    max_answer_tokens=main.max_answer_tokens,resize_mode=main.builder.resize_mode,video_logits='none')
                for frame in frames: branch.append(frame)
                branch.finish()
                generated = branch.ask(question,tokens)
            finally:
                if branch is not None: branch.close()
                main.model.model.rope_deltas = saved_rope
            if main.status() != before: raise AssertionError('Historical query changed main counters')
            if cache is not None:
                for name in FIELDS:
                    left,right = getattr(cache,name),getattr(main.main.cache,name)
                    if len(left) != len(right) or not all(
                        (a is None and b is None) or (a is not None and b is not None and torch.equal(a,b))
                        for a,b in zip(left,right)):
                        raise AssertionError('Historical query changed main mixed state')
                del cache
            answer.update(generated, target_time=timestamp, route='external_frame_memory',
                          retrieval_reason=reason, model_called=True,
                          evidence_interval=[p.timestamp for p in frames],
                          evidence_frame_ids=[p.frame_id for p in frames],
                          main_unchanged_bitwise=True if self.audit else None,
                          main_counters_unchanged=True)
            self.history_model_calls += 1
        self.history_queries += 1
        self.emit('history_answer',answer=answer)
        return answer

    def close(self):
        if not self.closed: self.frame_memory.clear()
        super().close()
