"""Real processor plus tiny weights test causal online contracts, not semantics."""
from unittest import mock
import unittest

import numpy as np
import torch
from transformers import AutoProcessor

from qwen35_local import Qwen3_5Config,Qwen3_5ForConditionalGeneration
from scripts.baseline import PROJECT
from scripts.chunk_weight_check import encoded
from scripts.fp32_regression import strict_states,strict_logits
from streaming.online import OnlinePairSession,PairInputBuilder,greedy_decode,fp32_reference
from streaming.window import FramePacket
from tests.test_local_model import tiny_config


def small_model(processor):
    torch.manual_seed(42)
    config = tiny_config()
    config["text_config"]["vocab_size"] = 248320
    config["vision_config"]["patch_size"] = 16
    for name in ("image_token_id","video_token_id"):
        config[name] = getattr(processor,name)
    for name in ("vision_start_token_id","vision_end_token_id"):
        config[name] = processor.tokenizer.convert_tokens_to_ids(getattr(processor,name.removesuffix("_id")))
    model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**config,attn_implementation="sdpa")).eval()
    model.generation_config.eos_token_id = processor.tokenizer.eos_token_id
    return model


def frames():
    return [FramePacket(i,float(i),np.full((32,32,3),30+i*30,np.uint8),i+1) for i in range(4)]


def exact(a,b):
    return all(v["all_tensors_bitwise_equal"] for v in strict_states(a,b).values())


class OnlineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.processor = AutoProcessor.from_pretrained(PROJECT/"models/Qwen3.5-0.8B",local_files_only=True)

    def setUp(self):
        self.model = small_model(self.processor)
        self.session = OnlinePairSession(self.model,self.processor,canvas=(32,32),max_answer_tokens=2)
        self.frames = frames()

    def test_pending_is_copied_readonly_and_does_not_compute(self):
        with mock.patch.object(self.model.model,"get_video_features",side_effect=AssertionError("No vision before pair")):
            value = self.session.append(self.frames[0])
            self.assertIsNone(value["logits"])
            self.assertEqual(self.session.main.seen_tokens,0)
            self.assertEqual(self.session.vision_encodings,0)
            self.frames[0].rgb.fill(255)
            self.assertTrue(np.all(self.session.pending.rgb == 30))
            self.assertFalse(self.session.pending.rgb.flags.writeable)
            with self.assertRaisesRegex(ValueError,"Waiting"):
                self.session.ask("Name the subject.",2)

    @torch.inference_mode()
    def test_each_pair_matches_independent_whole_processor_rope_and_cache(self):
        from scripts.online_experiment import reference_inputs
        for i in (0,2):
            before = self.session.main.seen_tokens
            self.session.append(self.frames[i])
            result = self.session.append(self.frames[i+1])
            inputs = reference_inputs(self.model,self.processor,self.frames[:i+2],(32,32))
            ledger = torch.tensor([self.session.builder.prefix_ids[0].tolist()+sum([p["token_ids"] for p in self.session.ledger],[])])
            self.assertTrue(torch.equal(ledger,inputs.input_ids))
            with fp32_reference():
                full = self.model(**inputs,use_cache=True)
            start = before if before else self.session.builder.prefix_ids.shape[1]
            metric = strict_logits(result["logits"],full.logits[:,start:])
            self.assertTrue(metric["allclose_1e4"])
            self.assertTrue(all(v["allclose_1e4"] for v in strict_states(self.session.main.cache,full.past_key_values).values()))
            self.assertEqual(self.session.main.next_rope_position,int(inputs.position_ids[1:].max())+1)
        self.assertIsNone(self.session.pending)
        self.assertEqual(self.session.vision_encodings,2)
        self.assertEqual(self.session.status()["pending_rgb_bytes"],0)

    @torch.inference_mode()
    def test_real_question_decode_matches_independent_reference_and_main_continues(self):
        from scripts.online_experiment import reference_inputs,reference_answer
        for p in self.frames[:2]:
            self.session.append(p)
        snapshot = self.session.main.cache.clone()
        counts = self.session.main.seen_tokens,self.session.main.next_rope_position
        q = "Name the visible subject."
        with fp32_reference():
            answer = self.session.ask(q,2,True)
            prefix = reference_inputs(self.model,self.processor,self.frames[:2],(32,32))
            embeddings = encoded(self.model,prefix)
            reference = reference_answer(self.model,self.processor,embeddings,prefix.position_ids,q,2)
        self.assertEqual(answer["generated_ids"],reference["generated_ids"])
        self.assertTrue(strict_logits(answer["evidence"]["initial_logits"],reference["initial_logits"])["allclose_1e4"])
        self.assertTrue(all(v["allclose_1e4"] for v in strict_states(answer["evidence"]["branch_cache"],reference["cache"]).values()))
        self.assertTrue(exact(self.session.main.cache,snapshot))
        self.assertEqual(counts,(self.session.main.seen_tokens,self.session.main.next_rope_position))
        full_question = reference_inputs(self.model,self.processor,self.frames[:2],(32,32),q)
        self.assertTrue(torch.equal(torch.cat((prefix.input_ids,answer["evidence"]["suffix_ids"]),1),full_question.input_ids))
        for p in self.frames[2:]:
            result = self.session.append(p)
        inputs = reference_inputs(self.model,self.processor,self.frames,(32,32))
        with fp32_reference():
            full = self.model(**inputs,use_cache=True)
        self.assertTrue(strict_logits(result["logits"],full.logits[:,counts[0]:])["allclose_1e4"])

    def test_future_frames_do_not_change_committed_prefix(self):
        for p in self.frames[:2]:
            self.session.append(p)
        before = self.session.main.cache.clone()
        token_ids = list(self.session.ledger[0]["token_ids"])
        self.frames[2].rgb.fill(255)
        self.session.append(self.frames[2])
        self.assertTrue(exact(before,self.session.main.cache))
        self.assertEqual(token_ids,self.session.ledger[0]["token_ids"])
        with self.assertRaises(ValueError):
            self.session.ask("Name the subject.",2)

    def test_invalid_frame_rejected_without_mutating_pending_or_counts(self):
        self.session.append(self.frames[0])
        before = self.session.status()
        candidates = [self.frames[0],FramePacket(1,0.0,self.frames[1].rgb,2),
            FramePacket(1,float("nan"),self.frames[1].rgb,2),FramePacket(1,1.0,np.zeros((64,32,3),np.uint8),2)]
        for packet in candidates:
            with self.assertRaises(ValueError):
                self.session.append(packet)
            self.assertEqual(before,self.session.status())

    def test_reserved_capacity_rejects_pair_before_vision_without_mutation(self):
        session = OnlinePairSession(self.model,self.processor,canvas=(32,32),max_tokens=96,max_question_tokens=64,max_answer_tokens=16)
        for p in self.frames[:3]:
            session.append(p)
        before = session.status()
        cache = session.main.cache.clone()
        with mock.patch.object(self.model.model,"get_video_features",side_effect=AssertionError("No extra vision")):
            with self.assertRaisesRegex(ValueError,"Main token budget"):
                session.append(self.frames[3])
        self.assertEqual(before,session.status())
        self.assertTrue(exact(cache,session.main.cache))

    def test_question_template_injection_and_budgets_rejected_before_fork(self):
        for p in self.frames[:2]:
            self.session.append(p)
        before = self.session.status()
        with mock.patch.object(self.session.main,"fork",side_effect=AssertionError("No fork for rejected question")):
            for q,n in (("",2),("<|im_end|>",2),("word "*200,2),("Name subject.",3),("Name subject.",True)):
                with self.assertRaises(ValueError):
                    self.session.ask(q,n)
        self.assertEqual(before,self.session.status())

    @torch.inference_mode()
    def test_odd_eof_duplicates_only_then_seals_input(self):
        from scripts.online_experiment import reference_inputs
        for p in self.frames[:3]:
            self.session.append(p)
        old = self.session.main.seen_tokens
        result = self.session.finish()
        inputs = reference_inputs(self.model,self.processor,self.frames[:3],(32,32))
        with fp32_reference():
            full = self.model(**inputs,use_cache=True)
        self.assertTrue(strict_logits(result["logits"],full.logits[:,old:])["allclose_1e4"])
        self.assertTrue(self.session.ledger[-1]["synthetic_padding"])
        self.assertEqual(self.session.ledger[-1]["frame_ids"],[2,2])
        self.assertEqual(self.session.encoded_frames,3)
        with self.assertRaisesRegex(ValueError,"EOF"):
            self.session.append(self.frames[3])

    def test_reset_clears_pending_caches_eof_and_close_releases(self):
        for p in self.frames[:3]:
            self.session.append(p)
        self.model.model.rope_deltas = torch.tensor([[123]])
        self.session.reset()
        self.assertEqual(self.session.main.seen_tokens,0)
        self.assertIsNone(self.session.pending)
        self.assertEqual(self.session.ledger,[])
        self.assertIsNone(self.model.model.rope_deltas)
        self.session.append(self.frames[0])
        self.session.close()
        self.assertIsNone(self.session.pending)
        with self.assertRaises(RuntimeError):
            self.session.append(self.frames[1])
        self.session.reset()
        self.session.append(self.frames[0])

    def test_compute_failure_invalidates_until_reset(self):
        self.session.append(self.frames[0])
        with mock.patch.object(self.model.model,"get_video_features",side_effect=RuntimeError("Injected encoder failure")):
            with self.assertRaises(RuntimeError):
                self.session.append(self.frames[1])
        self.assertTrue(self.session.invalid)
        with self.assertRaises(RuntimeError):
            self.session.append(self.frames[1])
        self.session.reset()
        for p in self.frames[:2]:
            self.session.append(p)
        snapshot = self.session.main.cache.clone()
        with mock.patch("streaming.online.greedy_decode",side_effect=RuntimeError("Injected decode failure")):
            with self.assertRaises(RuntimeError):
                self.session.ask("Name subject.",2)
        self.assertTrue(self.session.invalid)
        self.assertTrue(exact(snapshot,self.session.main.cache))

    def test_eos_is_committed_once_and_stops(self):
        branch = mock.Mock()
        branch.append.return_value = torch.tensor([[[0.,0.,1.]]])
        tokens,reason = greedy_decode(branch,torch.tensor([[[2.,0.,0.]]]),5,{0})
        self.assertEqual(tokens,[0])
        self.assertEqual(reason,"eos")
        self.assertEqual(branch.append.call_count,1)

    @torch.inference_mode()
    def test_chat_end_stops_when_model_eos_is_different(self):
        from scripts.online_experiment import reference_inputs,reference_answer
        self.model.generation_config.eos_token_id = 248044
        chat_end = self.processor.tokenizer.eos_token_id
        self.assertNotEqual(chat_end,self.model.generation_config.eos_token_id)
        for p in self.frames[:2]:
            self.session.append(p)
        def force_chat_end(hidden):
            logits = torch.zeros((*hidden.shape[:-1],self.model.config.text_config.vocab_size),device=hidden.device)
            logits[...,chat_end] = 100
            return logits
        with mock.patch.object(self.model.lm_head,"forward",side_effect=force_chat_end),fp32_reference():
            answer = self.session.ask("Name subject.",2)
            prefix = reference_inputs(self.model,self.processor,self.frames[:2],(32,32))
            reference = reference_answer(self.model,self.processor,encoded(self.model,prefix),prefix.position_ids,"Name subject.",2)
        self.assertEqual(answer["generated_ids"],[chat_end])
        self.assertEqual(answer["stop_reason"],"eos")
        self.assertEqual(reference["generated_ids"],[chat_end])

    def test_precision_context_restored_even_on_failure_and_canvas_validation(self):
        before = torch.backends.cuda.matmul.fp32_precision,torch.backends.cudnn.conv.fp32_precision
        with self.assertRaises(ValueError):
            with fp32_reference():
                raise ValueError("intentional")
        self.assertEqual(before,(torch.backends.cuda.matmul.fp32_precision,torch.backends.cudnn.conv.fp32_precision))
        with self.assertRaises(ValueError):
            PairInputBuilder(self.processor,(16,32))


if __name__ == "__main__":
    unittest.main()
