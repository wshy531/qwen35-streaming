import unittest

import torch

from qwen35_local import Qwen3_5Config,Qwen3_5ForConditionalGeneration
from streaming.hybrid import HybridSequence
from tests.test_local_model import tiny_config

FIELDS = ("key_cache","value_cache","conv_states","recurrent_states")


def snapshot(cache):
    return {field:[None if x is None else x.clone() for x in getattr(cache,field)] for field in FIELDS}


def assert_states(case,actual,expected,exact=False):
    for field in FIELDS:
        a,b = getattr(actual,field),getattr(expected,field) if not isinstance(expected,dict) else expected[field]
        case.assertEqual(len(a),len(b))
        for left,right in zip(a,b):
            if left is None or right is None:
                case.assertIs(left,right)
            else:
                torch.testing.assert_close(left,right,atol=0 if exact else 1e-5,rtol=0 if exact else 1e-4)


class MixedStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**tiny_config(),attn_implementation="eager")).eval()

    @torch.inference_mode()
    def compare_chunks(self,layout,batch=1,seed=7):
        torch.manual_seed(seed)
        tokens = torch.randint(1,100,(batch,sum(layout)))
        full = self.model(input_ids=tokens,use_cache=True)
        cache,parts,start = None,[],0
        for length in layout:
            output = self.model(input_ids=tokens[:,start:start+length],past_key_values=cache,
                cache_position=torch.arange(start,start+length),use_cache=True)
            cache = output.past_key_values
            parts.append(output.logits)
            start += length
            self.assertEqual(cache.get_seq_length(),start)
        torch.testing.assert_close(torch.cat(parts,dim=1),full.logits,atol=1e-5,rtol=1e-4)
        assert_states(self,cache,full.past_key_values)

    def test_short_and_convolution_boundary_chunks(self):
        for layout in ((2,2),(1,3),(3,1),(1,1,1,1),(1,1,3),(5,1,6),(2,3,7),(7,9,3)):
            with self.subTest(layout=layout):
                self.compare_chunks(layout)

    def test_internal_64_token_boundaries_and_multiple_seeds(self):
        for layout in ((3,62,5),(64,2),(63,1,2),(1,64,1),(17,48,65)):
            for seed in (7,99):
                with self.subTest(layout=layout,seed=seed):
                    self.compare_chunks(layout,seed=seed)

    def test_batch_two_unpadded(self):
        self.compare_chunks((2,5,3),batch=2)

    @torch.inference_mode()
    def test_causal_prefix_unaffected_by_future(self):
        a = self.model(input_ids=torch.tensor([[1,2,3,4,5,6]]),use_cache=False).logits
        b = self.model(input_ids=torch.tensor([[1,2,3,91,92,93]]),use_cache=False).logits
        torch.testing.assert_close(a[:,:3],b[:,:3],atol=0,rtol=0)

    @torch.inference_mode()
    def test_clone_copies_storage_for_all_four_fields(self):
        cache = self.model(input_ids=torch.tensor([[1,2,3]]),use_cache=True).past_key_values
        cloned = cache.clone()
        assert_states(self,cache,cloned,exact=True)
        for field in FIELDS:
            for original,copy in zip(getattr(cache,field),getattr(cloned,field)):
                if original is not None:
                    self.assertNotEqual(original.data_ptr(),copy.data_ptr())
                    copy.add_(123)
                    self.assertFalse(torch.equal(original,copy))

    @torch.inference_mode()
    def test_fork_question_branch_does_not_pollute_main(self):
        main = HybridSequence(self.model)
        main.append(input_ids=torch.tensor([[1,2,3]]))
        before = snapshot(main.cache)
        question = main.fork("question_1")
        question.append(input_ids=torch.tensor([[51,52,53]]))
        assert_states(self,main.cache,before,exact=True)
        self.assertEqual((main.seen_tokens,main.next_rope_position),(3,3))
        suffix = main.append(input_ids=torch.tensor([[4,5]]))
        full = self.model(input_ids=torch.tensor([[1,2,3,4,5]]),use_cache=True)
        torch.testing.assert_close(suffix,full.logits[:,-2:],atol=1e-5,rtol=1e-4)
        assert_states(self,main.cache,full.past_key_values)

    @torch.inference_mode()
    def test_repeated_forks_reuse_same_prefix_without_shared_storage(self):
        main = HybridSequence(self.model)
        main.append(input_ids=torch.tensor([[1,2]]))
        for tokens in (torch.tensor([[3,4]]),torch.tensor([[91,92,93]])):
            branch = main.fork()
            logits = branch.append(input_ids=tokens)
            expected = self.model(input_ids=torch.cat((torch.tensor([[1,2]]),tokens),dim=1),use_cache=False).logits
            torch.testing.assert_close(logits,expected[:,-tokens.shape[1]:],atol=1e-5,rtol=1e-4)
            self.assertEqual(main.seen_tokens,2)

    @torch.inference_mode()
    def test_embeddings_and_nonuniform_four_axis_positions(self):
        tokens = torch.tensor([[1,2,3,4,5,6]])
        embeddings = self.model.get_input_embeddings()(tokens)
        positions = torch.tensor([[[0,1,2,3,4,5]],[[0,1,2,2,2,3]],[[0,1,2,3,4,3]],[[0,1,2,4,3,3]]])
        full = self.model(inputs_embeds=embeddings,position_ids=positions,use_cache=True)
        stream = HybridSequence(self.model)
        a = stream.append(inputs_embeds=embeddings[:,:3],position_ids=positions[:,:,:3])
        b = stream.append(inputs_embeds=embeddings[:,3:],position_ids=positions[:,:,3:])
        torch.testing.assert_close(torch.cat((a,b),dim=1),full.logits,atol=1e-5,rtol=1e-4)
        assert_states(self,stream.cache,full.past_key_values)

    @torch.inference_mode()
    def test_text_after_visual_positions_uses_rope_offset(self):
        tokens = torch.tensor([[1,2,3,4,5,6]])
        positions = torch.tensor([[[0,1,2,3]],[[0,1,1,1]],[[0,1,1,1]],[[0,1,1,1]]])
        stream = HybridSequence(self.model)
        stream.append(inputs_embeds=self.model.get_input_embeddings()(tokens[:,:4]),position_ids=positions)
        self.assertEqual(stream.next_rope_position,2)
        suffix = stream.append(input_ids=tokens[:,4:])
        suffix_positions = torch.tensor([[[4,5]],[[2,3]],[[2,3]],[[2,3]]])
        full = self.model(input_ids=tokens,position_ids=torch.cat((positions,suffix_positions),dim=2),use_cache=True)
        torch.testing.assert_close(suffix,full.logits[:,-2:],atol=1e-5,rtol=1e-4)

    @torch.inference_mode()
    def test_reset_removes_real_states_counts_and_model_rope(self):
        stream = HybridSequence(self.model)
        stream.append(input_ids=torch.tensor([[1,2,3,4]]))
        self.model.model.rope_deltas = torch.tensor([[-17]])
        stream.reset("new_video")
        self.assertEqual((stream.seen_tokens,stream.next_rope_position),(0,0))
        self.assertEqual(stream.cache.get_seq_length(),0)
        self.assertFalse(stream.cache.has_previous_state)
        self.assertIsNone(self.model.model.rope_deltas)
        self.assertTrue(all(x is None for field in FIELDS for x in getattr(stream.cache,field)))
        actual = stream.append(input_ids=torch.tensor([[21,22]]))
        expected = self.model(input_ids=torch.tensor([[21,22]]),use_cache=False).logits
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)

    @torch.inference_mode()
    def test_budget_rejection_before_mutating_state(self):
        stream = HybridSequence(self.model,max_tokens=4)
        stream.append(input_ids=torch.tensor([[1,2,3]]))
        before = snapshot(stream.cache)
        with self.assertRaises(ValueError):
            stream.append(input_ids=torch.tensor([[4,5]]))
        assert_states(self,stream.cache,before,exact=True)
        self.assertEqual((stream.seen_tokens,stream.next_rope_position),(3,3))

    @torch.inference_mode()
    def test_invalid_positions_and_reset_leave_state_unchanged(self):
        stream = HybridSequence(self.model)
        stream.append(input_ids=torch.tensor([[1,2]]))
        before = snapshot(stream.cache)
        for operation in (lambda:stream.reset(""),lambda:stream.append(input_ids=torch.tensor([[3,4]]),
                position_ids=torch.arange(2)[None,None,:].expand(4,1,2))):
            with self.assertRaises(ValueError):
                operation()
            assert_states(self,stream.cache,before,exact=True)
        self.assertEqual(stream.seen_tokens,2)


if __name__ == "__main__":
    unittest.main()
