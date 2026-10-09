"""Small random models validate adaptation, not pretrained quality. No downloads."""
import inspect
import tempfile
import unittest
from pathlib import Path
import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config as UpstreamConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration as UpstreamModel
from qwen35_local import Qwen3_5Config, Qwen3_5ForConditionalGeneration


def tiny_config():
    return dict(
        text_config=dict(
            vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16, layer_types=["linear_attention", "full_attention"],
            linear_key_head_dim=16, linear_value_head_dim=16,
            linear_num_key_heads=2, linear_num_value_heads=2,
            rope_parameters={"rope_type": "default", "rope_theta": 10000.0,
                             "partial_rotary_factor": 1.0, "mrope_section": [2, 3, 3]},
        ),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=64,
                           num_heads=2, patch_size=2, spatial_merge_size=2,
                           temporal_patch_size=2, out_hidden_size=32,
                           num_position_embeddings=16),
        image_token_id=120, video_token_id=121,
        vision_start_token_id=122, vision_end_token_id=123,
        tie_word_embeddings=False,
    )


class LocalModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(1)
        self.local = Qwen3_5ForConditionalGeneration(
            Qwen3_5Config(**tiny_config(), attn_implementation="eager")
        ).eval()
        self.upstream = UpstreamModel(
            UpstreamConfig(**tiny_config(), attn_implementation="eager")
        ).eval()
        self.upstream.load_state_dict(self.local.state_dict(), strict=True)

    def test_local_modules_and_checkpoint_roundtrip(self):
        expected = Path(__file__).resolve().parents[1] / "qwen35_local"
        for module in [self.local, self.local.model.visual, self.local.model.language_model]:
            self.assertEqual(Path(inspect.getfile(type(module))).resolve().parent, expected)
        with tempfile.TemporaryDirectory() as directory:
            self.upstream.save_pretrained(directory)
            loaded, info = Qwen3_5ForConditionalGeneration.from_pretrained(
                directory, output_loading_info=True, attn_implementation="eager"
            )
            for key in ["missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"]:
                self.assertFalse(info.get(key), info)
            for key, value in loaded.state_dict().items():
                torch.testing.assert_close(value, self.upstream.state_dict()[key])

    @torch.inference_mode()
    def test_text_logits_and_incremental_cache(self):
        tokens = torch.tensor([[1, 2, 3, 4]])
        expected = self.upstream(input_ids=tokens, use_cache=False).logits
        actual = self.local(input_ids=tokens, use_cache=False).logits
        torch.testing.assert_close(actual, expected)
        lp = self.local(input_ids=tokens[:, :3], use_cache=True)
        up = self.upstream(input_ids=tokens[:, :3], use_cache=True)
        ln = self.local(input_ids=tokens[:, 3:], past_key_values=lp.past_key_values,
                        cache_position=torch.tensor([3]), use_cache=True)
        un = self.upstream(input_ids=tokens[:, 3:], past_key_values=up.past_key_values,
                           cache_position=torch.tensor([3]), use_cache=True)
        torch.testing.assert_close(ln.logits, un.logits)
        torch.testing.assert_close(ln.logits, actual[:, -1:], atol=1e-5, rtol=1e-4)

    @torch.inference_mode()
    def test_image_and_video_logits(self):
        for token, pixel_key, grid_key in [
            (120, "pixel_values", "image_grid_thw"),
            (121, "pixel_values_videos", "video_grid_thw"),
        ]:
            with self.subTest(modality=grid_key):
                inputs = dict(input_ids=torch.tensor([[1, 122, token, 123, 2]]),
                              use_cache=False,
                              **{pixel_key: torch.randn(4, 24), grid_key: torch.tensor([[1, 2, 2]])})
                torch.testing.assert_close(self.local(**inputs).logits, self.upstream(**inputs).logits)


if __name__ == "__main__":
    unittest.main()
