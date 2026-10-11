"""Same state contracts under the optional token-recurrent reference path."""
import tempfile
from pathlib import Path

import torch
from qwen35_local import Qwen3_5ForConditionalGeneration
from tests.test_mixed_state import MixedStateTests


class RecurrentReferenceTests(MixedStateTests):
    def setUp(self):
        super().setUp()
        self.model.config.text_config.streaming_recurrent_reference = True
        for layer in self.model.model.language_model.layers:
            if layer.layer_type == "linear_attention":
                layer.linear_attn.experimental_recurrent_reference = True

    @torch.inference_mode()
    def test_optional_rule_does_not_change_parameters_and_fp32_prefill_close(self):
        before = {name:value.clone() for name,value in self.model.state_dict().items()}
        tokens = torch.tensor([[1,2,3,4,5,6]])
        candidate = self.model(input_ids=tokens,use_cache=False).logits
        for layer in self.model.model.language_model.layers:
            if layer.layer_type == "linear_attention":
                layer.linear_attn.experimental_recurrent_reference = False
        original = self.model(input_ids=tokens,use_cache=False).logits
        torch.testing.assert_close(candidate,original,atol=1e-5,rtol=1e-4)
        self.assertEqual(before.keys(),self.model.state_dict().keys())
        for name,value in self.model.state_dict().items():
            torch.testing.assert_close(value,before[name],atol=0,rtol=0)

    def test_rule_config_survives_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            self.model.save_pretrained(directory)
            loaded = Qwen3_5ForConditionalGeneration.from_pretrained(Path(directory),attn_implementation="eager")
            self.assertTrue(loaded.config.text_config.streaming_recurrent_reference)
            for layer in loaded.model.language_model.layers:
                if layer.layer_type == "linear_attention":
                    self.assertTrue(layer.linear_attn.experimental_recurrent_reference)
