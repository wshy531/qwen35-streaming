"""Geometry, output/state equivalence, bounded viewer and collaboration policies."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.request

import numpy as np
import torch

from qwen35_local import Qwen3_5Config,Qwen3_5ForConditionalGeneration
from scripts.video_demo import branch_allowed,check_proof,source_hashes,request,budget
from scripts.baseline import file_hash
from streaming.hybrid import HybridSequence
from streaming.preprocessing import fit_canvas
from streaming.viewer import Viewer
from tests.test_local_model import tiny_config
from tests.test_mixed_state import assert_states


class VideoDemoTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_portrait_and_wide_geometry_no_stretch(self):
        for shape, bounds in (((200,100,3),(0,192,80,176)),((100,200,3),(32,160,0,256))):
            image = fit_canvas(np.full(shape,255,np.uint8),(192,256))
            self.assertEqual(image.shape,(192,256,3))
            y,x = np.where(image[:,:,0] != 0)
            self.assertEqual((y.min(),y.max()+1,x.min(),x.max()+1),bounds)

    def test_matching_aspect_has_no_padding_or_geometry_change(self):
        image = np.full((192,256,3),155,np.uint8)
        np.testing.assert_array_equal(fit_canvas(image,(192,256)),image)

    @torch.inference_mode()
    def test_logit_modes_same_state_and_last_prediction(self):
        torch.manual_seed(42)
        model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**tiny_config(),attn_implementation="eager")).eval()
        tokens = torch.tensor([[1,2,3,4]])
        full,last,none = [HybridSequence(model) for _ in range(3)]
        all_logits = full.append(input_ids=tokens)
        with mock.patch.object(model.lm_head,"forward",wraps=model.lm_head.forward) as head:
            last_logits = last.append(input_ids=tokens,logits_mode="last")
            self.assertEqual(head.call_args.args[0].shape[1],1)
        with mock.patch.object(model.lm_head,"forward",side_effect=AssertionError("No LM head for state-only append")):
            self.assertIsNone(none.append(input_ids=tokens,logits_mode="none"))
        torch.testing.assert_close(last_logits,all_logits[:,-1:],rtol=1e-5,atol=1e-6)
        self.assertEqual(last_logits.argmax(-1).item(),all_logits[:,-1:].argmax(-1).item())
        assert_states(self,last.cache,full.cache,exact=True)
        assert_states(self,none.cache,full.cache,exact=True)
        self.assertEqual(none.seen_tokens,full.seen_tokens)
        a = none.append(input_ids=torch.tensor([[5]]),logits_mode="last")
        b = full.append(input_ids=torch.tensor([[5]]),logits_mode="last")
        torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_invalid_logit_option_rejected_before_state(self):
        model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**tiny_config(),attn_implementation="eager")).eval()
        sequence = HybridSequence(model)
        with self.assertRaises(ValueError):
            sequence.append(input_ids=torch.tensor([[1]]),logits_mode="bad")
        self.assertEqual(sequence.seen_tokens,0)

    def test_collaborator_branch_config(self):
        self.assertTrue(branch_allowed("wshy/demo",[]))
        self.assertTrue(branch_allowed("36tar/s4-mixed-state",["36tar/s4-mixed-state"]))
        self.assertFalse(branch_allowed("wshy/demo",["36tar/s4-mixed-state"]))
        for branch in ("main","master","HEAD",""):
            self.assertFalse(branch_allowed(branch,[]))
            self.assertFalse(branch_allowed(branch,[branch]))

    def test_viewer_bounded_answers_and_safe_snapshot(self):
        viewer = Viewer("CPU interface test")
        viewer.frame(np.zeros((32,64,3),np.uint8))
        for i in range(20):
            viewer.event({"kind":"answer","status":{"last_received_pts":float(i),"memory_from_pts":10.,
                "resets":1,"received_real_frames":i+1,"encoded_real_frames":i+1,"session":{"seen_tokens":100},"finished":False},
                "answer":{"timestamp":float(i),"question":"</script><script>bad</script>","answer":"test"}})
        self.assertEqual(len(viewer.snapshot()["answers"]),8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"view.html"
            viewer.save(path)
            self.assertNotIn("</script><script>bad</script>",path.read_text(encoding="utf-8"))
            self.assertIn("\\u003c/script>",path.read_text(encoding="utf-8"))

    def test_loopback_http_current_state_and_cleanup(self):
        viewer = Viewer("CPU interface test")
        try:
            url = viewer.start(0)
            self.assertTrue(url.startswith("http://127.0.0.1:"))
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(url+"state",timeout=2) as reply:
                self.assertEqual(json.load(reply)["scope"],"CPU interface test")
        finally:
            viewer.close()
        self.assertIsNone(viewer.server)

    def test_missing_preflight_rejected_before_gpu(self):
        from argparse import Namespace
        with self.assertRaises(ValueError):
            check_proof(Namespace(preflight=None))

    def test_launcher_and_core_are_in_source_proof(self):
        hashes = source_hashes()
        self.assertIn("@launcher",hashes)
        self.assertIn("streaming/hybrid.py",hashes)
        self.assertIn("streaming/preprocessing.py",hashes)

    def test_proof_rejects_changed_input_source_request_and_failed_cpu(self):
        from argparse import Namespace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root/"model"
            model.mkdir()
            (model/"DOWNLOAD_MANIFEST.json").write_text("{}")
            video = root/"input.webm"
            video.write_bytes(b"controlled test bytes")
            args = Namespace(stage="gpu",preflight=root,out=root,model_dir=model,video=video,
                             fps=1.,duration=5.,max_frames=5,answer_tokens=2,max_tokens=320,qa_every=4,
                             wall_seconds=600,allow_branch=[],ui_port=-1,question="What is visible?")
            def write(name,value):
                (root/name).write_text(json.dumps(value),encoding="utf-8")
            write("status.json",{"status":"completed","stage":"preflight"})
            write("summary.json",{"schedule":[]})
            write("cpu_tests.json",{"passed":True})
            proof = {"request":request(args),"budget":budget(args),"source_hashes":{"test":"frozen"},
                     "summary_sha256":file_hash(root/"summary.json"),"cpu_tests_sha256":file_hash(root/"cpu_tests.json"),
                     "video_sha256":file_hash(video),"model_manifest_sha256":file_hash(model/"DOWNLOAD_MANIFEST.json"),
                     "assets":[],"cpu_tests_passed":True,"schedule":[]}
            write("proof.json",proof)
            with mock.patch("scripts.video_demo.source_hashes",return_value={"test":"frozen"}),\
                 mock.patch("scripts.video_demo.verify_assets",return_value={"files":[]}),\
                 mock.patch("scripts.video_demo.subprocess.check_output",return_value="collaborator/demo\n"):
                self.assertEqual(check_proof(args),proof)
                args.question = "Changed question"
                with self.assertRaisesRegex(ValueError,"request"):
                    check_proof(args)
                args.question = "What is visible?"
                video.write_bytes(b"changed video")
                with self.assertRaisesRegex(ValueError,"Input"):
                    check_proof(args)
                video.write_bytes(b"controlled test bytes")
                write("cpu_tests.json",{"passed":False})
                with self.assertRaisesRegex(ValueError,"evidence"):
                    check_proof(args)
                write("cpu_tests.json",{"passed":True})
                proof["source_hashes"] = {"test":"changed"}
                write("proof.json",proof)
                with self.assertRaisesRegex(ValueError,"source"):
                    check_proof(args)


if __name__ == "__main__":
    unittest.main()
