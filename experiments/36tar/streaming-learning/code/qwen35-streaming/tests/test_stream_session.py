import unittest
from types import SimpleNamespace
import numpy as np

from streaming.session import WindowSession
from streaming.window import FramePacket


def packet(i):
    return FramePacket(i,float(i),np.zeros((8,8,3),dtype=np.uint8),i+1)


class SessionTests(unittest.TestCase):
    def test_reset_removes_pixels_identity_question_count_and_rope(self):
        model = SimpleNamespace(model=SimpleNamespace(rope_deltas="sentinel"))
        session = WindowSession(model)
        for i in range(10):
            session.append(packet(i))
        session.questions = 5
        model.model.rope_deltas = "old video position"
        session.reset("B")
        self.assertEqual(session.window.evidence()["window_pixel_bytes"],0)
        self.assertEqual(session.questions,0)
        self.assertEqual(session.reset_count,1)
        self.assertIsNone(model.model.rope_deltas)
        session.append(packet(0))
        fresh = WindowSession(session_id="B")
        fresh.append(packet(0))
        self.assertEqual(session.window.evidence(),fresh.window.evidence())

    def test_invalid_reset_preserves_prior_state(self):
        model = SimpleNamespace(model=SimpleNamespace(rope_deltas=None))
        session = WindowSession(model)
        session.append(packet(0))
        model.model.rope_deltas = "must survive rejection"
        before = session.evidence()
        for invalid in ("",None,12):
            self.assertRaises(ValueError,session.reset,invalid)
            self.assertEqual(session.evidence(),before)
            self.assertEqual(model.model.rope_deltas,"must survive rejection")

    def test_many_resets_keep_numbering_and_capacity_valid(self):
        session = WindowSession()
        for cycle in range(20):
            session.reset(str(cycle))
            for i in range(4):
                session.append(packet(i))
            self.assertEqual(session.window.evidence()["frame_ids"],[0,1,2,3])
            self.assertEqual(session.reset_count,cycle+1)


if __name__ == "__main__":
    unittest.main()
