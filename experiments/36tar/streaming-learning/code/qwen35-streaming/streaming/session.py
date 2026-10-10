"""Reusable application session for independent-window inference and reset."""
from streaming.window import FrameWindow


class WindowSession:
    def __init__(self, model=None, capacity=4, session_id="continuous"):
        self.model = model
        self.window = FrameWindow(capacity)
        self.session_id = session_id
        self.questions = 0
        self.reset_count = 0
        self._clear_position_state()

    def _clear_position_state(self):
        if self.model is not None:
            self.model.model.rope_deltas = None

    def reset(self, session_id):
        if not isinstance(session_id,str) or not session_id:
            raise ValueError("New session ID must be a nonempty string")
        self.window.reset()
        self._clear_position_state()
        self.session_id = session_id
        self.questions = 0
        self.reset_count += 1

    def append(self, packet):
        self.window.append(packet)

    def evidence(self):
        return {"session_id":self.session_id,"questions":self.questions,
                "reset_count":self.reset_count,**self.window.evidence()}
