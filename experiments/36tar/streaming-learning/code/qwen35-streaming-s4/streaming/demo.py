"""Synchronous demo policy above the unchanged OnlinePairSession contract."""
import time

import torch

from streaming.window import FramePacket

FIELDS = ("key_cache", "value_cache", "conv_states", "recurrent_states")


class PairDemo:
    """At most two current RGB packets; events go to a caller-owned sink.

    Four real frames per epoch before first QA, then two more per QA. EOF
    padding never earns an extra QA. Capacity reset deliberately loses history.
    """
    def __init__(self, session, question, max_new_tokens=8, sink=None, audit=True,
                 first_qa_frames=4, qa_every_frames=2):
        if any(not isinstance(x,int) or isinstance(x,bool) or x < 1 for x in (first_qa_frames,qa_every_frames)):
            raise ValueError("Positive QA intervals required")
        self.first_qa_frames, self.qa_every_frames = first_qa_frames, qa_every_frames
        session.check_ready(for_input=True)
        if session.accepted_frames or session.main.seen_tokens or session.pending is not None:
            raise ValueError("Demo requires a fresh session")
        if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or not 1 <= max_new_tokens <= session.max_answer_tokens:
            raise ValueError("Invalid answer budget")
        if session.builder.question_ids(question).shape[1] > session.max_question_tokens:
            raise ValueError("Question exceeds reserved budget")
        self.session, self.question, self.max_new_tokens = session, question, max_new_tokens
        self.sink, self.audit = sink, audit
        self.pending = None
        self.received = self.encoded = self.pairs = self.questions = self.resets = 0
        self.epoch = self.last_asked_real = self.generated = 0
        self.last_id = self.last_pts = self.shape = None
        self.finished = self.closed = self.invalid = False
        self.event_index = self.max_buffered_bytes = 0
        self.memory_from_pts = None

    def status(self):
        return {"received_real_frames": self.received, "encoded_real_frames": self.encoded,
                "pairs": self.pairs, "questions": self.questions, "resets": self.resets,
                "epoch": self.epoch, "generated_tokens": self.generated,
                "memory_from_pts": self.memory_from_pts,
                "history_across_reset": False,
                "last_received_id": self.last_id, "last_received_pts": self.last_pts,
                "pending_frame_id": None if self.pending is None else self.pending.frame_id,
                "pending_rgb_bytes": 0 if self.pending is None else self.pending.rgb.nbytes,
                "max_buffered_rgb_bytes": self.max_buffered_bytes,
                "finished": self.finished, "closed": self.closed, "invalid": self.invalid,
                "session": self.session.status()}

    def emit(self, kind, **values):
        event = {"index": self.event_index, "kind": kind, **values, "status": self.status()}
        self.event_index += 1
        if self.sink is not None:
            self.sink(event)
        return event

    def ready(self):
        if self.closed or self.invalid or self.finished:
            raise RuntimeError("Demo closed, invalid or finished")

    def validate(self, packet):
        self.session.validate(packet)
        # The underlying session forgets these checks after reset; the reader does not.
        if self.last_id is not None and (packet.frame_id <= self.last_id or packet.timestamp <= self.last_pts):
            raise ValueError("Global frame IDs and PTS must strictly increase across resets")
        if self.shape is not None and packet.rgb.shape != self.shape:
            raise ValueError("Global source shape changed")

    def append(self, packet):
        self.ready()
        try:
            self.validate(packet)
        except Exception as error:
            self.emit("rejected", error_type=type(error).__name__, error=str(error))
            raise
        self.received += 1
        self.last_id, self.last_pts, self.shape = packet.frame_id, packet.timestamp, packet.rgb.shape
        buffered = packet.rgb.nbytes + (0 if self.pending is None else self.pending.rgb.nbytes)
        self.max_buffered_bytes = max(self.max_buffered_bytes, buffered)
        self.emit("received", frame_id=packet.frame_id, timestamp=packet.timestamp,
                  decoded_count=packet.decoded_count, decode_seconds=packet.decode_seconds)
        if self.pending is None:
            pixels = packet.rgb.copy()
            pixels.flags.writeable = False
            self.pending = FramePacket(packet.frame_id, packet.timestamp, pixels, packet.decoded_count, packet.decode_seconds)
            self.emit("waiting", frame_id=packet.frame_id)
            return
        try:
            self.encode(self.pending, packet, False)
            self.pending = None
            self.maybe_ask()
        except Exception as error:
            self.invalid = True
            self.emit("error", error_type=type(error).__name__, error=str(error))
            raise

    def encode(self, first, second, padding):
        session = self.session
        prefix = session.builder.prefix_ids.shape[1]
        offset = prefix if session.main.seen_tokens == 0 else 0
        # CPU-only preview of the actual new block, never a speculative GPU encode.
        inputs, record = session.builder.pair(first, second, session.main.seen_tokens + offset,
                                             session.main.next_rope_position + offset, padding)
        size = inputs.input_ids.shape[1]
        del inputs
        if prefix + size > session.main_budget:
            raise ValueError("New pair cannot fit even a fresh epoch")
        if session.main.seen_tokens + offset + size > session.main_budget:
            before = session.status()
            session.reset()
            self.resets += 1
            self.epoch += 1
            self.last_asked_real = 0
            self.memory_from_pts = first.timestamp
            self.emit("capacity_reset", reason="actual_new_block_exceeds_main_budget",
                      discarded_epoch=before, retained_frame_ids=record["frame_ids"],
                      synthetic_padding=padding, new_block_tokens=size)
        begin = time.perf_counter()
        if session.accepted_frames == 0:
            self.memory_from_pts = first.timestamp
        session.append(first)
        if padding:
            session.finish()
        else:
            session.append(second)
        self.encoded += 1 if padding else 2
        self.pairs += 1
        actual = session.ledger[-1]
        assert actual == record
        assert session.main.seen_tokens <= session.main_budget
        self.emit("encoded", record=actual, block_tokens=size, encode_wall_seconds=time.perf_counter() - begin)

    def maybe_ask(self):
        real = self.session.encoded_frames
        if real < self.first_qa_frames or (self.last_asked_real and real - self.last_asked_real < self.qa_every_frames):
            return
        before = self.session.status()
        snapshot = self.session.main.cache.clone() if self.audit else None
        answer = self.session.ask(self.question, self.max_new_tokens)
        unchanged = self.session.status() == before
        if snapshot is not None:
            for name in FIELDS:
                left, right = getattr(snapshot, name), getattr(self.session.main.cache, name)
                unchanged = unchanged and len(left) == len(right) and all(
                    (a is None and b is None) or (a is not None and b is not None and torch.equal(a, b))
                    for a, b in zip(left, right))
            del snapshot
        assert unchanged, "QA changed the main video state"
        self.last_asked_real = real
        self.questions += 1
        self.generated += len(answer["generated_ids"])
        self.emit("answer", answer=answer, epoch_real_frames=real,
                  main_unchanged_bitwise=True if self.audit else None,
                  main_counters_unchanged=unchanged)

    def finish(self, reason="source_eof"):
        self.ready()
        try:
            if self.pending is not None:
                self.encode(self.pending, self.pending, True)
                self.pending = None
                self.maybe_ask()
            else:
                self.session.finish()
            self.finished = True
            self.emit("eof", reason=reason, padding_earns_no_extra_real_frame=True)
        except Exception as error:
            self.invalid = True
            self.emit("error", error_type=type(error).__name__, error=str(error))
            raise

    def close(self):
        if not self.closed:
            before = self.status()
            self.pending = None
            self.session.close()
            self.closed = True
            self.emit("closed", before_close=before)
