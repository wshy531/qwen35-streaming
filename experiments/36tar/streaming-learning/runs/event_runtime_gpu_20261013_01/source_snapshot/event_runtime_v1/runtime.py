"""One optional bounded event-memory session composed from frozen experiment modules."""
import hashlib
import math

import numpy as np
import torch
from features import raw_block, answer
from memory import EventMemory
from routing import select
from streaming.window import FramePacket
from streaming.preprocessing import fit_canvas


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def bank_fingerprint(bank):
    rows = [g.representative for g in bank.groups] + ([] if bank.current is None else [bank.current])
    return {'status': bank.status(), 'tensors': [[tensor_sha(b.features), tensor_sha(b.mass), tensor_sha(b.xy)] for b in rows]}


class EventRuntime:
    """Single-consumer real pair capture, bounded bank, explicit evidence queries.

    Main capacity resets never change external bank state. Source frame IDs and
    received time retain a separate monotonic horizon across those resets.
    An odd EOF sample is reported and dropped, never duplicated as a second frame.
    """
    def __init__(self, session, threshold):
        self.session = session
        self.bank = EventMemory(threshold)
        self.pending = None
        self.last_received_id = self.last_received_pts = None
        self.received = self.encoded_pairs = self.capacity_resets = self.explicit_resets = 0
        self.epoch = 0
        self.current_in_epoch = False
        self.finished = self.invalid = self.closed = False
        self.dropped_eof = None

    def ready(self):
        if self.closed or self.invalid:
            raise RuntimeError('Runtime closed or invalid')

    def status(self):
        return {'received_frames': self.received, 'encoded_pairs': self.encoded_pairs,
                'last_received_id': self.last_received_id, 'last_received_pts': self.last_received_pts,
                'encoded_horizon': self.bank.last_end, 'pending_frame_id': None if self.pending is None else self.pending.frame_id,
                'pending_rgb_bytes': 0 if self.pending is None else self.pending.rgb.nbytes,
                'capacity_resets': self.capacity_resets, 'explicit_resets': self.explicit_resets,
                'epoch': self.epoch, 'current_in_epoch': self.current_in_epoch,
                'finished': self.finished, 'dropped_odd_eof_frame': self.dropped_eof,
                'bank': self.bank.status(), 'main': self.session.status()}

    def reset_main(self, reason='explicit', keep_pending=False):
        self.ready()
        before = bank_fingerprint(self.bank)
        if self.pending is not None and not keep_pending:
            raise ValueError('Complete pending pair before explicit reset')
        self.session.reset()
        if bank_fingerprint(self.bank) != before:
            raise AssertionError('Main reset changed external event evidence')
        self.epoch += 1
        self.current_in_epoch = False
        if reason == 'capacity': self.capacity_resets += 1
        else: self.explicit_resets += 1
        return {'reason': reason, 'epoch': self.epoch, 'bank_unchanged': True, 'main_seen_tokens': self.session.main.seen_tokens}

    @torch.inference_mode()
    def feed(self, packet):
        self.ready()
        if self.finished: raise ValueError('EOF already submitted')
        if (not isinstance(packet, FramePacket) or type(packet.frame_id) is not int or packet.frame_id < 0 or
            packet.decoded_count <= packet.frame_id or not math.isfinite(packet.timestamp) or packet.timestamp < 0 or
            not isinstance(packet.rgb, np.ndarray) or packet.rgb.dtype != np.uint8 or packet.rgb.ndim != 3 or
            packet.rgb.shape[-1] != 3 or min(packet.rgb.shape[:2]) < 1 or max(packet.rgb.shape[:2]) > 256):
            raise ValueError('Valid bounded RGB/source frame/PTS required')
        if self.last_received_id is not None and (packet.frame_id <= self.last_received_id or packet.timestamp <= self.last_received_pts):
            raise ValueError('Source horizon must increase even across main resets')
        rgb = fit_canvas(packet.rgb, (192, 256)); rgb.flags.writeable = False
        incoming = FramePacket(packet.frame_id, packet.timestamp, rgb, packet.decoded_count, packet.decode_seconds)
        if self.pending is None:
            self.pending = incoming
            self.last_received_id, self.last_received_pts = incoming.frame_id, incoming.timestamp
            self.received += 1
            return {'encoded': False, 'reset': None, 'status': self.status()}
        first = self.pending
        prefix = self.session.builder.prefix_ids.shape[1] if self.session.main.seen_tokens == 0 else 0
        inputs, _ = self.session.builder.pair(first, incoming, self.session.main.seen_tokens + prefix,
                                            self.session.main.next_rope_position + prefix)
        reset = None
        if self.session.main.seen_tokens + prefix + inputs.input_ids.shape[1] > self.session.main_budget:
            reset = self.reset_main('capacity', keep_pending=True)
        del inputs
        captured = []
        def capture(_module, _args, output):
            value = output.pooler_output
            if captured or not isinstance(value, torch.Tensor) or value.shape[0] != 48:
                raise ValueError('Exactly one 48-token real projected pair required')
            captured.append(value.detach().to('cpu', dtype=torch.float32).clone())
        handle = self.session.model.model.visual.register_forward_hook(capture)
        try:
            self.session.append(first); self.session.append(incoming)
            if len(captured) != 1 or self.session.last_encoded_id != incoming.frame_id:
                raise AssertionError('Main pair did not finish successfully')
            block = raw_block(captured.pop(), self.encoded_pairs, first.timestamp, incoming.timestamp)
            update = self.bank.append(block)
            self.encoded_pairs += 1
            self.pending = None
            self.current_in_epoch = True
            self.last_received_id, self.last_received_pts = incoming.frame_id, incoming.timestamp
            self.received += 1
            if self.bank.status()['tokens'] > 144 or self.bank.status()['tensor_bytes'] > 592704:
                raise RuntimeError('External bank budget exceeded')
            return {'encoded': True, 'reset': reset, 'group_update': update,
                    'feature_sha256': tensor_sha(block.features), 'status': self.status()}
        except Exception:
            self.invalid = True
            raise
        finally:
            handle.remove(); captured.clear()

    @torch.inference_mode()
    def ask(self, question, mode='current', target=None, max_new_tokens=32):
        self.ready()
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 32:
            raise ValueError('Answer budget 1..32 required')
        suffix = self.session.builder.question_ids(question)
        if suffix.shape[1] > 64: raise ValueError('Question exceeds reserved 64-token budget')
        if mode == 'current' and self.pending is not None:
            return {'model_called': False, 'reason': 'waiting_for_complete_pair'}
        if mode == 'current' and not self.current_in_epoch:
            return {'model_called': False, 'reason': 'no_current_pair_in_epoch'}
        metadata, blocks = select(self.bank, mode, target)
        if not blocks: return metadata
        result = answer(self.session.model, self.session.builder, blocks, question, max_new_tokens)
        return {'model_called': True, 'selection': metadata, 'result': result,
                'epoch': self.epoch, 'received_horizon': self.last_received_pts,
                'encoded_horizon': self.bank.last_end, 'eof_unencoded_frame': self.dropped_eof}

    def finish(self):
        self.ready()
        if self.finished: raise ValueError('EOF already submitted')
        if self.pending is not None:
            self.dropped_eof = {'frame_id': self.pending.frame_id, 'timestamp': self.pending.timestamp,
                                'reason': 'odd_sample_no_synthetic_pair'}
            self.pending = None
        self.finished = True
        return self.status()

    def close(self):
        self.session.close(); self.bank.clear(); self.pending = None
        self.closed = True
