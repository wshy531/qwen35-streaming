"""Causality, eviction, refusal and ownership, independent of semantic answers."""
import unittest
import numpy as np
from memory import TemporalFrameMemory
from streaming.window import FramePacket

def packet(i):return FramePacket(i,float(i),np.full((8,8,3),i%256,dtype=np.uint8),i+1)
class MemoryTests(unittest.TestCase):
    def test_retained_pixels_owned_and_returned_copy(self):
        bank=TemporalFrameMemory();a,b=packet(0),packet(1);bank.add_pair(a,b)
        a.rgb[:]=99;x,reason=bank.retrieve_at(.5);self.assertEqual(reason,'retained_interval');self.assertEqual(x[0].rgb[0,0,0],0)
        x[0].rgb.flags.writeable=True;x[0].rgb[:]=88
        self.assertEqual(bank.retrieve_at(0)[0][0].rgb[0,0,0],0)
    def test_gap_and_future_never_replaced_by_nearest(self):
        bank=TemporalFrameMemory()
        for i in range(0,12,2):bank.add_pair(packet(i),packet(i+1))
        self.assertEqual(bank.retrieve_at(4.5),(None,'not_retained'))
        self.assertEqual(bank.retrieve_at(100),(None,'not_observed'))
        self.assertEqual(bank.retrieve_at(8)[0][0].frame_id,8)
    def test_fifo_eviction_keeps_hard_limit(self):
        bank=TemporalFrameMemory(max_pairs=2,every_pairs=1,max_bytes=768)
        for i in range(0,100,2):bank.add_pair(packet(i),packet(i+1));self.assertLessEqual(bank.bytes,768)
        self.assertEqual(bank.retrieve_at(0),(None,'not_retained'));self.assertEqual(bank.evicted_pairs,48)
    def test_bad_pair_rejection_is_atomic(self):
        bank=TemporalFrameMemory();bank.add_pair(packet(0),packet(1));before=bank.evidence()
        for a,b in [(packet(1),packet(2)),(packet(3),packet(3)),(packet(5),packet(4))]:
            with self.assertRaises(ValueError):bank.add_pair(a,b)
            self.assertEqual(bank.evidence(),before)
    def test_oversized_anchor_refused_without_mutation(self):
        bank=TemporalFrameMemory(max_bytes=100);before=bank.evidence()
        with self.assertRaises(ValueError):bank.add_pair(packet(0),packet(1))
        self.assertEqual(bank.evidence(),before)
    def test_bad_query_and_limits_rejected(self):
        for limits in [dict(max_pairs=0),dict(every_pairs=True),dict(max_bytes=-1)]:
            with self.assertRaises(ValueError):TemporalFrameMemory(**limits)
        bank=TemporalFrameMemory()
        for t in [-1,float('nan'),float('inf'),True]:
            with self.assertRaises(ValueError):bank.retrieve_at(t)
    def test_clear_removes_evidence_and_identity(self):
        bank=TemporalFrameMemory();bank.add_pair(packet(0),packet(1));bank.clear()
        self.assertEqual(bank.bytes,0);self.assertEqual(bank.observed_pairs,0);self.assertEqual(bank.retrieve_at(0),(None,'not_observed'))
