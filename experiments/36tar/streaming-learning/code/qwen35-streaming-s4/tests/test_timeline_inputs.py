"""Bounded epoch metadata contracts independent of model output."""
import json
from pathlib import Path
import tempfile
import unittest
from streaming.timeline_inputs import contexts,select


class TimelineInputTests(unittest.TestCase):
    def path(self,rows,directory):
        p=Path(directory)/'events.jsonl';p.write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8');return p
    def pair(self,a,b,padding=False):
        return {'kind':'encoded','record':{'frame_ids':[a,b],'timestamps':[float(a),float(b)],'synthetic_padding':padding}}
    def answer(self,last,first):
        return {'kind':'answer','answer':{'frame_id':last,'timestamp':float(last),'question':'q','generated_ids':[1],'answer':'x'},
                'status':{'memory_from_pts':float(first)},'main_unchanged_bitwise':True}
    def test_reset_excludes_discarded_history_and_future(self):
        with tempfile.TemporaryDirectory() as d:
            rows=contexts(self.path([self.pair(0,1),self.answer(1,0),{'kind':'capacity_reset'},self.pair(2,3),self.answer(3,2),self.pair(4,5)],d))
            self.assertEqual([[x['frame_id'] for x in r['frames']] for r in rows],[[0,1],[2,3]])
    def test_padding_counts_one_real_frame(self):
        with tempfile.TemporaryDirectory() as d:
            rows=contexts(self.path([self.pair(0,0,True),self.answer(0,0)],d));self.assertEqual(len(rows[0]['frames']),1)
    def test_mismatched_memory_or_unbacked_answer_refused(self):
        with tempfile.TemporaryDirectory() as d:
            for answer in (self.answer(2,0),self.answer(1,1)):
                with self.assertRaises(ValueError):contexts(self.path([self.pair(0,1),answer],d))
    def test_excessive_or_unordered_selection_refused_before_decode(self):
        for ids in ([],[1,0],list(range(7))):
            with self.assertRaises(ValueError):select('must_not_open.webm',[{'frame_id':i,'timestamp':float(i)} for i in ids])
