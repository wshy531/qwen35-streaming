"""Protect causal inputs and prevent annotation/answer leakage into requests."""
import copy
import math
import unittest

from prefix_request import make_request


def fixture():
    row=dict(id='synthetic:0',source_path='toy.mp4',question='What happened first?',
        options=['Left','Right'],answer_index=0,query_time_s=4.,task='TC',
        evidence_intervals=[[0,2]],split='comparison')
    media=dict(source_path='toy.mp4',media_path='datasets/toy.mp4',sha256='a'*64)
    trace=dict(source_path='toy.mp4',source_sha256='a'*64,frames=[
        dict(frame_id=i,pts_s=t,rgb_sha256=str(i)*64) for i,t in enumerate([0.,2.,4.,6.])])
    return row,media,trace


class PrefixRequestTests(unittest.TestCase):
    def test_labels_annotations_and_task_cannot_change_input(self):
        row,m,t=fixture();before=make_request(row,m,t)
        row.update(answer_index=1,evidence_intervals=[[100,200]],task='HLD',dataset='answer_hint',
            review=dict(witnesses=['secret-GT']),source_timing_audit='oracle',split='holdout')
        self.assertEqual(before,make_request(row,m,t))
        self.assertEqual(set(before['model_visible']),{'question','options','query_time_s','frames',
            'completed_real_pairs','snapshot_tail_pair'})

    def test_each_question_has_own_cutoff_and_snapshot_only_tail(self):
        row,m,t=fixture();r=make_request(row,m,t)['model_visible']
        self.assertEqual([f['pts_s'] for f in r['frames']],[0,2,4])
        self.assertEqual(len(r['completed_real_pairs']),1)
        self.assertEqual(r['snapshot_tail_pair']['frame_ids'],[2,2])
        self.assertFalse(r['snapshot_tail_pair']['commit_to_main_stream'])
        row['query_time_s']=2.;earlier=make_request(row,m,t)['model_visible']
        self.assertIsNone(earlier['snapshot_tail_pair'])
        self.assertEqual([f['pts_s'] for f in earlier['frames']],[0,2])
        self.assertEqual(len(t['frames']),4) # original ongoing trace remains intact

    def test_later_stream_is_invariant_to_number_of_questions(self):
        row,m,t=fixture();original=copy.deepcopy(t)
        for query in (0,2,4,2,0,6):
            row['query_time_s']=query;make_request(row,m,t)
        self.assertEqual(t,original)

    def test_reject_invalid_times_or_nonmonotonic_real_frames(self):
        for bad in (-1,math.nan,math.inf):
            row,m,t=fixture();row['query_time_s']=bad
            with self.assertRaises(ValueError):make_request(row,m,t)
        for field in ('pts_s','frame_id'):
            row,m,t=fixture();t['frames'][1][field]=t['frames'][0][field]
            with self.assertRaises(ValueError):make_request(row,m,t)

    def test_reject_wrong_assets_and_path_escape(self):
        for change in ('sha','source','path'):
            row,m,t=fixture()
            if change=='sha':t['source_sha256']='b'*64
            elif change=='source':m['source_path']='other.mp4'
            else:m['media_path']='../outside.mp4'
            with self.assertRaises(ValueError):make_request(row,m,t)


if __name__=='__main__':unittest.main(verbosity=2)
