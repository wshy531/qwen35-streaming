"""Published boundaries and actual selection-log replay; not accuracy tests."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

CAP=Path(__file__).resolve().parents[1];sys.path[:0]=[str(CAP/'src'),str(CAP/'tools')]
from memory_policy import Pair,RecentLongHierarchy
from verify import verified_files,decode,audit_results


class DeliveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.files,cls.receipt=verified_files()

    def test_archived_actual_answers_recompute(self):
        receipt=audit_results(self.files)
        self.assertEqual(receipt['main_answers'],1415)
        self.assertEqual(receipt['new_model_loads'],0)

    def test_provenance_span_does_not_resurrect_discarded_event(self):
        bank=RecentLongHierarchy(4,4)
        for i in range(23):bank.append(Pair(i,4*i,4*i+2,payload=('actual_pair',i)))
        self.assertEqual([p.index for p in bank.read()],[0,16,17,18,19,20,21,22])
        self.assertEqual(bank.provenance()[0][:3],(0,15,0))
        self.assertNotIn(3,[p.index for p in bank.read()])
        self.assertEqual(bank.read()[0].payload,('actual_pair',0))

    def test_all_actual_question_selections_replay_without_answers(self):
        from selector import choose
        prefix='runs/08b_question_readout_new24_20261019_01/'
        requests={r['id']:r for r in decode(self.files[prefix+'requests.json'])}
        for record in decode(self.files[prefix+'memory/query_selection.json']):
            frames=requests[record['id']]['model_visible']['frames']
            bank=[SimpleNamespace(first=i,begin=frames[2*i]['pts_s'],end=frames[min(2*i+1,len(frames)-1)]['pts_s']) for i in record['bank_pairs']]
            scores={r['pair']:r['score_mean_logp'] for r in record['scores']}
            self.assertEqual([n.first for n in choose(bank,scores,kind='question')],record['selections']['Q4'])
            self.assertEqual([n.first for n in choose(bank,kind='uniform')],record['selections']['uniform4'])
            self.assertEqual([n.first for n in choose(bank,kind='recent')],record['selections']['recent4'])
            self.assertTrue(all(t['pts_s']<=requests[record['id']]['model_visible']['query_time_s'] for t in frames))


if __name__=='__main__':unittest.main(verbosity=2)
