"""Meaningful state/index/IO contracts, run only with the tiny CPU model."""
from pathlib import Path
from types import SimpleNamespace
import json,tempfile,unittest
from unittest.mock import patch
import numpy as np
from file_session import TimeFileSession
from streaming.window import FramePacket


class Contracts(unittest.TestCase):
    def make(self):return TimeFileSession(self.model,self.processor,3,256,True)
    def packet(self,i):
        rgb=np.full((96,128,3),i*19%256,dtype=np.uint8)
        return FramePacket(i,float(i),rgb,i+1,0.)

    def test_continuous_capacity_resets_and_bounded_eviction(self):
        wrapper=self.make()
        try:
            for i in range(20):
                wrapper.feed(self.packet(i));status=wrapper.status()
                self.assertLessEqual(status['retained_visual_tokens'],144)
                self.assertLessEqual(status['preview_bytes'],896*1024)
                self.assertLessEqual(wrapper.inner.session.main.seen_tokens,wrapper.inner.session.main_budget)
            self.assertGreater(wrapper.inner.capacity_resets,0)
            self.assertEqual([n.first for n in wrapper.index.nodes],[7,8,9])
        finally:wrapper.close()
        self.assertEqual(wrapper.status()['index_tensor_bytes'],0)
        self.assertEqual(wrapper.status()['preview_bytes'],0)

    def test_same_old_query_across_reset_then_continue(self):
        wrapper=self.make()
        try:
            for i in range(2):wrapper.feed(self.packet(i))
            q='Name the animal visible at 0.5 seconds.'
            first=wrapper.ask(q);fingerprint=wrapper.index.fingerprint()
            reset=wrapper.reset_main();self.assertEqual(reset['main_seen_tokens'],0)
            second=wrapper.ask(q)
            self.assertEqual(first['result']['generated_ids'],second['result']['generated_ids'])
            self.assertEqual(first['result']['four_axis_positions'],second['result']['four_axis_positions'])
            self.assertEqual(fingerprint,wrapper.index.fingerprint())
            second['selection']['sampled_frames'][0]['source_pts']=999
            self.assertEqual(wrapper.status()['retained_intervals'][0]['sampled_frames'][0]['source_pts'],0.)
            for i in (2,3):wrapper.feed(self.packet(i))
            self.assertEqual(wrapper.inner.encoded_pairs,2)
        finally:wrapper.close()

    def test_refusal_has_no_model_call_for_empty_future_gap_evicted_unsupported(self):
        wrapper=self.make()
        try:
            with patch('file_session.native_answer',side_effect=AssertionError('Refusal called model')) as call:
                self.assertFalse(wrapper.ask('Name the animal visible at 0.5 seconds.')['model_called'])
                self.assertFalse(wrapper.ask('What happened?')['model_called'])
                for i in range(6):wrapper.feed(self.packet(i))
                before=wrapper.status()
                for stamp in ('1.5','2e3','1000000000','-1'):
                    self.assertFalse(wrapper.ask(f'Name the animal visible at {stamp} seconds.')['model_called'])
                self.assertEqual(before,wrapper.status())
                for i in (6,7):wrapper.feed(self.packet(i))
                self.assertEqual(wrapper.ask('Name the animal visible at 0.5 seconds.')['reason'],'no_retained_exact_pair')
                self.assertFalse(call.called)
        finally:wrapper.close()

    def test_pending_eof_and_older_query_do_not_use_future_sample(self):
        wrapper=self.make()
        try:
            for i in range(3):wrapper.feed(self.packet(i))
            self.assertTrue(wrapper.ask('Name the animal visible at 0.5 seconds.')['model_called'])
            with self.assertRaises(ValueError):wrapper.reset_main()
            final=wrapper.finish()
            self.assertEqual(final['main']['dropped_odd_eof_frame']['frame_id'],2)
            self.assertEqual(final['encoded_horizon'],1.)
            self.assertEqual(wrapper.ask('Name the animal visible at 2.0 seconds.')['reason'],'future_or_empty')
        finally:wrapper.close()

    def test_failed_main_write_cannot_commit_index_and_preview_is_owned(self):
        wrapper=self.make()
        try:
            packet=self.packet(0);wrapper.feed(packet)
            before=wrapper.pending_display['images'][0];packet.rgb.fill(255)
            self.assertEqual(before,wrapper.pending_display['images'][0])
            hooks=len(self.model.model.visual._forward_hooks)
            with patch.object(wrapper.inner.session,'commit_pair',side_effect=RuntimeError('deliberate failure')):
                with self.assertRaises(RuntimeError):wrapper.feed(self.packet(1))
            self.assertEqual(len(wrapper.index.nodes),0)
            self.assertEqual(len(self.model.model.visual._forward_hooks),hooks)
            with self.assertRaises(RuntimeError):wrapper.ask('Name the animal visible at 0.5 seconds.')
        finally:wrapper.close()

    def test_report_escapes_text_and_plan_preserves_unreached_triggers(self):
        from time_file_entry import render_report,plan
        directory=Path(tempfile.mkdtemp()).resolve()
        try:
            args=SimpleNamespace(out=directory,video=Path('<video>.webm'),stage='preflight')
            row={'question':'<script>bad</script>','trigger':'eof','previews':[],
                 'status':{'main':{'epoch':0},'retained_intervals':[]},
                 'outcome':{'model_called':False,'reason_text':'<script>bad</script>'}}
            render_report(args,[row],{'retained_intervals':[]},[{'after_pairs':4}])
            text=(directory/'report.html').read_text(encoding='utf-8')
            self.assertIn('&lt;script&gt;',text);self.assertNotIn('<script>',text)
            args=SimpleNamespace(query_plan=None,question=['What happened?'],max_frames=9)
            self.assertEqual(plan(args)[0]['after_pairs'],'eof')
            args.question=[]
            with self.assertRaises(ValueError):plan(args)
        finally:
            (directory/'report.html').unlink(missing_ok=True);directory.rmdir()

    def test_short_input_keeps_unreached_query_pending(self):
        from time_file_entry import replay
        directory=Path(tempfile.mkdtemp()).resolve()
        self.assertTrue(directory.is_relative_to(Path(tempfile.gettempdir()).resolve()))
        try:
            query_plan=directory/'plan.json'
            query_plan.write_text(json.dumps([{'after_pairs':4,'question':'Name the animal visible at 0.5 seconds.'},
                                              {'after_pairs':'eof','question':'What happened?'}]),encoding='utf-8')
            args=SimpleNamespace(out=directory,video=Path('short.webm'),stage='preflight',max_frames=9,
                                 max_pairs=3,max_tokens=256,no_preview=True,query_plan=query_plan,
                                 question=[],reset_after_pairs=[])
            with patch('time_file_entry.packets',return_value=iter([self.packet(0),self.packet(1)])):
                result=replay(self.model,self.processor,args)
            self.assertEqual(result['request_attempts'],1);self.assertEqual(result['model_questions'],0)
            self.assertEqual(result['pending_unreached'][0]['after_pairs'],4)
        finally:
            for p in directory.iterdir():
                self.assertTrue(p.is_file());p.unlink()
            directory.rmdir()

    def test_file_pts_roundoff_boundary_does_not_snap_real_gap(self):
        wrapper=self.make()
        try:
            wrapper.feed(self.packet(0))
            p=self.packet(1)
            wrapper.feed(FramePacket(p.frame_id,.9999999999999999,p.rgb,p.decoded_count,p.decode_seconds))
            with patch('file_session.native_answer',return_value={'answer':'stub','generated_ids':[]}) as call:
                result=wrapper.ask('Name the animal visible at 1.0 seconds.')
                self.assertTrue(result['model_called']);self.assertTrue(call.called)
                self.assertLess(result['selection']['selected_interval'][1],1.)
                self.assertEqual(result['selection']['boundary_tolerance_seconds'],1e-9)
                self.assertFalse(wrapper.ask('Name the animal visible at 1.00001 seconds.')['model_called'])
                self.assertFalse(wrapper.ask('Name the animal visible at 1.5 seconds.')['model_called'])
        finally:wrapper.close()
