"""Bounded successful-write feature indexing with independent strict-time queries."""
from io import BytesIO
import hashlib
from PIL import Image
from dependencies import frozen,raw,native_answer,RawTimeIndex
from streaming.online import OnlinePairSession
from streaming.preprocessing import fit_canvas
from runtime import EventRuntime

REASONS={
    'unsupported_or_ambiguous_question':'问题不属于支持的时间问句，或含歧义。',
    'nonfinite_time':'查询时间必须是有限数值。',
    'future_or_empty':'该时间尚未成功编码，或还没有完整帧对。',
    'no_retained_exact_pair':'没有保留覆盖该时间的完整采样帧对；可能已淘汰，或位于帧对之间的空隙。',
    'question_token_limit':'问题超过预留长度。',
}


class TimeFileSession:
    def __init__(self,model,processor,max_pairs=3,max_tokens=256,preview=True):
        self.index=RawTimeIndex(max_pairs)
        session=OnlinePairSession(model,processor,max_tokens=max_tokens,max_question_tokens=64,
                                  max_answer_tokens=24,resize_mode='letterbox',video_logits='none')
        self.inner=EventRuntime(session,0.047536224126815796)
        self.preview_enabled=bool(preview)
        self.display={}
        self.pending_display=None
        self.invalid=self.closed=False

    def ready(self):
        if self.closed or self.invalid:raise RuntimeError('Time session closed or invalid')
        self.inner.ready()

    def status(self):
        return {'main':self.inner.status(),'retained_intervals':[
            {'pair_id':n.first,'interval':[n.begin,n.end],'sampled_frames':[dict(r) for r in self.display.get(n.first,{}).get('frames',[])]}
            for n in self.index.nodes],
            'retained_visual_tokens':sum(len(n.features) for n in self.index.nodes),
            'index_tensor_bytes':sum(n.metadata()['tensor_bytes'] for n in self.index.nodes),
            'preview_bytes':sum(len(b) for d in self.display.values() for b in d['images'])+
                            (0 if self.pending_display is None else sum(len(b) for b in self.pending_display['images'])),
            'encoded_horizon':self.index.horizon,'time_is_normalized_file_pts':True,
            'pair_interval_is_sampled_evidence_not_continuous_recording':True,
            'supported_questions':['Name the animal visible at T seconds.',
                "Is the animal's coat mostly light-colored or mostly dark-colored at T seconds? Answer light or dark."]}

    def display_frame(self,packet):
        images=[]
        if self.preview_enabled:
            image=Image.fromarray(fit_canvas(packet.rgb,(192,256)))
            with BytesIO() as buffer:
                image.save(buffer,format='JPEG',quality=78);data=buffer.getvalue()
            if len(data)>128*1024:raise RuntimeError('Preview byte bound')
            images=[data]
        return {'frames':[{'source_frame_id':packet.frame_id,'source_pts':packet.timestamp}], 'images':images}

    def feed(self,packet):
        self.ready()
        try:
            result=self.inner.feed(packet)
            current=self.display_frame(packet)
            if result['encoded']:
                if self.pending_display is None:raise AssertionError('Missing display pair start')
                node=raw(self.inner.bank.current)
                self.index.append(node)
                self.display[node.first]={'frames':self.pending_display['frames']+current['frames'],
                                          'images':self.pending_display['images']+current['images']}
                retained={n.first for n in self.index.nodes}
                self.display={i:d for i,d in self.display.items() if i in retained}
                self.pending_display=None
            else:self.pending_display=current
            if self.status()['preview_bytes']>896*1024:raise RuntimeError('Bounded previews exceeded')
            return result
        except Exception:
            self.invalid=True
            raise

    def reset_main(self):
        self.ready();before=self.index.fingerprint()
        result=self.inner.reset_main()
        if self.index.fingerprint()!=before:raise AssertionError('Reset changed time index')
        return result

    def ask(self,question):
        self.ready()
        if not isinstance(question,str) or len(question)>512:
            selected={'ok':False,'reason':'unsupported_or_ambiguous_question'}
        else:selected=self.index.query(question)
        # A file PTS such as 0.9999999999999999 must not exclude a requested 1.0.
        # This tolerance covers floating-point roundoff only; it never snaps across a sampled-pair gap.
        if (not selected['ok'] and selected['reason'] in ('future_or_empty','no_retained_exact_pair')
                and 'time' in selected):
            stamp=selected['time'];tolerance=1e-9
            boundary=[n for n in self.index.nodes if n.begin-tolerance<=stamp<=n.end+tolerance]
            if len(boundary)==1:
                n=boundary[0]
                selected={'ok':True,'time':stamp,'selected_pair':n.first,'selected_interval':[n.begin,n.end],
                          'selection_uses':'parsed_time_and_intervals_with_roundoff_tolerance',
                          'boundary_tolerance_seconds':tolerance,'node':n.copy()}
        if selected['ok'] and self.inner.session.builder.question_ids(question).shape[1]>64:
            selected={'ok':False,'reason':'question_token_limit'}
        if not selected['ok']:
            detail={}
            text=REASONS[selected['reason']]
            if selected['reason']=='no_retained_exact_pair' and self.index.nodes:
                earlier=selected['time']<self.index.nodes[0].begin
                detail['coverage_relation']='older_than_retained_range' if earlier else 'between_retained_pairs'
                text='该时间早于当前保留范围，无法取回旧帧对。' if earlier else '该时间位于已保留帧对之间的空隙，没有对应的完整采样帧对。'
            return {'model_called':False,**selected,**detail,'reason_text':text,
                    'coverage':self.status()['retained_intervals'],'encoded_horizon':self.index.horizon}
        node=selected.pop('node')
        protected=frozen.forest_ref.protected(self.inner,{})
        index_hash=self.index.fingerprint();ledger=list(self.inner.session.ledger)
        status=self.status()
        result=native_answer(self.inner.session.model,self.inner.session.builder,[node],question,24)
        result['visual_values_sha256']=frozen.history.file_tensor_sha(node.features)
        if (protected!=frozen.forest_ref.protected(self.inner,{}) or index_hash!=self.index.fingerprint()
                or ledger!=self.inner.session.ledger or status!=self.status()):
            raise AssertionError('History question changed main/index/display state')
        evidence={**selected,'sampled_frames':[dict(r) for r in self.display[node.first]['frames']],
                  'feature_sha256':result['visual_values_sha256'],
                  'time_is_normalized_file_pts':True,'not_an_exact_intermediate_frame':True}
        return {'model_called':True,'selection':evidence,'result':result,
                'main_and_index_unchanged':True,'epoch':self.inner.epoch,
                'encoded_horizon':self.index.horizon}

    def previews(self,pair_id):
        return list(self.display[pair_id]['images'])

    def finish(self):
        self.ready();self.inner.finish();self.pending_display=None
        return self.status()

    def close(self):
        self.inner.close();self.index.clear();self.display.clear();self.pending_display=None;self.closed=True
