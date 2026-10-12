"""Reuse frozen fair benchmark; apply size32 symmetrically and restore callables."""
import importlib.util
import inspect
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('benefit64',HERE.with_name('streaming_benefit_v1')/'benchmark.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
import torch
from scripts.chunk_size_probe import compare

base_sources,base_budget,base_preflight,base_check,base_gpu=b.sources,b.budget,b.preflight,b.check,b.gpu
def sources():return {**base_sources(),**{str(p.resolve()):b.file_hash(p) for p in HERE.iterdir() if p.suffix in ('.py','.md','.json')}}
def budget():return {**base_budget(),'execution_chunk_size':32,'chunk_rule_calls_max':4096,'default_size64_unchanged':True}

class Chunk32:
    def __init__(self,model):self.model=model;self.saved=[];self.calls=0
    def __enter__(self):
        origin=Path(inspect.getfile(type(self.model))).resolve()
        modules=[m for m in self.model.modules() if hasattr(m,'chunk_gated_delta_rule')]
        if not modules:raise ValueError('No GDN modules found')
        for m in modules:
            fn=m.chunk_gated_delta_rule
            if fn.__name__!='torch_chunk_gated_delta_rule' or Path(inspect.getsourcefile(fn)).resolve()!=origin:
                raise ValueError('Only registered local torch fallback allowed')
        for m in modules:
            fn=m.chunk_gated_delta_rule;self.saved.append((m,fn))
            def rule(*args,_original=fn,**kwargs):
                self.calls+=1
                if self.calls>4096:raise RuntimeError('Registered32 call budget exceeded')
                kwargs['chunk_size']=32
                return _original(*args,**kwargs)
            m.chunk_gated_delta_rule=rule
        return self
    def __exit__(self,*_error):
        for m,fn in self.saved:m.chunk_gated_delta_rule=fn
        if not all(m.chunk_gated_delta_rule is fn for m,fn in self.saved):raise AssertionError('Candidate callables not restored')

@torch.inference_mode()
def preflight(a):
    base_preflight(a)
    from tests.test_online import small_model
    processor=b.AutoProcessor.from_pretrained(a.model_dir,local_files_only=True)
    reference=small_model(processor);candidate=small_model(processor);candidate.load_state_dict(reference.state_dict(),strict=True)
    rows=[]
    for source in b.config()['videos']:
        frames=b.load_frames(source);left=b.session_for(reference,processor);right=b.session_for(candidate,processor)
        try:
            for size in b.config()['prefix_frames']:
                left.append(frames[size-2]);left.append(frames[size-1])
                with Chunk32(candidate):right.append(frames[size-2]);right.append(frames[size-1])
                left_before=b.snapshot(left);right_before=b.snapshot(right)
                q=left.builder.question_ids(b.config()['question'])
                with b.fp32_reference():
                    lb=left.main.fork();ll=lb.append(input_ids=q,logits_mode='last')
                    with Chunk32(candidate):
                        rb=right.main.fork();rl=rb.append(input_ids=q,logits_mode='last')
                    qa=compare(rl,ll,rb.cache,lb.cache)
                    fixed=torch.tensor([[1]],dtype=torch.long)
                    ll=lb.append(input_ids=fixed,logits_mode='last')
                    with Chunk32(candidate):rl=rb.append(input_ids=fixed,logits_mode='last')
                    decode=compare(rl,ll,rb.cache,lb.cache)
                b.unchanged(left,left_before);b.unchanged(right,right_before)
                if not qa['passed'] or not decode['passed']:raise AssertionError('Candidate32 failed existing FP32 numeric limits')
                row={'case_id':source['id']+'_'+str(size),'qa_last':qa,'fixed_next_token':decode,'main_unchanged_bitwise':True,'no_resets':True}
                rows.append(row);b.write(a.out/'candidate_numeric.json',{'passed':True,'cases':rows,'scope':'Tiny32 vs64, fixed suffix and next Token, not real-weight accuracy or full hidden-state equivalence'})
                del lb,rb,ll,rl,left_before,right_before
        finally:left.close();right.close()
    p=b.read(a.out/'proof.json');p['candidate_numeric_sha256']=b.file_hash(a.out/'candidate_numeric.json');b.write(a.out/'proof.json',p)
    print(json.dumps({'candidate32_passed':True,'numeric_cases':len(rows),'existing_limits_unchanged':True}),flush=True)
def check(a):
    p=b.read(a.preflight/'proof.json')
    if p.get('candidate_numeric_sha256')!=b.file_hash(a.preflight/'candidate_numeric.json') or not b.read(a.preflight/'candidate_numeric.json')['passed']:
        raise ValueError('Candidate numeric proof missing/changed')
    return base_check(a)
def gpu(a):
    if a.arm=='original':cls=b.original_package().Qwen3_5ForConditionalGeneration
    else:
        from qwen35_local import Qwen3_5ForConditionalGeneration
        cls=Qwen3_5ForConditionalGeneration
    sentinel=object();saved=cls.__dict__.get('from_pretrained',sentinel);loader=cls.from_pretrained;contexts=[]
    def load(_cls,*args,**kwargs):
        model=loader(*args,**kwargs);context=Chunk32(model);context.__enter__();contexts.append(context);return model
    cls.from_pretrained=classmethod(load)
    try:
        base_gpu(a)
    finally:
        for ctx in contexts:ctx.__exit__()
        if saved is sentinel:delattr(cls,'from_pretrained')
        else:cls.from_pretrained=saved
    if len(contexts)!=1:raise AssertionError('One candidate model load required')
    report=b.read(a.out/'summary.json');report['execution_candidate']={'chunk_size':32,'calls':contexts[0].calls,'callables_restored':True,'class_loader_restored':True,'default_source_not_modified':True}
    b.write(a.out/'summary.json',report)
b.sources,b.budget,b.preflight,b.check,b.gpu=sources,budget,preflight,check,gpu
if __name__=='__main__':b.main()
