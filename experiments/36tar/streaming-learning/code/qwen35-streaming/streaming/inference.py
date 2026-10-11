"""Instrumented original-model independent question with an explicit prompt."""
import time
import weakref

import torch

from scripts.baseline import add_video_positions
from scripts.window_experiment import ModelTrace
from streaming.inputs import make_window_inputs,tensor_fingerprint


def independent_question(model,processor,frames,prompt):
    model.model.rope_deltas = None
    start = time.perf_counter()
    inputs,info = make_window_inputs(processor,frames,prompt)
    info["position_adapter"] = add_video_positions(model,inputs)
    fingerprint = tensor_fingerprint(inputs)
    input_length = inputs.input_ids.shape[1]
    inputs = inputs.to("cuda")
    model.model.rope_deltas = model.model.rope_deltas.to("cuda")
    torch.cuda.synchronize()
    prepare_seconds = time.perf_counter()-start
    refs = []
    def capture(module,args,kwargs,output):
        cache = output.past_key_values
        if cache is not None and not any(x() is cache for x in refs):
            refs.append(weakref.ref(cache))
    handle = model.register_forward_hook(capture,with_kwargs=True)
    trace = ModelTrace(model)
    torch.cuda.reset_peak_memory_stats()
    try:
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.generate(**inputs,max_new_tokens=16,do_sample=False)
        torch.cuda.synchronize()
        generation_seconds = time.perf_counter()-start
        measurements = trace.summary()
        answer = processor.batch_decode(output[:,input_length:],skip_special_tokens=True)[0]
        record = {**info,"input_fingerprint":fingerprint,"input_tokens":input_length,
            "generated_tokens":output.shape[1]-input_length,"answer":answer,
            "prepare_and_transfer_seconds":prepare_seconds,"generation_seconds":generation_seconds,
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),"peak_reserved_bytes":torch.cuda.max_memory_reserved(),
            "cuda_intervals_ms":measurements["cuda_intervals_ms"],
            "calls":[{k:v for k,v in x.items() if k != "cache"} |
                ({"cache_total_bytes":x["cache"]["total_tensor_bytes"],"cache_seq_length":x["cache"]["seq_length"]} if "cache" in x else {})
                for x in measurements["module_calls"]]}
        return record,refs
    finally:
        handle.remove()
        trace.close()
