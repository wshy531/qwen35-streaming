"""S4-A independent run records for cache continuation correctness."""
import argparse
import json
from pathlib import Path
import tempfile
import traceback
import unittest

import torch

from qwen35_local import Qwen3_5Config,Qwen3_5ForConditionalGeneration
from scripts.baseline import PROJECT,provenance,write_json
from tests.test_local_model import tiny_config


def diagnostic(out):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = Qwen3_5ForConditionalGeneration(Qwen3_5Config(**tiny_config(),attn_implementation="eager")).eval()
    tokens = torch.tensor([[1,2,3,4]])
    with torch.inference_mode():
        expected = model(input_ids=tokens,use_cache=False).logits
        cases = []
        for layout in ((2,2),(1,3),(3,1),(1,1,1,1)):
            cache,parts,start = None,[],0
            for length in layout:
                output = model(input_ids=tokens[:,start:start+length],past_key_values=cache,
                    cache_position=torch.arange(start,start+length),use_cache=True)
                cache = output.past_key_values
                parts.append(output.logits)
                start += length
            actual = torch.cat(parts,dim=1)
            cases.append({"chunks":layout,"max_abs_diff":(actual-expected).abs().max().item(),
                          "equivalent":torch.allclose(actual,expected,atol=1e-5,rtol=1e-4)})
    write_json(out/"diagnostic_summary.json",{"scope":"random FP32 tiny model, no pretrained capability","cases":cases})
    print(json.dumps(cases),flush=True)


def cpu(out,recurrent_reference=False):
    from tests.test_mixed_state import MixedStateTests
    from tests.test_local_model import LocalModelTests
    classes = [LocalModelTests,MixedStateTests]
    if recurrent_reference:
        from tests.test_recurrent_reference import RecurrentReferenceTests
        classes.append(RecurrentReferenceTests)
    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(x) for x in classes])
    with (out/"unittest.log").open("w",encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report = {"tests":result.testsRun,"errors":len(result.errors),"failures":len(result.failures),
              "passed":result.wasSuccessful(),"recurrent_reference":recurrent_reference,
              "scope":"FP32 tiny-model state/logits equivalence and isolation, no quality claims"}
    write_json(out/"cpu_summary.json",report)
    print(json.dumps(report),flush=True)
    if not result.wasSuccessful():
        raise RuntimeError("Mixed-state CPU correctness tests failed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=("diagnostic","cpu"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--recurrent-reference",action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/"tmp").mkdir()
    tempfile.tempdir = str((args.out/"tmp").resolve())
    write_json(args.out/"provenance.json",provenance())
    try:
        if args.stage == "diagnostic":
            diagnostic(args.out)
        else:
            cpu(args.out,args.recurrent_reference)
        write_json(args.out/"status.json",{"status":"completed","stage":args.stage})
    except Exception:
        (args.out/"traceback.txt").write_text(traceback.format_exc(),encoding="utf-8")
        write_json(args.out/"status.json",{"status":"failed","stage":args.stage})
        raise


if __name__ == "__main__":
    main()
