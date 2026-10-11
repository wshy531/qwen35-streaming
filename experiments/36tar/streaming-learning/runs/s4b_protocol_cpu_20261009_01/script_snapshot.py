"""CPU-only pair preprocessing, tokenizer-boundary and causal position probe."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"code/qwen35-streaming-s4"))

import torch
from transformers import AutoProcessor
from transformers.feature_extraction_utils import BatchFeature
from scripts.baseline import provenance,write_json,file_hash,check_tiny_video_adapter,add_video_positions,check_position_oracle
from scripts.chunk_weight_check import PROMPT
from streaming.inputs import make_window_inputs,tensor_fingerprint
from streaming.window import read_packets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    write_json(args.out/"provenance.json",provenance())
    (args.out/"script_snapshot.py").write_bytes(Path(__file__).read_bytes())
    write_json(args.out/"script_snapshot.py.json",{"sha256":file_hash(Path(__file__))})
    try:
        torch.set_num_threads(1)
        processor = AutoProcessor.from_pretrained(ROOT/"models/Qwen3.5-0.8B",local_files_only=True)
        reader = read_packets(ROOT/"datasets/natural_commons_v1/clean_hands.webm",1)
        try:
            frames = tuple(itertools.islice(reader,4))
        finally:
            reader.close()
        whole,meta = make_window_inputs(processor,frames,PROMPT)
        pairs = [make_window_inputs(processor,frames[i:i+2],PROMPT)[0] for i in (0,2)]
        stacked = torch.cat([x.pixel_values_videos for x in pairs])
        grid = whole.video_grid_thw[0]
        per_pair = int(grid[1:].prod())//processor.video_processor.merge_size**2
        marker = processor.vision_start_token+processor.video_token+processor.vision_end_token
        template = processor.apply_chat_template([{"role":"user","content":[{"type":"video"},{"type":"text","text":PROMPT}]}],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        prefix,suffix = template.split(marker)
        blocks = [f"<{mean:.1f} seconds>"+processor.vision_start_token+processor.video_token*per_pair+processor.vision_end_token
                  for mean in meta["temporal_patch_means"]]
        tokenize = lambda text:processor.tokenizer(text,add_special_tokens=False,return_tensors="pt").input_ids
        components = [tokenize(x) for x in (prefix,*blocks,suffix)]
        concatenated = torch.cat(components,1)
        identical = torch.equal(concatenated,whole.input_ids)
        _,position_model = check_tiny_video_adapter()
        add_video_positions(position_model,whole)
        check_position_oracle(position_model,whole)
        first_ids = tokenize(prefix+blocks[0])
        first = BatchFeature({"input_ids":first_ids,"attention_mask":torch.ones_like(first_ids),
            "mm_token_type_ids":(first_ids == processor.video_token_id).long()*2,
            "video_grid_thw":pairs[0].video_grid_thw})
        add_video_positions(position_model,first)
        check_position_oracle(position_model,first)
        position_prefix_equal = (torch.equal(first_ids,whole.input_ids[:,:first_ids.shape[1]]) and
                                 torch.equal(first.position_ids,whole.position_ids[:,:,:first_ids.shape[1]]))
        nmain = sum(x.shape[1] for x in components[:-1])
        positions = whole.position_ids[:,:,nmain:]
        next_rope = int(whole.position_ids[1:,:,:nmain].max())+1
        suffix_expected = torch.stack((torch.arange(nmain,whole.input_ids.shape[1]),
            *[torch.arange(next_rope,next_rope+positions.shape[-1]) for _ in range(3)]))[:,None,:]
        report = {"scope":"CPU real processor/token/position interfaces; no official weights or GPU",
            "whole_grid":whole.video_grid_thw.tolist(),"pair_grids":[x.video_grid_thw.tolist() for x in pairs],
            "pair_pixels_concatenate_exactly_to_whole":stacked.shape == whole.pixel_values_videos.shape and torch.equal(stacked,whole.pixel_values_videos),
            "component_tokens_concatenate_exactly_to_native_template":identical,"component_token_counts":[x.shape[1] for x in components],
            "first_pair_tokens_and_positions_independent_of_second_pair":position_prefix_equal,
            "text_suffix_positions_match_incremental_rule":torch.equal(positions,suffix_expected),
            "native_question_suffix_text":suffix,"prefix_text":prefix,"pair_timestamps":meta["temporal_patch_means"],
            "whole_input_fingerprint_with_positions":tensor_fingerprint(whole),
            "processor_source_sha256":file_hash(ROOT/".venv/Lib/site-packages/transformers/models/qwen3_vl/video_processing_qwen3_vl.py"),
            "warnings":["Native smart_resize uses frame count: the four-frame result does not generalize to long videos.",
                "This probe does not exercise official-weight pair vision features, pending-frame state or actual QA generation."]}
        report["checked_short_protocol_interfaces_passed"] = all(report[k] for k in (
            "pair_pixels_concatenate_exactly_to_whole","component_tokens_concatenate_exactly_to_native_template",
            "first_pair_tokens_and_positions_independent_of_second_pair","text_suffix_positions_match_incremental_rule"))
        write_json(args.out/"protocol_summary.json",report)
        write_json(args.out/"status.json",{"status":"completed","stage":"cpu_protocol_probe"})
        print(json.dumps(report,ensure_ascii=False,indent=2))
    except Exception:
        (args.out/"traceback.txt").write_text(traceback.format_exc(),encoding="utf-8")
        write_json(args.out/"status.json",{"status":"failed","stage":"cpu_protocol_probe"})
        raise


if __name__ == "__main__":
    main()
