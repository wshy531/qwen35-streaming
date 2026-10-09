"""Explicit opt-in download of official weights; optional local image input."""
import argparse
import inspect
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

from qwen35_local import Qwen3_5ForConditionalGeneration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--revision", default=None, help="Override the pinned official model revision")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt", default="请用一句话介绍你自己。")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    args = parser.parse_args()
    revision = args.revision
    if revision is None and args.model == "Qwen/Qwen3.5-0.8B":
        revision = "2fc06364715b967f1860aea9cf38778875588b17"
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Install CUDA PyTorch, or use --device cpu (slow).")
    processor = AutoProcessor.from_pretrained(args.model, revision=revision)
    dtype = torch.float32 if args.device == "cpu" else (
        torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    )
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, revision=revision, dtype=dtype, attn_implementation="sdpa"
    ).to(args.device).eval()
    for name, module in [("model", model), ("vision", model.model.visual),
                         ("text", model.model.language_model)]:
        source = Path(inspect.getfile(type(module))).resolve()
        assert source.parent == Path(__file__).resolve().parents[1] / "qwen35_local", source
        print(f"{name}: {source}")
    print(f"Model revision: {getattr(model.config, '_commit_hash', None)}")
    content = []
    if args.image:
        with Image.open(args.image) as image:
            image = image.convert("RGB")
            image.thumbnail((336, 336))
            content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": args.prompt})
    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False,
        return_dict=True, return_tensors="pt",
    ).to(args.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    print(processor.batch_decode(output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0])
    if args.device == "cuda":
        print(f"Peak allocated VRAM: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
