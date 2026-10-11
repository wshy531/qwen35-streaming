# Upstream source and local modifications

The files `qwen35_local/modeling_qwen3_5.py` and
`qwen35_local/configuration_qwen3_5.py` are derived from Hugging Face Transformers
v5.3.0, commit `aad13b87ed59f2afcfaebc985f403301887a35fc`.

Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
Licensed under Apache-2.0; the full upstream license is preserved in
`third_party/transformers-LICENSE`.

Source URLs and original/adapted SHA-256 hashes are in `third_party/upstream.json`.
Initial adaptation replaces package-relative imports of Transformers utilities
with absolute `transformers.*` imports. Model/configuration code remains local.
No architectural or parameter-name changes are intended in this baseline.

The separate S4-A branch adds cache clone/reset and permits cached multi-token
GatedDeltaNet continuation using the existing torch convolution update and
recurrent initial state. It adds no trainable parameters or renamed weights.
The original baseline snapshot is retained separately. FP32 tiny-model checks
passed; the first official-weight BF16 numerical check failed its registered
limits. See `S4_A.md` for the recorded scope and reproduction entry points.
An optional, disabled-by-default `streaming_recurrent_reference` text config
selects the existing torch token-recurrent algorithm for correctness studies.
Its CPU contracts passed, but its official-weight GPU regression also failed;
it is not presented as a validated numerical fix or performance improvement.

The upstream files were generated from `modular_qwen3_5.py`. In this independent
project, the adapted files are maintained directly; upstream code-generation
tools are not used. Preserve the upstream header as provenance and review any
future upgrade explicitly rather than overwriting local changes.

Model weights are not included. Download official `Qwen/Qwen3.5-0.8B` weights
separately and follow their model license. Third-party license text does not
select a license for independently authored project files.
