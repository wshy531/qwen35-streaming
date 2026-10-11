"""Local Qwen3.5 implementation; weights remain in the official model repository."""
from importlib.metadata import version

if version("transformers") != "5.3.0":
    raise ImportError("qwen35_local requires transformers==5.3.0; install requirements.txt")

from .configuration_qwen3_5 import Qwen3_5Config
from .modeling_qwen3_5 import Qwen3_5ForConditionalGeneration

__all__ = ["Qwen3_5Config", "Qwen3_5ForConditionalGeneration"]
