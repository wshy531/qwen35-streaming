"""Reuse frozen state/visual capture/index/reader; default sources remain untouched."""
from pathlib import Path
import importlib.util
import sys
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'time_query_v1'
sys.path.insert(0,str(OLD))
spec=importlib.util.spec_from_file_location('file_time_frozen_reference',OLD/'probe.py')
frozen=importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)
from forest import raw
from reader import native_answer
from time_index import RawTimeIndex
