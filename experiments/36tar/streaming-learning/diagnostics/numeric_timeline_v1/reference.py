"""Portable numeric timeline wrapper; shared reference implementation read-only."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
CAP=next((p for p in HERE.parents if p.name=='streaming-learning' and (p/'code/qwen35-streaming-s4').is_dir()),None)
if CAP is None:CAP=HERE.parents[1]/'publish/qwen35-streaming/experiments/36tar/streaming-learning'
sys.path.insert(0,str(CAP/'code/qwen35-streaming-s4'))
from scripts import timeline_reference as reference
reference.CONFIG=HERE/'cases.json'
reference.PLAN=HERE/'方案.md'
reference.LAUNCHER=HERE/'run_reference.py'
base_sources=reference.sources
def sources():return {**base_sources(),'@numeric_reference_wrapper':reference.file_hash(HERE/'reference.py')}
reference.sources=sources
if __name__=='__main__':reference.main()
