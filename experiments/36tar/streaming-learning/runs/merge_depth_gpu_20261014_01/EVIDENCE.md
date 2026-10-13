# Evidence
GPU summary is directly readable JSON. CPU proof is exact original JSON encoded as gzip+base64 to keep the historical random-output snapshot small.
Restore without overwriting an existing file:
`python tools/restore_evidence.py runs/merge_depth_cpu_20261014_01/proof.json.gz-base64.json --out runs/my_restored_cpu_proof.json`
The envelope records original byte length/SHA256 and restoration verifies both. No model weights, video, environment or credentials are included.
