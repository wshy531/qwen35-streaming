# Evidence
Complete JSON evidence is stored as gzip+base64 with original byte length and SHA256. Every real answer is also readable in the console log and the audit answer_quality_review.json.
Restore to a fresh file using `python tools/restore_evidence.py ENVELOPE_PATH --out NEW_JSON_PATH`.
No model weights, source video, environment or credentials are included. Archive encodings do not change model results.
