# ForteGuard — local video analyzer starter

This is a local baseline for exploratory candidate discovery, NOT a detector of bank-rule violations. It does not invent legal citations or claim that unusual motion is a violation.

## Setup
Python 3.11 or 3.12 recommended:
```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

## Run without a model (checks video decoding)
```bash
python -m scripts.run "/path/to/X-1.mov" --output result.json
```

## Train baseline on normal videos only
```bash
python -m scripts.train --normal "/path/to/X-3.mov" "/path/to/X-5.mov" --out baseline.joblib
python -m scripts.run "/path/to/X-1.mov" --model baseline.joblib --output result.json
```

This very small training set is for a smoke test, not for accuracy claims. Score calibrations depend on the camera, lighting, and scene. No face recognition or audio analysis is done. The `candidates` field is for manual review and is not part of the required jury schema; keep it for your UI/debugging only. No trained model -> no candidate inference.

## Integrating with FastAPI
In your friend's background job worker (not inside the immediate upload request):
```python
from forte_ai import analyze_video
result = analyze_video(uploaded_video_path, model_path="baseline.joblib")
# save result to a job store, return only required schema to jury
response = {
    "job_id": job_id,
    "status": "done",
    "verdict": result["verdict"],
    "violations": result["violations"],
}
```

**Do not submit baseline verdicts as production-quality predictions**: the current `no_violation` is a protocol placeholder while classifier/rule checks are not implemented. Implement validated event categories, rule mappings, and frame evidence before jury use.

## Output
- `verdict`, `violations`: placeholder jury-compatible fields
- `candidates`: exploratory anomalous intervals, confidence-like anomaly score (NOT calibrated violation probabilities), base64 JPEG snapshots
- `review_required`: signals the user must review and classifier isn't ready

Security: validate content type and upload sizes upstream; randomize file names; clean files after retention period; load `joblib` weights only from your trusted training script (pickle is unsafe for untrusted files).
# Lighthouse-AI
