"""Train a basic anomaly baseline on NORMAL videos only; do not include violation clips."""
import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from forte_ai.pipeline import extract_features

p = argparse.ArgumentParser()
p.add_argument("--normal", nargs="+", required=True, help="Paths of videos labeled no_violation")
p.add_argument("--out", default="baseline.joblib")
a = p.parse_args()
all_features = []
for file in a.normal:
    X, _ = extract_features(file)
    print(f"{file}: {len(X)} sampled frames")
    all_features.append(X)
X = np.concatenate(all_features)
model = IsolationForest(n_estimators=120, contamination=0.08, random_state=42, n_jobs=-1)
model.fit(X)
# Threshold chosen on training data; NOT calibrated on unseen violations.
threshold = float(np.quantile(-model.decision_function(X), 0.92))
joblib.dump({"schema_version": 1, "model": model, "threshold": threshold}, a.out)
print(f"Saved {Path(a.out).resolve()} | frames={len(X)} threshold={threshold:.5f}")
