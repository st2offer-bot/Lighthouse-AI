"""Video analysis baseline, fully local. Visual anomaly != proven cash-discipline violation."""
from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np


def _frame_features(frame: np.ndarray, previous: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """Small scene/motion descriptor; camera must be reasonably stable."""
    frame = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Coarse spatial light levels, colors, edges, and inter-frame motion.
    cells = cv2.resize(gray, (8, 5), interpolation=cv2.INTER_AREA).flatten() / 255.0
    hue = cv2.calcHist([hsv], [0], None, [8], [0, 180]).flatten()
    hue = hue / max(float(hue.sum()), 1)
    edges = cv2.Canny(gray, 70, 150)
    edge_fraction = float(np.count_nonzero(edges)) / edges.size
    motion = (float(np.mean(cv2.absdiff(gray, previous))) / 255.0) if previous is not None else 0.0
    features = np.concatenate([cells, hue, [edge_fraction, motion]]).astype(np.float32)
    return features, gray


def extract_features(video_path: str | Path, sample_fps: float = 2.0) -> tuple[np.ndarray, list[float]]:
    """Return frame-level descriptors and exact sampled frame timestamps."""
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise ValueError("Video frame rate unavailable")
    features: list[np.ndarray] = []
    times: list[float] = []
    previous: np.ndarray | None = None
    next_time = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = max(0.0, (cap.get(cv2.CAP_PROP_POS_FRAMES) - 1) / fps)
            if timestamp + (0.5 / fps) < next_time:
                continue
            vec, previous = _frame_features(frame, previous)
            features.append(vec)
            times.append(round(timestamp, 3))
            next_time += 1.0 / sample_fps
    finally:
        cap.release()
    if not features:
        raise ValueError("No video frames decoded")
    return np.stack(features), times


def evidence_frame(video_path: str | Path, timestamp_sec: float, max_width: int = 640) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Cannot open video for evidence")
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0, timestamp_sec) * 1000)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Cannot read evidence at {timestamp_sec}s")
        h, w = frame.shape[:2]
        if w > max_width:
            frame = cv2.resize(frame, (max_width, max(1, round(h * max_width / w))))
        success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if not success:
            raise ValueError("Cannot JPEG-encode evidence")
        return {"timestamp_sec": round(timestamp_sec, 2), "image_base64": base64.b64encode(encoded.tobytes()).decode("ascii")}
    finally:
        cap.release()


def _merge_candidates(scores: np.ndarray, times: list[float], threshold: float, gap_sec: float = 1.5) -> list[dict[str, Any]]:
    flagged = np.where(scores >= threshold)[0]
    groups: list[list[int]] = []
    for i in flagged:
        if not groups or times[i] - times[groups[-1][-1]] > gap_sec:
            groups.append([int(i)])
        else:
            groups[-1].append(int(i))
    results = []
    for group in groups:
        best = max(group, key=lambda j: scores[j])
        results.append({"timecode_sec": round(times[best], 2), "start_sec": round(times[group[0]], 2), "end_sec": round(times[group[-1]], 2), "anomaly_score": round(float(scores[best]), 5)})
    return sorted(results, key=lambda c: c["anomaly_score"], reverse=True)


def analyze_video(video_path: str | Path, model_path: str | Path | None = None, max_candidates: int = 5) -> dict[str, Any]:
    """Return non-adjudicative anomaly candidates and safe jury-schema fields.

    IMPORTANT: anomaly != violation. Anomalies are intentionally NOT emitted as violations.
    Add a separately evaluated classifier + verified legal rules before publishing verdicts.
    """
    features, times = extract_features(video_path)
    result: dict[str, Any] = {
        "verdict": "no_violation", "violations": [],
        "review_required": True,
        "diagnostic": "Exploratory visual-anomaly scan; cannot establish compliance or a violation.",
        "video_duration_sec": round(times[-1], 2),
        "frames_sampled": len(times),
        "candidates": [],
    }
    if model_path is None:
        result["diagnostic"] = "No trained model loaded. Only decoding and feature extraction were performed."
        return result
    payload = joblib.load(model_path)  # Only load files produced by your own trusted training script.
    if payload.get("schema_version") != 1:
        raise ValueError("Unexpected model schema")
    model = payload["model"]
    # IsolationForest: negative decision_function values are the most anomalous.
    scores = -model.decision_function(features)
    threshold = float(payload["threshold"])
    candidates = _merge_candidates(scores, times, threshold)[:max_candidates]
    for candidate in candidates:
        candidate["evidence_frames"] = [evidence_frame(video_path, candidate["timecode_sec"])]
    result["candidates"] = candidates
    return result
