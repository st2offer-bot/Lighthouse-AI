from backend.rules import evaluate_event


def detect_events(frames, min_confidence=0.25):
    candidates = []

    for frame in frames:
        timestamp = frame["timecode_sec"]

        for obj in frame["objects"]:
            if obj["class"] != "cash":
                continue

            confidence = float(obj["confidence"])

            if confidence < min_confidence:
                continue

            candidates.append({
                "timecode_sec": timestamp,
                "event_type": "cash_near_counter",
                "confidence": confidence,
                "bbox": obj["bbox"],
                "source": "object_detection",
            })

    return [evaluate_event(event) for event in candidates]
