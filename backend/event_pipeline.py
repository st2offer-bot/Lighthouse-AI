from backend.rules import evaluate_event

BANK_CLASSES = {"cash", "document", "phone", "personal_item", "collection_bag"}

def build_events(frames, confidence=0.25, max_gap=2.5):
    observations = []

    for frame in frames:
        timestamp = float(frame.get("timecode_sec", 0))

        for obj in frame.get("objects", []):
            name = obj.get("class")
            score = float(obj.get("confidence", 0))

            if name not in BANK_CLASSES or score < confidence:
                continue

            bbox = obj.get("bbox", [])

            if len(bbox) != 4:
                continue

            observations.append({
                "time": timestamp,
                "class": name,
                "confidence": score,
                "bbox": bbox,
            })

    observations.sort(key=lambda x: (x["class"], x["time"]))

    groups = []

    for obj in observations:
        if (
            groups
            and groups[-1]["class"] == obj["class"]
            and obj["time"] - groups[-1]["end"] <= max_gap
        ):
            group = groups[-1]
            group["end"] = obj["time"]
            group["detections"].append(obj)
        else:
            groups.append({
                "class": obj["class"],
                "start": obj["time"],
                "end": obj["time"],
                "detections": [obj],
            })

    events = []

    for group in groups:
        detections = group["detections"]
        name = group["class"]
        event_type = "cash_near_counter" if name == "cash" else "object_observed"

        event = {
            "event_id": f"event-{len(events) + 1:03d}",
            "event_type": event_type,
            "object_class": name,
            "start_sec": round(group["start"], 2),
            "end_sec": round(group["end"], 2),
            "confidence": round(max(x["confidence"] for x in detections), 3),
            "observations": len(detections),
            "bbox": max(detections, key=lambda x: x["confidence"])["bbox"],
            "source": "yolox_bank",
            "status": "requires_review",
        }

        events.append(evaluate_event(event))

    return sorted(events, key=lambda x: x["start_sec"])
