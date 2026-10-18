from datetime import datetime, timezone
from backend.event_pipeline import build_events

def generate_report(job_id, diagnostic):
    frames = diagnostic.get("frames", [])
    events = build_events(frames)

    return {
        "job_id": job_id,
        "status": "completed",
        "verdict": "requires_review",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": diagnostic.get("model"),
        "bank_model": "yolox_bank_v3",
        "frames_analyzed": diagnostic.get("frames_analyzed", 0),
        "processing_time_sec": diagnostic.get("processing_time_sec"),
        "events_count": len(events),
        "events": events,
        "confirmed_violations": [],
        "requires_human_review": True,
        "limitations": [
            "Object detection does not establish a banking violation",
            "Events require human verification",
            "Normative references have not been verified",
        ],
    }
