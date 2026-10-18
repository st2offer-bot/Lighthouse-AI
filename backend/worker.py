from backend.bank_detector import BankDetector
from backend.events import detect_events
from backend.interactions import detect_interactions
from backend.temporal import analyze_movement
import time

from backend.yolo_detector import analyze_video
from backend.evidence import create_evidence


def analyze_video_file(video_path: str, job_id="local") -> dict:
    start = time.perf_counter()

    result = analyze_video(
        video_path,
        sample_fps=2,
    )

    import cv2

    bank_detector = BankDetector()
    cap = cv2.VideoCapture(str(video_path))
    bank_objects = 0

    if cap.isOpened():
        fps = cap.get(cv2.CAP_PROP_FPS)
        for frame_data in result["frames"]:
            if fps <= 0:
                break

            timestamp = frame_data["timecode_sec"]
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, image = cap.read()

            if not ok:
                continue

            detections = bank_detector.detect(image)
            frame_data["objects"].extend(detections)
            bank_objects += len(detections)

    cap.release()

    movement_events = analyze_movement(result["frames"])

    interaction_events = detect_interactions(result["frames"])

    events = detect_events(result["frames"])

    evidence = create_evidence(
        video_path,
        result["frames"],
        job_id=job_id,
        limit=3,
    )

    elapsed = time.perf_counter() - start

    return {
        "verdict": "no_violation",
        "violations": [],
        "diagnostic": {
            "mode": "yolox_object_detection",
            "ai_inference_performed": True,
            "compliance_classification_performed": False,
            "verdict_is_placeholder": True,
            "model": result["model"],
            "device": result["device"],
            "frames_analyzed": result["frames_analyzed"],
            "processing_time_sec": round(elapsed, 2),
            "frames": result["frames"],
            "events": events,
            "movement_events": movement_events,
            "interaction_events": interaction_events,
            "evidence": evidence,
        },
    }
