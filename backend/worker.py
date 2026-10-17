from pathlib import Path

import cv2

from backend.yolo_detector import analyze_video


def analyze_video_file(video_path: str) -> dict:
    result = analyze_video(video_path, sample_fps=2)

    return {
        "verdict": "no_violation",
        "violations": [],
        "diagnostic": {
            "mode": "yolox_object_detection",
            "ai_inference_performed": True,
            "model": result["model"],
            "device": result["device"],
            "frames_analyzed": result["frames_analyzed"],
            "frames": result["frames"],
            "compliance_classification_performed": False,
        },
    }
