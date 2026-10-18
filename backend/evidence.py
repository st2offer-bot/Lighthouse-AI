import base64
from pathlib import Path

import cv2


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_evidence(video_path, frames, job_id="local", limit=3):
    candidates = sorted(
        frames,
        key=lambda item: (
            len(item["objects"]),
            max(
                (o["confidence"] for o in item["objects"]),
                default=0
            )
        ),
        reverse=True,
    )

    selected = candidates[:limit]
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video for evidence")

    evidence = []
    folder = OUTPUT_DIR / job_id
    folder.mkdir(parents=True, exist_ok=True)

    try:
        for index, item in enumerate(selected):
            timestamp = item["timecode_sec"]

            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
            ok, original = cap.read()

            if not ok:
                continue

            annotated = original.copy()
            height, width = annotated.shape[:2]

            for obj in item["objects"]:
                x1, y1, x2, y2 = obj["bbox"]
                x1 = max(0, min(width - 1, x1))
                x2 = max(0, min(width - 1, x2))
                y1 = max(0, min(height - 1, y1))
                y2 = max(0, min(height - 1, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                label = (
                    f"{obj['class']} "
                    f"{obj['confidence']:.2f}"
                )

                cv2.putText(
                    annotated,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

            original_path = folder / f"original_{index}.jpg"
            annotated_path = folder / f"annotated_{index}.jpg"

            cv2.imwrite(str(original_path), original)
            cv2.imwrite(str(annotated_path), annotated)

            ok_original, original_jpg = cv2.imencode(
                ".jpg", original
            )
            ok_annotated, annotated_jpg = cv2.imencode(
                ".jpg", annotated
            )

            if not ok_original or not ok_annotated:
                continue

            evidence.append({
                "timestamp_sec": timestamp,
                "objects": item["objects"],
                "image_base64": base64.b64encode(
                    original_jpg.tobytes()
                ).decode("ascii"),
                "annotated_image_base64": base64.b64encode(
                    annotated_jpg.tobytes()
                ).decode("ascii"),
                "original_path": str(original_path),
                "annotated_path": str(annotated_path),
            })

    finally:
        cap.release()

    return evidence
