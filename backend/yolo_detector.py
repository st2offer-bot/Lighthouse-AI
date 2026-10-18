import json
import sys

import cv2
import numpy as np
import torch

from yolox.exp import get_exp
from yolox.data.data_augment import preproc
from yolox.utils import postprocess
from yolox.data.datasets import COCO_CLASSES


DEVICE = "cpu"


class YOLODetector:
    def __init__(self, checkpoint="models/yolox_nano.pth"):
        self.exp = get_exp(exp_name="yolox_nano")
        self.model = self.exp.get_model()

        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(DEVICE)
        self.model.eval()

        self.input_size = self.exp.test_size

    def detect(self, frame):
        height, width = frame.shape[:2]

        image, ratio = preproc(frame, self.input_size)
        tensor = torch.from_numpy(image).unsqueeze(0).float().to(DEVICE)

        with torch.inference_mode():
            predictions = self.model(tensor)

        # NMS on CPU for better compatibility with Apple MPS.
        predictions = predictions.detach().cpu().clone()

        outputs = postprocess(
            predictions,
            self.exp.num_classes,
            conf_thre=0.25,
            nms_thre=0.45,
            class_agnostic=True,
        )

        detections = []

        if outputs[0] is None:
            return detections

        for detection in outputs[0]:
            x1, y1, x2, y2 = (detection[:4] / ratio).tolist()

            confidence = float(detection[4] * detection[5])
            class_id = int(detection[6])

            box = [
                max(0, min(width, round(x1))),
                max(0, min(height, round(y1))),
                max(0, min(width, round(x2))),
                max(0, min(height, round(y2))),
            ]

            detections.append({
                "class": COCO_CLASSES[class_id],
                "confidence": round(confidence, 3),
                "bbox": box,
            })

        return detections


def analyze_video(video_path, sample_fps=2):
    detector = YOLODetector()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise RuntimeError("Invalid FPS")

    results = []
    frame_index = 0
    next_sample = 0.0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            timestamp = frame_index / fps
            frame_index += 1

            if timestamp + (0.5 / fps) < next_sample:
                continue

            objects = detector.detect(frame)

            results.append({
                "timecode_sec": round(timestamp, 2),
                "objects": objects,
            })

            next_sample += 1.0 / sample_fps

    finally:
        cap.release()

    return {
        "device": DEVICE,
        "model": "YOLOX-Nano",
        "frames_analyzed": len(results),
        "frames": results,
    }


if __name__ == "__main__":
    result = analyze_video(sys.argv[1])

    with open("yolo_result.json", "w") as f:
        json.dump(result, f, indent=2)

    print("Device:", result["device"])
    print("Frames analyzed:", result["frames_analyzed"])

    for frame in result["frames"]:
        if frame["objects"]:
            print(frame["timecode_sec"], frame["objects"])

    print("Saved yolo_result.json")
