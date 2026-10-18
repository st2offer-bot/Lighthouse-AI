import argparse
import json
import time
from pathlib import Path

import cv2
import torch

from yolox.exp import get_exp
from yolox.data.data_augment import preproc
from yolox.utils import postprocess

CLASSES = ["cash", "document", "phone", "personal_item", "collection_bag"]

parser = argparse.ArgumentParser()
parser.add_argument("video")
parser.add_argument("--interval", type=float, default=1.0)
parser.add_argument("--conf", type=float, default=0.25)
args = parser.parse_args()

if args.interval <= 0:
    parser.error("interval must be positive")

torch.set_num_threads(4)

exp = get_exp(exp_name="yolox_nano")
exp.num_classes = 5

model = exp.get_model()
ckpt = torch.load(
    "training/models/yolox_bank_v3.pth",
    map_location="cpu",
    weights_only=False,
)
model.load_state_dict(ckpt["model"])
model.eval()

cap = cv2.VideoCapture(args.video)

if not cap.isOpened():
    raise SystemExit("Cannot open video")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    cap.release()
    raise SystemExit("Invalid FPS")

folder = Path("training/new_video_test") / Path(args.video).stem
folder.mkdir(parents=True, exist_ok=True)

results = []
frame_index = 0
next_sample = 0.0
started = time.perf_counter()

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        timestamp = frame_index / fps
        frame_index += 1

        if timestamp + 0.001 < next_sample:
            continue

        next_sample += args.interval

        height, width = frame.shape[:2]
        top = int(height * 0.28)
        bottom = int(height * 0.80)

        crop = frame[top:bottom, :]
        if crop.size == 0:
            continue

        tensor_image, ratio = preproc(crop, exp.test_size)
        tensor = torch.from_numpy(tensor_image).unsqueeze(0).float()

        with torch.no_grad():
            raw = model(tensor)

        prediction = postprocess(
            raw.detach().cpu().clone(),
            5,
            conf_thre=args.conf,
            nms_thre=0.45,
            class_agnostic=True,
        )[0]

        detections = []

        if prediction is not None:
            for row in prediction:
                cls = int(row[6])
                confidence = float(row[4] * row[5])

                if not 0 <= cls < len(CLASSES):
                    continue

                x1, y1, x2, y2 = (row[:4] / ratio).tolist()

                x1 = max(0, min(width - 1, round(x1)))
                y1 = max(0, min(height - 1, round(y1 + top)))
                x2 = max(0, min(width - 1, round(x2)))
                y2 = max(0, min(height - 1, round(y2 + top)))

                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append({
                    "class": CLASSES[cls],
                    "confidence": round(confidence, 3),
                    "bbox": [x1, y1, x2, y2],
                })

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{CLASSES[cls]} {confidence:.2f}",
                    (x1, max(20, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        filename = f"frame_{len(results):04d}.jpg"
        cv2.imwrite(str(folder / filename), frame)

        results.append({
            "timecode_sec": round(timestamp, 2),
            "detections": detections,
            "image": filename,
        })

finally:
    cap.release()

elapsed = time.perf_counter() - started

summary = {
    "video": args.video,
    "model": "yolox_bank_v3",
    "frames_analyzed": len(results),
    "processing_time_sec": round(elapsed, 2),
    "detections": sum(len(r["detections"]) for r in results),
    "frames": results,
}

(folder / "results.json").write_text(json.dumps(summary, indent=2))

print("Frames:", summary["frames_analyzed"])
print("Detections:", summary["detections"])
print("Processing time:", summary["processing_time_sec"])
print("Output:", folder)

for frame in results:
    if frame["detections"]:
        print(frame["timecode_sec"], frame["detections"])
