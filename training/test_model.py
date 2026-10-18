import json
from pathlib import Path

import cv2
import torch

from yolox.exp import get_exp
from yolox.data.data_augment import preproc
from yolox.utils import postprocess

CLASSES = [
    "cash",
    "document",
    "phone",
    "personal_item",
    "collection_bag",
]

ROOT = Path("training/yolox_dataset/val")
OUTPUT = Path("training/predictions")
OUTPUT.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(4)

exp = get_exp(exp_name="yolox_nano")
exp.num_classes = len(CLASSES)

model = exp.get_model()

checkpoint = torch.load(
    "training/models/yolox_bank_pilot.pth",
    map_location="cpu",
    weights_only=False,
)

model.load_state_dict(checkpoint["model"])
model.eval()

summary = []

for path in sorted(ROOT.glob("*.jpg")):
    frame = cv2.imread(str(path))

    if frame is None:
        continue

    height, width = frame.shape[:2]
    processed, ratio = preproc(frame, exp.test_size)

    tensor = torch.from_numpy(processed).unsqueeze(0).float()

    with torch.no_grad():
        predictions = model(tensor)

    outputs = postprocess(
        predictions.detach().cpu().clone(),
        len(CLASSES),
        conf_thre=0.25,
        nms_thre=0.45,
        class_agnostic=True,
    )

    detections = []

    if outputs[0] is not None:
        for row in outputs[0]:
            x1, y1, x2, y2 = (row[:4] / ratio).tolist()
            confidence = float(row[4] * row[5])
            class_id = int(row[6])

            x1 = max(0, min(width - 1, round(x1)))
            x2 = max(0, min(width - 1, round(x2)))
            y1 = max(0, min(height - 1, round(y1)))
            y2 = max(0, min(height - 1, round(y2)))

            if x2 <= x1 or y2 <= y1:
                continue

            name = CLASSES[class_id]

            detections.append({
                "class": name,
                "confidence": round(confidence, 3),
                "bbox": [x1, y1, x2, y2],
            })

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"{name} {confidence:.2f}",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    cv2.imwrite(str(OUTPUT / path.name), frame)

    summary.append({
        "image": path.name,
        "detections": detections,
    })

    print(path.name, detections, flush=True)

(OUTPUT / "results.json").write_text(
    json.dumps(summary, indent=2)
)

print("Test completed")
print("Images:", len(summary))
print("Predictions:", sum(len(x["detections"]) for x in summary))
