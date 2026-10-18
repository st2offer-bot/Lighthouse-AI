import json
import shutil
from pathlib import Path

import cv2

SOURCE = Path("training/merged_dataset")
DEST = Path("training/roi_dataset")
ANN = DEST / "annotations"

ANN.mkdir(parents=True, exist_ok=True)

for split in ("train", "val"):
    (DEST / split).mkdir(parents=True, exist_ok=True)

    source_json = SOURCE / "annotations" / f"instances_{split}.json"
    data = json.loads(source_json.read_text())

    for info in data["images"]:
        src = SOURCE / split / info["file_name"]
        dst = DEST / split / info["file_name"]
        shutil.copy2(src, dst)

    (ANN / f"instances_{split}.json").write_text(
        json.dumps(data, indent=2)
    )

data = json.loads(
    (ANN / "instances_train.json").read_text()
)

by_image = {}

for ann in data["annotations"]:
    by_image.setdefault(ann["image_id"], []).append(ann)

original_images = list(data["images"])
next_image_id = max((i["id"] for i in original_images), default=0) + 1
next_ann_id = max((a["id"] for a in data["annotations"]), default=0) + 1

added_images = 0
added_boxes = 0

for info in original_images:
    if not info["file_name"].startswith("local_"):
        continue

    path = DEST / "train" / info["file_name"]
    frame = cv2.imread(str(path))

    if frame is None:
        continue

    height, width = frame.shape[:2]

    regions = [
        (0, int(height * 0.25), width, int(height * 0.78)),
        (0, int(height * 0.38), width, int(height * 0.91)),
    ]

    for index, (left, top, right, bottom) in enumerate(regions):
        if bottom <= top:
            continue

        crop = frame[top:bottom, left:right]

        if crop.size == 0:
            continue

        image_id = next_image_id
        next_image_id += 1

        name = f"roi_{index}_{info['file_name']}"
        new_annotations = []

        for ann in by_image.get(info["id"], []):
            x, y, bw, bh = ann["bbox"]

            x1 = max(float(left), float(x))
            y1 = max(float(top), float(y))
            x2 = min(float(right), float(x + bw))
            y2 = min(float(bottom), float(y + bh))

            if x2 <= x1 or y2 <= y1:
                continue

            visible = (x2 - x1) * (y2 - y1) / max(bw * bh, 1e-9)

            if visible < 0.8:
                continue

            new_annotations.append({
                "id": next_ann_id,
                "image_id": image_id,
                "category_id": ann["category_id"],
                "bbox": [
                    x1 - left,
                    y1 - top,
                    x2 - x1,
                    y2 - y1,
                ],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            })

            next_ann_id += 1

        if not new_annotations:
            continue

        if not cv2.imwrite(str(DEST / "train" / name), crop):
            raise RuntimeError(f"Cannot save {name}")

        data["images"].append({
            "id": image_id,
            "file_name": name,
            "width": crop.shape[1],
            "height": crop.shape[0],
        })

        data["annotations"].extend(new_annotations)

        added_images += 1
        added_boxes += len(new_annotations)

(ANN / "instances_train.json").write_text(
    json.dumps(data, indent=2)
)

print("ROI images added:", added_images)
print("ROI boxes added:", added_boxes)
print("Train images:", len(data["images"]))
print("Train annotations:", len(data["annotations"]))
print("Validation remains unchanged")
