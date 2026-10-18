import json
import shutil
from pathlib import Path
import cv2

root = Path("training")
source = root / "relabel"
labels = root / "relabel_verified"
dest = root / "v3_dataset"
classes = (root / "classes.txt").read_text().splitlines()

for split in ("train", "val"):
    (dest / split).mkdir(parents=True, exist_ok=True)

(dest / "annotations").mkdir(parents=True, exist_ok=True)

for split in ("train", "val"):
    images = []
    annotations = []

    for label in sorted(labels.glob("X-*.txt")):
        video = label.stem.split("_")[0]
        target_split = "val" if video == "X-4" else "train"

        if target_split != split:
            continue

        image_path = source / (label.stem + ".jpg")
        frame = cv2.imread(str(image_path))

        if frame is None:
            print("Missing:", image_path)
            continue

        h, w = frame.shape[:2]
        image_id = len(images) + 1
        pending = []

        for line in label.read_text().splitlines():
            parts = line.split()

            if len(parts) != 5:
                continue

            cls = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])

            if cls != 0:
                continue

            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h

            x1 = max(0.0, min(float(w), x1))
            y1 = max(0.0, min(float(h), y1))
            x2 = max(0.0, min(float(w), x2))
            y2 = max(0.0, min(float(h), y2))

            if x2 <= x1 or y2 <= y1:
                continue

            pending.append({
                "id": len(annotations) + len(pending) + 1,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "area": (x2 - x1) * (y2 - y1),
                "iscrowd": 0,
            })

        if not pending:
            continue

        shutil.copy2(image_path, dest / split / image_path.name)

        images.append({
            "id": image_id,
            "file_name": image_path.name,
            "width": w,
            "height": h,
        })

        annotations.extend(pending)

    data = {
        "info": {"description": "Lighthouse AI bank camera pilot"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": i + 1,
                "name": name,
                "supercategory": "bank",
            }
            for i, name in enumerate(classes)
        ],
    }

    path = dest / "annotations" / f"instances_{split}.json"
    path.write_text(json.dumps(data, indent=2))

    print(split, "images:", len(images), "objects:", len(annotations))
