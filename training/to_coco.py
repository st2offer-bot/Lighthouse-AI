import json
from pathlib import Path

import cv2

ROOT = Path("training")
IMAGES = ROOT / "images"
LABELS = ROOT / "labels"
OUTPUT = ROOT / "coco"
OUTPUT.mkdir(parents=True, exist_ok=True)

classes = (ROOT / "classes.txt").read_text().splitlines()
review = set((ROOT / "needs_review.txt").read_text().splitlines())

train_videos = {"X-1", "X-2", "X-3", "X-10", "X-11", "X-12", "X-13", "X-14"}
val_videos = {"X-4", "X-5", "X-15"}

categories = [
    {"id": i + 1, "name": name, "supercategory": "bank_object"}
    for i, name in enumerate(classes)
]

for split, videos in [("train", train_videos), ("val", val_videos)]:
    images = []
    annotations = []
    image_id = 1
    annotation_id = 1

    for label in sorted(LABELS.glob("X-*.txt")):
        if label.stem in review:
            continue

        video_id = label.stem.rsplit("_", 1)[0]
        if video_id not in videos:
            continue

        image_path = IMAGES / f"{label.stem}.jpg"
        image = cv2.imread(str(image_path))

        if image is None:
            continue

        height, width = image.shape[:2]

        image_annotations = []

        for line in label.read_text().splitlines():
            values = line.split()

            if len(values) != 5:
                continue

            class_id = int(values[0])
            cx, cy, bw, bh = map(float, values[1:])

            box_width = bw * width
            box_height = bh * height

            x = (cx - bw / 2) * width
            y = (cy - bh / 2) * height

            image_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": class_id + 1,
                "bbox": [x, y, box_width, box_height],
                "area": box_width * box_height,
                "iscrowd": 0,
            })

            annotation_id += 1

        if not image_annotations:
            continue

        images.append({
            "id": image_id,
            "file_name": image_path.name,
            "width": width,
            "height": height,
        })

        annotations.extend(image_annotations)
        image_id += 1

    data = {
        "info": {"description": "Lighthouse-AI pilot dataset"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    target = OUTPUT / f"{split}.json"
    target.write_text(json.dumps(data, indent=2))

    print(split, "images:", len(images), "objects:", len(annotations))
    print("Saved:", target)
