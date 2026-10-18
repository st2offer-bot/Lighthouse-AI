import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path("training")
DEST = ROOT / "merged_dataset"
DOWNLOADS = Path.home() / "Downloads"
CLASSES = (ROOT / "classes.txt").read_text().splitlines()

archives = {
    "banknote": "banknote-yolo.v3i.coco.zip",
    "recognition": "Banknotes Recognition.v2i.coco.zip",
}

categories = [
    {"id": i + 1, "name": name, "supercategory": "bank"}
    for i, name in enumerate(CLASSES)
]

output = {
    split: {
        "info": {"description": "Lighthouse-AI combined dataset"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": categories,
    }
    for split in ("train", "val")
}

for split in output:
    (DEST / split).mkdir(parents=True, exist_ok=True)

seen_hashes = set()
stats = Counter()


def add_image(split, filename, content, boxes, source):
    digest = hashlib.sha256(content).hexdigest()

    if digest in seen_hashes:
        stats["duplicates"] += 1
        return

    import cv2
    import numpy as np

    image = cv2.imdecode(
        np.frombuffer(content, dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        stats["invalid_images"] += 1
        return

    height, width = image.shape[:2]
    clean = []

    for class_id, box in boxes:
        if class_id < 1 or class_id > len(CLASSES):
            continue

        x, y, bw, bh = map(float, box)

        x1 = max(0.0, min(float(width), x))
        y1 = max(0.0, min(float(height), y))
        x2 = max(0.0, min(float(width), x + bw))
        y2 = max(0.0, min(float(height), y + bh))

        if x2 <= x1 or y2 <= y1:
            continue

        clean.append((class_id, [x1, y1, x2 - x1, y2 - y1]))

    if not clean:
        stats["empty_or_invalid_annotations"] += 1
        return

    image_id = len(output[split]["images"]) + 1
    new_name = f"{source}_{digest[:16]}.jpg"

    cv2.imwrite(str(DEST / split / new_name), image)

    output[split]["images"].append({
        "id": image_id,
        "file_name": new_name,
        "width": width,
        "height": height,
    })

    for class_id, box in clean:
        output[split]["annotations"].append({
            "id": len(output[split]["annotations"]) + 1,
            "image_id": image_id,
            "category_id": class_id,
            "bbox": box,
            "area": box[2] * box[3],
            "iscrowd": 0,
        })

    seen_hashes.add(digest)
    stats[f"{source}_{split}"] += 1


def import_external(key, filename):
    matches = list(DOWNLOADS.rglob(filename))

    if len(matches) != 1:
        print("Archive not uniquely found:", filename)
        return

    with zipfile.ZipFile(matches[0]) as archive:
        members = archive.namelist()
        annotation = "train/_annotations.coco.json"

        if annotation not in members:
            print("Missing annotation:", filename)
            return

        data = json.loads(archive.read(annotation))
        names = {
            item["id"]: item["name"]
            for item in data["categories"]
        }

        by_image = {}

        for ann in data["annotations"]:
            name = names.get(ann["category_id"], "").lower()

            if name in {"objects", "object", "background"}:
                continue

            if key == "banknote":
                if name != "banknote":
                    continue
            elif "vnd" not in name and "banknote" not in name:
                continue

            by_image.setdefault(ann["image_id"], []).append(
                (1, ann["bbox"])
            )

        for info in data["images"]:
            boxes = by_image.get(info["id"], [])

            if not boxes:
                continue

            basename = Path(info["file_name"]).name
            candidates = [
                member for member in members
                if member.startswith("train/")
                and Path(member).name == basename
            ]

            if len(candidates) != 1:
                stats["missing_archive_images"] += 1
                continue

            content = archive.read(candidates[0])

            add_image(
                "train",
                basename,
                content,
                boxes,
                key,
            )


def import_local(split):
    source = ROOT / "yolox_dataset"
    annotation = source / "annotations" / f"instances_{split}.json"

    data = json.loads(annotation.read_text())
    by_image = {}

    for ann in data["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(
            (ann["category_id"], ann["bbox"])
        )

    for info in data["images"]:
        path = source / split / info["file_name"]

        if not path.is_file():
            continue

        add_image(
            split,
            path.name,
            path.read_bytes(),
            by_image.get(info["id"], []),
            "local",
        )


import_local("val")
import_external("banknote", archives["banknote"])
import_external("recognition", archives["recognition"])
import_local("train")

annotation_dir = DEST / "annotations"
annotation_dir.mkdir(parents=True, exist_ok=True)

for split, data in output.items():
    target = annotation_dir / f"instances_{split}.json"
    target.write_text(json.dumps(data, indent=2))

    print(
        split,
        "images:", len(data["images"]),
        "objects:", len(data["annotations"]),
    )

print("Statistics:", dict(stats))
print("Saved:", DEST)
