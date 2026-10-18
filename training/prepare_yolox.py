import json
import shutil
from pathlib import Path

root = Path("training")
source_images = root / "images"
source_annotations = root / "coco"
destination = root / "yolox_dataset"

annotations_dir = destination / "annotations"
annotations_dir.mkdir(parents=True, exist_ok=True)

for split in ("train", "val"):
    data = json.loads(
        (source_annotations / f"{split}.json").read_text()
    )

    image_dir = destination / split
    image_dir.mkdir(parents=True, exist_ok=True)

    for image in data["images"]:
        source = source_images / image["file_name"]
        target = image_dir / image["file_name"]

        if not source.exists():
            raise FileNotFoundError(source)

        shutil.copy2(source, target)

    target_json = annotations_dir / f"instances_{split}.json"
    target_json.write_text(json.dumps(data, indent=2))

    print(split, "images:", len(data["images"]))
    print("Annotations:", len(data["annotations"]))
    print("Saved:", target_json)
