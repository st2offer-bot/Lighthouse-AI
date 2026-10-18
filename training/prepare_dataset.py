from pathlib import Path
import shutil

root = Path("training")
images = root / "images"
labels = root / "labels"
dataset = root / "dataset"

train_videos = {"X-1", "X-2", "X-3", "X-5"}
val_videos = {"X-4"}

counts = {"train": 0, "val": 0}

for label in sorted(labels.glob("X-*.txt")):
    video_id = label.stem.rsplit("_", 1)[0]

    if video_id in train_videos:
        split = "train"
    elif video_id in val_videos:
        split = "val"
    else:
        continue

    image = images / f"{label.stem}.jpg"

    if not image.exists():
        print("Missing:", image)
        continue

    target_images = dataset / "images" / split
    target_labels = dataset / "labels" / split

    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    shutil.copy2(image, target_images / image.name)
    shutil.copy2(label, target_labels / label.name)

    counts[split] += 1

print("Train:", counts["train"])
print("Validation:", counts["val"])
