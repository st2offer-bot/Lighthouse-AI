import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from yolox.exp import get_exp
from yolox.data.data_augment import preproc


ROOT = Path("training/roi_dataset")
WEIGHTS = Path("models/yolox_nano.pth")
OUTPUT = Path("training/models")
OUTPUT.mkdir(parents=True, exist_ok=True)


def load_samples():
    data = json.loads(
        (ROOT / "annotations/instances_train.json").read_text()
    )

    annotations = {}

    for ann in data["annotations"]:
        annotations.setdefault(ann["image_id"], []).append(ann)

    samples = []

    for info in data["images"]:
        path = ROOT / "train" / info["file_name"]
        if path.is_file():
            samples.append((path, annotations.get(info["id"], [])))

    return samples


def prepare_sample(path, annotations, image_size):
    image = cv2.imread(str(path))

    if image is None:
        raise RuntimeError(f"Cannot read {path}")

    processed, ratio = preproc(image, image_size)
    tensor = torch.from_numpy(processed).float().unsqueeze(0)

    targets = torch.zeros((1, 30, 5), dtype=torch.float32)

    if len(annotations) > 30:
        raise RuntimeError("More than 30 annotations in one image")

    for index, ann in enumerate(annotations):
        x, y, w, h = ann["bbox"]

        targets[0, index, 0] = ann["category_id"] - 1
        targets[0, index, 1] = (x + w / 2) * ratio
        targets[0, index, 2] = (y + h / 2) * ratio
        targets[0, index, 3] = w * ratio
        targets[0, index, 4] = h * ratio

    return tensor, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0001)
    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError("epochs must be positive")

    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)

    exp = get_exp(exp_name="yolox_nano")
    exp.num_classes = 5

    model = exp.get_model()

    checkpoint = torch.load(
        WEIGHTS,
        map_location="cpu",
        weights_only=False,
    )

    original = checkpoint["model"]
    current = model.state_dict()

    compatible = {
        key: value
        for key, value in original.items()
        if key in current and current[key].shape == value.shape
    }

    model.load_state_dict(compatible, strict=False)
    model.train()

    print("Transferred tensors:", len(compatible), flush=True)
    print("Total model tensors:", len(current), flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.0001,
    )

    samples = load_samples()

    if not samples:
        raise RuntimeError("Training dataset is empty")

    print("Training images:", len(samples), flush=True)

    image_size = (416, 416)

    for epoch in range(args.epochs):
        np.random.shuffle(samples)
        losses = []

        for index, (path, annotations) in enumerate(samples):
            images, targets = prepare_sample(
                path,
                annotations,
                image_size,
            )

            optimizer.zero_grad(set_to_none=True)

            outputs = model(images, targets)

            if isinstance(outputs, dict):
                loss = outputs["total_loss"]
            elif isinstance(outputs, (tuple, list)):
                loss = outputs[0]
            else:
                loss = outputs

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss: {path}")

            loss.backward()
            optimizer.step()

            value = float(loss.detach())
            losses.append(value)

            print(
                f"Epoch {epoch + 1}/{args.epochs} "
                f"Step {index + 1}/{len(samples)} "
                f"Loss {value:.4f}",
                flush=True,
            )

        average = sum(losses) / len(losses)

        torch.save(
            {
                "model": model.state_dict(),
                "num_classes": 5,
                "classes": [
                    "cash",
                    "document",
                    "phone",
                    "personal_item",
                    "collection_bag",
                ],
                "epoch": epoch + 1,
                "pilot_only": True,
            },
            OUTPUT / "yolox_bank_roi.pth",
        )

        print(
            f"Epoch {epoch + 1} completed. "
            f"Average loss: {average:.4f}",
            flush=True,
        )

    print("Training completed")
    print("Saved training/models/yolox_bank_pilot.pth")


if __name__ == "__main__":
    main()
