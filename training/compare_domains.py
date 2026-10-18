import json
from pathlib import Path

import cv2
import torch

from yolox.exp import get_exp
from yolox.data.data_augment import preproc
from yolox.utils import postprocess

torch.set_num_threads(4)

root = Path("training/merged_dataset")
out = Path("training/domain_test")
out.mkdir(parents=True, exist_ok=True)

exp = get_exp(exp_name="yolox_nano")
exp.num_classes = 5
model = exp.get_model()

checkpoint = torch.load(
    "training/models/yolox_bank_v2.pth",
    map_location="cpu",
    weights_only=False
)

model.load_state_dict(checkpoint["model"])
model.eval()

for split, limit in [("train", 10), ("val", 4)]:
    data = json.loads(
        (root / "annotations" / f"instances_{split}.json").read_text()
    )

    images = data["images"]
    if split == "train":
        images = [
            item for item in images
            if "local_" not in item["file_name"]
        ][:limit]
    else:
        images = images[:limit]

    for item in images:
        path = root / split / item["file_name"]
        frame = cv2.imread(str(path))
        if frame is None:
            continue

        processed, ratio = preproc(frame, exp.test_size)
        tensor = torch.from_numpy(processed).unsqueeze(0).float()

        with torch.no_grad():
            prediction = model(tensor)

        result = postprocess(
            prediction.detach().cpu().clone(),
            5,
            conf_thre=0.05,
            nms_thre=0.45,
            class_agnostic=True
        )[0]

        detected = 0

        if result is not None:
            for row in result:
                x1, y1, x2, y2 = (row[:4] / ratio).tolist()
                score = float(row[4] * row[5])
                cls = int(row[6])

                if cls != 0:
                    continue

                detected += 1
                a = (round(x1), round(y1))
                b = (round(x2), round(y2))
                cv2.rectangle(frame, a, b, (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"cash {score:.2f}",
                    (a[0], max(18, a[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        destination = out / f"{split}_{path.name}"
        cv2.imwrite(str(destination), frame)
        print(split, path.name, "cash detections:", detected)

print("Saved:", out)
