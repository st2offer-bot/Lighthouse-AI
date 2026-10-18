from pathlib import Path
import cv2
import torch

from yolox.exp import get_exp
from yolox.data.data_augment import preproc
from yolox.utils import postprocess

torch.set_num_threads(4)

CLASSES = ["cash", "document", "phone", "personal_item", "collection_bag"]

MODELS = {
    "v2": "training/models/yolox_bank_v2.pth",
    "roi": "training/models/yolox_bank_roi.pth",
}

IMAGES = Path("training/roi_test")
OUTPUT = Path("training/model_comparison")
OUTPUT.mkdir(parents=True, exist_ok=True)

exp = get_exp(exp_name="yolox_nano")
exp.num_classes = len(CLASSES)

for model_name, checkpoint_path in MODELS.items():
    model = exp.get_model()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])
    model.eval()

    total = 0

    for path in sorted(IMAGES.glob("*.jpg")):
        frame = cv2.imread(str(path))
        if frame is None:
            continue

        height, width = frame.shape[:2]

        processed, ratio = preproc(frame, exp.test_size)
        tensor = torch.from_numpy(processed).unsqueeze(0).float()

        with torch.no_grad():
            predictions = model(tensor)

        results = postprocess(
            predictions.detach().cpu().clone(),
            len(CLASSES),
            conf_thre=0.05,
            nms_thre=0.45,
            class_agnostic=True,
        )[0]

        detections = 0

        if results is not None:
            for row in results:
                cls = int(row[6])
                confidence = float(row[4] * row[5])

                if cls != 0:
                    continue

                x1, y1, x2, y2 = (row[:4] / ratio).tolist()

                x1 = max(0, min(width - 1, round(x1)))
                y1 = max(0, min(height - 1, round(y1)))
                x2 = max(0, min(width - 1, round(x2)))
                y2 = max(0, min(height - 1, round(y2)))

                if x2 <= x1 or y2 <= y1:
                    continue

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    f"cash {confidence:.2f}",
                    (x1, max(20, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                detections += 1

        target = OUTPUT / f"{model_name}_{path.name}"
        cv2.imwrite(str(target), frame)

        total += detections
        print(model_name, path.name, "detections:", detections)

    print(model_name, "TOTAL:", total)
