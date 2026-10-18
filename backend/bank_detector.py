from pathlib import Path

import cv2
import torch

from yolox.exp import get_exp
from yolox.data.data_augment import preproc
from yolox.utils import postprocess


CLASSES = ["cash", "document", "phone", "personal_item", "collection_bag"]
WEIGHTS = Path("training/models/yolox_bank_v3.pth")


class BankDetector:
    def __init__(self, confidence=0.25):
        torch.set_num_threads(4)

        self.confidence = confidence
        self.exp = get_exp(exp_name="yolox_nano")
        self.exp.num_classes = len(CLASSES)
        self.model = self.exp.get_model()

        checkpoint = torch.load(
            WEIGHTS,
            map_location="cpu",
            weights_only=False,
        )

        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    def detect(self, frame):
        height, width = frame.shape[:2]

        top = int(height * 0.28)
        bottom = int(height * 0.80)

        crop = frame[top:bottom, :]

        if crop.size == 0:
            return []

        processed, ratio = preproc(crop, self.exp.test_size)
        tensor = torch.from_numpy(processed).unsqueeze(0).float()

        with torch.no_grad():
            prediction = self.model(tensor)

        output = postprocess(
            prediction.detach().cpu().clone(),
            len(CLASSES),
            conf_thre=self.confidence,
            nms_thre=0.45,
            class_agnostic=True,
        )[0]

        objects = []

        if output is None:
            return objects

        for row in output:
            class_id = int(row[6])
            confidence = float(row[4] * row[5])

            x1, y1, x2, y2 = (row[:4] / ratio).tolist()

            x1 = max(0, min(width - 1, round(x1)))
            y1 = max(0, min(height - 1, round(y1 + top)))
            x2 = max(0, min(width - 1, round(x2)))
            y2 = max(0, min(height - 1, round(y2 + top)))

            if x2 <= x1 or y2 <= y1:
                continue

            objects.append({
                "class": CLASSES[class_id],
                "confidence": round(confidence, 3),
                "bbox": [x1, y1, x2, y2],
            })

        return objects
