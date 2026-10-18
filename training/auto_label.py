from pathlib import Path
import cv2

from backend.yolo_detector import YOLODetector

source = Path("training/images")
output = Path("training/auto_preview")
output.mkdir(parents=True, exist_ok=True)

detector = YOLODetector()

allowed = {"person", "cell phone", "handbag"}

count = 0

for path in sorted(source.glob("*.jpg")):
    image = cv2.imread(str(path))

    if image is None:
        continue

    detections = detector.detect(image)

    for obj in detections:
        if obj["class"] not in allowed:
            continue

        if obj["confidence"] < 0.5:
            continue

        x1, y1, x2, y2 = obj["bbox"]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            image,
            obj["class"],
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    cv2.imwrite(str(output / path.name), image)
    count += 1

print("Processed images:", count)
print("Output:", output)
