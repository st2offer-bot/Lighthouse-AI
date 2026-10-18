import csv
from pathlib import Path

root = Path("training")
classes = (root / "classes.txt").read_text().splitlines()

valid_labels = 0
invalid_labels = 0
objects = {name: 0 for name in classes}

for label in sorted((root / "labels").glob("X-*.txt")):
    image = root / "images" / f"{label.stem}.jpg"

    if not image.exists():
        print("MISSING IMAGE:", label.name)
        invalid_labels += 1
        continue

    for line in label.read_text().splitlines():
        try:
            values = line.split()
            if len(values) != 5:
                raise ValueError("Invalid format")

            class_id = int(values[0])
            x, y, w, h = map(float, values[1:])

            if not 0 <= class_id < len(classes):
                raise ValueError("Invalid class")

            if not (0 < w <= 1 and 0 < h <= 1):
                raise ValueError("Invalid dimensions")

            if not (0 <= x-w/2 and x+w/2 <= 1):
                raise ValueError("Invalid X bounds")

            if not (0 <= y-h/2 and y+h/2 <= 1):
                raise ValueError("Invalid Y bounds")

            valid_labels += 1
            objects[classes[class_id]] += 1

        except ValueError as error:
            print("INVALID:", label.name, error)
            invalid_labels += 1

with (root / "events.csv").open(newline="") as f:
    events = list(csv.DictReader(f))

print("Valid objects:", valid_labels)
print("Invalid objects:", invalid_labels)
print("Classes:", objects)
print("Events:", len(events))
print("Verified events:", sum(
    row["verified"].lower() == "true" for row in events
))
