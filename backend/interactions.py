import math


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def detect_interactions(frames, max_distance=130):
    events = []

    for frame in frames:
        timestamp = frame["timecode_sec"]
        objects = frame["objects"]

        persons = [
            obj for obj in objects
            if obj["class"] == "person"
            and obj["confidence"] >= 0.5
        ]

        items = [
            obj for obj in objects
            if obj["class"] in {
                "cell phone",
                "handbag",
                "backpack",
            }
            and obj["confidence"] >= 0.5
        ]

        for item in items:
            item_center = box_center(item["bbox"])

            for person in persons:
                person_center = box_center(person["bbox"])

                distance = math.dist(
                    item_center,
                    person_center
                )

                if distance <= max_distance:
                    events.append({
                        "timecode_sec": timestamp,
                        "event_type": "person_near_object",
                        "object_class": item["class"],
                        "distance_px": round(distance, 2),
                        "review_required": True
                    })

    return events
