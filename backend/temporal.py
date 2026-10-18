import math


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def distance(a, b):
    return math.dist(a, b)


def analyze_movement(frames, threshold=40, max_match_distance=90):
    events = []
    tracks = {}
    next_id = 1

    for frame in frames:
        timestamp = frame["timecode_sec"]
        objects = frame["objects"]
        used_tracks = set()
        new_tracks = {}

        for obj in objects:
            position = center(obj["bbox"])
            label = obj["class"]

            candidates = []

            for track_id, track in tracks.items():
                if track_id in used_tracks:
                    continue

                if track["class"] != label:
                    continue

                d = distance(position, track["center"])

                if d <= max_match_distance:
                    candidates.append((d, track_id))

            if candidates:
                candidates.sort()
                movement, track_id = candidates[0]
                used_tracks.add(track_id)

                if movement >= threshold:
                    events.append({
                        "timecode_sec": timestamp,
                        "event_type": "object_movement",
                        "track_id": track_id,
                        "object_class": label,
                        "distance_px": round(movement, 2),
                        "review_required": True,
                    })

            else:
                track_id = next_id
                next_id += 1

            new_tracks[track_id] = {
                "class": label,
                "center": position,
            }

        tracks = new_tracks

    return events
