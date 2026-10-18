import argparse
from pathlib import Path

import cv2


def extract(video_path, output_dir, interval=1.0):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise RuntimeError("Invalid FPS")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    next_time = 0.0
    saved = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            timestamp = frame_index / fps
            frame_index += 1

            if timestamp < next_time:
                continue

            name = f"{Path(video_path).stem}_{int(timestamp * 1000):07d}.jpg"
            cv2.imwrite(str(output_dir / name), frame)

            saved += 1
            next_time += interval

    finally:
        cap.release()

    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("videos", nargs="+")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--output", default="training/images")

    args = parser.parse_args()

    if args.interval <= 0:
        parser.error("--interval must be positive")

    for video in args.videos:
        count = extract(video, args.output, args.interval)
        print(f"{video}: {count} frames")
