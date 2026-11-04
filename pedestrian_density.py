
import argparse
import csv
from pathlib import Path
from collections import defaultdict, deque

import cv2


ZONES = [
    "top_left", "top_center", "top_right",
    "middle_left", "middle_center", "middle_right",
    "bottom_left", "bottom_center", "bottom_right"
]


def get_zone(x, y, width, height):
    col = min(2, max(0, int(x / (width / 3))))
    row = min(2, max(0, int(y / (height / 3))))
    return row * 3 + col


def main():
    parser = argparse.ArgumentParser(
        description="Pedestrian density analysis"
    )

    parser.add_argument("--source", required=True,
                        help="Original video")
    parser.add_argument("--tracks-csv", required=True,
                        help="CSV from combined module")
    parser.add_argument("--output-dir", default="runs/density")
    parser.add_argument("--window", type=float, default=5.0,
                        help="Moving average window in seconds")

    args = parser.parse_args()

    video_path = Path(args.source)
    tracks_path = Path(args.tracks_csv)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not tracks_path.exists():
        raise FileNotFoundError(f"CSV not found: {tracks_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read video properties
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    # Read tracking CSV
    frame_data = defaultdict(list)

    with open(
        tracks_path,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file, delimiter=";")

        required = {
            "frame", "track_id", "class",
            "foot_x", "foot_y"
        }

        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                "CSV does not contain required columns: "
                + ", ".join(required)
            )

        for row in reader:
            if row["class"].strip().lower() != "person":
                continue

            frame = int(row["frame"])

            frame_data[frame].append({
                "track_id": int(row["track_id"]),
                "x": float(row["foot_x"]),
                "y": float(row["foot_y"])
            })

    output_path = output_dir / "pedestrian_density.csv"

    # Rolling average of pedestrian count
    window_frames = max(1, int(args.window * fps))
    history = deque()

    with open(
        output_path,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.writer(file, delimiter=";")

        writer.writerow([
            "frame",
            "time_sec",
            "pedestrian_count",
            "unique_track_ids",
            "moving_average",
            "most_crowded_zone",
            *ZONES
        ])

        for frame in range(1, total_frames + 1):

            pedestrians = frame_data.get(frame, [])

            zone_counts = [0] * 9
            track_ids = set()

            for person in pedestrians:

                zone = get_zone(
                    person["x"],
                    person["y"],
                    # Coordinates are clamped to the frame
                    # dimensions below using the video size.
                    width,
                    height
                )

                zone_counts[zone] += 1

                if person["track_id"] >= 0:
                    track_ids.add(person["track_id"])

            count = len(pedestrians)
            time_sec = frame / fps

            history.append(count)

            if len(history) > window_frames:
                history.popleft()

            average = sum(history) / len(history)

            if count == 0:
                busiest_zone = "none"
            else:
                busiest_zone = ZONES[
                    zone_counts.index(max(zone_counts))
                ]

            writer.writerow([
                frame,
                round(time_sec, 3),
                count,
                len(track_ids),
                round(average, 2),
                busiest_zone,
                *zone_counts
            ])

            if frame % 500 == 0:
                print(
                    f"Analyzed {frame}/{total_frames} frames"
                )

    print("\nAnalysis completed!")
    print("Output:", output_path.resolve())


if __name__ == "__main__":
    # Obtain frame dimensions before starting analysis
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", required=True)
    known, _ = parser.parse_known_args()

    cap = cv2.VideoCapture(known.source)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    main()
