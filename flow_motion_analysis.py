"""
Модуль анализа движения по оптическому потоку для проекта оценки стиля
вождения СИМ.

Назначение:
- оценка эго-движения камеры (СИМ) по оптическому потоку фоновых точек;
- компенсация эго-движения в траекториях объектов из combined-модуля;
- расчёт скорости объекта относительно сцены (пикс/с);
- оценочный TTC по скорости расширения рамки.

Вход:
    --source       исходное видео
    --tracks-csv   CSV из combine_segmentation_tracking.py

Выход:
    runs/flow/<имя_видео>_flow.csv

Запуск:
    python flow_motion_analysis.py --source video.mp4 \
        --tracks-csv runs/combined/video_combined.csv

Важно: TTC является оценочным (см. README_RU.md, раздел 8). Значения
скорости даны в пикселях в секунду, а не в м/с: один видеопоток с
движущейся камерой не даёт метрической скорости без калибровки сцены.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np


# Параметры Lucas-Kanade для фоновых точек.
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

# Сетка фоновых точек: шаг в пикселях.
GRID_STEP = 40

# Запас при маскировке рамок объектов (доля размера рамки).
BOX_DILATE = 0.15

# Окно сглаживания высоты рамки для оценочного TTC (кадров).
TTC_WINDOW = 7

# Минимальное число согласованных точек для оценки движения камеры.
MIN_INLIERS = 12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Оптический поток: компенсация эго-движения и оценочный TTC"
    )
    parser.add_argument("--source", required=True, help="Путь к видеофайлу")
    parser.add_argument("--tracks-csv", required=True,
                        help="CSV из combine_segmentation_tracking.py")
    parser.add_argument("--output-dir", default="runs/flow",
                        help="Папка для результатов")
    return parser.parse_args()


def load_tracks(csv_path: Path, width: int, height: int):
    """Читает combined-CSV: рамки и точки ног по кадрам и трекам."""
    per_frame = defaultdict(list)   # frame -> [x1, y1, x2, y2, ...]
    per_track = defaultdict(dict)   # track_id -> frame -> запись

    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        required = {"frame", "track_id", "x1", "y1", "x2", "y2",
                    "foot_x", "foot_y"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError(
                "CSV не содержит нужных колонок: " + ", ".join(sorted(required))
            )

        for row in reader:
            frame = int(row["frame"])
            track_id = int(row["track_id"])
            if track_id < 0:
                continue
            rec = {
                "x1": int(row["x1"]), "y1": int(row["y1"]),
                "x2": int(row["x2"]), "y2": int(row["y2"]),
                "foot_x": float(row["foot_x"]), "foot_y": float(row["foot_y"]),
            }
            per_frame[frame].append(rec)
            per_track[track_id][frame] = rec

    return per_frame, per_track


def background_grid(width: int, height: int, boxes, dilate: float):
    """Сетка точек по кадру за вычетом расширенных рамок объектов."""
    mask = np.ones((height, width), dtype=np.uint8)

    for b in boxes:
        w_box = b["x2"] - b["x1"]
        h_box = b["y2"] - b["y1"]
        dx = int(w_box * dilate)
        dy = int(h_box * dilate)
        x1 = max(0, b["x1"] - dx)
        y1 = max(0, b["y1"] - dy)
        x2 = min(width - 1, b["x2"] + dx)
        y2 = min(height - 1, b["y2"] + dy)
        mask[y1:y2 + 1, x1:x2 + 1] = 0

    ys, xs = np.mgrid[GRID_STEP // 2:height:GRID_STEP,
                      GRID_STEP // 2:width:GRID_STEP]
    keep = mask[ys, xs] > 0
    pts = np.stack([xs[keep], ys[keep]], axis=1).astype(np.float32)
    return pts.reshape(-1, 1, 2)


def estimate_camera_motion(prev_gray, gray, prev_boxes):
    """Аффинное движение камеры prev->current по фоновому потоку."""
    h, w = prev_gray.shape
    p0 = background_grid(w, h, prev_boxes, BOX_DILATE)

    if len(p0) < MIN_INLIERS:
        return None

    p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None,
                                             **LK_PARAMS)
    if p1 is None:
        return None

    good0 = p0[status.ravel() == 1]
    good1 = p1[status.ravel() == 1]
    if len(good0) < MIN_INLIERS:
        return None

    matrix, inliers = cv2.estimateAffinePartial2D(
        good0, good1, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if matrix is None or inliers is None:
        return None
    if int(inliers.sum()) < MIN_INLIERS:
        return None
    return matrix


def main():
    args = parse_args()
    source = Path(args.source)
    tracks_path = Path(args.tracks_csv)

    if not source.exists():
        raise FileNotFoundError(f"Видео не найдено: {source}")
    if not tracks_path.exists():
        raise FileNotFoundError(f"CSV не найден: {tracks_path}")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    per_frame, per_track = load_tracks(tracks_path, width, height)

    # Первый проход: движение камеры на каждом кадре.
    camera_motion = {}  # frame -> матрица 2x3 (prev -> current)
    ok, prev = cap.read()
    if not ok:
        raise RuntimeError("Видео пустое")
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

    frame_number = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_number += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        matrix = estimate_camera_motion(
            prev_gray, gray, per_frame.get(frame_number - 1, [])
        )
        camera_motion[frame_number] = matrix
        prev_gray = gray

        if frame_number % 100 == 0:
            print(f"Поток посчитан для кадра {frame_number}")

    cap.release()

    # Второй проход: компенсированные скорости и оценочный TTC.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{source.stem}_flow.csv"

    height_hist = defaultdict(lambda: deque(maxlen=TTC_WINDOW))
    rows = 0

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "frame", "time_sec", "track_id",
            "vx_px_s", "vy_px_s", "speed_px_s",
            "box_h", "ttc_sec_est",
        ])

        for track_id, frames in per_track.items():
            prev_frame = None
            for frame in sorted(frames):
                rec = frames[frame]
                box_h = rec["y2"] - rec["y1"]
                height_hist[track_id].append(box_h)

                if prev_frame is None or frame != prev_frame + 1:
                    prev_frame = frame
                    continue

                prev_rec = frames[prev_frame]
                matrix = camera_motion.get(frame)

                # Ожидаемая позиция ноги при чистом движении камеры.
                if matrix is not None:
                    pt = np.array([prev_rec["foot_x"], prev_rec["foot_y"], 1.0])
                    exp_x, exp_y = matrix @ pt
                else:
                    # Оценка камеры не удалась: fallback без компенсации.
                    exp_x, exp_y = prev_rec["foot_x"], prev_rec["foot_y"]

                vx = (rec["foot_x"] - exp_x) * fps
                vy = (rec["foot_y"] - exp_y) * fps
                speed = float(np.hypot(vx, vy))

                # Оценочный TTC: h / dh/dt по сглаженной высоте рамки.
                ttc_est = ""
                hist = height_hist[track_id]
                if len(hist) == TTC_WINDOW:
                    h_smooth = sum(hist) / len(hist)
                    dh = (hist[-1] - hist[0]) / (len(hist) - 1)
                    if dh > 0.5:  # рамка растёт — объект приближается
                        ttc_frames = h_smooth / dh
                        ttc_est = f"{ttc_frames / fps:.2f}"

                writer.writerow([
                    frame, f"{frame / fps:.3f}", track_id,
                    f"{vx:.2f}", f"{vy:.2f}", f"{speed:.2f}",
                    box_h, ttc_est,
                ])
                rows += 1
                prev_frame = frame

    print("\nАнализ завершён.")
    print(f"Строк записано: {rows}")
    print(f"CSV: {out_path.resolve()}")
    print("Примечание: скорости в пикс/с; TTC является оценочным.")


if __name__ == "__main__":
    main()
