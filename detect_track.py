"""
Модуль обнаружения и отслеживания участников движения для проекта оценки
стиля вождения СИМ.

По умолчанию:
- модель: YOLOv8n (предобученные веса COCO; при первом запуске Ultralytics
  может скачать yolov8n.pt);
- трекер: ByteTrack;
- классы COCO: person, bicycle, car, motorcycle, bus, truck.

Запуск:
    python detect_track.py --source video.mp4
    python detect_track.py --source video.mp4 --model yolov8s.pt --conf 0.35

Результаты:
    runs/tracking/<имя_видео>_tracked.mp4
    runs/tracking/<имя_видео>_tracks.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import cv2
from ultralytics import YOLO


# Идентификаторы классов COCO:
# 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck.
COCO_CLASSES = [0, 1, 2, 3, 5, 7]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Детекция и трекинг объектов на видео с камеры СИМ"
    )
    parser.add_argument("--source", required=True, help="Путь к видеофайлу")
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Веса модели детекции (по умолчанию yolov8n.pt)",
    )
    parser.add_argument("--conf", type=float, default=0.30,
                        help="Порог уверенности детекции")
    parser.add_argument("--iou", type=float, default=0.50,
                        help="Порог IoU для NMS")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Размер изображения для модели")
    parser.add_argument("--tracker", default="bytetrack.yaml",
                        help="Конфигурация трекера Ultralytics")
    parser.add_argument("--device", default=None,
                        help="Например: cpu, 0. Если не задано — автоопределение")
    parser.add_argument("--output-dir", default="runs/tracking",
                        help="Папка для результатов")
    parser.add_argument(
        "--all-coco",
        action="store_true",
        help="Не фильтровать классы; показывать все классы COCO",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source)

    if not source.exists():
        raise FileNotFoundError(f"Видео не найдено: {source}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = source.stem
    video_out = output_dir / f"{stem}_tracked.mp4"
    csv_out = output_dir / f"{stem}_tracks.csv"

    model = YOLO(args.model)

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео: {source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Не удалось определить размер видео")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_out), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Не удалось создать выходное видео: {video_out}")

    classes = None if args.all_coco else COCO_CLASSES
    frame_index = 0
    total_detections = 0

    # CSV хранит координаты рамки и идентификатор трека на каждом кадре.
    with csv_out.open("w", newline="", encoding="utf-8-sig") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=";")
        csv_writer.writerow([
            "frame", "time_sec", "track_id", "class_id", "class_name",
            "confidence", "x1", "y1", "x2", "y2", "center_x", "center_y"
        ])

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame_index += 1
                time_sec = frame_index / fps

                track_kwargs = dict(
                    source=frame,
                    persist=True,
                    tracker=args.tracker,
                    conf=args.conf,
                    iou=args.iou,
                    imgsz=args.imgsz,
                    verbose=False,
                )
                if classes is not None:
                    track_kwargs["classes"] = classes
                if args.device is not None:
                    track_kwargs["device"] = args.device

                results = model.track(**track_kwargs)
                result = results[0]
                annotated = frame.copy()

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    class_ids = boxes.cls.cpu().numpy().astype(int)

                    if boxes.id is not None:
                        track_ids = boxes.id.cpu().numpy().astype(int)
                    else:
                        # Детекция есть, но трекер пока не присвоил ID.
                        track_ids = [-1] * len(xyxy)

                    for box, confidence, class_id, track_id in zip(
                        xyxy, confs, class_ids, track_ids
                    ):
                        x1, y1, x2, y2 = map(int, box)
                        center_x = (x1 + x2) / 2.0
                        center_y = (y1 + y2) / 2.0
                        class_name = model.names.get(
                            int(class_id), str(class_id)
                        )

                        label = f"{class_name} ID:{track_id} {confidence:.2f}"
                        cv2.rectangle(
                            annotated, (x1, y1), (x2, y2),
                            (40, 220, 40), 2
                        )
                        cv2.putText(
                            annotated, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (40, 220, 40), 2, cv2.LINE_AA
                        )

                        csv_writer.writerow([
                            frame_index, f"{time_sec:.3f}", int(track_id),
                            int(class_id), class_name,
                            f"{float(confidence):.4f}",
                            x1, y1, x2, y2,
                            f"{center_x:.2f}", f"{center_y:.2f}"
                        ])
                        total_detections += 1

                cv2.putText(
                    annotated, f"Frame: {frame_index}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA
                )
                writer.write(annotated)

                if frame_index % 100 == 0:
                    print(f"Обработано кадров: {frame_index}")

        finally:
            cap.release()
            writer.release()

    print("\nОбработка завершена.")
    print(f"Кадров обработано: {frame_index}")
    print(f"Строк с обнаружениями: {total_detections}")
    print(f"Видео: {video_out.resolve()}")
    print(f"CSV:   {csv_out.resolve()}")


if __name__ == "__main__":
    main()
