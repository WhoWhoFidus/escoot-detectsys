
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROAD_USERS = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--seg-weights", default="best.pt")
    parser.add_argument("--det-model", default="yolov8n.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Видео не найдено: {source}")

    if not Path(args.seg_weights).exists():
        raise FileNotFoundError(
            f"Модель сегментации не найдена: {args.seg_weights}"
        )

    output_dir = Path("runs/combined")
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / f"{source.stem}_combined.mp4"
    csv_path = output_dir / f"{source.stem}_combined.csv"

    print("Загрузка модели сегментации...")
    seg_model = YOLO(args.seg_weights)

    print("Классы сегментации:", seg_model.names)

    if "bike_lane" not in seg_model.names.values():
        print("ВНИМАНИЕ: класс bike_lane не найден!")

    print("Загрузка модели детекции...")
    det_model = YOLO(args.det_model)

    cap = cv2.VideoCapture(str(source))

    if not cap.isOpened():
        raise RuntimeError("Не удалось открыть видео")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    if not writer.isOpened():
        raise RuntimeError("Не удалось создать выходное видео")

    csv_file = open(
        csv_path, "w", newline="", encoding="utf-8-sig"
    )
    csv_writer = csv.writer(csv_file, delimiter=";")

    csv_writer.writerow([
        "frame", "time_sec", "track_id", "class",
        "confidence", "x1", "y1", "x2", "y2",
        "foot_x", "foot_y", "foot_in_lane",
        "lane_ratio"
    ])

    frame_number = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_number += 1
            time_sec = frame_number / fps

            # 1. СЕГМЕНТАЦИЯ ВЕЛОДОРОЖКИ

            seg_result = seg_model.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False
            )[0]

            lane_mask = np.zeros(
                (height, width), dtype=np.uint8
            )

            if (
                seg_result.masks is not None
                and seg_result.boxes is not None
            ):
                masks = seg_result.masks.data.cpu().numpy()
                classes = (
                    seg_result.boxes.cls.cpu().numpy().astype(int)
                )

                for mask, class_id in zip(masks, classes):
                    class_name = seg_model.names[int(class_id)]

                    if class_name != "bike_lane":
                        continue

                    resized_mask = cv2.resize(
                        mask.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST
                    )

                    lane_mask = np.maximum(
                        lane_mask, resized_mask
                    )

            # 2. ДЕТЕКЦИЯ И ТРЕКИНГ

            track_result = det_model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False
            )[0]

            output = frame.copy()

            # 3. ОТРИСОВКА МАСКИ

            if np.any(lane_mask):
                green = np.zeros_like(output)
                green[:, :, 1] = 255

                mask_bool = lane_mask > 0

                output[mask_bool] = cv2.addWeighted(
                    output[mask_bool],
                    0.65,
                    green[mask_bool],
                    0.35,
                    0
                )

            # 4. АНАЛИЗ ОБЪЕКТОВ

            if (
                track_result.boxes is not None
                and len(track_result.boxes) > 0
            ):
                boxes = track_result.boxes.xyxy.cpu().numpy()
                classes = (
                    track_result.boxes.cls.cpu().numpy().astype(int)
                )
                confidences = (
                    track_result.boxes.conf.cpu().numpy()
                )

                ids = None

                if track_result.boxes.id is not None:
                    ids = (
                        track_result.boxes.id.cpu()
                        .numpy().astype(int)
                    )

                for i, (box, class_id, confidence) in enumerate(
                    zip(boxes, classes, confidences)
                ):
                    if class_id not in ROAD_USERS:
                        continue

                    x1, y1, x2, y2 = map(int, box)

                    x1 = max(0, min(width - 1, x1))
                    x2 = max(0, min(width - 1, x2))
                    y1 = max(0, min(height - 1, y1))
                    y2 = max(0, min(height - 1, y2))

                    if x2 <= x1 or y2 <= y1:
                        continue

                    track_id = (
                        int(ids[i]) if ids is not None else -1
                    )

                    class_name = ROAD_USERS[class_id]

                    # Нижняя центральная точка рамки
                    foot_x = (x1 + x2) // 2
                    foot_y = y2

                    foot_x = max(0, min(width - 1, foot_x))
                    foot_y = max(0, min(height - 1, foot_y))

                    foot_in_lane = bool(
                        lane_mask[foot_y, foot_x] > 0
                    )

                    # Доля маски в нижних 35% рамки
                    lower_y = y1 + int((y2 - y1) * 0.65)

                    lower_crop = lane_mask[
                        lower_y:y2 + 1,
                        x1:x2 + 1
                    ]

                    lane_ratio = (
                        np.count_nonzero(lower_crop)
                        / lower_crop.size
                        if lower_crop.size else 0
                    )

                    if foot_in_lane:
                        status = "LANE"
                        color = (0, 255, 0)
                    else:
                        status = "OUT"
                        color = (0, 165, 255)

                    # Рамка и подпись
                    cv2.rectangle(
                        output,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    label = (
                        f"{class_name} ID:{track_id} {status}"
                    )

                    cv2.putText(
                        output,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2
                    )

                    cv2.circle(
                        output,
                        (foot_x, foot_y),
                        5,
                        color,
                        -1
                    )

                    csv_writer.writerow([
                        frame_number,
                        round(time_sec, 3),
                        track_id,
                        class_name,
                        round(float(confidence), 4),
                        x1, y1, x2, y2,
                        foot_x, foot_y,
                        int(foot_in_lane),
                        round(float(lane_ratio), 4)
                    ])

            cv2.putText(
                output,
                f"Frame: {frame_number}/{total}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            writer.write(output)

            if frame_number % 30 == 0:
                print(
                    f"Обработано кадров: {frame_number}/{total}"
                )

    finally:
        cap.release()
        writer.release()
        csv_file.close()

    print("\nОбработка завершена!")
    print("Видео:", video_path.resolve())
    print("CSV:", csv_path.resolve())


if __name__ == "__main__":
    main()
