"""
Проверка пользовательской модели сегментации best.pt.

Скрипт выводит задачу, архитектуру (если доступна), названия классов
и позволяет проверить модель на одном изображении или видео.

Примеры:
    python inspect_segmentation.py --weights best.pt
    python inspect_segmentation.py --weights best.pt --source test.jpg
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="best.pt")
    parser.add_argument("--source", default=None,
                        help="Необязательное изображение или видео для проверки")
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Файл весов не найден: {weights}")

    model = YOLO(str(weights))

    print("Задача модели:", getattr(model, "task", "не определена"))
    print("Классы модели:")
    names = model.names
    if isinstance(names, dict):
        for class_id, class_name in names.items():
            print(f"  {class_id}: {class_name}")
    else:
        for class_id, class_name in enumerate(names):
            print(f"  {class_id}: {class_name}")

    print("Количество классов:", len(names))

    if args.source:
        source = Path(args.source)
        if not source.exists():
            raise FileNotFoundError(f"Источник не найден: {source}")
        results = model.predict(
            source=str(source),
            conf=args.conf,
            save=True,
            project="runs/segmentation_check",
            name="prediction",
        )
        print("Проверка завершена. Результаты сохранены в runs/segmentation_check.")
        print("Количество обработанных результатов:", len(results))


if __name__ == "__main__":
    main()
