"""Thin wrapper around Ultralytics YOLO training.

Shared entrypoint for local smoke-testing (CPU, few epochs, sanity checks)
and for the Colab notebook (training/train_colab.ipynb), which invokes the
same logic once per candidate model variant (yolov8n/s/m).

Note: the horizontal bbox-center imbalance documented in the EDA (mean
cx=0.27, left-skewed) means Ultralytics' default fliplr=0.5 augmentation is
worth watching -- it is left enabled here by default (it should help the
model generalize to right-side placements it has seen little of), but if
validation results show it hurting, pass --fliplr 0 to disable it as an
experiment.
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def train(
    data_yaml: Path,
    model: str = "yolov8n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    patience: int = 20,
    project: Path = Path("runs/train"),
    name: str = "probe",
    seed: int = 42,
    device: str = "cpu",
    fliplr: float = 0.5,
) -> Path:
    """Run training and return the path to the best checkpoint."""
    yolo = YOLO(model)
    results = yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=str(project),
        name=name,
        seed=seed,
        device=device,
        fliplr=fliplr,
    )
    return Path(results.save_dir) / "weights" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("yolo_dataset/data.yaml"))
    parser.add_argument("--model", default="yolov8n.pt", help="yolov8n.pt / yolov8s.pt / yolov8m.pt / ...")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--project", type=Path, default=Path("runs/train"))
    parser.add_argument("--name", default="probe")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="'cpu', '0' for first GPU, etc.")
    parser.add_argument("--fliplr", type=float, default=0.5)
    args = parser.parse_args()

    best_path = train(
        data_yaml=args.data,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.name,
        seed=args.seed,
        device=args.device,
        fliplr=args.fliplr,
    )
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
