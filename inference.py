"""Run probe detection on a folder of images.

Usage:
    python inference.py <input_folder> [--weights weights/best.pt] [--conf 0.22]
                         [--out outputs/] [--show] [--no-json] [--recursive]

Iterates through each image in the input folder, detects the probe bounding
box if present, draws it on the image, and either saves the result to an
output folder (default behavior) or shows it in a window (--show). If no
detection reaches the confidence threshold, the image is explicitly
annotated "No probe detected" rather than silently skipped.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class DetectionResult:
    file_name: str
    probe_detected: bool
    confidence: float | None
    bbox_xyxy: list[float] | None


def load_model(weights_path: str) -> YOLO:
    return YOLO(str(weights_path))


def predict_top_box(
    model: YOLO, image_bgr: np.ndarray, imgsz: int = 640
) -> tuple[np.ndarray | None, float | None]:
    """Run the model and return the single highest-confidence box (xyxy, conf),
    or (None, None) if the model found no candidates at all."""
    results = model.predict(source=image_bgr, conf=0.001, imgsz=imgsz, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, None

    confs = boxes.conf.cpu().numpy()
    best_idx = int(confs.argmax())
    box_xyxy = boxes.xyxy.cpu().numpy()[best_idx]
    return box_xyxy, float(confs[best_idx])


def draw_result(
    image_bgr: np.ndarray,
    box_xyxy: np.ndarray | None,
    conf: float | None,
    conf_threshold: float,
) -> np.ndarray:
    annotated = image_bgr.copy()
    if box_xyxy is not None and conf is not None and conf >= conf_threshold:
        x1, y1, x2, y2 = box_xyxy.astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"probe {conf:.2f}"
        text_y = max(15, y1 - 8)
        cv2.putText(annotated, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(
            annotated, "No probe detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
        )
    return annotated


def run_on_folder(
    input_dir: Path,
    weights: str,
    conf: float = 0.22,
    out_dir: Path | None = None,
    show: bool = False,
    save_json: bool = True,
    recursive: bool = False,
    imgsz: int = 640,
) -> list[DetectionResult]:
    model = load_model(weights)

    pattern = "**/*" if recursive else "*"
    paths = sorted(
        p for p in Path(input_dir).glob(pattern) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        print(f"No images found in {input_dir}")
        return []

    if out_dir is None and not show:
        out_dir = Path("outputs")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    results: list[DetectionResult] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"WARNING: could not read {path}, skipping")
            continue

        box_xyxy, det_conf = predict_top_box(model, image, imgsz=imgsz)
        probe_detected = box_xyxy is not None and det_conf is not None and det_conf >= conf
        annotated = draw_result(image, box_xyxy, det_conf, conf)

        if out_dir is not None:
            cv2.imwrite(str(out_dir / path.name), annotated)
        if show:
            cv2.imshow("Probe detection", annotated)
            key = cv2.waitKey(0)
            if key == 27:  # Esc quits early
                break

        results.append(
            DetectionResult(
                file_name=path.name,
                probe_detected=probe_detected,
                confidence=det_conf if probe_detected else None,
                bbox_xyxy=box_xyxy.tolist() if probe_detected and box_xyxy is not None else None,
            )
        )
        print(f"{path.name}: {'probe detected' if probe_detected else 'no probe detected'}"
              + (f" (conf={det_conf:.2f})" if det_conf is not None else ""))

    if show:
        cv2.destroyAllWindows()

    if save_json and out_dir is not None:
        json_path = out_dir / "detections.json"
        json_path.write_text(json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8")
        print(f"Wrote {json_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_folder", type=Path, help="Folder containing images to process")
    parser.add_argument("--weights", default="weights/best.pt")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.22,
        help="Confidence threshold for a detection to count (default: 0.22, calibrated for <=5%% FP rate)",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output folder (default: outputs/ unless --show is used)")
    parser.add_argument("--show", action="store_true", help="Show each result in a window instead of / as well as saving")
    parser.add_argument("--no-json", action="store_true", help="Skip writing outputs/detections.json")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders of input_folder")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    run_on_folder(
        input_dir=args.input_folder,
        weights=args.weights,
        conf=args.conf,
        out_dir=args.out,
        show=args.show,
        save_json=not args.no_json,
        recursive=args.recursive,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
