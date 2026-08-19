"""Evaluate a trained probe detector on the val split.

Implements the three-part protocol agreed for this project:
  1. Localization quality on positive (probe-present) val images:
     mean IoU, Precision/Recall/F1 @ IoU=0.5 at the operating confidence
     threshold, and AP@0.5 (threshold-independent, VOC-style continuous
     interpolation).
  2. No-probe behavior on synthetic-negative val images: false-positive
     rate vs. confidence threshold, and a suggested operating threshold
     for a target FP-rate.
  3. Runtime on CPU (ms/image, FPS), matching inference.py's actual
     batch=1 usage pattern.

Deliberately hand-rolled rather than using Ultralytics' model.val(), since
the core need here -- AP/IoU on positives vs. false-positive rate on true
negatives -- isn't how stock validation is structured (it has no notion of
a GT-free "negative image" used purely to calibrate false positives).

Usage:
    python evaluate.py --weights weights/best.pt
"""

import argparse
import json
import platform
import time
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

from probe_common.bbox import iou_xyxy, xywh_to_xyxy


def _get_raw_boxes(model: YOLO, image_bgr: np.ndarray, imgsz: int) -> list[tuple[float, np.ndarray]]:
    """Return every raw detection [(conf, box_xyxy), ...] at a very low
    confidence floor, so downstream code can apply its own thresholds."""
    results = model.predict(source=image_bgr, conf=0.001, imgsz=imgsz, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    return list(zip(confs.tolist(), xyxy))


def _top_box(dets: list[tuple[float, np.ndarray]]) -> tuple[float | None, np.ndarray | None]:
    if not dets:
        return None, None
    conf, box = max(dets, key=lambda d: d[0])
    return conf, box


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """VOC2012-style continuous (all-points) AP integration."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_localization(
    model: YOLO,
    positive_records: list[dict],
    images_dir: Path,
    imgsz: int,
    conf_threshold: float,
    iou_thresh: float,
    extra_threshold: float | None = None,
) -> tuple[dict, list[tuple[float, float]]]:
    n_positive = len(positive_records)
    ious_top: list[float] = []
    tp_at_conf = fp_at_conf = 0
    tp_at_extra = 0
    pos_conf_iou: list[tuple[float, float]] = []  # (top_conf, iou) per image, for ROC sweeps

    all_dets: list[tuple[str, float, np.ndarray]] = []  # (file_name, conf, box)
    gt_by_file: dict[str, np.ndarray] = {}

    for rec in positive_records:
        image = cv2.imread(str(images_dir / rec["file_name"]))
        gt_xyxy = np.array(xywh_to_xyxy(tuple(rec["bbox"])))
        gt_by_file[rec["file_name"]] = gt_xyxy

        dets = _get_raw_boxes(model, image, imgsz)
        for conf, box in dets:
            all_dets.append((rec["file_name"], conf, box))

        top_conf, top_box = _top_box(dets)
        if top_box is None:
            ious_top.append(0.0)
            pos_conf_iou.append((0.0, 0.0))
            continue

        iou = iou_xyxy(top_box, gt_xyxy)
        ious_top.append(iou)
        pos_conf_iou.append((top_conf, iou))

        fired = top_conf >= conf_threshold
        if fired and iou >= iou_thresh:
            tp_at_conf += 1
        elif fired:
            fp_at_conf += 1

        if extra_threshold is not None and top_conf >= extra_threshold and iou >= iou_thresh:
            tp_at_extra += 1

    precision = tp_at_conf / (tp_at_conf + fp_at_conf) if (tp_at_conf + fp_at_conf) else 0.0
    recall = tp_at_conf / n_positive if n_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    # AP@0.5, pooling every raw detection across all positive val images.
    all_dets.sort(key=lambda d: d[1], reverse=True)
    matched_gt: set[str] = set()
    tps, fps = [], []
    for file_name, _conf, box in all_dets:
        gt_xyxy = gt_by_file[file_name]
        if file_name in matched_gt:
            tps.append(0)
            fps.append(1)
            continue
        if iou_xyxy(box, gt_xyxy) >= iou_thresh:
            tps.append(1)
            fps.append(0)
            matched_gt.add(file_name)
        else:
            tps.append(0)
            fps.append(1)

    ap = 0.0
    if all_dets:
        cum_tp = np.cumsum(tps)
        cum_fp = np.cumsum(fps)
        recalls = cum_tp / n_positive
        precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1)
        ap = compute_ap(recalls, precisions)

    result = {
        "n_positive_images": n_positive,
        "mean_iou_top_detection": float(np.mean(ious_top)) if ious_top else 0.0,
        "operating_conf_threshold": conf_threshold,
        "iou_match_threshold": iou_thresh,
        "precision_at_conf": precision,
        "recall_at_conf": recall,
        "f1_at_conf": f1,
        "ap_at_iou_0.5": ap,
    }
    if extra_threshold is not None:
        # Recall at the threshold calibrated against negatives (see
        # evaluate_negatives): unlike the FP-rate itself, which the
        # calibration procedure pins close to fp_target for every model by
        # construction, this recall genuinely differs across models -- it's
        # the real cost (in missed probes) of reaching that shared FP-rate
        # target, and is what actually differentiates models on the
        # no-probe axis.
        result["fp_calibrated_threshold"] = extra_threshold
        result["recall_at_fp_calibrated_threshold"] = tp_at_extra / n_positive if n_positive else 0.0
    return result, pos_conf_iou


def _trapezoid_area(y: np.ndarray, x: np.ndarray) -> float:
    """Trapezoidal-rule integral of y over x. Hand-rolled instead of using
    numpy's built-in: it was renamed between numpy versions (trapz ->
    trapezoid), and different environments (this machine vs. Colab) turned
    out to have versions on opposite sides of that rename -- two lines here
    avoids depending on either name."""
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2))


def compute_roc(
    thresholds: np.ndarray,
    fp_rates: np.ndarray,
    pos_conf_iou: list[tuple[float, float]],
    iou_thresh: float,
) -> tuple[np.ndarray, float]:
    """True-positive rate (recall, requiring IoU>=iou_thresh) at each of the
    same thresholds used for the negative FP-rate sweep, so both share an
    x-axis -- a standard ROC curve (FPR vs TPR). AUC via the trapezoidal
    rule, integrating TPR over FPR sorted ascending.
    """
    if not pos_conf_iou:
        return np.zeros_like(thresholds), 0.0
    confs = np.array([c for c, _ in pos_conf_iou])
    matched = np.array([iou >= iou_thresh for _, iou in pos_conf_iou])
    tpr = np.array([((confs >= t) & matched).mean() for t in thresholds])
    order = np.argsort(fp_rates)
    auc = _trapezoid_area(tpr[order], fp_rates[order])
    return tpr, auc


def evaluate_negatives(
    model: YOLO,
    negative_records: list[dict],
    images_dir: Path,
    imgsz: int,
    fp_target: float,
    n_thresholds: int = 101,
) -> tuple[dict, np.ndarray, np.ndarray]:
    max_confs = []
    for rec in negative_records:
        image = cv2.imread(str(images_dir / rec["file_name"]))
        dets = _get_raw_boxes(model, image, imgsz)
        max_confs.append(max((c for c, _ in dets), default=0.0))
    max_confs = np.array(max_confs)

    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    fp_rates = np.array([(max_confs >= t).mean() if len(max_confs) else 0.0 for t in thresholds])

    below_target = np.where(fp_rates <= fp_target)[0]
    if len(below_target) > 0:
        suggested_idx = below_target[0]
        suggested_threshold = float(thresholds[suggested_idx])
        achieved_fp_rate = float(fp_rates[suggested_idx])
        target_reached = True
    else:
        suggested_threshold = 1.0
        achieved_fp_rate = float(fp_rates[-1])
        target_reached = False

    summary = {
        "n_negative_images": len(negative_records),
        "fp_target": fp_target,
        "target_reached": target_reached,
        "suggested_threshold": suggested_threshold,
        "achieved_fp_rate_at_suggested_threshold": achieved_fp_rate,
    }
    return summary, thresholds, fp_rates


def evaluate_runtime(
    model: YOLO, records: list[dict], images_dir: Path, imgsz: int, n_warmup: int = 5, n_timed: int = 50
) -> dict:
    sample = records[: n_warmup + n_timed]
    images = [cv2.imread(str(images_dir / r["file_name"])) for r in sample]

    for image in images[:n_warmup]:
        model.predict(source=image, conf=0.25, imgsz=imgsz, verbose=False, device="cpu")

    timings_ms = []
    for image in images[n_warmup:]:
        start = time.perf_counter()
        model.predict(source=image, conf=0.25, imgsz=imgsz, verbose=False, device="cpu")
        timings_ms.append((time.perf_counter() - start) * 1000)

    timings_ms.sort()
    n = len(timings_ms)
    return {
        "device": "cpu",
        "cpu_model": platform.processor() or platform.machine(),
        "n_images_timed": n,
        "mean_ms": float(np.mean(timings_ms)) if n else None,
        "median_ms": float(np.median(timings_ms)) if n else None,
        "p90_ms": float(timings_ms[int(0.9 * (n - 1))]) if n else None,
        "fps": float(1000 / np.mean(timings_ms)) if n else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="weights/best.pt",
                         help="Path/name of the model to evaluate. Can be a fine-tuned checkpoint "
                              "(e.g. weights/best.pt) or a stock pretrained one (e.g. yolov8n.pt) to "
                              "get an out-of-the-box baseline -- detections aren't filtered by class, "
                              "so any COCO-class box overlapping the true probe location still counts "
                              "for the localization metrics below, even though the model has no "
                              "'probe' class of its own.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val",
                         help="Use 'val' during development (model/threshold selection); "
                              "use 'test' only once, at the end, for the final reported numbers.")
    parser.add_argument("--manifest", type=Path, default=Path("yolo_dataset/splits_manifest.json"))
    parser.add_argument("--images-dir", type=Path, default=None,
                         help="Defaults to yolo_dataset/images/<split>")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--fp-target", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", type=Path, default=Path("report/eval_results.json"))
    args = parser.parse_args()

    images_dir = args.images_dir or Path(f"yolo_dataset/images/{args.split}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    split_records = [r for r in manifest if r["split"] == args.split]
    positive_records = [r for r in split_records if r["origin"] == "original"]
    negative_records = [r for r in split_records if r["origin"] == "synthetic_negative"]

    model = YOLO(args.weights)

    print(f"Evaluating on {args.split}: {len(positive_records)} positive + {len(negative_records)} negative images...")

    negatives_summary, thresholds, fp_rates = evaluate_negatives(
        model, negative_records, images_dir, args.imgsz, args.fp_target
    )
    localization, pos_conf_iou = evaluate_localization(
        model, positive_records, images_dir, args.imgsz, args.conf, args.iou_thresh,
        extra_threshold=negatives_summary["suggested_threshold"],
    )
    runtime = evaluate_runtime(model, split_records, images_dir, args.imgsz)
    tpr_rates, roc_auc = compute_roc(thresholds, fp_rates, pos_conf_iou, args.iou_thresh)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "weights": str(args.weights),
        "split": args.split,
        "localization": localization,
        "no_probe_behavior": negatives_summary,
        "runtime": runtime,
        "roc_auc": roc_auc,
    }
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    fig_path = args.out.parent / "figures" / f"fp_rate_vs_threshold_{args.out.stem}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, fp_rates, color="steelblue")
    plt.axhline(args.fp_target, color="indianred", linestyle="--", label=f"target FP rate = {args.fp_target}")
    plt.xlabel("Confidence threshold")
    plt.ylabel(f"False positive rate ({args.split} negative images)")
    plt.title("No-probe false-positive rate vs. confidence threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=110)

    roc_fig_path = args.out.parent / "figures" / f"roc_curve_{args.out.stem}.png"
    plt.figure(figsize=(6, 6))
    plt.plot(fp_rates, tpr_rates, color="steelblue", label=f"ROC (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="chance")
    plt.scatter(
        [negatives_summary["achieved_fp_rate_at_suggested_threshold"]],
        [localization.get("recall_at_fp_calibrated_threshold", 0.0)],
        color="indianred", zorder=5,
        label=f"operating point (target FP={args.fp_target})",
    )
    plt.xlabel("False positive rate (negative images)")
    plt.ylabel(f"True positive rate / recall (positives, IoU>={args.iou_thresh})")
    plt.title("ROC curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_fig_path, dpi=110)

    print(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")
    print(f"Wrote {fig_path}")
    print(f"Wrote {roc_fig_path}")


if __name__ == "__main__":
    main()
