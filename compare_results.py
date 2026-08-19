"""Compare multiple evaluate.py result files with bar charts.

Typical use: compare the out-of-the-box pretrained baseline against the
fine-tuned model, and/or compare candidate YOLOv8 variants (n/s/m) against
each other, on the same val split.

Usage:
    python compare_results.py \
        --result baseline=report/eval_results_baseline.json \
        --result yolov8n=report/eval_results_yolov8n.json \
        --result yolov8s=report/eval_results_yolov8s.json \
        --result yolov8m=report/eval_results_yolov8m.json \
        --out report/figures/model_comparison.png
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_result_arg(value: str) -> tuple[str, Path]:
    name, _, path = value.partition("=")
    if not path:
        raise argparse.ArgumentTypeError(f"Expected NAME=PATH, got: {value!r}")
    return name, Path(path)


def load_results(named_paths: list[tuple[str, Path]]) -> dict[str, dict]:
    results = {}
    for name, path in named_paths:
        results[name] = json.loads(path.read_text(encoding="utf-8"))
    return results


def plot_comparison(results: dict[str, dict], out_path: Path) -> None:
    names = list(results.keys())

    ap = [results[n]["localization"]["ap_at_iou_0.5"] for n in names]
    f1 = [results[n]["localization"]["f1_at_conf"] for n in names]
    # NOT the raw FP-rate: evaluate.py's threshold calibration pins that close
    # to the same fp_target for every model by construction, so it doesn't
    # differentiate them. Recall at that same calibrated threshold does --
    # it's the real cost, in missed probes, of reaching the shared FP-rate
    # target, and is what actually varies across models.
    recall_at_fp_target = [results[n]["localization"]["recall_at_fp_calibrated_threshold"] for n in names]
    runtime_ms = [results[n]["runtime"]["mean_ms"] for n in names]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].bar(names, ap, color="steelblue")
    axes[0, 0].set_title("AP@IoU=0.5 (higher is better)")
    axes[0, 0].set_ylim(0, 1)

    axes[0, 1].bar(names, f1, color="seagreen")
    axes[0, 1].set_title("F1 @ operating confidence threshold (higher is better)")
    axes[0, 1].set_ylim(0, 1)

    axes[1, 0].bar(names, recall_at_fp_target, color="indianred")
    axes[1, 0].set_title("Recall at FP-rate-calibrated threshold (higher is better)")
    axes[1, 0].set_ylim(0, 1)

    axes[1, 1].bar(names, runtime_ms, color="goldenrod")
    axes[1, 1].set_title("Mean inference time, CPU (ms/image, lower is better)")

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result", dest="results", action="append", type=_parse_result_arg, required=True,
        metavar="NAME=PATH", help="Repeatable: a label and an evaluate.py --out JSON path",
    )
    parser.add_argument("--out", type=Path, default=Path("report/figures/model_comparison.png"))
    args = parser.parse_args()

    results = load_results(args.results)
    plot_comparison(results, args.out)

    print(f"{'model':<12} {'AP@0.5':>8} {'F1':>6} {'Recall@FPtgt':>13} {'ROC-AUC':>8} {'ms/img':>8}")
    for name, r in results.items():
        print(
            f"{name:<12} {r['localization']['ap_at_iou_0.5']:>8.3f} "
            f"{r['localization']['f1_at_conf']:>6.3f} "
            f"{r['localization']['recall_at_fp_calibrated_threshold']:>13.3f} "
            f"{r.get('roc_auc', float('nan')):>8.3f} "
            f"{r['runtime']['mean_ms']:>8.1f}"
        )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
