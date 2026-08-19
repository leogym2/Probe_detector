"""Analyze already-trained candidate checkpoints: evaluate each on val,
compare them, auto-pick the winner by AP@0.5, copy it to weights/best.pt,
and run the final one-time test-split check.

Deliberately decoupled from training/train_colab.ipynb: this only reads
checkpoints that already exist on disk (under --weights-dir/<candidate>/
best.pt) and re-runs evaluate.py/compare_results.py on them. Re-run it any
time -- e.g. after tweaking evaluate.py's metrics, as happened repeatedly
during development -- without retraining anything. Runs entirely on CPU and
needs only `ultralytics` (no LaMa/simple-lama-inpainting), so it can run
locally in the main .venv just as well as in Colab.

Usage:
    python analyze_results.py \\
        --weights-dir weights \\
        --candidates yolov8n yolov8n_noflip yolov8s yolov8s_noflip yolov8m yolov8m_noflip
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run_evaluate(
    weights_path: Path | str,
    label: str,
    split: str,
    out_dir: Path,
    conf: float,
    iou_thresh: float,
    fp_target: float,
    imgsz: int,
) -> Path:
    out_path = out_dir / f"eval_results_{label}.json"
    cmd = [
        sys.executable, "evaluate.py",
        "--weights", str(weights_path),
        "--split", split,
        "--conf", str(conf),
        "--iou-thresh", str(iou_thresh),
        "--fp-target", str(fp_target),
        "--imgsz", str(imgsz),
        "--out", str(out_path),
    ]
    print(f"\n=== {label} ===")
    subprocess.run(cmd, check=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument(
        "--candidates", nargs="+", required=True,
        help="Subfolder names under --weights-dir, each containing best.pt, "
             "e.g. yolov8n yolov8n_noflip yolov8s yolov8s_noflip yolov8m yolov8m_noflip",
    )
    parser.add_argument("--baseline-weights", default="yolov8n.pt",
                         help="Stock pretrained checkpoint for the out-of-the-box comparison")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"],
                         help="Split used for the comparison/auto-pick (test is reserved for the final check below)")
    parser.add_argument("--out-dir", type=Path, default=Path("report"))
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--fp-target", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--final-checkpoint-out", type=Path, default=Path("weights/best.pt"))
    parser.add_argument("--skip-final-test", action="store_true",
                         help="Skip the one-time test-split check (e.g. while still iterating)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    result_paths: dict[str, Path] = {}
    result_paths["baseline"] = run_evaluate(
        args.baseline_weights, "baseline", args.split, args.out_dir,
        args.conf, args.iou_thresh, args.fp_target, args.imgsz,
    )

    for name in args.candidates:
        weights_path = args.weights_dir / name / "best.pt"
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {weights_path}")
        result_paths[name] = run_evaluate(
            weights_path, name, args.split, args.out_dir,
            args.conf, args.iou_thresh, args.fp_target, args.imgsz,
        )

    compare_cmd = [sys.executable, "compare_results.py"]
    for label, path in result_paths.items():
        compare_cmd += ["--result", f"{label}={path}"]
    compare_cmd += ["--out", str(args.out_dir / "figures" / "model_comparison.png")]
    subprocess.run(compare_cmd, check=True)

    scores = {
        name: json.loads(result_paths[name].read_text())["localization"]["ap_at_iou_0.5"]
        for name in args.candidates
    }
    print("\nCandidate AP@0.5 (val):")
    for name, ap in scores.items():
        print(f"  {name:20s} {ap:.3f}")
    winner = max(scores, key=scores.get)
    print(f"-> winner: {winner}")

    winner_weights = args.weights_dir / winner / "best.pt"
    args.final_checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(winner_weights, args.final_checkpoint_out)
    print(f"Copied {winner_weights} -> {args.final_checkpoint_out}")

    if not args.skip_final_test:
        run_evaluate(
            args.final_checkpoint_out, "final_test", "test", args.out_dir,
            args.conf, args.iou_thresh, args.fp_target, args.imgsz,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
