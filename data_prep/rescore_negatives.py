"""Recompute quality_score for already-generated synthetic negatives.

Does NOT re-run LaMa: the fill images on disk don't change, only the score
does. Useful because score_negative_quality() itself has already been
revised more than once -- only needs OpenCV, so unlike the rest of
data_prep this runs fine in the main .venv (no LaMa dependency).
"""

import argparse
import json
from pathlib import Path

import cv2

from data_prep.generate_negatives import score_negative_quality


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("yolo_dataset/splits_manifest.json"))
    parser.add_argument("--images-root", type=Path, default=Path("yolo_dataset/images"))
    parser.add_argument("--pad-frac", type=float, default=0.15)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    positives_by_id = {r["source_image_id"]: r for r in manifest if r["origin"] == "original"}

    updated = []
    for r in manifest:
        if r["origin"] != "synthetic_negative":
            continue
        src = positives_by_id[r["source_image_id"]]
        img_path = args.images_root / r["split"] / r["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(img_path)
        r["quality_score"] = score_negative_quality(image, tuple(src["bbox"]), pad_frac=args.pad_frac)
        updated.append(r)

    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    scores = [r["quality_score"] for r in updated]
    print(f"Updated quality_score for {len(updated)} synthetic negatives")
    if scores:
        print(f"mean={sum(scores)/len(scores):.2f}, median={sorted(scores)[len(scores)//2]:.2f}")
        print("Lowest-scoring (review these first, not a bad/good verdict):")
        for r in sorted(updated, key=lambda r: r["quality_score"])[:5]:
            print(f"  {r['quality_score']:.2f}  {r['file_name']}")
    print(f"Wrote {args.manifest}")


if __name__ == "__main__":
    main()
