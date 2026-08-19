"""Prune already-generated synthetic negatives down to a target ratio,
without re-running LaMa on anything -- just removes files and manifest
entries for a random subset.

Train negatives don't add localization learning signal (no GT box to
regress against): they only teach the classification/objectness branch.
A 1:1 ratio there mostly dilutes how often the model sees real positive
examples per epoch without buying much. val/test negatives, on the other
hand, are what calibrate the confidence threshold and measure the
false-positive rate -- shrinking those makes that measurement noisier. So
by default this prunes train negatives down to a target ratio while
leaving val/test untouched.
"""

import argparse
import json
import random
from pathlib import Path


def prune_negatives(
    manifest: list[dict],
    images_root: Path,
    labels_root: Path,
    target_ratio: float,
    splits: list[str],
    seed: int = 42,
) -> list[dict]:
    """Return a new manifest with a random subset of synthetic negatives
    removed (and their files deleted from disk) for the given splits, so
    that each targeted split keeps target_ratio of its original negative
    count."""
    rng = random.Random(seed)

    remove_indices: set[int] = set()
    for split in splits:
        candidates = [
            i for i, r in enumerate(manifest)
            if r["split"] == split and r["origin"] == "synthetic_negative"
        ]
        n_keep = round(len(candidates) * target_ratio)
        keep = set(rng.sample(candidates, k=n_keep)) if candidates else set()
        remove_indices.update(i for i in candidates if i not in keep)

    for i in remove_indices:
        r = manifest[i]
        img_path = images_root / r["split"] / r["file_name"]
        label_path = labels_root / r["split"] / f"{Path(r['file_name']).stem}.txt"
        img_path.unlink(missing_ok=True)
        label_path.unlink(missing_ok=True)

    return [r for i, r in enumerate(manifest) if i not in remove_indices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("yolo_dataset/splits_manifest.json"))
    parser.add_argument("--images-root", type=Path, default=Path("yolo_dataset/images"))
    parser.add_argument("--labels-root", type=Path, default=Path("yolo_dataset/labels"))
    parser.add_argument("--target-ratio", type=float, default=0.3)
    parser.add_argument("--splits", nargs="+", default=["train"],
                         help="Which splits to prune (default: train only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    n_before = sum(1 for r in manifest if r["origin"] == "synthetic_negative")

    new_manifest = prune_negatives(
        manifest, args.images_root, args.labels_root, args.target_ratio, args.splits, args.seed
    )

    n_after = sum(1 for r in new_manifest if r["origin"] == "synthetic_negative")
    args.manifest.write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")

    print(f"Synthetic negatives: {n_before} -> {n_after} (pruned {n_before - n_after} from {args.splits})")
    for split in ("train", "val", "test"):
        n_pos = sum(1 for r in new_manifest if r["split"] == split and r["origin"] == "original")
        n_neg = sum(1 for r in new_manifest if r["split"] == split and r["origin"] == "synthetic_negative")
        print(f"  {split:5s}: {n_pos} positive, {n_neg} negative")
    print(f"Wrote {args.manifest}")


if __name__ == "__main__":
    main()
