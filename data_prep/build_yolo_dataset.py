"""Orchestrator: builds a complete Ultralytics-ready dataset from the raw
probe_dataset/ folder.

Produces yolo_dataset/{images,labels}/{train,val,test}/, data.yaml,
group_split.json and splits_manifest.json. Regenerable at any time from the
seed alone -- nothing in yolo_dataset/ is meant to be hand-edited.

Requires the .venv-lama environment (see requirements-dataprep-lama.txt),
since synthetic negatives are generated with LaMa generative inpainting.

Negative ratio is per-split, not a single global fraction: train negatives
don't add localization learning signal (no GT box to regress against), so a
large 1:1 ratio there mostly dilutes how often the model sees real positive
examples per epoch. val/test negatives, on the other hand, are what
calibrate the confidence threshold and measure the false-positive rate --
shrinking those makes that measurement noisier. Default: 0.3 for train,
1.0 (full) for val/test.

Which negatives to keep at a ratio below 1.0 is not random: a LaMa
candidate is generated for every positive image in the split, scored with
score_negative_quality(), and the kept fraction is the candidates closest
to a quality_score of 1.0 (i.e. hardest to distinguish from real
surrounding texture). See negative_selection.json, written alongside the
dataset, for the resulting positives/candidates/selected counts per split.
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2

from probe_common.coco_io import load_coco
from data_prep.coco_to_yolo import write_yolo_label
from data_prep.generate_negatives import generate_negative, score_negative_quality
from data_prep.split_groups import DEFAULT_FRACTIONS, compute_group_split, group_key_str

SPLITS = ("train", "val", "test")
DEFAULT_NEG_RATIOS = {"train": 0.3, "val": 1.0, "test": 1.0}


def build(
    labels_path: Path,
    images_dir: Path,
    out_dir: Path,
    fractions: dict[str, float] = DEFAULT_FRACTIONS,
    neg_ratios: dict[str, float] = DEFAULT_NEG_RATIOS,
    pad_frac: float = 0.15,
    seed: int = 42,
    trials: int = 5000,
    clean: bool = False,
) -> None:
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)

    images_out = {split: out_dir / "images" / split for split in SPLITS}
    labels_out = {split: out_dir / "labels" / split for split in SPLITS}
    for d in list(images_out.values()) + list(labels_out.values()):
        d.mkdir(parents=True, exist_ok=True)

    records = load_coco(labels_path)
    assignment, split_meta = compute_group_split(
        records, fractions=fractions, seed=seed, trials=trials
    )

    group_split_path = out_dir / "group_split.json"
    group_split_path.write_text(
        json.dumps(
            {
                "groups": {group_key_str(*k): v for k, v in assignment.items()},
                "meta": {**split_meta, "seed": seed, "trials": trials},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest: list[dict] = []
    counts: dict[tuple[str, str, str], int] = {}  # (split, origin, device) -> n

    def bump(split: str, origin: str, device: str) -> None:
        key = (split, origin, device)
        counts[key] = counts.get(key, 0) + 1

    # --- positives: copy image, write real label ---
    for r in records:
        split = assignment[(r.device, r.sequence)]
        src = images_dir / r.file_name
        dst = images_out[split] / r.file_name
        shutil.copyfile(src, dst)

        label_path = labels_out[split] / f"{Path(r.file_name).stem}.txt"
        write_yolo_label(label_path, bbox_xywh=r.bbox, img_w=r.width, img_h=r.height)

        manifest.append(
            {
                "file_name": r.file_name,
                "split": split,
                "group": group_key_str(r.device, r.sequence),
                "origin": "original",
                "source_image_id": r.image_id,
                "bbox": list(r.bbox),
                "cluster": r.cluster,
            }
        )
        bump(split, "original", r.device)

    # --- synthetic negatives: one LaMa candidate is generated per positive
    # in the split, scored with score_negative_quality(), and only the
    # candidates closest to a quality_score of 1.0 are kept, per neg_ratios
    # (see module docstring for the selection rule and for why train
    # differs from val/test) ---
    records_by_split: dict[str, list] = {s: [] for s in SPLITS}
    for r in records:
        records_by_split[assignment[(r.device, r.sequence)]].append(r)

    neg_quality: list[tuple[float, str]] = []  # (score, file_name) of kept negatives
    selection_summary: dict[str, dict] = {}
    for split in SPLITS:
        split_records = records_by_split[split]
        ratio = neg_ratios.get(split, 1.0)

        candidates = []
        for r in split_records:
            src = images_dir / r.file_name
            image_bgr = cv2.imread(str(src))
            if image_bgr is None:
                raise FileNotFoundError(src)

            neg_image = generate_negative(image_bgr, r.bbox, pad_frac=pad_frac)
            quality_score = score_negative_quality(neg_image, r.bbox, pad_frac=pad_frac)
            candidates.append((abs(quality_score - 1.0), quality_score, r, neg_image))

        candidates.sort(key=lambda c: c[0])
        n_keep = len(candidates) if ratio >= 1.0 else round(len(split_records) * ratio)
        selection_summary[split] = {
            "positives": len(split_records),
            "candidates": len(candidates),
            "selected": n_keep,
        }

        for rank, (score_distance, quality_score, r, neg_image) in enumerate(candidates[:n_keep], start=1):
            neg_file_name = f"{Path(r.file_name).stem}_neg.jpg"
            cv2.imwrite(str(images_out[split] / neg_file_name), neg_image)

            label_path = labels_out[split] / f"{Path(neg_file_name).stem}.txt"
            write_yolo_label(label_path, bbox_xywh=None)

            manifest.append(
                {
                    "file_name": neg_file_name,
                    "split": split,
                    "group": group_key_str(r.device, r.sequence),
                    "origin": "synthetic_negative",
                    "source_image_id": r.image_id,
                    "bbox": None,
                    "cluster": None,
                    "neg_method": "lama",
                    "quality_score": quality_score,
                    "score_distance_from_one": score_distance,
                    "selection_rank": rank,
                }
            )
            bump(split, "synthetic_negative", r.device)
            neg_quality.append((quality_score, neg_file_name))

    manifest_path = out_dir / "splits_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    selection_path = out_dir / "negative_selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "ratios": neg_ratios,
                "selection_rule": "minimum abs(quality_score - 1.0)",
                **selection_summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    data_yaml_path = out_dir / "data.yaml"
    data_yaml_path.write_text(
        "path: " + str(out_dir.resolve()).replace("\\", "/") + "\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names: [\"probe\"]\n",
        encoding="utf-8",
    )

    _print_summary(counts, split_meta, neg_quality)
    print(f"\nWrote {group_split_path}")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {selection_path}")
    print(f"Wrote {data_yaml_path}")


def _print_summary(
    counts: dict[tuple[str, str, str], int], split_meta: dict, neg_quality: list[tuple[float, str]]
) -> None:
    devices = sorted({device for _, _, device in counts})
    origins = sorted({origin for _, origin, _ in counts})
    print("Dataset build summary")
    print("=" * 60)
    for split in SPLITS:
        print(
            f"{split:5s} image fraction achieved: {split_meta['fractions_achieved'][split]:.3f} "
            f"(top/bottom ratio {split_meta['top_ratios'][split]:.3f} vs global "
            f"{split_meta['global_top_ratio']:.3f})"
        )
    if split_meta["constraints_relaxed"] > 0:
        print(f"NOTE: split constraints relaxed to stage {split_meta['constraints_relaxed']}")
    print()
    for split in SPLITS:
        for origin in origins:
            total = sum(counts.get((split, origin, d), 0) for d in devices)
            per_device = ", ".join(f"{d}={counts.get((split, origin, d), 0)}" for d in devices)
            print(f"  {split:5s} / {origin:24s}: {total:4d}  ({per_device})")

    if neg_quality:
        scores = [s for s, _ in neg_quality]
        print(
            f"\nSynthetic negative quality score (filled-region sharpness / context sharpness): "
            f"mean={sum(scores)/len(scores):.2f}, median={sorted(scores)[len(scores)//2]:.2f}"
        )
        # No absolute pass/fail cutoff -- generated fills are inherently a
        # bit smoother than real sensor noise, so low scores aren't reliably
        # "bad". Use this as a relative ranking of what to review first.
        print("Lowest-scoring (review these first, not a bad/good verdict):")
        for score, file_name in sorted(neg_quality)[:5]:
            print(f"  {score:.2f}  {file_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("probe_dataset/probe_labels.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("probe_dataset/probe_images"))
    parser.add_argument("--out-dir", type=Path, default=Path("yolo_dataset"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--train-neg-ratio", type=float, default=DEFAULT_NEG_RATIOS["train"])
    parser.add_argument("--val-neg-ratio", type=float, default=DEFAULT_NEG_RATIOS["val"])
    parser.add_argument("--test-neg-ratio", type=float, default=DEFAULT_NEG_RATIOS["test"])
    parser.add_argument("--pad-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--clean", action="store_true", help="Remove --out-dir before building")
    args = parser.parse_args()

    fractions = {
        "train": 1.0 - args.val_fraction - args.test_fraction,
        "val": args.val_fraction,
        "test": args.test_fraction,
    }
    neg_ratios = {
        "train": args.train_neg_ratio,
        "val": args.val_neg_ratio,
        "test": args.test_neg_ratio,
    }
    build(
        labels_path=args.labels,
        images_dir=args.images_dir,
        out_dir=args.out_dir,
        fractions=fractions,
        neg_ratios=neg_ratios,
        pad_frac=args.pad_frac,
        seed=args.seed,
        trials=args.trials,
        clean=args.clean,
    )


if __name__ == "__main__":
    main()
