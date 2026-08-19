"""QA contact sheet: before/after comparison for a sample of synthetic negatives.

Run this and manually inspect the output before trusting synthetic negatives
at full scale. Uses the same generate_negative() (LaMa) as the real
pipeline, so what's shown here is exactly what build_yolo_dataset.py will
produce. Requires the .venv-lama environment.
"""

import argparse
import random
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

from probe_common.coco_io import load_coco
from data_prep.generate_negatives import generate_negative, score_negative_quality


def build_contact_sheet(
    labels_path: Path,
    images_dir: Path,
    out_path: Path,
    n_samples: int = 12,
    pad_frac: float = 0.15,
    seed: int = 42,
    cell_w: int = 320,
    cell_h: int = 220,
) -> None:
    records = load_coco(labels_path)

    # Bias the sample towards large bboxes (the known risk case) while still
    # including a spread of sizes, so QA actually exercises the worst case.
    by_area = sorted(records, key=lambda r: r.bbox[2] * r.bbox[3], reverse=True)
    n_large = min(n_samples // 2, len(by_area))
    large_sample = by_area[:n_large]

    rng = random.Random(seed)
    remaining = [r for r in records if r not in large_sample]
    random_sample = rng.sample(remaining, k=min(n_samples - n_large, len(remaining)))

    sample = large_sample + random_sample

    cols = 2  # before | after, one row per sampled image
    rows = len(sample)
    sheet = Image.new("RGB", (cell_w * cols, cell_h * rows), "white")

    for row, r in enumerate(sample):
        img_path = images_dir / r.file_name
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            raise FileNotFoundError(img_path)

        neg_bgr = generate_negative(image_bgr, r.bbox, pad_frac=pad_frac)
        quality_score = score_negative_quality(neg_bgr, r.bbox, pad_frac=pad_frac)

        before = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        after = Image.fromarray(cv2.cvtColor(neg_bgr, cv2.COLOR_BGR2RGB))

        x, y, w, h = r.bbox
        draw_before = ImageDraw.Draw(before)
        draw_before.rectangle([x, y, x + w, y + h], outline="red", width=2)

        thumb_w, thumb_h = cell_w, cell_h - 15
        before_thumb = before.resize((thumb_w, thumb_h))
        after_thumb = after.resize((thumb_w, thumb_h))

        cell_before = Image.new("RGB", (cell_w, cell_h), "white")
        cell_before.paste(before_thumb, (0, 15))
        ImageDraw.Draw(cell_before).text((2, 2), f"BEFORE  {r.file_name[:28]}", fill="black")

        cell_after = Image.new("RGB", (cell_w, cell_h), "white")
        cell_after.paste(after_thumb, (0, 15))
        area_pct = (w * h) / (r.width * r.height) * 100
        ImageDraw.Draw(cell_after).text(
            (2, 2), f"AFTER  area {area_pct:.1f}%  quality {quality_score:.2f}", fill="black"
        )

        sheet.paste(cell_before, (0, row * cell_h))
        sheet.paste(cell_after, (cell_w, row * cell_h))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("probe_dataset/probe_labels.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("probe_dataset/probe_images"))
    parser.add_argument("--out", type=Path, default=Path("data_prep/negatives_qa/contact_sheet.png"))
    parser.add_argument("--n-samples", type=int, default=12)
    parser.add_argument("--pad-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_contact_sheet(
        labels_path=args.labels,
        images_dir=args.images_dir,
        out_path=args.out,
        n_samples=args.n_samples,
        pad_frac=args.pad_frac,
        seed=args.seed,
    )
    print(f"Wrote contact sheet: {args.out}")


if __name__ == "__main__":
    main()
