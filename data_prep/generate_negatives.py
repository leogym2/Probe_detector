"""Synthetic "no probe" negative image generation via generative inpainting (LaMa).

The dataset has zero images without a probe, but the final system must be
able to report "no probe detected". Since the exact bbox is known for every
image, we can build a "probe absent" version of each scene at near-zero
additional labeling cost: mask out the (padded) probe region and fill it
with LaMa (Large Mask Inpainting), a model trained specifically to
hallucinate plausible content for large masked regions.

An earlier version of this script used cv2.inpaint (classical texture
propagation, not generative). QA showed it looked convincing for small
bboxes but produced a visibly blurred band for large ones (up to ~33% of
image area here), which risked teaching the model to associate a blur
artifact -- rather than genuine scene understanding -- with "no probe". LaMa
handles both cases well, confirmed on the worst-case image in this dataset,
so a single method is now used uniformly (also avoids a second, subtler
shortcut: two visually different negative "styles" correlated with the
source bbox size).

Note: this module requires the `simple-lama-inpainting` package, which is
NOT in the main project requirements.txt (it forces older numpy/opencv/
pillow versions that conflict with the rest of the stack). It lives in the
separate .venv-lama environment -- see requirements-dataprep-lama.txt --
since it's a one-time offline dataset-prep tool, not a training/inference
dependency. Run this script (and build_yolo_dataset.py / inspect_negatives.py)
with .venv-lama's python.

Use inspect_negatives.py to visually review a sample before trusting the
output at scale.
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from probe_common.bbox import pad_bbox_xyxy
from probe_common.coco_io import Record, load_coco

_lama_model = None


def _get_lama_model():
    """Lazily load the LaMa model once (loading it is expensive)."""
    global _lama_model
    if _lama_model is None:
        from simple_lama_inpainting import SimpleLama

        _lama_model = SimpleLama()
    return _lama_model


def generate_negative(
    image_bgr: np.ndarray,
    bbox_xywh: tuple[float, float, float, float],
    pad_frac: float = 0.15,
) -> np.ndarray:
    """Return a copy of image_bgr with the (padded) bbox region filled in by
    LaMa generative inpainting."""
    img_h, img_w = image_bgr.shape[:2]
    x1, y1, x2, y2 = pad_bbox_xyxy(bbox_xywh, pad_frac, img_w, img_h)

    mask_arr = np.zeros((img_h, img_w), dtype=np.uint8)
    mask_arr[y1:y2, x1:x2] = 255

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = _get_lama_model()(Image.fromarray(image_rgb), Image.fromarray(mask_arr))
    return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


def _block_median_sharpness(
    gray_patch: np.ndarray,
    block_size: int = 8,
    exclude_box: tuple[int, int, int, int] | None = None,
) -> float:
    """Median Laplacian-variance across non-overlapping blocks of a patch.

    More robust than a single variance computed over the whole patch: one
    small high-contrast region (e.g. a hard real-world edge cutting through
    part of the patch) can dominate a whole-patch variance and doesn't
    represent the "typical" local texture, but barely moves the median
    across many blocks.

    If exclude_box=(ex1, ey1, ex2, ey2) is given (in gray_patch's local
    coordinates), any block overlapping it is skipped -- used to turn a
    padded outer box into a proper ring around the inner region, rather
    than a context measurement that includes the inner region itself.
    """
    h, w = gray_patch.shape
    if h < block_size or w < block_size:
        return float(cv2.Laplacian(gray_patch, cv2.CV_64F).var())

    lap = cv2.Laplacian(gray_patch, cv2.CV_64F)
    ex1, ey1, ex2, ey2 = exclude_box if exclude_box is not None else (0, 0, 0, 0)
    block_vars = []
    for by in range(0, h - block_size + 1, block_size):
        for bx in range(0, w - block_size + 1, block_size):
            if exclude_box is not None and bx < ex2 and bx + block_size > ex1 and by < ey2 and by + block_size > ey1:
                continue  # block overlaps the excluded inner region
            block_vars.append(lap[by : by + block_size, bx : bx + block_size].var())
    return float(np.median(block_vars)) if block_vars else float(lap.var())


def score_negative_quality(
    image_bgr: np.ndarray,
    bbox_xywh: tuple[float, float, float, float],
    pad_frac: float = 0.15,
    context_frac: float = 0.5,
    block_size: int = 8,
) -> float:
    """Heuristic quality score for a generated negative.

    Ratio of the sharpness (block-median Laplacian variance, see
    _block_median_sharpness) of the filled region to that of a surrounding
    context patch in the same image. Close to 1 means the fill is about as
    sharp as its real surroundings; lower means it's smoother/flatter.

    No fixed pass/fail cutoff: generated fills are inherently a bit
    smoother than real sensor noise/grain even when they look visually
    convincing, so an absolute threshold (e.g. "below 0.5 is bad") isn't
    reliable -- confirmed empirically, genuinely fine-looking fills scored
    well below 0.5 once the context measurement was corrected to properly
    exclude the inner region (see below). Treat this as a *relative
    ranking* across a batch (sort ascending, review the lowest few), not a
    verdict on any single image.

    Also deliberately compares *within the same image* rather than against
    a fixed global threshold in another sense: background texture varies a
    lot across scenes (e.g. a smooth, evenly-lit metal surface is naturally
    low-texture even where it's real), so comparing to some other image's
    sharpness would misfire.

    Uses a block-median rather than a single whole-patch variance
    specifically because of an observed failure mode: when the context
    patch mixes textures (e.g. mostly smooth sky plus a hard building
    edge), a whole-patch variance is pulled up by the sharp minority,
    flagging even a genuinely good, smooth fill as suspicious. The median
    across small blocks stays representative of the majority texture
    instead of being dominated by a locally sharp outlier region.

    The context region is a proper ring: the padded outer box minus the
    inner box (blocks overlapping the inner box are excluded), not just a
    bigger box that happens to include the inner region.
    """
    img_h, img_w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    x1, y1, x2, y2 = pad_bbox_xyxy(bbox_xywh, pad_frac, img_w, img_h)
    inner = gray[y1:y2, x1:x2]
    if inner.size == 0:
        return 1.0
    inner_sharpness = _block_median_sharpness(inner, block_size)

    cx1, cy1, cx2, cy2 = pad_bbox_xyxy(bbox_xywh, pad_frac + context_frac, img_w, img_h)
    context = gray[cy1:cy2, cx1:cx2]
    # Inner box in the context patch's local coordinates, to exclude it.
    local_inner = (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)
    context_sharpness = _block_median_sharpness(context, block_size, exclude_box=local_inner)

    if context_sharpness < 1e-6:
        return 1.0  # surrounding area itself has ~no texture; can't judge

    return float(inner_sharpness / context_sharpness)


def generate_all_negatives(
    records: list[Record],
    images_dir: Path,
    out_dir: Path,
    neg_ratio: float = 1.0,
    pad_frac: float = 0.15,
    seed: int = 42,
) -> list[dict]:
    """Generate a LaMa negative for a (sub)set of records.

    Returns a list of {source_file_name, neg_file_name, source_image_id,
    quality_score}.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if neg_ratio >= 1.0:
        selected = list(records)
    else:
        rng = random.Random(seed)
        k = round(len(records) * neg_ratio)
        selected = rng.sample(records, k=k)

    neg_records = []
    for r in selected:
        img_path = images_dir / r.file_name
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        neg_image = generate_negative(image, r.bbox, pad_frac=pad_frac)
        # Scored on the *result* image -- the context patch is unaffected by
        # inpainting (only the inner masked region changed), and the inner
        # patch needs to reflect what LaMa actually produced there.
        quality_score = score_negative_quality(neg_image, r.bbox, pad_frac=pad_frac)

        neg_file_name = f"{Path(r.file_name).stem}_neg.jpg"
        cv2.imwrite(str(out_dir / neg_file_name), neg_image)

        neg_records.append(
            {
                "source_file_name": r.file_name,
                "neg_file_name": neg_file_name,
                "source_image_id": r.image_id,
                "quality_score": quality_score,
            }
        )
    return neg_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("probe_dataset/probe_labels.json"))
    parser.add_argument("--images-dir", type=Path, default=Path("probe_dataset/probe_images"))
    parser.add_argument("--out-dir", type=Path, default=Path("data_prep/negatives_cache"))
    parser.add_argument("--pad-frac", type=float, default=0.15)
    parser.add_argument("--neg-ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = load_coco(args.labels)
    neg_records = generate_all_negatives(
        records,
        images_dir=args.images_dir,
        out_dir=args.out_dir,
        neg_ratio=args.neg_ratio,
        pad_frac=args.pad_frac,
        seed=args.seed,
    )

    manifest_path = args.out_dir / "negatives_manifest.json"
    manifest_path.write_text(json.dumps(neg_records, indent=2), encoding="utf-8")

    print(f"Generated {len(neg_records)} synthetic negatives in {args.out_dir}")
    if neg_records:
        scores = [r["quality_score"] for r in neg_records]
        print(f"Quality score: mean={sum(scores)/len(scores):.2f}, "
              f"median={sorted(scores)[len(scores)//2]:.2f}")
        # No absolute pass/fail cutoff -- generated fills are inherently a
        # bit smoother than real sensor noise, so low scores aren't reliably
        # "bad". Use this as a relative ranking of what to review first.
        worst = sorted(neg_records, key=lambda r: r["quality_score"])[:5]
        print("Lowest-scoring (review these first, not a bad/good verdict):")
        for r in worst:
            print(f"  {r['quality_score']:.2f}  {r['neg_file_name']}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
