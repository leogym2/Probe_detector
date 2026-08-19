"""Export a score-ordered QA sample from the complete synthetic-negative cache."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw


def fit(image, max_width=640, max_height=480):
    result = image.copy()
    result.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    return result


def make_pair(original_path, negative_path, record, rank, total):
    with Image.open(original_path) as source, Image.open(negative_path) as negative:
        original = fit(source.convert("RGB"))
        synthetic = fit(negative.convert("RGB"))
    header, gap, margin = 48, 12, 12
    canvas = Image.new("RGB", (original.width + synthetic.width + gap + 2 * margin,
        max(original.height, synthetic.height) + header + 2 * margin), "white")
    canvas.paste(original, (margin, header + margin))
    canvas.paste(synthetic, (margin + original.width + gap, header + margin))
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 8), f"ORIGINAL | {record['source_file_name']}", fill="black")
    draw.text((margin + original.width + gap, 8), "SYNTHETIC NEGATIVE (LaMa)", fill="black")
    draw.text((margin, 28), f"quality score: {record['quality_score']:.3f} | rank: {rank}/{total} (ordered by score)", fill="black")
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data_prep/negatives_cache/negatives_manifest.json"))
    parser.add_argument("--originals", type=Path, default=Path("probe_dataset/probe_images"))
    parser.add_argument("--negatives", type=Path, default=Path("data_prep/negatives_cache"))
    parser.add_argument("--out", type=Path, default=Path("data_prep/negatives_qa/all_pairs"))
    args = parser.parse_args()
    records = json.loads(args.manifest.read_text(encoding="utf-8"))
    records.sort(key=lambda record: record["quality_score"])
    total = len(records)
    ranks = set(range(1, min(10, total) + 1))
    ranks.update(range(20, total + 1, 10))
    ranks.add(total)
    args.out.mkdir(parents=True, exist_ok=True)
    for rank in sorted(ranks):
        record = records[rank - 1]
        pair = make_pair(args.originals / record["source_file_name"], args.negatives / record["neg_file_name"], record, rank, total)
        pair.save(args.out / f"{rank:03d}_score-{record['quality_score']:06.3f}_{Path(record['neg_file_name']).stem}.jpg", quality=95)
    (args.out / "README.txt").write_text(
        f"QA sample from {total} synthetic negatives, ordered by quality score.\n"
        "Includes ranks 1-10, then every tenth rank, plus the final rank.\n"
        "Score is filled-region sharpness / surrounding-context sharpness. It is a QA ranking only; selection is applied separately.\n",
        encoding="utf-8")
    print(f"Wrote {len(ranks)} QA pairs from {total} cached negatives to {args.out}")


if __name__ == "__main__":
    main()
