"""Materialize the best cached synthetic negatives in yolo_dataset.

Edit NEGATIVE_RATIOS below, then run this module.  LaMa is not run again.
"""

import json
import shutil
from pathlib import Path

from data_prep.coco_to_yolo import write_yolo_label

SPLITS = ("train", "val", "test")

# Change these values in code, then rerun this module. Keep val/test at 1.0
# unless intentionally changing the false-positive evaluation set.
NEGATIVE_RATIOS = {"train": 0.30, "val": 1.0, "test": 1.0}


def select_negatives(manifest_path, cache_manifest_path, cache_dir, images_root, labels_root, ratios):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    originals = [record for record in manifest if record["origin"] == "original"]
    source_by_id = {record["source_image_id"]: record for record in originals}

    candidates_by_split = {split: [] for split in SPLITS}
    for cached in cache:
        source = source_by_id.get(cached["source_image_id"])
        if source is None:
            raise KeyError(f"Cached negative has no source: {cached['source_image_id']}")
        candidate = {**cached, "split": source["split"], "group": source["group"]}
        candidate["score_distance_from_one"] = abs(candidate["quality_score"] - 1.0)
        candidates_by_split[candidate["split"]].append(candidate)

    selected = []
    summary = {"ratios": ratios, "selection_rule": "minimum abs(quality_score - 1.0)"}
    for split in SPLITS:
        candidates = sorted(candidates_by_split[split], key=lambda record: record["score_distance_from_one"])
        positives = sum(record["split"] == split for record in originals)
        keep = round(positives * ratios[split])
        if keep > len(candidates):
            raise ValueError(f"Requested {keep} negatives for {split}, cache has {len(candidates)}")
        for rank, candidate in enumerate(candidates[:keep], start=1):
            candidate["selection_rank"] = rank
            selected.append(candidate)
        summary[split] = {"positives": positives, "candidates": len(candidates), "selected": keep}

    # The yolo_dataset directory is generated. Remove only its prior synthetic negatives.
    for record in manifest:
        if record["origin"] == "synthetic_negative":
            (images_root / record["split"] / record["file_name"]).unlink(missing_ok=True)
            (labels_root / record["split"] / f"{Path(record['file_name']).stem}.txt").unlink(missing_ok=True)

    new_manifest = originals.copy()
    for candidate in selected:
        negative_name = candidate["neg_file_name"]
        shutil.copyfile(cache_dir / negative_name, images_root / candidate["split"] / negative_name)
        write_yolo_label(labels_root / candidate["split"] / f"{Path(negative_name).stem}.txt", bbox_xywh=None)
        new_manifest.append({
            "file_name": negative_name,
            "split": candidate["split"],
            "group": candidate["group"],
            "origin": "synthetic_negative",
            "source_image_id": candidate["source_image_id"],
            "bbox": None,
            "cluster": None,
            "neg_method": "lama",
            "quality_score": candidate["quality_score"],
            "score_distance_from_one": candidate["score_distance_from_one"],
            "selection_rank": candidate["selection_rank"],
        })

    manifest_path.write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")
    (manifest_path.parent / "negative_selection.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    if any(not 0 <= ratio <= 1 for ratio in NEGATIVE_RATIOS.values()):
        raise ValueError("All NEGATIVE_RATIOS values must be between 0 and 1.")
    summary = select_negatives(
        Path("yolo_dataset/splits_manifest.json"),
        Path("data_prep/negatives_cache/negatives_manifest.json"),
        Path("data_prep/negatives_cache"),
        Path("yolo_dataset/images"),
        Path("yolo_dataset/labels"),
        NEGATIVE_RATIOS,
    )
    for split in SPLITS:
        values = summary[split]
        print(f"{split}: {values['selected']}/{values['candidates']} candidates selected")


if __name__ == "__main__":
    main()
