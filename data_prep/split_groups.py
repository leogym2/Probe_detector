"""Group-stratified train/val/test split.

The 308 images come from only 21 (device, flight-sequence) groups. Frames
within a group are temporally close (same wall, same lighting), so splitting
per-image would leak near-duplicate frames across splits. This module splits
per group instead, using a seeded random search (only 21 groups, so exact
bin-packing is unnecessary) that targets given split fractions while keeping
every device represented in every split and preserving the global top/bottom
probe-position ratio in each split.

Note on the test split: with only 21 groups total, a held-out test split
ends up with just a handful of groups (~3, ~45 images). This is small enough
that its metrics carry real sampling noise (e.g. one extra false positive on
~20 negative test images moves the FP-rate estimate by ~5 percentage
points) -- documented as an explicit limitation in the report rather than
presented as a fully precise number. val is still used during development
(model-variant selection, confidence-threshold calibration); test is only
touched once, at the end, for the final reported numbers.
"""

import argparse
import json
import random
from pathlib import Path

from probe_common.coco_io import load_coco

DEFAULT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def group_key_str(device: str, sequence: str) -> str:
    return f"{device}|{sequence}"


def _build_groups(records) -> dict[tuple[str, str], dict]:
    groups: dict[tuple[str, str], dict] = {}
    for r in records:
        key = (r.device, r.sequence)
        g = groups.setdefault(key, {"n": 0, "device": r.device, "cluster": r.cluster})
        if g["cluster"] != r.cluster:
            raise ValueError(
                f"Group {key} is not cluster-homogeneous: found both "
                f"{g['cluster']!r} and {r.cluster!r}"
            )
        g["n"] += 1
    return groups


def compute_group_split(
    records,
    fractions: dict[str, float] = DEFAULT_FRACTIONS,
    seed: int = 42,
    trials: int = 5000,
    tol_fraction: float = 0.07,
    tol_cluster_pp: float = 0.12,
) -> tuple[dict[tuple[str, str], str], dict]:
    """Assign each (device, sequence) group to one of fractions.keys().

    Each trial shuffles the group order, then streams through it assigning
    each group to whichever split is currently furthest below its target
    image-fraction (greedy deficit allocation -- generalizes cleanly to
    more than two splits, unlike a single prefix cut). Falls back through
    progressively relaxed constraints (first drop per-split device
    coverage, then drop cluster-ratio tolerance) only if no candidate
    satisfies the full constraint set within the trial budget.
    """
    assert abs(sum(fractions.values()) - 1.0) < 1e-6, "fractions must sum to 1.0"
    split_names = list(fractions.keys())

    groups = _build_groups(records)
    group_keys = list(groups.keys())
    devices = sorted({g["device"] for g in groups.values()})

    total_images = sum(g["n"] for g in groups.values())
    global_top_images = sum(g["n"] for g in groups.values() if g["cluster"] == "top")
    global_top_ratio = global_top_images / total_images

    rng = random.Random(seed)
    best = None  # (score, assignment, meta)

    for stage in range(3):  # 0 = full constraints, 1 = drop device coverage, 2 = drop cluster tol too
        for _ in range(trials):
            order = group_keys[:]
            rng.shuffle(order)

            achieved_n = {s: 0 for s in split_names}
            assignment: dict[tuple[str, str], str] = {}
            for key in order:
                n = groups[key]["n"]
                deficits = {
                    s: fractions[s] - achieved_n[s] / total_images for s in split_names
                }
                chosen = max(deficits, key=deficits.get)
                assignment[key] = chosen
                achieved_n[chosen] += n

            fractions_achieved = {s: achieved_n[s] / total_images for s in split_names}
            if any(abs(fractions_achieved[s] - fractions[s]) > tol_fraction for s in split_names):
                continue

            top_n = {s: 0 for s in split_names}
            for key, split in assignment.items():
                if groups[key]["cluster"] == "top":
                    top_n[split] += groups[key]["n"]
            top_ratios = {
                s: (top_n[s] / achieved_n[s] if achieved_n[s] else 0.0) for s in split_names
            }
            if stage < 2 and any(
                abs(top_ratios[s] - global_top_ratio) > tol_cluster_pp for s in split_names
            ):
                continue

            if stage < 1:
                devices_per_split = {
                    s: {groups[k]["device"] for k, v in assignment.items() if v == s}
                    for s in split_names
                }
                if any(devices_per_split[s] != set(devices) for s in split_names):
                    continue

            score = sum(abs(fractions_achieved[s] - fractions[s]) for s in split_names) + sum(
                abs(top_ratios[s] - global_top_ratio) for s in split_names
            )
            if best is None or score < best[0]:
                meta = {
                    "fractions_achieved": fractions_achieved,
                    "top_ratios": top_ratios,
                    "global_top_ratio": global_top_ratio,
                    "constraints_relaxed": stage,
                }
                best = (score, dict(assignment), meta)

        if best is not None:
            break

    if best is None:
        raise RuntimeError("No valid group split found even after relaxing constraints")

    _, assignment, meta = best
    return assignment, meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("probe_dataset/probe_labels.json"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=Path("yolo_dataset/group_split.json"))
    args = parser.parse_args()

    fractions = {
        "train": 1.0 - args.val_fraction - args.test_fraction,
        "val": args.val_fraction,
        "test": args.test_fraction,
    }
    records = load_coco(args.labels)
    assignment, meta = compute_group_split(
        records, fractions=fractions, seed=args.seed, trials=args.trials
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "groups": {group_key_str(*k): v for k, v in assignment.items()},
        "meta": {**meta, "seed": args.seed, "trials": args.trials},
    }
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for split in ("train", "val", "test"):
        n_groups = sum(1 for v in assignment.values() if v == split)
        print(
            f"{split:5s}: {n_groups:2d} groups, image fraction "
            f"{meta['fractions_achieved'][split]:.3f} (target {fractions[split]:.3f}), "
            f"top/bottom ratio {meta['top_ratios'][split]:.3f} (global {meta['global_top_ratio']:.3f})"
        )
    if meta["constraints_relaxed"] > 0:
        print(f"WARNING: constraints relaxed to stage {meta['constraints_relaxed']} to find a valid split")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
