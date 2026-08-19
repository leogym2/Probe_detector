"""Loading of the provided COCO-like probe_labels.json into flat records."""

import json
from dataclasses import dataclass
from pathlib import Path

from probe_common.naming import parse_filename


@dataclass
class Record:
    file_name: str
    image_id: int
    width: int
    height: int
    bbox: tuple[float, float, float, float]  # x, y, w, h (COCO convention)
    device: str
    sequence: str
    cluster: str  # "top" or "bottom", based on bbox vertical center


def _cluster_for(bbox: tuple[float, float, float, float], height: int) -> str:
    x, y, w, h = bbox
    cy_norm = (y + h / 2) / height
    return "top" if cy_norm < 0.5 else "bottom"


def load_coco(json_path: Path) -> list[Record]:
    """Load probe_labels.json and join images[] with annotations[] on id.

    Assumes a 1:1 relationship between images and annotations, as verified
    during the EDA phase (every image has exactly one probe annotation).
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    images_by_id = {im["id"]: im for im in data["images"]}

    records = []
    for ann in data["annotations"]:
        image = images_by_id[ann["image_id"]]
        bbox = tuple(ann["bbox"])
        parts = parse_filename(image["file_name"])
        records.append(
            Record(
                file_name=image["file_name"],
                image_id=image["id"],
                width=image["width"],
                height=image["height"],
                bbox=bbox,
                device=parts["device"],
                sequence=parts["sequence"],
                cluster=_cluster_for(bbox, image["height"]),
            )
        )
    return records
