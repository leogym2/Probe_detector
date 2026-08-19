"""Writing of YOLO-format label files."""

from pathlib import Path

from probe_common.bbox import coco_to_yolo


def write_yolo_label(
    path: Path,
    bbox_xywh: tuple[float, float, float, float] | None = None,
    img_w: int | None = None,
    img_h: int | None = None,
    class_id: int = 0,
) -> None:
    """Write a YOLO label .txt file.

    If bbox_xywh is None, writes an empty file -- Ultralytics treats images
    with an empty (or missing) label file as background/negative examples.
    """
    if bbox_xywh is None:
        path.write_text("", encoding="utf-8")
        return

    if img_w is None or img_h is None:
        raise ValueError("img_w and img_h are required when bbox_xywh is provided")

    cx, cy, w, h = coco_to_yolo(bbox_xywh, img_w, img_h)
    path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n", encoding="utf-8")
