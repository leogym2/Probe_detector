"""Bounding box format conversions and geometry helpers."""


def coco_to_yolo(
    bbox_xywh: tuple[float, float, float, float], img_w: int, img_h: int
) -> tuple[float, float, float, float]:
    """Convert [x, y, w, h] (top-left + size, pixels) to YOLO's normalized
    [cx, cy, w, h] (center, fraction of image size)."""
    x, y, w, h = bbox_xywh
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return cx, cy, w / img_w, h / img_h


def xywh_to_xyxy(
    bbox_xywh: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    x, y, w, h = bbox_xywh
    return x, y, x + w, y + h


def iou_xyxy(
    box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]
) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def bbox_area_fraction(
    bbox_xywh: tuple[float, float, float, float], img_w: int, img_h: int
) -> float:
    """Fraction of the image area covered by the box."""
    _x, _y, w, h = bbox_xywh
    return (w * h) / (img_w * img_h)


def pad_bbox_xyxy(
    bbox_xywh: tuple[float, float, float, float],
    pad_frac: float,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Pad a [x, y, w, h] box by pad_frac of its own size on each side,
    clipped to image bounds, returned as integer pixel [x1, y1, x2, y2].

    Used to make sure inpainting fully covers antialiased edges and thin
    cable/arm fringes that the tight annotation box might clip.
    """
    x, y, w, h = bbox_xywh
    pad_x = w * pad_frac
    pad_y = h * pad_frac

    x1 = int(max(0, round(x - pad_x)))
    y1 = int(max(0, round(y - pad_y)))
    x2 = int(min(img_w, round(x + w + pad_x)))
    y2 = int(min(img_h, round(y + h + pad_y)))
    return x1, y1, x2, y2
