"""Image loading and optional OpenCV preprocessing."""

from pathlib import Path
from typing import Any


def preprocess_image(path: str | Path, *, max_dimension: int | None = None) -> tuple[Any, float]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for image preprocessing") from exc
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to read image: {path}")
    scale = 1.0
    if max_dimension and max(image.shape[:2]) > max_dimension:
        scale = max_dimension / max(image.shape[:2])
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # CLAHE improves faint diagrams without erasing thin arrowheads.
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR), scale

