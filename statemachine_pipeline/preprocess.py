"""Image loading and OpenCV preprocessing shared by detection stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for image preprocessing") from exc
    return cv2


def preprocess_image(path: str | Path, *, max_dimension: int | None = None) -> tuple[Any, float]:
    cv2 = _cv2()
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


def load_gray(path_or_image: str | Path | Any) -> Any:
    cv2 = _cv2()
    if isinstance(path_or_image, (str, Path)):
        image = cv2.imread(str(path_or_image), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Unable to read image: {path_or_image}")
        return image
    if path_or_image.ndim == 3:
        return cv2.cvtColor(path_or_image, cv2.COLOR_BGR2GRAY)
    return path_or_image


def ink_mask(gray: Any) -> Any:
    """Binary mask (uint8 0/1) of pen/printer ink, robust to uneven lighting.

    The background is estimated with a large morphological closing and divided
    out, so shadows and paper gradients in photos do not become ink; Otsu then
    picks the ink threshold on the flattened image.
    """
    cv2 = _cv2()
    import numpy as np

    kernel_size = max(15, (min(gray.shape[:2]) // 20) | 1)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)))
    flat = cv2.divide(gray, np.maximum(background, 1), scale=255)
    flat = cv2.GaussianBlur(flat, (3, 3), 0)
    threshold, mask = cv2.threshold(flat, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if threshold > 235:
        # Nearly blank image: Otsu splits noise. Require clearly dark pixels.
        mask = (flat < 200).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return mask.astype(np.uint8)


def stroke_width(mask: Any) -> float:
    """Typical stroke thickness in pixels (twice the median ridge distance)."""
    cv2 = _cv2()
    import numpy as np

    if not mask.any():
        return 2.0
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    # Ridge pixels: local maxima of the distance transform.
    dilated = cv2.dilate(distance, np.ones((3, 3), np.uint8))
    ridge = distance[(distance >= dilated - 1e-6) & (distance > 0)]
    if ridge.size == 0:
        return 2.0
    return float(max(1.0, 2 * np.median(ridge)))
