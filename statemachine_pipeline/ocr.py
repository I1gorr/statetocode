"""Tesseract OCR for state names and transition labels, plus text-box finding.

OCR is optional: when Tesseract or pytesseract is missing every call returns
``(None, None)`` and the pipeline records a warning instead of failing.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

MIN_CONFIDENCE = 0.45


@dataclass(frozen=True)
class OcrResult:
    text: str | None
    confidence: float | None


@lru_cache(maxsize=1)
def ocr_available() -> bool:
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:  # ImportError, TesseractNotFoundError, OSError
        return False


def _clean(text: str) -> str | None:
    text = "".join(ch for ch in text if ch.isprintable())
    text = " ".join(text.split())
    # Tesseract often reads the outline of a box/circle as stray bars or quotes.
    text = re.sub(r"^[|‘’'\"`.,_~-]+|[|‘’'\"`.,_~-]+$", "", text).strip()
    return text or None


def _glyph_image(gray_crop: Any, keep: Any = None) -> Any | None:
    """Upscale, binarize and keep only glyph-sized components.

    Components touching the crop border (or the border of ``keep``) and
    components spanning most of the crop (state outlines, inner final-state
    rings, arrow fragments) are dropped.
    """
    import cv2
    import numpy as np

    if gray_crop is None or gray_crop.size == 0:
        return None
    scale = 3.0 if gray_crop.shape[0] < 80 else 2.0
    big = cv2.resize(gray_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    big = cv2.GaussianBlur(big, (3, 3), 0)
    _, ink = cv2.threshold(big, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if keep is not None:
        keep_big = cv2.resize(keep.astype(np.uint8), (big.shape[1], big.shape[0]), interpolation=cv2.INTER_NEAREST)
        ink &= keep_big
    if ink.mean() > 0.5:  # light text on dark background
        ink = 1 - ink
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if count <= 1:
        return None
    height, width = ink.shape
    keep_ids = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if x == 0 or y == 0 or x + w >= width or y + h >= height:
            continue
        if w > 0.6 * width and h > 0.45 * height:
            continue  # ring / outline
        if h > 0.7 * height:
            continue  # vertical outline fragment
        if area < 4 * scale:
            continue  # noise speck
        keep_ids.append(label)
    if not keep_ids:
        return None
    glyphs = np.isin(labels, keep_ids)
    # Keep the dominant text row: drop specks far above/below the main line.
    ys, xs = np.nonzero(glyphs)
    clean = np.where(glyphs, 0, 255).astype(np.uint8)[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return clean


def _normalise_label(text: str) -> str:
    # Tesseract confuses the closing guard bracket with ')' / '}' and '[' with '{'.
    text = text.replace("{", "[")
    return re.sub(r"\[([^\[\]()]*)[)}]", r"[\1]", text)


def read_text(gray_crop: Any, *, keep: Any = None, label: bool = False) -> OcrResult:
    """OCR a grayscale crop holding one line of dark text on a light background."""
    if not ocr_available():
        return OcrResult(None, None)
    import cv2
    import pytesseract

    clean = _glyph_image(gray_crop, keep)
    if clean is None:
        return OcrResult(None, None)
    image = cv2.copyMakeBorder(clean, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=255)
    try:
        data = pytesseract.image_to_data(image, config="--psm 7", output_type=pytesseract.Output.DICT)
    except Exception:
        return OcrResult(None, None)
    words, confidences = [], []
    for word, conf in zip(data.get("text", []), data.get("conf", [])):
        try:
            value = float(conf)
        except (TypeError, ValueError):
            continue
        if value < 0 or not str(word).strip():
            continue
        words.append(str(word))
        confidences.append(value / 100.0)
    text = _clean(" ".join(words))
    if text is None:
        return OcrResult(None, None)
    if label:
        text = _normalise_label(text)
    return OcrResult(text, round(sum(confidences) / len(confidences), 3))


def read_state_label(gray: Any, bbox: tuple[float, float, float, float], shape: str,
                     double_ring: bool = False) -> OcrResult:
    """OCR the interior of a state; its outline is removed as an oversized component."""
    h, w = gray.shape
    x1, y1, x2, y2 = bbox
    pad = 2
    crop = gray[max(0, int(y1) - pad):min(h, int(math.ceil(y2)) + pad + 1),
                max(0, int(x1) - pad):min(w, int(math.ceil(x2)) + pad + 1)]
    return read_text(crop)


def read_box(gray: Any, box: tuple[float, float, float, float]) -> OcrResult:
    h, w = gray.shape
    x1, y1, x2, y2 = box
    crop = gray[max(0, int(y1) - 3):min(h, int(math.ceil(y2)) + 4), max(0, int(x1) - 3):min(w, int(math.ceil(x2)) + 4)]
    return read_text(crop, label=True)


def find_text_boxes(free_ink: Any, *, max_glyph_height: int = 28) -> list[tuple[float, float, float, float]]:
    """Group small free-standing ink components into text-line boxes.

    Used when the detector has no ``label`` class. Arrows are long connected
    components and are excluded by size; glyphs are small and arranged in rows.
    """
    import cv2

    count, _, stats, _ = cv2.connectedComponentsWithStats(free_ink.astype("uint8"), connectivity=8)
    glyphs = []
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        if h > max_glyph_height or w > max_glyph_height * 1.6 or area < 3:
            continue
        if h < 4 and w < 4:
            continue
        glyphs.append([float(x), float(y), float(x + w), float(y + h)])
    glyphs.sort(key=lambda b: (b[1], b[0]))
    lines: list[list[float]] = []
    for box in glyphs:
        height = box[3] - box[1]
        merged = False
        for line in lines:
            line_height = line[3] - line[1]
            vertical_overlap = min(line[3], box[3]) - max(line[1], box[1])
            gap = max(box[0] - line[2], line[0] - box[2])
            if vertical_overlap > 0.4 * min(height, line_height) and gap < 0.9 * max(height, line_height, 8):
                line[0], line[1] = min(line[0], box[0]), min(line[1], box[1])
                line[2], line[3] = max(line[2], box[2]), max(line[3], box[3])
                merged = True
                break
        if not merged:
            lines.append(list(box))
    # Second pass: merge lines that touch after growth.
    changed = True
    while changed:
        changed = False
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                a, b = lines[i], lines[j]
                overlap = min(a[3], b[3]) - max(a[1], b[1])
                gap = max(b[0] - a[2], a[0] - b[2])
                if overlap > 0.4 * min(a[3] - a[1], b[3] - b[1]) and gap < 0.9 * max(a[3] - a[1], b[3] - b[1], 8):
                    lines[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del lines[j]
                    changed = True
                    break
            if changed:
                break
    # A single isolated small blob (noise, arrowhead fragment) is not text.
    return [tuple(line) for line in lines if (line[2] - line[0]) >= 6 and (line[3] - line[1]) >= 6]  # type: ignore[misc]
