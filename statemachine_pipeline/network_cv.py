"""High-precision baseline for networks drawn as circles joined by strokes.

This deliberately does not use the state-machine YOLO weights: those weights have
different classes and using them for networks lowers recall and creates false edges.
"""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path

from .network import NetworkEdge, NetworkGraph, NetworkNode


def _ocr(image, x: int, y: int, radius: int) -> str | None:
    try:
        import cv2
        import pytesseract
        crop = image[max(0, y - radius + 4):y + radius - 3, max(0, x - radius + 4):x + radius - 3]
        text = pytesseract.image_to_string(crop, config="--psm 7").strip()
        return " ".join(text.split()) or None
    except (ImportError, RuntimeError, OSError):
        return None


def detect_network(image_path: str | Path, *, min_radius: int = 15) -> NetworkGraph:
    """Detect circles and connecting straight strokes from a photographed sketch."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("Network detection requires opencv-python and numpy") from exc
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    blurred = cv2.medianBlur(image, 5)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=max(2 * min_radius, 24),
        param1=90, param2=28, minRadius=min_radius, maxRadius=max(min_radius + 10, min(image.shape) // 3),
    )
    candidates: list[tuple[float, float, float]] = []
    if circles is not None:
        candidates = [tuple(map(float, circle)) for circle in np.round(circles[0]).astype(int)]
    # A contour fallback recovers imperfect, hand-drawn circles that Hough misses.
    if not candidates:
        _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter and 0.55 < 4 * math.pi * area / perimeter**2 < 1.25:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if radius >= min_radius:
                    candidates.append((x, y, radius))
    # Deduplicate overlapping detections, retaining the stronger/larger circle.
    candidates.sort(key=lambda value: value[2], reverse=True)
    edge_map = cv2.Canny(image, 50, 150, apertureSize=3)
    verified: list[tuple[float, float, float]] = []
    for x, y, radius in candidates:
        angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        xs = np.clip(np.rint(x + radius * np.cos(angles)).astype(int), 0, edge_map.shape[1] - 1)
        ys = np.clip(np.rint(y + radius * np.sin(angles)).astype(int), 0, edge_map.shape[0] - 1)
        # Hough can fit large circles around connector crossings. Requiring
        # visible support around the circumference removes those false nodes.
        if float((edge_map[ys, xs] > 0).mean()) >= 0.20:
            verified.append((x, y, radius))
    candidates = verified
    selected: list[tuple[float, float, float]] = []
    for candidate in candidates:
        if all(math.hypot(candidate[0] - other[0], candidate[1] - other[1]) > min(candidate[2], other[2]) for other in selected):
            selected.append(candidate)
    nodes = tuple(
        NetworkNode(
            f"N{index}", _ocr(image, int(x), int(y), int(radius)),
            {"x": x, "y": y}, radius, min(1.0, 0.55 + min(radius / 100, 0.4)),
        )
        for index, (x, y, radius) in enumerate(sorted(selected, key=lambda value: (value[1], value[0])))
    )
    edges: list[NetworkEdge] = []
    if len(nodes) >= 2:
        edges = _find_edges(image, nodes)
    warnings = []
    if not nodes:
        warnings.append("No circular components detected; improve contrast or use larger circles")
    if any(node.label is None for node in nodes):
        warnings.append("One or more component labels could not be read; review labels in the editor")
    if len(nodes) >= 2 and not edges:
        warnings.append("No connecting strokes detected between circular components")
    return NetworkGraph(
        "network-1.0", Path(image_path).stem, nodes, tuple(edges),
        {"source": "image", "detector": "opencv-hough-circles", "warnings": warnings},
    )


def _find_edges(image, nodes: tuple[NetworkNode, ...]) -> list[NetworkEdge]:
    import cv2
    import numpy as np
    edges = []
    threshold = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(threshold, 1, np.pi / 180, threshold=25,
                            minLineLength=25, maxLineGap=18)
    if lines is None:
        return edges
    seen: set[tuple[str, str]] = set()
    for raw in np.asarray(lines).reshape(-1, 4):
        x1, y1, x2, y2 = map(float, raw)
        length = math.hypot(x2 - x1, y2 - y1)
        if length < 25:
            continue
        matches: list[tuple[float, NetworkNode]] = []
        for node in nodes:
            d1 = abs(math.hypot(x1 - node.position["x"], y1 - node.position["y"]) - node.radius)
            d2 = abs(math.hypot(x2 - node.position["x"], y2 - node.position["y"]) - node.radius)
            if min(d1, d2) <= max(10, node.radius * 0.45):
                matches.append((min(d1, d2), node))
        if len(matches) != 2 or matches[0][1].id == matches[1][1].id:
            continue
        first, second = (matches[0][1], matches[1][1]) if matches[0][0] <= matches[1][0] else (matches[1][1], matches[0][1])
        key = tuple(sorted((first.id, second.id)))
        if key in seen:
            continue
        seen.add(key)
        confidence = max(0.5, min(0.98, 0.65 + min(length / max(image.shape), 0.3)))
        edges.append(NetworkEdge(f"E{len(edges)}", first.id, second.id, False, None, confidence))
    # Hough segments are often split where a hand-drawn stroke meets a circle.
    # A pairwise center-line check recovers those joins without accepting empty
    # space as an edge.
    ink = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY_INV)[1]
    for index, first in enumerate(nodes):
        for second in nodes[index + 1:]:
            key = tuple(sorted((first.id, second.id)))
            if key in seen:
                continue
            dx = second.position["x"] - first.position["x"]
            dy = second.position["y"] - first.position["y"]
            distance = math.hypot(dx, dy)
            if distance <= first.radius + second.radius + 10:
                continue
            start = (first.position["x"] + dx * first.radius / distance,
                     first.position["y"] + dy * first.radius / distance)
            end = (second.position["x"] - dx * second.radius / distance,
                   second.position["y"] - dy * second.radius / distance)
            samples = 160
            values = []
            normal = (-dy / distance, dx / distance)
            for step in range(samples):
                ratio = step / (samples - 1)
                x, y = start[0] + (end[0] - start[0]) * ratio, start[1] + (end[1] - start[1]) * ratio
                pixels = []
                for offset in (-3, 0, 3):
                    px = int(round(x + normal[0] * offset))
                    py = int(round(y + normal[1] * offset))
                    if 0 <= py < ink.shape[0] and 0 <= px < ink.shape[1]:
                        pixels.append(ink[py, px] > 0)
                values.append(any(pixels))
            support = sum(values) / len(values)
            if support >= 0.42:
                seen.add(key)
                edges.append(NetworkEdge(f"E{len(edges)}", first.id, second.id, False, None,
                                         min(0.95, 0.55 + support * 0.4)))
    return edges
