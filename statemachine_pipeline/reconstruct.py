"""Convert detector output into canonical states and transitions."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Iterable
from .models import Position, State, StateMachine, Transition


@dataclass(frozen=True)
class Detection:
    kind: str
    bbox: tuple[float, float, float, float]
    confidence: float
    orientation: float | None = None
    text: str | None = None

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)


@dataclass
class Reconstruction:
    machine: StateMachine
    warnings: list[str]


@dataclass(frozen=True)
class ArrowGeometry:
    """Image-derived arrow axis and tip, with a confidence independent of YOLO."""

    tail: tuple[float, float]
    tip: tuple[float, float]
    confidence: float


def _image_arrow_geometry(image_path: str | Path, bbox: tuple[float, float, float, float]) -> ArrowGeometry | None:
    """Find the arrowhead end using ink width along the arrow's PCA axis.

    The trained model currently has only ``state`` and ``arrow`` classes, so
    arrow direction cannot come from a class label. Arrowheads widen the ink
    near the tip; comparing the perpendicular ink width at both PCA extremes
    is more reliable than choosing the closest state center.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    x1, y1, x2, y2 = (max(0, int(v)) for v in bbox)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2 + 1, x1:x2 + 1]
    if crop.size < 25:
        return None
    # Otsu preserves antialiased arrow pixels while removing the white field.
    _, mask = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(mask)
    if len(xs) < 12:
        return None
    points = np.column_stack((xs, ys)).astype(float)
    centroid = points.mean(axis=0)
    points -= centroid
    _, _, vt = np.linalg.svd(points, full_matrices=False)
    axis = vt[0]
    if axis[0] < 0 or (axis[0] == 0 and axis[1] < 0):
        axis = -axis
    projections = points @ axis
    span = float(projections.max() - projections.min())
    if span < 8:
        return None
    perpendicular = points @ np.array([-axis[1], axis[0]])
    window = max(5.0, span * 0.22)
    scores: list[float] = []
    for extreme in (projections.min(), projections.max()):
        selected = np.abs(projections - extreme) <= window
        if not np.any(selected):
            scores.append(0)
            continue
        width = float(np.percentile(perpendicular[selected], 95) -
                      np.percentile(perpendicular[selected], 5))
        density = float(np.count_nonzero(selected)) / window
        scores.append(width * 0.75 + density * 0.25)
    difference = abs(scores[0] - scores[1])
    confidence = min(1.0, difference / max(scores) * 2.0) if max(scores) else 0.0
    tip_projection = projections.max() if scores[1] > scores[0] else projections.min()
    tail_projection = projections.min() if scores[1] > scores[0] else projections.max()
    origin = np.array([x1, y1], dtype=float) + centroid
    tip = origin + axis * tip_projection
    tail = origin + axis * tail_projection
    return ArrowGeometry(tuple(tail), tuple(tip), confidence)


def _boundary(center: tuple[float, float], bbox: tuple[float, float, float, float], toward: tuple[float, float]) -> tuple[float, float]:
    dx, dy = toward[0] - center[0], toward[1] - center[1]
    length = math.hypot(dx, dy) or 1
    # Ellipse intersection is more accurate than using the nearest center.
    rx, ry = (bbox[2] - bbox[0]) / 2, (bbox[3] - bbox[1]) / 2
    scale = 1 / math.sqrt((dx / rx) ** 2 + (dy / ry) ** 2)
    return center[0] + dx * scale / length, center[1] + dy * scale / length


def _oriented_endpoints(
    bbox: tuple[float, float, float, float], angle: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the two extreme points of a box along an arrow axis."""
    x1, y1, x2, y2 = bbox
    axis = (math.cos(angle), math.sin(angle))
    corners = ((x1, y1), (x1, y2), (x2, y1), (x2, y2))
    projections = [point[0] * axis[0] + point[1] * axis[1] for point in corners]
    return corners[projections.index(min(projections))], corners[projections.index(max(projections))]


def _associate_arrowheads(
    detections: list[Detection],
) -> list[Detection]:
    """Attach explicit arrowhead detections to their nearest shaft."""
    shafts = [d for d in detections if d.kind in {"arrow", "transition"}]
    heads = [d for d in detections if d.kind == "arrowhead"]
    if not heads or not shafts:
        return [d for d in detections if d.kind != "arrowhead"]
    result = list(shafts)
    used: set[int] = set()
    for head in heads:
        hx, hy = head.center
        candidates = [
            (math.hypot(hx - shaft.center[0], hy - shaft.center[1]), index, shaft)
            for index, shaft in enumerate(shafts)
            if index not in used
        ]
        if not candidates:
            continue
        distance, index, shaft = min(candidates)
        shaft_size = max(shaft.bbox[2] - shaft.bbox[0], shaft.bbox[3] - shaft.bbox[1])
        if distance > max(shaft_size * 0.75, 30):
            continue
        angle = math.degrees(math.atan2(hy - shaft.center[1], hx - shaft.center[0]))
        result[index] = replace(shaft, orientation=angle,
                                confidence=min(shaft.confidence, head.confidence))
        used.add(index)
    return result


def reconstruct(
    detections: Iterable[Detection],
    diagram_id: str = "diagram_001",
    image_path: str | Path | None = None,
) -> Reconstruction:
    states_raw = [d for d in detections if d.kind in {"state", "initial", "final"}]
    arrows = _associate_arrowheads([
        d for d in detections if d.kind in {"arrow", "transition", "arrowhead"}
    ])
    states: list[State] = []
    for index, detection in enumerate(sorted(states_raw, key=lambda d: (d.center[1], d.center[0]))):
        state_type = detection.kind if detection.kind in {"initial", "final"} else "normal"
        states.append(State(
            id=f"S{index}",
            label=detection.text,
            type=state_type,
            position=Position(*detection.center),
            confidence=detection.confidence,
        ))
    warnings: list[str] = []
    transitions: list[Transition] = []
    for index, arrow in enumerate(arrows):
        if len(states) == 0:
            warnings.append(f"T{index}: no states available for endpoint mapping")
            continue
        x1, y1, x2, y2 = arrow.bbox
        geometry = _image_arrow_geometry(image_path, arrow.bbox) if image_path else None
        start_point, end_point = (x1, y1), (x2, y2)
        if arrow.orientation is not None:
            angle = math.radians(arrow.orientation)
            start_point, end_point = _oriented_endpoints(arrow.bbox, angle)
        elif geometry is not None:
            start_point, end_point = geometry.tail, geometry.tip
        distances = []
        for state in states:
            assert state.position is not None
            distances.append((math.hypot(start_point[0] - state.position.x, start_point[1] - state.position.y), state))
        source = min(distances, key=lambda item: item[0])[1]
        distances = []
        for state in states:
            assert state.position is not None
            distances.append((math.hypot(end_point[0] - state.position.x, end_point[1] - state.position.y), state))
        target = min(distances, key=lambda item: item[0])[1]
        if source.id == target.id:
            direction = "loop"
        elif arrow.orientation is None and (geometry is None or geometry.confidence < 0.18):
            direction = "unknown"
            warnings.append(f"T{index}: arrow direction is ambiguous; review endpoint mapping")
        else:
            direction = "forward"
        if geometry is not None and geometry.confidence < 0.45:
            warnings.append(f"T{index}: arrowhead evidence is weak; review direction")
        confidence = arrow.confidence
        if geometry is not None:
            confidence = min(1.0, confidence * (0.7 + 0.3 * geometry.confidence))
        transitions.append(Transition(
            id=f"T{index}", source=source.id, target=target.id,
            confidence=confidence, direction=direction, label=arrow.text,
            event=arrow.text,
        ))
    return Reconstruction(
        StateMachine("1.0", diagram_id, tuple(states), tuple(transitions),
                     {"source": "image", "warnings": warnings}),
        warnings,
    )
