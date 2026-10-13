"""Convert detector output (plus the image) into the canonical state machine.

With an image, endpoints come from stroke tracing (``arrows.py``) and labels
from OCR (``ocr.py``). Without an image only box geometry is available; that
legacy mode is kept for callers that have detections only and records an
explicit uncertainty for every endpoint it infers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable
from .models import (
    SCHEMA_VERSION, Position, SourceImage, State, StateMachine, Transition, Uncertainty,
)

STATE_KINDS = {"state", "initial", "final", "final_state"}
ARROW_KINDS = {"arrow", "transition"}


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
    debug: dict[str, Any] | None = None


@dataclass(frozen=True)
class ArrowGeometry:
    """Image-derived arrow axis and tip, with a confidence independent of YOLO."""

    tail: tuple[float, float]
    tip: tuple[float, float]
    confidence: float


def _image_arrow_geometry(image_path: str | Path, bbox: tuple[float, float, float, float]) -> ArrowGeometry | None:
    """Find the arrowhead end using ink width along the arrow's PCA axis.

    Used only in box-only mode (straight arrows). Arrowheads widen the ink near
    the tip; comparing perpendicular ink width at both PCA extremes is more
    reliable than choosing the closest state center.
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


def _associate_arrowheads(detections: list[Detection]) -> list[Detection]:
    """Attach explicit arrowhead detections to their nearest shaft (box-only mode)."""
    shafts = [d for d in detections if d.kind in ARROW_KINDS]
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


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _dedupe_states(detections: list[Detection]) -> list[Detection]:
    """Cross-class NMS: the detector may emit both ``state`` and ``final_state``."""
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda d: -d.confidence):
        if all(_iou(detection.bbox, other.bbox) < 0.5 for other in kept):
            kept.append(detection)
    return kept


def _state_geometry(mask: Any, bbox: tuple[float, float, float, float]) -> tuple[str, bool]:
    """Classify outline shape and detect a double ring (final state)."""
    h, w = mask.shape
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    rx, ry = (x2 - x1) / 2, (y2 - y1) / 2

    def ink_at(px: float, py: float) -> bool:
        ix, iy = int(round(px)), int(round(py))
        return 0 <= ix < w and 0 <= iy < h and bool(mask[iy, ix])

    # Rounded rectangles have ink along the box diagonal close to the corners;
    # circles do not (the diagonal at 0.85 of the half-diagonal lies outside).
    corner_hits = 0
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        if any(ink_at(cx + sx * rx * f, cy + sy * ry * f) for f in (0.82, 0.86, 0.9, 0.94)):
            corner_hits += 1
    aspect = (x2 - x1) / max(1.0, y2 - y1)
    shape = "rounded_rect" if corner_hits >= 3 or (aspect > 1.45 and corner_hits >= 2) else "circle"
    if shape != "circle":
        return shape, False
    # Radial profile: fraction of angles with ink at each relative radius.
    # Radial ink profiles: ``plain`` samples one radius per angle; ``banded``
    # counts an angle if any radius in a small band is ink, so thin blurred
    # inner rings still register. The gap between rings uses ``plain``.
    angles = [2 * math.pi * k / 90 for k in range(90)]

    def fraction(factor: float, band: tuple[float, ...]) -> float:
        return sum(any(ink_at(cx + rx * (factor + d) * math.cos(a), cy + ry * (factor + d) * math.sin(a))
                       for d in band) for a in angles) / 90

    factors = [step / 100 for step in range(40, 106, 2)]
    plain = {f: fraction(f, (0.0,)) for f in factors}
    banded = {f: fraction(f, (-0.015, 0.0, 0.015)) for f in factors}
    outer = max((f for f in factors if f >= 0.85), key=lambda f: (plain[f], -abs(f - 1.0)))
    inner_candidates = [f for f in factors if 0.55 < f < outer - 0.1]
    if not inner_candidates:
        return shape, False
    inner = max(inner_candidates, key=lambda f: banded[f])
    gap = min(plain[f] for f in factors if inner < f < outer)
    return shape, banded[inner] > 0.6 and gap < 0.5 * min(banded[inner], plain[outer])


def _point_segment_distance(p, a, b) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    length2 = dx * dx + dy * dy
    if length2 == 0:
        return math.dist(p, a)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / length2))
    return math.dist(p, (ax + t * dx, ay + t * dy))


def _box_to_path(box: tuple[float, float, float, float], path: list[tuple[float, float]]) -> float:
    corners_and_center = [
        ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
        (box[0], box[1]), (box[2], box[1]), (box[0], box[3]), (box[2], box[3]),
        ((box[0] + box[2]) / 2, box[1]), ((box[0] + box[2]) / 2, box[3]),
        (box[0], (box[1] + box[3]) / 2), (box[2], (box[1] + box[3]) / 2),
    ]
    best = math.inf
    for point in corners_and_center:
        for a, b in zip(path, path[1:]):
            best = min(best, _point_segment_distance(point, a, b))
    return best


def reconstruct(
    detections: Iterable[Detection],
    diagram_id: str = "diagram_001",
    image_path: str | Path | None = None,
    *,
    ocr: bool = True,
    config: dict[str, float] | None = None,
) -> Reconstruction:
    detections = list(detections)
    if image_path is not None:
        try:
            return _reconstruct_from_image(detections, diagram_id, Path(image_path), ocr=ocr, config=config)
        except (ImportError, ValueError) as exc:
            result = _reconstruct_from_boxes(detections, diagram_id, image_path)
            result.warnings.insert(0, f"image analysis unavailable ({exc}); used box geometry only")
            result.machine.metadata["warnings"] = result.warnings
            return result
    return _reconstruct_from_boxes(detections, diagram_id, None)


def _reconstruct_from_boxes(detections: list[Detection], diagram_id: str,
                            image_path: str | Path | None) -> Reconstruction:
    """Legacy box-only association. Every inferred endpoint is flagged for review."""
    states_raw = _dedupe_states([d for d in detections if d.kind in STATE_KINDS])
    arrows = _associate_arrowheads([d for d in detections if d.kind in ARROW_KINDS | {"arrowhead"}])
    states: list[State] = []
    for index, detection in enumerate(sorted(states_raw, key=lambda d: (d.center[1], d.center[0]))):
        state_type = {"initial": "initial", "final": "final", "final_state": "final"}.get(detection.kind, "normal")
        states.append(State(
            id=f"S{index}", label=detection.text, type=state_type,  # type: ignore[arg-type]
            position=Position(*detection.center), confidence=detection.confidence,
            bbox=detection.bbox, provenance="detector",
        ))
    warnings: list[str] = []
    uncertainties: list[Uncertainty] = []
    transitions: list[Transition] = []
    for index, arrow in enumerate(arrows):
        if len(states) == 0:
            warnings.append(f"T{index}: no states available for endpoint mapping")
            continue
        x1, y1, x2, y2 = arrow.bbox
        geometry = _image_arrow_geometry(image_path, arrow.bbox) if image_path else None
        start_point, end_point = (x1, y1), (x2, y2)
        if arrow.orientation is not None:
            start_point, end_point = _oriented_endpoints(arrow.bbox, math.radians(arrow.orientation))
        elif geometry is not None:
            start_point, end_point = geometry.tail, geometry.tip

        def nearest(point: tuple[float, float]) -> State:
            return min(states, key=lambda s: math.hypot(point[0] - s.position.x, point[1] - s.position.y))  # type: ignore[union-attr]

        source, target = nearest(start_point), nearest(end_point)
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
        uncertainties.append(Uncertainty(
            f"T{index}", "source", "endpoints inferred from box geometry only (no stroke tracing)",
            (source.id, target.id),
        ))
        transitions.append(Transition(
            id=f"T{index}", source=source.id, target=target.id,
            confidence=confidence, direction=direction, label=arrow.text,  # type: ignore[arg-type]
            event=arrow.text, provenance="detector",
        ))
    return Reconstruction(
        StateMachine(SCHEMA_VERSION, diagram_id, tuple(states), tuple(transitions),
                     {"source": "image", "warnings": warnings, "association": "box-geometry"},
                     uncertainties=tuple(uncertainties)),
        warnings,
    )


def _reconstruct_from_image(detections: list[Detection], diagram_id: str, image_path: Path, *,
                            ocr: bool, config: dict[str, float] | None) -> Reconstruction:
    import numpy as np

    from .arrows import StateRegion, build_region_map, load_config, trace_arrows
    from .ocr import find_text_boxes, ocr_available, read_box, read_state_label
    from .preprocess import ink_mask, load_gray
    from .semantics import annotate_machine

    config = config or load_config()
    gray = load_gray(image_path)
    mask = ink_mask(gray)
    height, width = gray.shape
    warnings: list[str] = []
    uncertainties: list[Uncertainty] = []

    # --- states -----------------------------------------------------------
    raw_states = _dedupe_states([d for d in detections if d.kind in STATE_KINDS])
    raw_states.sort(key=lambda d: (round(d.center[1] / 40), d.center[0]))
    regions: list[StateRegion] = []
    states: list[State] = []
    use_ocr = ocr and ocr_available()
    if ocr and not use_ocr:
        warnings.append("Tesseract OCR is unavailable; state and transition labels are left null")
    for index, detection in enumerate(raw_states):
        shape, double_ring = _state_geometry(mask, detection.bbox)
        final = detection.kind in {"final", "final_state"} or double_ring
        regions.append(StateRegion(index, detection.bbox, shape))
        label, label_confidence = detection.text, None
        if label is None and use_ocr:
            result = read_state_label(gray, detection.bbox, shape, final)
            label, label_confidence = result.text, result.confidence
        state_id = f"S{index}"
        if label is None:
            uncertainties.append(Uncertainty(state_id, "label", "state name could not be read"))
        elif label_confidence is not None and label_confidence < 0.6:
            uncertainties.append(Uncertainty(state_id, "label", f"OCR confidence {label_confidence:.2f} is low"))
        if final and detection.kind not in {"final", "final_state"}:
            uncertainties.append(Uncertainty(state_id, "type", "final state inferred from a double outline"))
        states.append(State(
            id=state_id, label=label, type="final" if final else "normal",
            position=Position(round(detection.center[0], 1), round(detection.center[1], 1)),
            confidence=round(detection.confidence, 4),
            bbox=tuple(round(v, 1) for v in detection.bbox),  # type: ignore[arg-type]
            shape="double_circle" if final and shape == "circle" else shape,  # type: ignore[arg-type]
            label_confidence=label_confidence,
            provenance="ocr" if label_confidence is not None else "detector",
        ))
    if not states:
        warnings.append("no states detected; check the model weights and image quality")

    # --- text labels --------------------------------------------------------
    label_boxes = [d.bbox for d in detections if d.kind == "label"]
    if not label_boxes and states:
        region_map = build_region_map(mask.shape, regions, config, mask)
        label_boxes = find_text_boxes((mask > 0) & (region_map < 0))
        if label_boxes:
            warnings.append("detector has no 'label' class; text boxes were found by connected components")

    # --- arrows ---------------------------------------------------------------
    arrowheads = [(d.bbox, d.confidence) for d in detections if d.kind == "arrowhead"]
    markers = [(d.bbox, d.confidence) for d in detections if d.kind == "initial_marker"]
    traced, debug = trace_arrows(mask, regions, arrowheads=arrowheads, initial_markers=markers,
                                 exclude_boxes=label_boxes, config=config) if regions else ([], {})

    initial_candidates: list[tuple[float, int, str]] = []
    edges = []
    for arrow in traced:
        if arrow.is_initial_marker and arrow.head_state is not None:
            initial_candidates.append((arrow.direction_confidence, arrow.head_state, "traced marker"))
        elif arrow.head_state is not None and arrow.tail_state is not None:
            edges.append(arrow)
        else:
            state = arrow.tail_state if arrow.tail_state is not None else arrow.head_state
            if state is not None:
                uncertainties.append(Uncertainty(
                    f"S{state}", "transitions",
                    "an arrow from this state ends in empty space; a state or marker may be missing",
                ))
    # Detector-driven fallback: a marker box adjacent to a state marks it initial.
    if markers and regions:
        region_map = debug.get("region_map")
        if region_map is None:
            region_map = build_region_map(mask.shape, regions, config, mask)
        for box, confidence in markers:
            x1, y1 = max(0, int(box[0]) - 4), max(0, int(box[1]) - 4)
            x2, y2 = min(width, int(math.ceil(box[2])) + 5), min(height, int(math.ceil(box[3])) + 5)
            window = region_map[y1:y2, x1:x2]
            touched = window[window >= 0]
            if touched.size:
                values, counts = np.unique(touched, return_counts=True)
                state = int(values[int(np.argmax(counts))])
                if not any(c[1] == state for c in initial_candidates):
                    initial_candidates.append((confidence * 0.9, state, "detected marker"))
    initial_state: int | None = None
    if initial_candidates:
        initial_candidates.sort(reverse=True)
        initial_state = initial_candidates[0][1]
        others = sorted({c[1] for c in initial_candidates[1:] if c[1] != initial_state})
        if others:
            uncertainties.append(Uncertainty(
                f"S{initial_state}", "type", "several initial markers were found",
                tuple(f"S{i}" for i in [initial_state, *others]),
            ))
    elif states:
        warnings.append("no initial-state marker found; mark the initial state in the editor")
    if initial_state is not None:
        current = states[initial_state]
        if current.type == "final":
            uncertainties.append(Uncertainty(current.id, "type", "state has an initial marker and a double outline"))
        states[initial_state] = replace(current, type="initial")

    # --- transitions + labels ---------------------------------------------
    edges.sort(key=lambda a: (a.tail_state, a.head_state, a.path[0][1], a.path[0][0]))
    transitions: list[Transition] = []
    for arrow in edges:
        source, target = states[arrow.tail_state], states[arrow.head_state]
        transition_id = f"T{len(transitions)}"
        if source.id == target.id:
            direction = "loop"
        elif arrow.direction_known:
            direction = "forward"
        else:
            direction = "unknown"
            uncertainties.append(Uncertainty(
                transition_id, "direction", "; ".join(arrow.notes) or "arrowhead not found",
                (f"{source.id}->{target.id}", f"{target.id}->{source.id}"),
            ))
        if not arrow.confirmed_both_ends and source.id != target.id:
            uncertainties.append(Uncertainty(
                transition_id, "target", "arrow was traced from one end only (crossing or broken stroke)",
            ))
        confidence = min(source.confidence or 1.0, target.confidence or 1.0)
        confidence *= 0.55 + 0.45 * max(arrow.quality, 0.0)
        confidence *= 0.7 + 0.3 * (arrow.direction_confidence if direction != "loop" else 1.0)
        if not arrow.confirmed_both_ends:
            confidence *= 0.85
        step = max(1, len(arrow.path) // 16)
        points = arrow.path[::step]
        if points[-1] != arrow.path[-1]:
            points.append(arrow.path[-1])
        transitions.append(Transition(
            id=transition_id, source=source.id, target=target.id,
            confidence=round(max(0.0, min(1.0, confidence)), 4), direction=direction,  # type: ignore[arg-type]
            points=tuple(Position(round(x, 1), round(y, 1)) for x, y in points),
            provenance="detector",
        ))

    # Assign every text box to the nearest transition path; ambiguity is recorded.
    assignments: dict[int, list[tuple[float, tuple[float, float, float, float]]]] = {}
    for box in label_boxes:
        if any(_iou(box, s.bbox) > 0.3 for s in states if s.bbox):
            continue
        scored = sorted(
            (_box_to_path(box, [(p.x, p.y) for p in t.points]), i) for i, t in enumerate(transitions) if len(t.points) >= 2
        )
        if not scored:
            continue
        best_distance, best_index = scored[0]
        limit = max(45.0, (box[3] - box[1]) * 3)
        if best_distance > limit:
            warnings.append(f"text at {tuple(round(v) for v in box)} is not near any transition")
            continue
        assignments.setdefault(best_index, []).append((best_distance, box))
        if len(scored) > 1 and scored[1][0] < best_distance * 1.25 + 4 and scored[1][0] <= limit:
            uncertainties.append(Uncertainty(
                transitions[best_index].id, "label", "label is about equally close to two transitions",
                (transitions[best_index].id, transitions[scored[1][1]].id),
            ))
    for index, boxes in assignments.items():
        boxes.sort(key=lambda item: (item[1][1], item[1][0]))
        texts, confidences = [], []
        for _, box in boxes:
            result = read_box(gray, box) if use_ocr else None
            if result is not None and result.text:
                texts.append(result.text)
                confidences.append(result.confidence or 0.0)
        union = (min(b[0] for _, b in boxes), min(b[1] for _, b in boxes),
                 max(b[2] for _, b in boxes), max(b[3] for _, b in boxes))
        text = " ".join(texts) or None
        label_confidence = round(sum(confidences) / len(confidences), 3) if confidences else None
        transition = transitions[index]
        if text is None:
            uncertainties.append(Uncertainty(transition.id, "label", "label text could not be read"))
        elif label_confidence is not None and label_confidence < 0.6:
            uncertainties.append(Uncertainty(transition.id, "label", f"OCR confidence {label_confidence:.2f} is low"))
        transitions[index] = replace(
            transition, label=text, label_confidence=label_confidence,
            label_bbox=tuple(round(v, 1) for v in union),  # type: ignore[arg-type]
            provenance="ocr" if text else transition.provenance,
        )

    # Detector arrows that no trace explains are surfaced, never silently dropped.
    detector_arrows = [d for d in detections if d.kind in ARROW_KINDS]
    for arrow in detector_arrows:
        explained = any(
            sum(arrow.bbox[0] - 6 <= x <= arrow.bbox[2] + 6 and arrow.bbox[1] - 6 <= y <= arrow.bbox[3] + 6
                for x, y in t.path) >= 0.6 * len(t.path)
            for t in traced
        )
        if not explained:
            warnings.append(f"detected arrow at {tuple(round(v) for v in arrow.bbox)} was not connected to states")

    sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest() if image_path.is_file() else None
    machine = StateMachine(
        SCHEMA_VERSION, diagram_id, tuple(states), tuple(transitions),
        {"source": "image", "warnings": warnings, "association": "stroke-tracing",
         "stroke_width": round(float(debug.get("stroke_width", 0.0)), 2) if debug else None},
        uncertainties=tuple(dict.fromkeys(uncertainties)),
        source_image=SourceImage(str(image_path), width, height, sha256),
    )
    machine = annotate_machine(machine)
    return Reconstruction(machine, warnings, debug)
