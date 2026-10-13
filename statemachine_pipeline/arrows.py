"""Stroke tracing from state boundaries: arrow endpoints, paths and direction.

This replaces "map each arrow end to the nearest state centre". The pipeline is:

1. State regions (slightly expanded detector boxes) are removed from the ink
   mask, so arrows become free strokes whose ends touch state *boundaries*.
2. A *port* is where ink touches the ring just outside a state region.
3. From every port, a direction-following tracer walks along the stroke. It
   prefers going straight, so it passes through crossing edges instead of
   turning onto them, follows curves, and bridges small gaps (labels or
   hand-drawn breaks). The trace ends when it enters a state region (an edge),
   returns to its own state (a self-loop), or runs out of ink (a free end,
   e.g. an initial-state marker).
4. Each arrow is traced from both ends; agreeing traces are merged and
   disagreements are reported as uncertainty rather than resolved silently.
5. Direction comes from arrowhead detections when the detector has that
   class, otherwise from local ink mass (arrowheads add ink near the tip).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, float] = {
    "region_pad_ratio": 0.10,      # state region expansion, relative to radius
    "region_pad_min": 3.0,         # ... plus at least this many pixels
    "port_ring": 3.0,              # ring width (px) where ports are sampled
    "step": 2.0,                   # tracer step (px)
    "max_turn_deg": 50.0,          # tracer search half-angle per step
    "straight_bias": 0.004,        # score penalty per degree of turning
    "min_ink_fraction": 0.5,       # look-ahead ray must be this inky
    "bridge_gap": 14.0,            # max gap (px, plus stroke width) bridged
    "endpoint_tolerance": 14.0,    # px: two traces describe the same arrow
    "head_ratio": 2.2,             # head cross-section / shaft width ratio for an arrowhead
    "direction_margin": 0.25,      # min relative headness difference
    "min_free_length": 18.0,       # px: shorter free traces are noise / wings
    "overlap_reject": 0.4,         # drop candidates whose path is this covered by accepted arrows
}

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "association.json"


def load_config(path: str | Path | None = None) -> dict[str, float]:
    """Defaults overridden by ``config/association.json`` (written by ``main.py tune``)."""
    config = dict(DEFAULT_CONFIG)
    target = Path(path) if path else _CONFIG_PATH
    if target.is_file():
        try:
            stored = json.loads(target.read_text(encoding="utf-8"))
            config.update({k: float(v) for k, v in stored.items() if k in DEFAULT_CONFIG})
        except (OSError, ValueError, TypeError):
            pass
    return config


@dataclass(frozen=True)
class StateRegion:
    index: int
    bbox: tuple[float, float, float, float]
    shape: str  # "circle" or "rounded_rect"

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def radius(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(x2 - x1, y2 - y1) / 2


@dataclass(frozen=True)
class Port:
    state: int
    point: tuple[float, float]
    width: float


@dataclass
class Trace:
    start_state: int
    start: tuple[float, float]
    path: list[tuple[float, float]]
    end_state: int | None  # None: free end
    bridged: int = 0
    quality: float = 1.0

    @property
    def end(self) -> tuple[float, float]:
        return self.path[-1]

    @property
    def length(self) -> float:
        return sum(math.dist(a, b) for a, b in zip(self.path, self.path[1:]))


@dataclass
class TracedArrow:
    """One physical arrow, endpoints ordered tail -> head when direction is known."""
    tail_state: int | None
    head_state: int | None
    path: list[tuple[float, float]]
    direction_confidence: float
    direction_known: bool
    confirmed_both_ends: bool
    quality: float
    is_initial_marker: bool = False
    notes: list[str] = field(default_factory=list)
    head_evidence: tuple[float, float] = (0.0, 0.0)


def _ring_pad(mask: Any, region: StateRegion, config: dict[str, float]) -> float:
    """Distance beyond the box edge where the state's outline ink ends.

    Detector boxes are rarely pixel-exact and outlines can be thick or blurred;
    stopping the region at the box edge would leave outline fragments that look
    like arrow ink. The outline ends where the ink fraction around the
    (expanded) outline drops.
    """
    h, w = mask.shape
    x1, y1, x2, y2 = region.bbox
    cx, cy = region.center
    a, b = (x2 - x1) / 2, (y2 - y1) / 2
    minimum = max(config["region_pad_min"], region.radius * config["region_pad_ratio"])
    limit = max(minimum + 2, region.radius * 0.35)
    samples = 120
    k = -0.1 * region.radius
    seen_ring = False
    while k <= limit:
        hits = 0
        for i in range(samples):
            theta = 2 * math.pi * i / samples
            dx, dy = math.cos(theta), math.sin(theta)
            if region.shape == "rounded_rect":
                scale = min((a + k) / abs(dx) if dx else 1e9, (b + k) / abs(dy) if dy else 1e9)
                px, py = cx + dx * scale, cy + dy * scale
            else:
                px, py = cx + (a + k) * dx, cy + (b + k) * dy
            ix, iy = int(round(px)), int(round(py))
            if 0 <= ix < w and 0 <= iy < h and mask[iy, ix]:
                hits += 1
        fraction = hits / samples
        if fraction >= 0.45:
            seen_ring = True
        elif seen_ring and fraction < 0.2:
            return max(minimum, k + 2.0)
        k += 1.0
    return minimum if not seen_ring else limit


def build_region_map(shape: tuple[int, int], regions: list[StateRegion], config: dict[str, float],
                     mask: Any = None) -> Any:
    """int16 map: -1 background, otherwise the index of the (expanded) state region."""
    import cv2
    import numpy as np

    region_map = np.full(shape, -1, np.int16)
    for region in sorted(regions, key=lambda r: -r.radius):
        x1, y1, x2, y2 = region.bbox
        if mask is not None:
            pad = _ring_pad(mask, region, config)
        else:
            pad = max(config["region_pad_min"], region.radius * config["region_pad_ratio"])
        canvas = np.zeros(shape, np.uint8)
        if region.shape == "rounded_rect":
            cv2.rectangle(canvas, (int(x1 - pad), int(y1 - pad)), (int(math.ceil(x2 + pad)), int(math.ceil(y2 + pad))), 1, -1)
        else:
            cx, cy = region.center
            axes = (int(math.ceil((x2 - x1) / 2 + pad)), int(math.ceil((y2 - y1) / 2 + pad)))
            cv2.ellipse(canvas, (int(round(cx)), int(round(cy))), axes, 0, 0, 360, 1, -1)
        region_map[canvas > 0] = region.index
    return region_map


def find_ports(mask: Any, region_map: Any, regions: list[StateRegion], config: dict[str, float]) -> list[Port]:
    import cv2
    import numpy as np

    ports: list[Port] = []
    ring_width = max(2, int(config["port_ring"]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_width + 1, 2 * ring_width + 1))
    background = region_map < 0
    for region in regions:
        inside = (region_map == region.index).astype(np.uint8)
        ring = (cv2.dilate(inside, kernel) > 0) & background
        touching = (ring & (mask > 0)).astype(np.uint8)
        count, _, stats, centroids = cv2.connectedComponentsWithStats(touching, connectivity=8)
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] < 2:
                continue
            width = float(max(stats[label, cv2.CC_STAT_WIDTH], stats[label, cv2.CC_STAT_HEIGHT]))
            ports.append(Port(region.index, (float(centroids[label][0]), float(centroids[label][1])), width))
    return ports


def _ink_fraction(mask: Any, x: float, y: float, angle: float, start: float, length: float) -> float:
    h, w = mask.shape
    hits = total = 0
    dx, dy = math.cos(angle), math.sin(angle)
    distance = start
    while distance <= start + length:
        px, py = int(round(x + dx * distance)), int(round(y + dy * distance))
        total += 1
        if 0 <= px < w and 0 <= py < h and mask[py, px]:
            hits += 1
        distance += 1.0
    return hits / total if total else 0.0


def _recenter(mask: Any, x: float, y: float, angle: float, half: int, max_width: float) -> tuple[float, float]:
    """Shift the point to the centre of the ink run across the stroke."""
    h, w = mask.shape
    nx, ny = -math.sin(angle), math.cos(angle)
    inside = []
    for offset in range(-half, half + 1):
        px, py = int(round(x + nx * offset)), int(round(y + ny * offset))
        inside.append(0 <= px < w and 0 <= py < h and bool(mask[py, px]))
    centre = half
    if not inside[centre]:
        return x, y
    low = high = centre
    while low > 0 and inside[low - 1]:
        low -= 1
    while high < len(inside) - 1 and inside[high + 1]:
        high += 1
    if high - low + 1 > max_width:
        return x, y  # junction, arrowhead or blob: do not snap
    shift = (low + high) / 2 - half
    return x + nx * shift, y + ny * shift


def trace_from(mask: Any, region_map: Any, regions: list[StateRegion], start: tuple[float, float],
               angle: float, source: int, stroke: float, config: dict[str, float],
               max_length: float | None = None, excluded: Any = None) -> Trace:
    h, w = mask.shape
    x, y = start
    path = [(x, y)]
    step = config["step"]
    lookahead = max(6.0, 2.0 * stroke)
    max_turn = math.radians(config["max_turn_deg"])
    bias = config["straight_bias"]
    gap = config["bridge_gap"] + stroke
    max_length = max_length or 4.0 * (h + w)
    source_region = regions[source] if 0 <= source < len(regions) else None
    travelled = 0.0
    bridged = 0
    weak_steps = 0
    angles = [math.radians(d) for d in range(0, int(math.degrees(max_turn)) + 1, 5)]
    offsets = sorted({0.0, *angles, *(-a for a in angles)}, key=abs)
    while travelled < max_length:
        best = None
        for offset in offsets:
            candidate = angle + offset
            fraction = _ink_fraction(mask, x, y, candidate, 1.0, lookahead)
            if fraction < config["min_ink_fraction"]:
                continue
            score = fraction - bias * abs(math.degrees(offset))
            if best is None or score > best[0]:
                best = (score, candidate, fraction)
        if best is None:
            # Bridge a gap (broken stroke, label or crossing removed from the mask).
            jumped = False
            for offset in (0.0, 0.1, -0.1, 0.2, -0.2):
                candidate = angle + offset
                distance = lookahead
                budget = lookahead + gap
                while distance <= budget:
                    px, py = x + math.cos(candidate) * distance, y + math.sin(candidate) * distance
                    ix, iy = int(round(px)), int(round(py))
                    if not (0 <= ix < w and 0 <= iy < h):
                        break
                    if excluded is not None and excluded[iy, ix] and budget < lookahead + gap + 160:
                        budget += 1.0  # crossing a blanked text box does not use the gap budget
                    if region_map[iy, ix] >= 0 and (region_map[iy, ix] != source or travelled > 2 * stroke + 6):
                        x, y = px, py
                        path.append((x, y))
                        travelled += distance
                        jumped = True
                        break
                    if mask[iy, ix] and _ink_fraction(mask, px, py, candidate, 0.0, 4.0) >= 0.75:
                        x, y = px, py
                        path.append((x, y))
                        travelled += distance
                        bridged += 1
                        jumped = True
                        break
                    distance += 1.0
                if jumped:
                    break
            if not jumped:
                # Arrowhead blobs often stall the tracer just short of the target
                # outline; snap to a state region within reach of the stall point.
                snapped = _nearby_region(region_map, x, y, stroke * 1.5 + 8)
                quality = 1.0 - weak_steps / max(1, len(path))
                if snapped is not None and snapped != source:
                    return Trace(source, start, path, snapped, bridged, quality * 0.9)
                if snapped == source and source_region is not None and travelled > source_region.radius and \
                        max(math.dist(p, source_region.center) for p in path) > source_region.radius * 1.35 + 6:
                    return Trace(source, start, path, source, bridged, quality * 0.9)
                return Trace(source, start, path, None, bridged, quality)
        else:
            _, chosen, fraction = best
            if fraction < 0.8:
                weak_steps += 1
            x += math.cos(chosen) * step
            y += math.sin(chosen) * step
            travelled += step
            # Smooth the heading so single noisy pixels do not steer the trace.
            angle = math.atan2(0.65 * math.sin(angle) + 0.35 * math.sin(chosen),
                               0.65 * math.cos(angle) + 0.35 * math.cos(chosen))
            x, y = _recenter(mask, x, y, angle, int(stroke + 4), 2.2 * stroke + 2)
            path.append((x, y))
        ix, iy = int(round(x)), int(round(y))
        if not (0 <= ix < w and 0 <= iy < h):
            return Trace(source, start, path, None, bridged, 0.5)
        region = int(region_map[iy, ix])
        if region >= 0:
            if region != source:
                return Trace(source, start, path, region, bridged, 1.0 - weak_steps / max(1, len(path)))
            # Re-entering the source is a self-loop only after leaving its vicinity.
            assert source_region is not None
            furthest = max(math.dist(p, source_region.center) for p in path)
            if furthest > source_region.radius * 1.35 + 6 and travelled > source_region.radius:
                return Trace(source, start, path, source, bridged, 1.0 - weak_steps / max(1, len(path)))
            return Trace(source, start, path, None, bridged, 0.0)
    return Trace(source, start, path, None, bridged, 0.0)


def _nearby_region(region_map: Any, x: float, y: float, radius: float) -> int | None:
    h, w = region_map.shape
    best = None
    r = int(math.ceil(radius))
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            distance = math.hypot(dx, dy)
            if distance > radius:
                continue
            px, py = int(round(x)) + dx, int(round(y)) + dy
            if 0 <= px < w and 0 <= py < h and region_map[py, px] >= 0:
                if best is None or distance < best[0]:
                    best = (distance, int(region_map[py, px]))
    return best[1] if best else None


def _exit_directions(mask: Any, region_map: Any, port: Port, region: StateRegion, radius: float) -> list[float]:
    """Directions of ink leaving the port, found as ink runs on a small circle."""
    h, w = mask.shape
    samples = 72
    inks = []
    for k in range(samples):
        theta = 2 * math.pi * k / samples
        px = int(round(port.point[0] + radius * math.cos(theta)))
        py = int(round(port.point[1] + radius * math.sin(theta)))
        inks.append(0 <= px < w and 0 <= py < h and bool(mask[py, px]) and region_map[py, px] < 0)
    if all(inks):
        return [math.atan2(port.point[1] - region.center[1], port.point[0] - region.center[0])]
    runs: list[list[int]] = []
    start = inks.index(False)
    current: list[int] = []
    for offset in range(1, samples + 1):
        k = (start + offset) % samples
        if inks[k]:
            current.append(k)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    directions = []
    for run in runs:
        mid = run[len(run) // 2]
        directions.append(2 * math.pi * mid / samples)
    outward = math.atan2(port.point[1] - region.center[1], port.point[0] - region.center[0])
    if not directions:
        directions = [outward]
    # Most outward-pointing first; ties broken deterministically.
    directions.sort(key=lambda a: abs(math.atan2(math.sin(a - outward), math.cos(a - outward))))
    return directions


def _span(mask: Any, region_map: Any, point: tuple[float, float], angle: float, half: int) -> float:
    """Extent of ink across the path (outermost ink pixels along the normal)."""
    h, w = mask.shape
    nx, ny = -math.sin(angle), math.cos(angle)
    hits = []
    for offset in range(-half, half + 1):
        px, py = int(round(point[0] + nx * offset)), int(round(point[1] + ny * offset))
        if 0 <= px < w and 0 <= py < h and mask[py, px] and region_map[py, px] < 0:
            hits.append(offset)
    return float(max(hits) - min(hits) + 1) if hits else 0.0


def _arc_samples(path: list[tuple[float, float]], from_end: bool, start: float, stop: float, spacing: float = 1.5):
    """Points (with local heading) along the path between two arc lengths from one end."""
    sequence = path[::-1] if from_end else path
    out = []
    travelled = 0.0
    next_mark = start
    for a, b in zip(sequence, sequence[1:]):
        segment = math.dist(a, b)
        if segment == 0:
            continue
        while travelled + segment >= next_mark and next_mark <= stop:
            t = (next_mark - travelled) / segment
            out.append(((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t), math.atan2(b[1] - a[1], b[0] - a[0])))
            next_mark += spacing
        travelled += segment
        if next_mark > stop:
            break
    return out


def shaft_span(mask: Any, region_map: Any, path: list[tuple[float, float]], stroke: float) -> float:
    """Median cross-section width of the middle of the path (the plain shaft)."""
    import numpy as np

    length = sum(math.dist(a, b) for a, b in zip(path, path[1:]))
    margin = min(length * 0.3, 4.0 * stroke + 24.0)
    samples = _arc_samples(path, False, margin, length - margin, 3.0)
    spans = [_span(mask, region_map, p, a, int(stroke + 4)) for p, a in samples]
    spans = [s for s in spans if s > 0]
    return float(np.median(spans)) if spans else max(stroke, 1.0)


def headness(mask: Any, region_map: Any, path: list[tuple[float, float]], from_end: bool, stroke: float,
             shaft: float | None = None) -> float:
    """How arrowhead-like one end of the path is: max cross-section width near
    that end relative to the shaft width (≈1 for a plain end, ≥2 for a head)."""
    shaft = shaft or shaft_span(mask, region_map, path, stroke)
    reach = 3.0 * stroke + 22.0
    half = int(1.5 * stroke + 14)
    spans = sorted(
        (_span(mask, region_map, p, a, half) for p, a in _arc_samples(path, from_end, 0.0, reach)),
        reverse=True,
    )
    if not spans:
        return 0.0
    robust = spans[1] if len(spans) > 2 else spans[0]  # ignore a single noisy cross-section
    return robust / max(shaft, 1.0)


def _point_near_box(point: tuple[float, float], box: tuple[float, float, float, float], slack: float) -> bool:
    return box[0] - slack <= point[0] <= box[2] + slack and box[1] - slack <= point[1] <= box[3] + slack


def _path_toward(path: list[tuple[float, float]], from_end: bool, distance: float) -> tuple[float, float]:
    sequence = path[::-1] if from_end else path
    travelled = 0.0
    for a, b in zip(sequence, sequence[1:]):
        travelled += math.dist(a, b)
        if travelled >= distance:
            return b
    return sequence[-1]


def trace_arrows(
    mask: Any,
    regions: list[StateRegion],
    *,
    arrowheads: list[tuple[tuple[float, float, float, float], float]] = (),  # type: ignore[assignment]
    initial_markers: list[tuple[tuple[float, float, float, float], float]] = (),  # type: ignore[assignment]
    exclude_boxes: list[tuple[float, float, float, float]] = (),  # type: ignore[assignment]
    config: dict[str, float] | None = None,
) -> tuple[list[TracedArrow], dict[str, Any]]:
    """Trace every arrow touching a state. Returns arrows and debug info.

    ``exclude_boxes`` (typically text-label boxes) are blanked before tracing so
    text next to a stroke is not mistaken for an arrowhead; the tracer's gap
    bridging carries strokes across them.
    """
    import numpy as np
    from .preprocess import stroke_width

    config = config or load_config()
    region_map = build_region_map(mask.shape, regions, config, mask)
    free_mask = (mask > 0) & (region_map < 0)
    excluded = np.zeros(mask.shape, bool)
    import cv2
    for x1, y1, x2, y2 in exclude_boxes:
        # Remove only glyphs fully inside the label box. Strokes passing through
        # or touching the box reach its border and are kept.
        bx1, by1 = max(0, int(x1) - 1), max(0, int(y1) - 1)
        bx2, by2 = min(mask.shape[1], int(math.ceil(x2)) + 2), min(mask.shape[0], int(math.ceil(y2)) + 2)
        if bx2 - bx1 < 3 or by2 - by1 < 3:
            continue
        crop = free_mask[by1:by2, bx1:bx2].astype(np.uint8)
        count, labels = cv2.connectedComponents(crop, connectivity=8)
        border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))) - {0}
        glyphs = (labels > 0) & ~np.isin(labels, list(border))
        excluded[by1:by2, bx1:bx2] |= glyphs
    free_mask &= ~excluded
    stroke = stroke_width(free_mask.astype(np.uint8))
    ports = find_ports(free_mask.astype(np.uint8) | (mask & (region_map >= 0)).astype(np.uint8), region_map, regions, config)
    traces: list[Trace] = []
    probe = 2.0 * stroke + 6.0
    for port in ports:
        region = regions[port.state]
        found: list[Trace] = []
        for direction in _exit_directions(free_mask, region_map, port, region, probe)[:4]:
            trace = trace_from(free_mask, region_map, regions, port.point, direction, port.state, stroke, config,
                               excluded=excluded)
            if trace.end_state is None and trace.length < config["min_free_length"] + 2 * stroke:
                continue  # arrowhead wing or noise
            found.append(trace)
        # Prefer traces that reach a state; keep distinct ones (merged ports of two arrows).
        reaching = [t for t in found if t.end_state is not None]
        candidates = reaching or sorted(found, key=lambda t: -t.length)[:1]
        kept: list[Trace] = []
        for trace in candidates:
            if all(math.dist(trace.end, other.end) > config["endpoint_tolerance"] for other in kept):
                kept.append(trace)
        traces.extend(kept)

    tolerance = config["endpoint_tolerance"] + stroke

    def port_of(state: int | None, point: tuple[float, float]) -> int | None:
        if state is None:
            return None
        best = None
        for index, port in enumerate(ports):
            if port.state != state:
                continue
            distance = math.dist(port.point, point)
            if distance <= tolerance + port.width / 2 and (best is None or distance < best[0]):
                best = (distance, index)
        return best[1] if best else None

    # Merge traces of the same physical arrow (found from both ends).
    candidates_out: list[tuple[TracedArrow, tuple[int | None, int | None]]] = []
    used = [False] * len(traces)
    for i, trace in enumerate(traces):
        if used[i]:
            continue
        used[i] = True
        partner = None
        if trace.end_state is not None:
            for j in range(len(traces)):
                other = traces[j]
                if used[j] or other.start_state != trace.end_state or other.end_state != trace.start_state:
                    continue
                if math.dist(other.start, trace.end) <= tolerance * 1.5 and math.dist(other.end, trace.start) <= tolerance * 1.5:
                    partner = j
                    break
        if partner is not None:
            used[partner] = True
        arrow = _orient(trace, partner is not None, traces[partner] if partner is not None else None,
                        free_mask, region_map, stroke, arrowheads, initial_markers, config)
        ends = (port_of(trace.start_state, trace.start), port_of(trace.end_state, trace.end))
        candidates_out.append((arrow, ends))

    # Keep the most trustworthy arrows first; a candidate whose path mostly runs
    # over already-accepted arrows is a duplicate or a crossing mis-trace.
    ordered = sorted(candidates_out, key=lambda item: (
        not item[0].confirmed_both_ends, item[0].is_initial_marker, -item[0].quality,
        sum(math.dist(a, b) for a, b in zip(item[0].path, item[0].path[1:])),
    ))
    coverage = np.zeros(mask.shape, np.uint8)
    radius = int(math.ceil(stroke / 2 + 2))
    arrows: list[TracedArrow] = []
    for arrow, _ in ordered:
        points = arrow.path[1:-1] or arrow.path
        covered = sum(
            1 for x, y in points
            if 0 <= int(y) < coverage.shape[0] and 0 <= int(x) < coverage.shape[1] and coverage[int(y), int(x)]
        )
        if covered / len(points) > config["overlap_reject"]:
            continue
        arrows.append(arrow)
        import cv2
        cv2.polylines(coverage, [np.array(arrow.path, np.int32)], False, 1, 2 * radius + 1)
    debug = {"stroke_width": stroke, "ports": [p.__dict__ for p in ports], "traces": len(traces),
             "region_map": region_map}
    return arrows, debug


def _same_arrow(a: TracedArrow, b: TracedArrow, tolerance: float) -> bool:
    ends_a = {a.path[0], a.path[-1]}
    ends_b = [b.path[0], b.path[-1]]
    return all(any(math.dist(p, q) <= tolerance for q in ends_a) for p in ends_b) and \
        {a.tail_state, a.head_state} == {b.tail_state, b.head_state}


def _orient(trace: Trace, confirmed: bool, partner: Trace | None, mask: Any, region_map: Any, stroke: float,
            arrowheads, initial_markers, config: dict[str, float]) -> TracedArrow:
    path = trace.path
    start, end = path[0], path[-1]
    quality = trace.quality if partner is None else (trace.quality + partner.quality) / 2
    shaft = shaft_span(mask, region_map, path, stroke)
    h_start = headness(mask, region_map, path, False, stroke, shaft)
    h_end = headness(mask, region_map, path, True, stroke, shaft)
    notes: list[str] = []
    # Detector arrowheads are strong evidence when present.
    slack = stroke + 4
    reach = 3.0 * stroke + 22.0
    start_points = [p for p, _ in _arc_samples(path, False, 0.0, reach)] or [start]
    end_points = [p for p, _ in _arc_samples(path, True, 0.0, reach)] or [end]

    def detected(points: list[tuple[float, float]]) -> float:
        # A detected arrowhead belongs to this end only if this arrow's own end
        # segment runs through it (adjacent arrows of a bidirectional pair do not).
        best = 0.0
        for box, conf in arrowheads:
            inside = sum(_point_near_box(p, box, 2.0) for p in points)
            if inside >= 2:
                best = max(best, conf)
        return best

    det_start = detected(start_points)
    det_end = detected(end_points)
    score_start = h_start / config["head_ratio"] + 1.5 * det_start
    score_end = h_end / config["head_ratio"] + 1.5 * det_end

    if trace.end_state is None:
        # Free end: an initial marker if the head is at the state end.
        marker_hit = max((conf for box, conf in initial_markers if _point_near_box(start, box, slack)
                          or _point_near_box(end, box, slack)), default=0.0)
        head_at_state = score_start >= 1.0 or marker_hit > 0
        if head_at_state:
            return TracedArrow(None, trace.start_state, path[::-1],
                               min(1.0, max(score_start, marker_hit)), True, False, quality,
                               is_initial_marker=True, head_evidence=(h_end, h_start))
        notes.append("arrow leaves a state and ends in empty space (missing state or unrecognised marker)")
        return TracedArrow(trace.start_state, None, path, min(1.0, score_end / 2), score_end >= 1.0, False,
                           quality, notes=notes, head_evidence=(h_start, h_end))

    total = score_start + score_end
    difference = abs(score_end - score_start) / total if total else 0.0
    if trace.end_state == trace.start_state:
        # Self-loop: source == target, direction along the loop only matters for drawing.
        head_at_end = score_end >= score_start
        oriented = path if head_at_end else path[::-1]
        return TracedArrow(trace.start_state, trace.start_state, oriented, max(difference, 0.5), True,
                           confirmed, quality, notes=notes, head_evidence=(h_start, h_end))
    known = difference >= config["direction_margin"] and max(score_start, score_end) >= 1.0
    if not known:
        notes.append(f"arrowhead evidence is ambiguous (start={score_start:.2f}, end={score_end:.2f})")
    if score_end >= score_start:
        tail, head, oriented = trace.start_state, trace.end_state, path
    else:
        tail, head, oriented = trace.end_state, trace.start_state, path[::-1]
    if not confirmed:
        notes.append("arrow was traced from one end only; the path may be confused by a crossing")
    return TracedArrow(tail, head, oriented, min(1.0, difference), known, confirmed, quality,
                       notes=notes, head_evidence=(h_start, h_end))
