"""Synthetic state-diagram renderer with exact ground truth.

Each sample yields an image, YOLO labels and the canonical JSON the image was
drawn from, so detection *and* graph reconstruction can be evaluated.
Covered cases: labelled circle / rounded-rect states, double-circle finals,
initial markers (plain arrow or UML dot), straight / curved / bidirectional /
self-loop transitions, crossing edges, ``event [guard] / action`` labels,
network-protocol vocabularies, line-width jitter, blur and noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from pathlib import Path
from typing import Any

from .models import SCHEMA_VERSION, Position, State, StateMachine, Transition
from .semantics import annotate_machine

YOLO_CLASSES = ("state", "arrow", "final_state", "arrowhead", "initial_marker", "label")
CLASS_ID = {name: index for index, name in enumerate(YOLO_CLASSES)}

GENERIC_STATES = ("Idle", "Ready", "Running", "Paused", "Done", "Error", "Init", "Wait",
                  "Open", "Closed", "Locked", "Active", "Busy", "Stopped", "Start", "End")
NETWORK_STATES = ("Disconnected", "Connecting", "Connected", "Listen", "SynSent", "SynRcvd",
                  "Established", "Retry", "Closing", "TimeWait", "Failed", "Sending", "Waiting",
                  "Recovery", "Offline", "Online")
GENERIC_EVENTS = ("start", "stop", "pause", "resume", "reset", "done", "go", "next", "tick",
                  "submit", "cancel", "finish", "enable", "disable", "lock", "unlock")
NETWORK_EVENTS = ("connect", "timeout", "SYN", "SYN-ACK", "ACK", "FIN", "send", "recv",
                  "disconnect", "retry", "linkDown", "linkUp", "error", "data", "close", "accept")
GUARDS = ("ok", "n<3", "valid", "retries<5", "ready", "auth", "x>0", "!busy")
ACTIONS = ("log", "send ACK", "init()", "save", "notify", "inc n", "reset()", "close()")


@dataclass
class Sample:
    image: Any  # numpy BGR array
    labels: list[tuple[int, float, float, float, float]]  # class, x1, y1, x2, y2 (pixels)
    machine: StateMachine
    meta: dict[str, Any] = field(default_factory=dict)


def _font(size: int):
    from PIL import ImageFont
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _bezier(p0, p1, p2, steps: int = 40) -> list[tuple[float, float]]:
    return [
        ((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t * t * p2[0],
         (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t * t * p2[1])
        for t in (i / steps for i in range(steps + 1))
    ]


def _cubic(p0, p1, p2, p3, steps: int = 40) -> list[tuple[float, float]]:
    out = []
    for i in range(steps + 1):
        t = i / steps
        a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t ** 3
        out.append((a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
                    a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]))
    return out


@dataclass
class _Node:
    x: float
    y: float
    r: float  # circle radius, or half-size for rounded rects
    shape: str
    label: str | None
    kind: str  # normal / initial / final

    def boundary(self, angle: float) -> tuple[float, float]:
        if self.shape == "rounded_rect":
            # half-width 1.35r, half-height 0.8r
            hw, hh = self.r * 1.35, self.r * 0.8
            dx, dy = math.cos(angle), math.sin(angle)
            scale = min(hw / abs(dx) if dx else 1e9, hh / abs(dy) if dy else 1e9)
            return self.x + dx * scale, self.y + dy * scale
        return self.x + self.r * math.cos(angle), self.y + self.r * math.sin(angle)

    def bbox(self) -> tuple[float, float, float, float]:
        if self.shape == "rounded_rect":
            return (self.x - self.r * 1.35, self.y - self.r * 0.8,
                    self.x + self.r * 1.35, self.y + self.r * 0.8)
        return (self.x - self.r, self.y - self.r, self.x + self.r, self.y + self.r)

    def contains(self, point, margin: float = 0.0) -> bool:
        x1, y1, x2, y2 = self.bbox()
        return x1 - margin <= point[0] <= x2 + margin and y1 - margin <= point[1] <= y2 + margin


def _boxes_overlap(a, b, margin: float = 0.0) -> bool:
    return not (a[2] + margin < b[0] or b[2] + margin < a[0] or a[3] + margin < b[1] or b[3] + margin < a[1])


def _path_bbox(points) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def render_sample(rng: random.Random, *, size: int = 640, network_bias: float = 0.5) -> Sample | None:
    """Render one diagram. Returns ``None`` when a layout attempt fails (caller retries)."""
    import cv2
    import numpy as np
    from PIL import Image, ImageDraw

    network = rng.random() < network_bias
    state_names = list(NETWORK_STATES if network else GENERIC_STATES)
    event_names = list(NETWORK_EVENTS if network else GENERIC_EVENTS)
    rng.shuffle(state_names)

    count = rng.randint(2, 5)
    min_distance = 175 if count <= 4 else 155
    thickness = rng.choice((2, 2, 3, 3, 4))
    initial_index = rng.randrange(count)
    final_index = None
    if count >= 3 and rng.random() < 0.6:
        final_index = rng.choice([i for i in range(count) if i != initial_index])
    measure = ImageDraw.Draw(Image.new("L", (1, 1)))
    nodes: list[_Node] = []
    attempts = 0
    while len(nodes) < count and attempts < 400:
        attempts += 1
        index = len(nodes)
        kind = "initial" if index == initial_index else "final" if index == final_index else "normal"
        label = state_names[index] if rng.random() > 0.05 else None
        shape = "rounded_rect" if kind != "final" and rng.random() < 0.2 else "circle"
        r = rng.randint(30, 46)
        if label is not None:
            # Grow the state so its name fits at a legible size (>= 13px).
            left, _, right, _ = measure.textbbox((0, 0), label, font=_font(13))
            needed = (right - left) / 2 + 5 + thickness + (7 + thickness if kind == "final" else 0)
            r = max(r, math.ceil(needed / 1.3) if shape == "rounded_rect" else math.ceil(needed))
        margin = r * 1.35 + 40
        x = rng.uniform(margin, size - margin)
        y = rng.uniform(margin, size - margin)
        if all(math.hypot(x - n.x, y - n.y) >= min_distance for n in nodes):
            nodes.append(_Node(x, y, r, shape, label, kind))
    if len(nodes) < count:
        return None
    ink = rng.randint(0, 60)
    background = rng.randint(225, 255)
    image = np.full((size, size, 3), background, np.uint8)
    color = (ink, ink, ink)
    labels: list[tuple[int, float, float, float, float]] = []
    occupied: list[tuple[float, float, float, float]] = [n.bbox() for n in nodes]

    def draw_polyline(points):
        pts = np.array([[round(p[0]), round(p[1])] for p in points], np.int32)
        cv2.polylines(image, [pts], False, color, thickness, cv2.LINE_AA)

    def draw_head(tip, before) -> tuple[float, float, float, float]:
        angle = math.atan2(tip[1] - before[1], tip[0] - before[0])
        length = rng.uniform(11, 17) + thickness
        spread = math.radians(rng.uniform(22, 30))
        left = (tip[0] - length * math.cos(angle - spread), tip[1] - length * math.sin(angle - spread))
        right = (tip[0] - length * math.cos(angle + spread), tip[1] - length * math.sin(angle + spread))
        if rng.random() < 0.5:
            triangle = np.array([tip, left, right], np.float32).round().astype(np.int32)
            cv2.fillPoly(image, [triangle], color, cv2.LINE_AA)
        else:
            for wing in (left, right):
                cv2.line(image, (round(tip[0]), round(tip[1])), (round(wing[0]), round(wing[1])),
                         color, thickness, cv2.LINE_AA)
        pad = thickness / 2 + 1
        return (min(tip[0], left[0], right[0]) - pad, min(tip[1], left[1], right[1]) - pad,
                max(tip[0], left[0], right[0]) + pad, max(tip[1], left[1], right[1]) + pad)

    def path_is_clear(points, endpoints: tuple[int, ...]) -> bool:
        for index, node in enumerate(nodes):
            if index in endpoints:
                continue
            if any(node.contains(p, margin=8) for p in points):
                return False
        return all(0 <= p[0] < size and 0 <= p[1] < size for p in points)

    # --- states -----------------------------------------------------------
    for node in nodes:
        if node.shape == "rounded_rect":
            x1, y1, x2, y2 = (round(v) for v in node.bbox())
            radius = 12
            cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
            cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
            cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
            cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
            for cx, cy, start in ((x1 + radius, y1 + radius, 180), (x2 - radius, y1 + radius, 270),
                                  (x2 - radius, y2 - radius, 0), (x1 + radius, y2 - radius, 90)):
                cv2.ellipse(image, (cx, cy), (radius, radius), 0, start, start + 90, color, thickness, cv2.LINE_AA)
        else:
            cv2.circle(image, (round(node.x), round(node.y)), round(node.r), color, thickness, cv2.LINE_AA)
            if node.kind == "final":
                cv2.circle(image, (round(node.x), round(node.y)), round(node.r - 6 - thickness),
                           color, max(1, thickness - 1), cv2.LINE_AA)
        labels.append((CLASS_ID["final_state" if node.kind == "final" else "state"], *node.bbox()))

    # --- initial marker -----------------------------------------------------
    initial = nodes[initial_index]
    for _ in range(30):
        angle = rng.uniform(0, 2 * math.pi)
        tip = initial.boundary(angle)
        length = rng.uniform(45, 75)
        start = (tip[0] + length * math.cos(angle), tip[1] + length * math.sin(angle))
        points = [start, tip]
        segment = [(start[0] + (tip[0] - start[0]) * t / 10, start[1] + (tip[1] - start[1]) * t / 10) for t in range(11)]
        if path_is_clear(segment, (initial_index,)) and 12 <= start[0] < size - 12 and 12 <= start[1] < size - 12:
            break
    else:
        return None
    dot = rng.random() < 0.4
    if dot:
        cv2.circle(image, (round(start[0]), round(start[1])), 6 + thickness // 2, color, -1, cv2.LINE_AA)
    draw_polyline(points)
    head = draw_head(tip, start)
    marker_box = _path_bbox(points)
    marker_box = (min(marker_box[0], head[0]) - (8 if dot else 2), min(marker_box[1], head[1]) - (8 if dot else 2),
                  max(marker_box[2], head[2]) + (8 if dot else 2), max(marker_box[3], head[3]) + (8 if dot else 2))
    labels.append((CLASS_ID["initial_marker"], *marker_box))
    occupied.append(marker_box)

    # --- transitions ------------------------------------------------------
    pairs: list[tuple[int, int]] = [(i, i + 1) for i in range(count - 1)]
    rng.shuffle(pairs)
    pairs = [(b, a) if rng.random() < 0.3 else (a, b) for a, b in pairs]
    for _ in range(rng.randint(0, count)):
        a, b = rng.randrange(count), rng.randrange(count)
        if a == b and rng.random() < 0.5:
            continue
        pairs.append((a, b))
    if rng.random() < 0.45:
        looped = rng.randrange(count)
        pairs.append((looped, looped))
    final_ids = {i for i, n in enumerate(nodes) if n.kind == "final"}
    pairs = [(a, b) for a, b in pairs if a not in final_ids]
    for final in final_ids:
        if not any(b == final for _, b in pairs):
            pairs.append((rng.choice([i for i in range(count) if i != final]), final))
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[int, int]] = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            unique.append(pair)
    pairs = unique

    pil_jobs: list[tuple[tuple[float, float], str, Any]] = []
    transitions: list[Transition] = []
    loop_angles: dict[int, list[float]] = {}

    def place_label(points, text: str, font) -> tuple[tuple[float, float], tuple[float, float, float, float]] | None:
        from PIL import ImageDraw as _D
        probe = _D.Draw(Image.new("L", (1, 1)))
        left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
        width, height = right - left, bottom - top
        mid = points[len(points) // 2]
        before = points[max(0, len(points) // 2 - 2)]
        after = points[min(len(points) - 1, len(points) // 2 + 2)]
        dx, dy = after[0] - before[0], after[1] - before[1]
        norm = math.hypot(dx, dy) or 1
        normal = (-dy / norm, dx / norm)
        for side in (1, -1):
            for gap in (6, 12, 20):
                offset = gap + (abs(normal[0]) * width + abs(normal[1]) * height) / 2
                cx, cy = mid[0] + normal[0] * offset * side, mid[1] + normal[1] * offset * side
                box = (cx - width / 2 - 2, cy - height / 2 - 2, cx + width / 2 + 2, cy + height / 2 + 2)
                if box[0] < 2 or box[1] < 2 or box[2] > size - 2 or box[3] > size - 2:
                    continue
                if any(_boxes_overlap(box, other, 3) for other in occupied):
                    continue
                path_hits = sum(box[0] <= p[0] <= box[2] and box[1] <= p[1] <= box[3] for p in drawn_points)
                if path_hits:
                    continue
                return (cx - width / 2 - left, cy - height / 2 - top), box
        return None

    drawn_points: list[tuple[float, float]] = []
    for a, b in pairs:
        source, target = nodes[a], nodes[b]
        if a == b:
            used = loop_angles.setdefault(a, [])
            options = [k * math.pi / 4 for k in range(8)]
            rng.shuffle(options)
            path = None
            for base in options:
                if any(abs((base - u + math.pi) % (2 * math.pi) - math.pi) < math.pi / 2 for u in used):
                    continue
                spread = math.radians(24)
                p0 = source.boundary(base - spread)
                p3 = source.boundary(base + spread)
                reach = source.r * 2.1 + 10
                c1 = (source.x + reach * math.cos(base - spread * 1.6), source.y + reach * math.sin(base - spread * 1.6))
                c2 = (source.x + reach * math.cos(base + spread * 1.6), source.y + reach * math.sin(base + spread * 1.6))
                candidate = _cubic(p0, c1, c2, p3)
                if path_is_clear(candidate, (a,)) and not any(
                    _boxes_overlap(_path_bbox(candidate[3:-3]), box, 2) for box in occupied[len(nodes):]
                ):
                    path = candidate
                    used.append(base)
                    break
            if path is None:
                continue
            direction = "loop"
        else:
            angle = math.atan2(target.y - source.y, target.x - source.x)
            bidirectional = (b, a) in seen
            curvatures = [0.0] if not bidirectional and rng.random() < 0.6 else []
            bend = rng.uniform(0.18, 0.35)
            curvatures += [bend, -bend] if not bidirectional else [bend]
            path = None
            for curvature in curvatures:
                offset = math.radians(25) * (1 if curvature > 0 else -1) if curvature else 0.0
                start = source.boundary(angle - offset)
                end = target.boundary(angle + math.pi + offset)
                distance = math.hypot(end[0] - start[0], end[1] - start[1])
                mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
                normal = (-(end[1] - start[1]) / distance, (end[0] - start[0]) / distance)
                control = (mid[0] + normal[0] * distance * curvature, mid[1] + normal[1] * distance * curvature)
                candidate = _bezier(start, control, end) if curvature else [start, end]
                dense = candidate if curvature else [(start[0] + (end[0] - start[0]) * t / 30,
                                                       start[1] + (end[1] - start[1]) * t / 30) for t in range(31)]
                if path_is_clear(dense, (a, b)):
                    path = candidate
                    break
            if path is None:
                continue
            direction = "forward"
        draw_polyline(path)
        head = draw_head(path[-1], path[-3] if len(path) > 3 else path[0])
        dense_path = path if len(path) > 2 else [(path[0][0] + (path[1][0] - path[0][0]) * t / 30,
                                                    path[0][1] + (path[1][1] - path[0][1]) * t / 30) for t in range(31)]
        drawn_points.extend(dense_path)
        x1, y1, x2, y2 = _path_bbox(path)
        pad = thickness / 2 + 1
        arrow_box = (min(x1 - pad, head[0]), min(y1 - pad, head[1]), max(x2 + pad, head[2]), max(y2 + pad, head[3]))
        labels.append((CLASS_ID["arrow"], *arrow_box))
        labels.append((CLASS_ID["arrowhead"], *head))

        # label text
        label_text = None
        label_box = None
        if rng.random() < 0.85:
            event = rng.choice(event_names)
            form = rng.random()
            if form < 0.5:
                label_text = event
            elif form < 0.7:
                label_text = f"{event} / {rng.choice(ACTIONS)}"
            elif form < 0.88:
                label_text = f"{event} [{rng.choice(GUARDS)}]"
            else:
                label_text = f"{event} [{rng.choice(GUARDS)}] / {rng.choice(ACTIONS)}"
            font = _font(rng.randint(13, 16))
            placement = place_label(dense_path, label_text, font)
            if placement is None:
                label_text = None
            else:
                origin, label_box = placement
                pil_jobs.append((origin, label_text, font))
                occupied.append(label_box)
                labels.append((CLASS_ID["label"], *label_box))
        transitions.append(Transition(
            id=f"T{len(transitions)}", source=f"S{a}", target=f"S{b}",
            event=None, label=label_text, confidence=1.0, direction=direction,
            points=tuple(Position(round(p[0], 1), round(p[1], 1)) for p in path[:: max(1, len(path) // 12)] + [path[-1]]),
            label_bbox=label_box, provenance="synthetic",
        ))
    if not transitions:
        return None

    # --- text ---------------------------------------------------------------
    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    for node in nodes:
        if node.label is None:
            continue
        inner = (node.r - 7 - thickness) * 2 if node.kind == "final" else node.r * 2 - 8 - 2 * thickness
        if node.shape == "rounded_rect":
            inner = node.r * 2.7 - 10
        for font_size in (16, 15, 14, 13, 12):
            font = _font(font_size)
            left, top, right, bottom = draw.textbbox((0, 0), node.label, font=font)
            if right - left <= inner:
                break
        else:
            node.label = None
            continue
        draw.text((node.x - (right - left) / 2 - left, node.y - (bottom - top) / 2 - top),
                  node.label, fill=color, font=font)
    for origin, text, font in pil_jobs:
        draw.text(origin, text, fill=color, font=font)
    image = np.array(pil)

    # --- degradation --------------------------------------------------------
    if rng.random() < 0.5:
        k = rng.choice((3, 3, 5))
        image = cv2.GaussianBlur(image, (k, k), 0)
    if rng.random() < 0.6:
        noise = np.random.default_rng(rng.randrange(2**32)).normal(0, rng.uniform(2, 9), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        quality = rng.randint(35, 80)
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    states = tuple(
        State(
            id=f"S{i}", label=node.label, type=node.kind,  # type: ignore[arg-type]
            position=Position(round(node.x, 1), round(node.y, 1)), confidence=1.0,
            bbox=tuple(round(v, 1) for v in node.bbox()),  # type: ignore[arg-type]
            shape="double_circle" if node.kind == "final" else node.shape,  # type: ignore[arg-type]
            provenance="synthetic",
        )
        for i, node in enumerate(nodes)
    )
    machine = annotate_machine(StateMachine(
        SCHEMA_VERSION, "synthetic", states, tuple(transitions),
        {"source": "synthetic", "warnings": [], "network_vocabulary": network,
         "initial_marker_bbox": [round(v, 1) for v in marker_box]},
    ))
    return Sample(image, labels, machine, {"network": network})


def generate_dataset(output: str | Path, *, count: int, seed: int = 7,
                     splits: tuple[float, float, float] = (0.7, 0.15, 0.15), size: int = 640) -> dict[str, int]:
    """Write ``images/{split}``, ``labels/{split}`` (YOLO) and ``graphs/{split}`` (canonical JSON)."""
    import cv2
    from dataclasses import replace

    output = Path(output)
    rng = random.Random(seed)
    totals = {"train": 0, "val": 0, "test": 0}
    boundaries = (splits[0] * count, (splits[0] + splits[1]) * count)
    for split in totals:
        for kind in ("images", "labels", "graphs"):
            (output / kind / split).mkdir(parents=True, exist_ok=True)
    index = 0
    while index < count:
        sample = render_sample(rng, size=size)
        if sample is None:
            continue
        split = "train" if index < boundaries[0] else "val" if index < boundaries[1] else "test"
        name = f"diagram_{index:05d}"
        image_path = output / "images" / split / f"{name}.png"
        cv2.imwrite(str(image_path), sample.image)
        height, width = sample.image.shape[:2]
        with (output / "labels" / split / f"{name}.txt").open("w", encoding="utf-8") as stream:
            for cls, x1, y1, x2, y2 in sample.labels:
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(width), x2), min(float(height), y2)
                stream.write(f"{cls} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
                             f"{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}\n")
        machine = replace(sample.machine, diagram_id=name)
        (output / "graphs" / split / f"{name}.json").write_text(machine.to_json() + "\n", encoding="utf-8")
        totals[split] += 1
        index += 1
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(YOLO_CLASSES))
    (output / "dataset.yaml").write_text(
        f"path: {output.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n{names}\n",
        encoding="utf-8",
    )
    return totals
