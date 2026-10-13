"""Draw small diagram fixtures with exact geometry for vision tests."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from statemachine_pipeline.reconstruct import Detection


class Canvas:
    def __init__(self, width: int = 640, height: int = 480, thickness: int = 3) -> None:
        self.image = np.full((height, width, 3), 255, np.uint8)
        self.thickness = thickness
        self.detections: list[Detection] = []

    def state(self, x: int, y: int, r: int = 40, *, final: bool = False, text: str | None = None) -> tuple[int, int, int]:
        cv2.circle(self.image, (x, y), r, (0, 0, 0), self.thickness, cv2.LINE_AA)
        if final:
            cv2.circle(self.image, (x, y), r - 8, (0, 0, 0), 2, cv2.LINE_AA)
        if text:
            scale = 0.5
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            cv2.putText(self.image, text, (x - w // 2, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
        self.detections.append(Detection("final_state" if final else "state", (x - r, y - r, x + r, y + r), 0.95))
        return x, y, r

    def _head(self, tip, before, filled: bool = True) -> None:
        angle = math.atan2(tip[1] - before[1], tip[0] - before[0])
        length, spread = 16, math.radians(26)
        left = (tip[0] - length * math.cos(angle - spread), tip[1] - length * math.sin(angle - spread))
        right = (tip[0] - length * math.cos(angle + spread), tip[1] - length * math.sin(angle + spread))
        pts = np.array([tip, left, right], np.float32).round().astype(np.int32)
        if filled:
            cv2.fillPoly(self.image, [pts], (0, 0, 0), cv2.LINE_AA)
        else:
            for wing in (left, right):
                cv2.line(self.image, tuple(map(round, tip)), tuple(map(round, wing)), (0, 0, 0), self.thickness, cv2.LINE_AA)
        xs, ys = [p[0] for p in (tip, left, right)], [p[1] for p in (tip, left, right)]
        self.detections.append(Detection("arrowhead", (min(xs) - 2, min(ys) - 2, max(xs) + 2, max(ys) + 2), 0.9))

    @staticmethod
    def boundary(state, toward) -> tuple[float, float]:
        x, y, r = state
        a = math.atan2(toward[1] - y, toward[0] - x)
        return x + r * math.cos(a), y + r * math.sin(a)

    def arrow(self, source, target, *, bend: float = 0.0, head: bool = True, filled: bool = True,
              detect_head: bool = True) -> list[tuple[float, float]]:
        start = self.boundary(source, target[:2])
        end = self.boundary(target, source[:2])
        if bend:
            mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            control = (mid[0] - dy / length * length * bend, mid[1] + dx / length * length * bend)
            start = self.boundary(source, control)
            end = self.boundary(target, control)
            points = [((1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t * t * end[0],
                       (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t * t * end[1])
                      for t in (i / 40 for i in range(41))]
        else:
            points = [start, end]
        return self._stroke(points, head, filled, detect_head)

    def _stroke(self, points, head, filled, detect_head) -> list[tuple[float, float]]:
        pts = np.array(points, np.float32).round().astype(np.int32)
        cv2.polylines(self.image, [pts], False, (0, 0, 0), self.thickness, cv2.LINE_AA)
        count = len(self.detections)
        if head:
            self._head(points[-1], points[-3] if len(points) > 3 else points[0], filled)
            if not detect_head:
                self.detections = self.detections[:count]
        xs, ys = [p[0] for p in points], [p[1] for p in points]
        self.detections.append(Detection("arrow", (min(xs) - 10, min(ys) - 10, max(xs) + 10, max(ys) + 10), 0.9))
        return points

    def self_loop(self, state, angle: float = -math.pi / 2) -> list[tuple[float, float]]:
        x, y, r = state
        a1, a2 = angle - 0.45, angle + 0.45
        p0 = (x + r * math.cos(a1), y + r * math.sin(a1))
        p3 = (x + r * math.cos(a2), y + r * math.sin(a2))
        reach = r * 2.2 + 10
        c1 = (x + reach * math.cos(a1 - 0.3), y + reach * math.sin(a1 - 0.3))
        c2 = (x + reach * math.cos(a2 + 0.3), y + reach * math.sin(a2 + 0.3))
        points = []
        for i in range(41):
            t = i / 40
            a, b, c, d = (1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t ** 3
            points.append((a * p0[0] + b * c1[0] + c * c2[0] + d * p3[0], a * p0[1] + b * c1[1] + c * c2[1] + d * p3[1]))
        return self._stroke(points, True, True, True)

    def initial_marker(self, state, angle: float = math.pi) -> None:
        x, y, r = state
        tip = (x + r * math.cos(angle), y + r * math.sin(angle))
        start = (x + (r + 60) * math.cos(angle), y + (r + 60) * math.sin(angle))
        cv2.circle(self.image, tuple(map(round, start)), 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.line(self.image, tuple(map(round, start)), tuple(map(round, tip)), (0, 0, 0), self.thickness, cv2.LINE_AA)
        self._head(tip, start)
        xs, ys = (start[0], tip[0]), (start[1], tip[1])
        self.detections.append(Detection("initial_marker", (min(xs) - 9, min(ys) - 9, max(xs) + 9, max(ys) + 9), 0.9))

    def text(self, x: int, y: int, text: str) -> None:
        scale = 0.5
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.putText(self.image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)
        self.detections.append(Detection("label", (x - 2, y - h - 3, x + w + 2, y + 5), 0.9))

    def save(self, path: Path) -> Path:
        assert cv2.imwrite(str(path), self.image)
        return path

    def without(self, *kinds: str) -> list[Detection]:
        return [d for d in self.detections if d.kind not in kinds]
