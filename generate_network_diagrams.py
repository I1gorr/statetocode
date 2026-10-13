"""Generate reproducible synthetic network diagrams for CV evaluation."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from statemachine_pipeline.network import NetworkEdge, NetworkGraph, NetworkNode


def generate(output: Path, count: int, seed: int) -> None:
    rng = random.Random(seed)
    output.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        width, height = 900, 640
        image = np.full((height, width), 255, np.uint8)
        centers = [(150, 150), (450, 120), (750, 180), (250, 430), (600, 440)]
        count_nodes = rng.randint(3, 5)
        centers = centers[:count_nodes]
        nodes = []
        for node_index, (x, y) in enumerate(centers):
            radius = rng.randint(34, 48)
            cv2.circle(image, (x, y), radius, 0, 4, cv2.LINE_AA)
            label = f"C{node_index}"
            cv2.putText(image, label, (x - 17, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 0, 2)
            nodes.append(NetworkNode(
                f"N{node_index}", label, {"x": float(x), "y": float(y)}, float(radius), 1.0,
            ))
        edges = []
        for edge_index in range(count_nodes - 1):
            source, target = nodes[edge_index], nodes[edge_index + 1]
            dx = target.position["x"] - source.position["x"]
            dy = target.position["y"] - source.position["y"]
            length = max(1.0, (dx * dx + dy * dy) ** 0.5)
            start = (int(source.position["x"] + dx * source.radius / length),
                     int(source.position["y"] + dy * source.radius / length))
            end = (int(target.position["x"] - dx * target.radius / length),
                   int(target.position["y"] - dy * target.radius / length))
            cv2.line(image, start, end, 0, 4, cv2.LINE_AA)
            edges.append(NetworkEdge(f"E{edge_index}", source.id, target.id, False, None, 1.0))
        stem = output / f"network_{index:04d}"
        cv2.imwrite(str(stem.with_suffix(".png")), image)
        graph = NetworkGraph("network-1.0", stem.name, tuple(nodes), tuple(edges),
                             {"source": "synthetic", "seed": seed})
        stem.with_suffix(".json").write_text(graph.to_json() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic network diagrams and labels")
    parser.add_argument("--output", type=Path, default=Path("network_dataset"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    generate(args.output, args.count, args.seed)
    print(f"Generated {args.count} network diagrams in {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
