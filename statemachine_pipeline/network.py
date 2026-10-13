"""Canonical contract and classical CV reconstruction for hand-drawn networks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NetworkNode:
    id: str
    label: str | None
    position: dict[str, float]
    radius: float
    confidence: float | None = None


@dataclass(frozen=True)
class NetworkEdge:
    id: str
    source: str
    target: str
    directed: bool = False
    label: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class NetworkGraph:
    schema_version: str
    diagram_id: str
    nodes: tuple[NetworkNode, ...]
    edges: tuple[NetworkEdge, ...]
    metadata: dict[str, Any] = field(default_factory=lambda: {"source": "image", "warnings": []})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "NetworkGraph":
        nodes = tuple(
            NetworkNode(
                str(item["id"]), item.get("label"),
                {"x": float(item["position"]["x"]), "y": float(item["position"]["y"])},
                float(item.get("radius", 30)), item.get("confidence"),
            )
            for item in value.get("nodes", [])
        )
        edges = tuple(
            NetworkEdge(
                str(item["id"]), str(item["source"]), str(item["target"]),
                bool(item.get("directed", False)), item.get("label"), item.get("confidence"),
            )
            for item in value.get("edges", [])
        )
        return cls(str(value.get("schema_version", "")), str(value.get("diagram_id", "")),
                   nodes, edges, dict(value.get("metadata") or {}))


def validate_network_graph(graph: NetworkGraph | dict[str, Any]) -> list[str]:
    """Return structural errors; warnings remain in graph metadata for review."""
    errors: list[str] = []
    if isinstance(graph, dict):
        if not isinstance(graph.get("nodes"), list):
            errors.append("$.nodes: required array")
        if not isinstance(graph.get("edges"), list):
            errors.append("$.edges: required array")
        try:
            graph = NetworkGraph.from_dict(graph)
        except (KeyError, TypeError, ValueError) as exc:
            return [f"$: malformed JSON representation ({exc})"]
    if graph.schema_version != "network-1.0":
        errors.append(f"$.schema_version: unsupported version {graph.schema_version!r}")
    if not graph.diagram_id:
        errors.append("$.diagram_id: required non-empty string")
    node_ids = {node.id for node in graph.nodes}
    if len(node_ids) != len(graph.nodes):
        errors.append("$.nodes.id: duplicate node id")
    edge_ids = [edge.id for edge in graph.edges]
    if len(set(edge_ids)) != len(edge_ids):
        errors.append("$.edges.id: duplicate edge id")
    for index, edge in enumerate(graph.edges):
        if edge.source not in node_ids:
            errors.append(f"$.edges[{index}].source: unknown node {edge.source!r}")
        if edge.target not in node_ids:
            errors.append(f"$.edges[{index}].target: unknown node {edge.target!r}")
    return errors


def _distance(a: NetworkNode, b: NetworkNode) -> float:
    return math.hypot(a.position["x"] - b.position["x"], a.position["y"] - b.position["y"])


def _line_circle_intersection(
    start: tuple[float, float], end: tuple[float, float],
    node: NetworkNode,
) -> float:
    """Distance from a segment endpoint to the node circumference."""
    cx, cy, radius = node.position["x"], node.position["y"], node.radius
    length = math.hypot(end[0] - start[0], end[1] - start[1]) or 1
    return min(
        abs(math.hypot(start[0] - cx, start[1] - cy) - radius),
        abs(math.hypot(end[0] - cx, end[1] - cy) - radius),
    ) / length
