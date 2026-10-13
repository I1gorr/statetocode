"""Canonical Petri-net model, validation, reconstruction, and simulation."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import math
from typing import Any, Iterable, Literal

PetriNodeType = Literal["place", "transition"]
ArcType = Literal["normal", "inhibitor", "reset"]


@dataclass(frozen=True)
class PetriPlace:
    id: str
    label: str | None = None
    position: dict[str, float] | None = None
    capacity: int | None = None
    initial_tokens: int = 0
    confidence: float | None = None


@dataclass(frozen=True)
class PetriTransition:
    id: str
    label: str | None = None
    position: dict[str, float] | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class PetriArc:
    id: str
    source: str
    target: str
    weight: int = 1
    arc_type: ArcType = "normal"
    confidence: float | None = None


@dataclass(frozen=True)
class PetriNet:
    schema_version: str
    diagram_id: str
    places: tuple[PetriPlace, ...]
    transitions: tuple[PetriTransition, ...]
    arcs: tuple[PetriArc, ...]
    metadata: dict[str, Any] = field(default_factory=lambda: {"source": "image", "warnings": []})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PetriNet":
        def position(raw: Any) -> dict[str, float] | None:
            return None if raw is None else {"x": float(raw["x"]), "y": float(raw["y"])}

        return cls(
            str(value.get("schema_version", "")),
            str(value.get("diagram_id", "")),
            tuple(PetriPlace(
                str(item["id"]), item.get("label"), position(item.get("position")),
                item.get("capacity"), int(item.get("initial_tokens", 0)), item.get("confidence"),
            ) for item in value.get("places", [])),
            tuple(PetriTransition(
                str(item["id"]), item.get("label"), position(item.get("position")),
                item.get("confidence"),
            ) for item in value.get("transitions", [])),
            tuple(PetriArc(
                str(item["id"]), str(item["source"]), str(item["target"]),
                int(item.get("weight", 1)), item.get("arc_type", "normal"), item.get("confidence"),
            ) for item in value.get("arcs", [])),
            dict(value.get("metadata") or {}),
        )


@dataclass
class PetriValidation:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("Invalid Petri net:\n" + "\n".join(self.errors))


def validate_petri_net(net: PetriNet | dict[str, Any]) -> PetriValidation:
    report = PetriValidation()
    if isinstance(net, dict):
        try:
            net = PetriNet.from_dict(net)
        except (KeyError, TypeError, ValueError) as exc:
            report.errors.append(f"$: malformed Petri-net JSON ({exc})")
            return report
    if net.schema_version not in {"1.0", "petri-net-1.0"}:
        report.errors.append(f"$.schema_version: unsupported version {net.schema_version!r}")
    if not net.diagram_id:
        report.errors.append("$.diagram_id: required non-empty string")
    place_ids = {p.id for p in net.places}
    transition_ids = {t.id for t in net.transitions}
    if len(place_ids) != len(net.places):
        report.errors.append("$.places: duplicate place IDs")
    if len(transition_ids) != len(net.transitions):
        report.errors.append("$.transitions: duplicate transition IDs")
    if place_ids & transition_ids:
        report.errors.append("$: place and transition IDs must be disjoint")
    for i, place in enumerate(net.places):
        if place.initial_tokens < 0:
            report.errors.append(f"$.places[{i}].initial_tokens: must be non-negative")
        if place.capacity is not None and (place.capacity < 0 or place.capacity < place.initial_tokens):
            report.errors.append(f"$.places[{i}].capacity: must contain initial marking")
    arc_ids: set[str] = set()
    for i, arc in enumerate(net.arcs):
        if arc.id in arc_ids:
            report.errors.append(f"$.arcs[{i}].id: duplicate arc ID")
        arc_ids.add(arc.id)
        if arc.source not in place_ids | transition_ids:
            report.errors.append(f"$.arcs[{i}].source: unknown node {arc.source!r}")
        if arc.target not in place_ids | transition_ids:
            report.errors.append(f"$.arcs[{i}].target: unknown node {arc.target!r}")
        if arc.source in place_ids and arc.target in place_ids:
            report.errors.append(f"$.arcs[{i}]: place-to-place arcs are unsupported")
        if arc.source in transition_ids and arc.target in transition_ids:
            report.errors.append(f"$.arcs[{i}]: transition-to-transition arcs are unsupported")
        if arc.weight < 1:
            report.errors.append(f"$.arcs[{i}].weight: must be >= 1")
        if arc.arc_type not in {"normal", "inhibitor", "reset"}:
            report.errors.append(f"$.arcs[{i}].arc_type: unsupported arc type")
    return report


def enabled_transitions(net: PetriNet, marking: dict[str, int]) -> tuple[str, ...]:
    validate_petri_net(net).raise_for_errors()
    result = []
    for transition in net.transitions:
        incoming = [a for a in net.arcs if a.target == transition.id]
        enabled = True
        for arc in incoming:
            tokens = marking.get(arc.source, 0)
            if arc.arc_type == "inhibitor" and tokens >= arc.weight:
                enabled = False
            elif arc.arc_type == "normal" and tokens < arc.weight:
                enabled = False
        if enabled:
            result.append(transition.id)
    return tuple(result)


def fire(net: PetriNet, marking: dict[str, int], transition_id: str) -> dict[str, int]:
    if transition_id not in enabled_transitions(net, marking):
        raise ValueError(f"transition {transition_id!r} is not enabled")
    next_marking = dict(marking)
    transition = next(t for t in net.transitions if t.id == transition_id)
    for arc in net.arcs:
        if arc.target == transition.id and arc.arc_type == "normal":
            next_marking[arc.source] = next_marking.get(arc.source, 0) - arc.weight
        elif arc.target == transition.id and arc.arc_type == "reset":
            next_marking[arc.source] = 0
        elif arc.source == transition.id and arc.arc_type == "normal":
            next_marking[arc.target] = next_marking.get(arc.target, 0) + arc.weight
    for place in net.places:
        if place.capacity is not None and next_marking.get(place.id, 0) > place.capacity:
            raise ValueError(f"transition {transition_id!r} exceeds capacity of {place.id!r}")
    return next_marking


def reconstruct_petri(
    detections: Iterable[Any], diagram_id: str = "diagram_001",
) -> PetriNet:
    """Build a Petri net from place/transition/arc detections.

    Arc endpoints are assigned by nearest compatible node; ambiguous or
    unsupported mappings are retained as warnings instead of becoming edges.
    """
    places = [d for d in detections if d.kind in {"place", "petri_place"}]
    transitions = [d for d in detections if d.kind in {"petri_transition", "transition"}]
    nodes = places + transitions
    place_models = tuple(PetriPlace(
        f"P{i}", d.text, {"x": d.center[0], "y": d.center[1]}, None, 0, d.confidence
    ) for i, d in enumerate(places))
    transition_models = tuple(PetriTransition(
        f"PT{i}", d.text, {"x": d.center[0], "y": d.center[1]}, d.confidence
    ) for i, d in enumerate(transitions))
    node_map: dict[int, str] = {}
    for index, detection in enumerate(places):
        node_map[id(detection)] = place_models[index].id
    for index, detection in enumerate(transitions):
        node_map[id(detection)] = transition_models[index].id
    warnings: list[str] = []
    arcs: list[PetriArc] = []
    for i, detection in enumerate(d for d in detections if d.kind in {"arc", "petri_arc"}):
        x, y = detection.center
        candidates = sorted(
            ((math.hypot(x - d.center[0], y - d.center[1]), d) for d in nodes),
            key=lambda item: item[0],
        )
        if len(candidates) < 2:
            warnings.append(f"A{i}: insufficient nodes for arc reconstruction")
            continue
        first, second = candidates[0][1], candidates[1][1]
        source = node_map[id(first)]
        target = node_map[id(second)]
        place_ids = {place.id for place in place_models}
        transition_ids = {transition.id for transition in transition_models}
        if (source in place_ids and target in place_ids) or (
            source in transition_ids and target in transition_ids
        ):
            warnings.append(f"A{i}: ambiguous same-type endpoints; omitted")
            continue
        arcs.append(PetriArc(f"A{i}", source, target, 1, "normal", detection.confidence))
    return PetriNet("petri-net-1.0", diagram_id, place_models, transition_models, tuple(arcs),
                    {"source": "image", "warnings": warnings})
