"""Canonical, deterministic state-machine representation.

Schema 1.1 extends 1.0 with image evidence (bounding boxes, arrow paths),
provenance, explicit uncertainties, and optional computer-network semantics.
Every 1.0 document is a valid 1.1 document once defaults are filled in.

Terminology kept deliberately distinct:
- ``State``: a node of the state machine (e.g. ``Connected``).
- ``Transition``: a directed change between two states, triggered by an event.
- ``NetworkEvent``: optional semantics attached to a transition when its event
  is a network occurrence (packet received, timeout, ...).
- ``NetworkDevice`` / ``NetworkLink``: physical/logical topology, i.e. which
  devices exist and which are connected. A link is not a transition.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal
import json

SCHEMA_VERSION = "1.1"
SUPPORTED_SCHEMAS = ("1.0", "1.1")

StateType = Literal["initial", "normal", "final"]
Direction = Literal["forward", "backward", "loop", "unknown"]
Shape = Literal["circle", "double_circle", "rounded_rect", "unknown"]
Provenance = Literal["detector", "ocr", "user", "llm", "synthetic", "unknown"]
TransitionKind = Literal["state_change", "network_event", "internal"]
NetworkEventType = Literal[
    "connect", "accept", "send", "receive", "ack", "timeout",
    "failure", "retry", "close", "reset", "other",
]
BBox = tuple[float, float, float, float]


def _normalise_box(instance: object, name: str) -> None:
    """Store boxes as float tuples so serialization is canonical (0 vs 0.0)."""
    value = getattr(instance, name)
    if value is not None:
        object.__setattr__(instance, name, tuple(float(v) for v in value))


@dataclass(frozen=True)
class Position:
    x: float
    y: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", float(self.x))
        object.__setattr__(self, "y", float(self.y))


@dataclass(frozen=True)
class State:
    id: str
    label: str | None
    type: StateType = "normal"
    position: Position | None = None
    confidence: float | None = None
    bbox: BBox | None = None
    shape: Shape = "unknown"
    label_confidence: float | None = None
    provenance: Provenance = "unknown"

    def __post_init__(self) -> None:
        _normalise_box(self, "bbox")


@dataclass(frozen=True)
class NetworkEvent:
    event_type: NetworkEventType
    protocol: str | None = None
    packet: str | None = None


@dataclass(frozen=True)
class Transition:
    id: str
    source: str
    target: str
    event: str | None = None
    guard: str | None = None
    action: str | None = None
    label: str | None = None
    confidence: float | None = None
    direction: Direction = "unknown"
    points: tuple[Position, ...] = ()
    label_bbox: BBox | None = None
    label_confidence: float | None = None
    kind: TransitionKind = "state_change"
    network: NetworkEvent | None = None
    provenance: Provenance = "unknown"

    def __post_init__(self) -> None:
        _normalise_box(self, "label_bbox")


@dataclass(frozen=True)
class NetworkDevice:
    id: str
    label: str | None
    role: str | None = None
    state_ids: tuple[str, ...] = ()
    position: Position | None = None


@dataclass(frozen=True)
class NetworkLink:
    id: str
    source: str
    target: str
    medium: str | None = None
    label: str | None = None


@dataclass(frozen=True)
class NetworkTopology:
    devices: tuple[NetworkDevice, ...] = ()
    links: tuple[NetworkLink, ...] = ()


@dataclass(frozen=True)
class Uncertainty:
    element_id: str
    field: str
    reason: str
    candidates: tuple[str, ...] = ()


@dataclass(frozen=True)
class SourceImage:
    path: str
    width: int
    height: int
    sha256: str | None = None


def _default_metadata() -> dict[str, Any]:
    return {"source": "image", "warnings": []}


@dataclass(frozen=True)
class StateMachine:
    schema_version: str
    diagram_id: str
    states: tuple[State, ...]
    transitions: tuple[Transition, ...]
    metadata: dict[str, Any] = field(default_factory=_default_metadata)
    network: NetworkTopology | None = None
    uncertainties: tuple[Uncertainty, ...] = ()
    source_image: SourceImage | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)

    def state(self, state_id: str) -> State | None:
        return next((s for s in self.states if s.id == state_id), None)

    def initial_state(self) -> str | None:
        return next((s.id for s in self.states if s.type == "initial"), None)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StateMachine":
        """Build from JSON; 1.0 documents get 1.1 defaults for new fields."""

        def position(raw: Any) -> Position | None:
            if raw is None:
                return None
            return Position(float(raw["x"]), float(raw["y"]))

        def bbox(raw: Any) -> BBox | None:
            if raw is None:
                return None
            values = tuple(float(v) for v in raw)
            if len(values) != 4:
                raise ValueError(f"bbox must have 4 numbers, got {raw!r}")
            return values  # type: ignore[return-value]

        def network_event(raw: Any) -> NetworkEvent | None:
            if raw is None:
                return None
            return NetworkEvent(raw["event_type"], raw.get("protocol"), raw.get("packet"))

        states = tuple(
            State(
                id=str(item["id"]),
                label=item.get("label"),
                type=item.get("type", "normal"),
                position=position(item.get("position")),
                confidence=item.get("confidence"),
                bbox=bbox(item.get("bbox")),
                shape=item.get("shape", "unknown"),
                label_confidence=item.get("label_confidence"),
                provenance=item.get("provenance", "unknown"),
            )
            for item in value.get("states", [])
        )
        transitions = tuple(
            Transition(
                id=str(item["id"]),
                source=str(item["source"]),
                target=str(item["target"]),
                event=item.get("event"),
                guard=item.get("guard"),
                action=item.get("action"),
                label=item.get("label"),
                confidence=item.get("confidence"),
                direction=item.get("direction", "unknown"),
                points=tuple(position(p) for p in item.get("points") or ()),  # type: ignore[misc]
                label_bbox=bbox(item.get("label_bbox")),
                label_confidence=item.get("label_confidence"),
                kind=item.get("kind", "state_change"),
                network=network_event(item.get("network")),
                provenance=item.get("provenance", "unknown"),
            )
            for item in value.get("transitions", [])
        )
        network = None
        if value.get("network") is not None:
            raw_network = value["network"]
            network = NetworkTopology(
                devices=tuple(
                    NetworkDevice(
                        str(d["id"]), d.get("label"), d.get("role"),
                        tuple(str(s) for s in d.get("state_ids") or ()),
                        position(d.get("position")),
                    )
                    for d in raw_network.get("devices", [])
                ),
                links=tuple(
                    NetworkLink(str(l["id"]), str(l["source"]), str(l["target"]),
                                l.get("medium"), l.get("label"))
                    for l in raw_network.get("links", [])
                ),
            )
        uncertainties = tuple(
            Uncertainty(str(u["element_id"]), str(u["field"]), str(u["reason"]),
                        tuple(str(c) for c in u.get("candidates") or ()))
            for u in value.get("uncertainties") or ()
        )
        source_image = None
        if value.get("source_image") is not None:
            raw_image = value["source_image"]
            source_image = SourceImage(str(raw_image["path"]), int(raw_image["width"]),
                                       int(raw_image["height"]), raw_image.get("sha256"))
        return cls(
            schema_version=str(value.get("schema_version", "")),
            diagram_id=str(value.get("diagram_id", "")),
            states=states,
            transitions=transitions,
            metadata=dict(value.get("metadata") or {}),
            network=network,
            uncertainties=uncertainties,
            source_image=source_image,
        )
