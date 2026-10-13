"""Canonical, deterministic state-machine representation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal
import json

StateType = Literal["initial", "normal", "final"]
Direction = Literal["forward", "backward", "loop", "unknown"]


@dataclass(frozen=True)
class Position:
    x: float
    y: float


@dataclass(frozen=True)
class State:
    id: str
    label: str | None
    type: StateType = "normal"
    position: Position | None = None
    confidence: float | None = None


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


@dataclass(frozen=True)
class StateMachine:
    schema_version: str
    diagram_id: str
    states: tuple[State, ...]
    transitions: tuple[Transition, ...]
    metadata: dict[str, Any] = field(default_factory=lambda: {"source": "image", "warnings": []})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StateMachine":
        def position(raw: Any) -> Position | None:
            if raw is None:
                return None
            return Position(float(raw["x"]), float(raw["y"]))

        states = tuple(
            State(
                id=str(item["id"]),
                label=item.get("label"),
                type=item.get("type", "normal"),
                position=position(item.get("position")),
                confidence=item.get("confidence"),
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
            )
            for item in value.get("transitions", [])
        )
        return cls(
            schema_version=str(value.get("schema_version", "")),
            diagram_id=str(value.get("diagram_id", "")),
            states=states,
            transitions=transitions,
            metadata=dict(value.get("metadata") or {}),
        )

