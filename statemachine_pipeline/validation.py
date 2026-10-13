"""Structural and semantic validation for the canonical contract."""

from dataclasses import dataclass, field
from typing import Any
from .models import StateMachine

SUPPORTED_SCHEMA = "1.0"
STATE_TYPES = {"initial", "normal", "final"}
DIRECTIONS = {"forward", "backward", "loop", "unknown"}


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("Invalid state machine:\n" + "\n".join(self.errors))


def validate_state_machine(machine: StateMachine | dict[str, Any]) -> ValidationReport:
    report = ValidationReport()
    if isinstance(machine, dict):
        if not isinstance(machine.get("states"), list):
            report.errors.append("$.states: required array")
        if not isinstance(machine.get("transitions"), list):
            report.errors.append("$.transitions: required array")
        for collection, name in (("states", "states"), ("transitions", "transitions")):
            for i, item in enumerate(machine.get(collection, [])):
                if not isinstance(item, dict):
                    report.errors.append(f"$.{name}[{i}]: expected object")
                    continue
                for field_name in ("id",) if name == "states" else ("id", "source", "target"):
                    if not isinstance(item.get(field_name), str) or not item[field_name]:
                        report.errors.append(f"$.{name}[{i}].{field_name}: required string")
                for field_name in ("event", "guard", "action", "label") if name == "transitions" else ("label",):
                    if item.get(field_name) is not None and not isinstance(item[field_name], str):
                        report.errors.append(f"$.{name}[{i}].{field_name}: expected string or null")
        try:
            machine = StateMachine.from_dict(machine)
        except (KeyError, TypeError, ValueError) as exc:
            report.errors.append(f"$: malformed JSON representation ({exc})")
            return report
    if machine.schema_version != SUPPORTED_SCHEMA:
        report.errors.append(f"$.schema_version: unsupported version {machine.schema_version!r}")
    if not machine.diagram_id:
        report.errors.append("$.diagram_id: required non-empty string")
    state_ids: set[str] = set()
    for i, state in enumerate(machine.states):
        path = f"$.states[{i}]"
        if not state.id:
            report.errors.append(f"{path}.id: required non-empty string")
        if state.id in state_ids:
            report.errors.append(f"{path}.id: duplicate state id {state.id!r}")
        state_ids.add(state.id)
        if state.type not in STATE_TYPES:
            report.errors.append(f"{path}.type: unsupported state type {state.type!r}")
        if state.confidence is not None and not 0 <= state.confidence <= 1:
            report.errors.append(f"{path}.confidence: must be between 0 and 1")
        if state.label is None:
            report.warnings.append(f"{path}.label: label is unknown")
    transition_ids: set[str] = set()
    initial_count = sum(state.type == "initial" for state in machine.states)
    if initial_count > 1:
        report.errors.append("$.states: at most one initial state is supported")
    for i, transition in enumerate(machine.transitions):
        path = f"$.transitions[{i}]"
        if transition.id in transition_ids:
            report.errors.append(f"{path}.id: duplicate transition id {transition.id!r}")
        transition_ids.add(transition.id)
        if transition.source not in state_ids:
            report.errors.append(f"{path}.source: unknown state {transition.source!r}")
        if transition.target not in state_ids:
            report.errors.append(f"{path}.target: unknown state {transition.target!r}")
        if transition.direction not in DIRECTIONS:
            report.errors.append(f"{path}.direction: unsupported direction {transition.direction!r}")
        if transition.confidence is not None and not 0 <= transition.confidence <= 1:
            report.errors.append(f"{path}.confidence: must be between 0 and 1")
        if transition.event is None and transition.guard is None and transition.action is None:
            report.warnings.append(f"{path}: transition has no event, guard, or action semantics")
        if transition.confidence is not None and transition.confidence < 0.5:
            report.warnings.append(f"{path}.confidence: low-confidence transition requires review")
    return report
