"""Structural and semantic validation for the canonical contract.

Errors make a document unusable (dangling references, schema violations).
Warnings are semantic findings a reviewer should see (unreachable states,
non-determinism, unlabelled transitions) but that are legitimate in a diagram.
"""

from collections import defaultdict, deque
import json
from dataclasses import dataclass, field
from typing import Any
from .models import SUPPORTED_SCHEMAS, StateMachine
from .schema import schema_errors

SUPPORTED_SCHEMA = "1.0"  # kept for backwards compatibility of imports
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

    def to_dict(self) -> dict[str, Any]:
        return {"valid": self.valid, "errors": self.errors, "warnings": self.warnings}


def _reachable(machine: StateMachine) -> set[str]:
    adjacency: dict[str, list[str]] = defaultdict(list)
    for transition in machine.transitions:
        adjacency[transition.source].append(transition.target)
    roots = [s.id for s in machine.states if s.type == "initial"]
    seen: set[str] = set()
    queue = deque(roots)
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        queue.extend(adjacency[current])
    return seen


def validate_state_machine(machine: StateMachine | dict[str, Any]) -> ValidationReport:
    report = ValidationReport()
    if isinstance(machine, dict):
        if not isinstance(machine.get("states"), list):
            report.errors.append("$.states: required array")
        if not isinstance(machine.get("transitions"), list):
            report.errors.append("$.transitions: required array")
        for collection in ("states", "transitions"):
            for i, item in enumerate(machine.get(collection) or []):
                if not isinstance(item, dict):
                    report.errors.append(f"$.{collection}[{i}]: expected object")
                    continue
                for field_name in ("id",) if collection == "states" else ("id", "source", "target"):
                    if not isinstance(item.get(field_name), str) or not item[field_name]:
                        report.errors.append(f"$.{collection}[{i}].{field_name}: required string")
                for field_name in ("event", "guard", "action", "label") if collection == "transitions" else ("label",):
                    if item.get(field_name) is not None and not isinstance(item[field_name], str):
                        report.errors.append(f"$.{collection}[{i}].{field_name}: expected string or null")
        if report.errors:
            return report
        try:
            machine = StateMachine.from_dict(machine)
        except (KeyError, TypeError, ValueError) as exc:
            report.errors.append(f"$: malformed JSON representation ({exc})")
            return report
    for error in schema_errors(json.loads(machine.to_json())):
        if error not in report.errors:
            report.errors.append(error)
    if machine.schema_version not in SUPPORTED_SCHEMAS:
        message = f"$.schema_version: unsupported version {machine.schema_version!r}"
        if not any(e.startswith("$.schema_version") for e in report.errors):
            report.errors.append(message)
    if not machine.diagram_id and not any(e.startswith("$.diagram_id") for e in report.errors):
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
    initial_count = sum(state.type == "initial" for state in machine.states)
    if initial_count > 1:
        report.errors.append("$.states: at most one initial state is supported")
    elif initial_count == 0 and machine.states:
        report.warnings.append("$.states: no initial state; the first state will be used at runtime")

    final_ids = {s.id for s in machine.states if s.type == "final"}
    transition_ids: set[str] = set()
    triggers: dict[tuple[str, str, str | None], list[str]] = defaultdict(list)
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
        if transition.direction == "unknown":
            report.warnings.append(f"{path}.direction: arrow direction is unconfirmed")
        if transition.source in final_ids:
            report.warnings.append(f"{path}: leaves final state {transition.source!r}; final states reject events at runtime")
        if transition.kind == "network_event" and transition.network is None:
            report.warnings.append(f"{path}.network: network_event transition has no network event type")
        trigger = transition.event if transition.event is not None else transition.id
        triggers[(transition.source, trigger, transition.guard)].append(transition.id)
    for (source, trigger, guard), ids in sorted(triggers.items(), key=lambda kv: kv[1]):
        if len(ids) > 1:
            report.warnings.append(
                f"$.transitions: non-deterministic transitions {ids} from {source!r} on "
                f"event {trigger!r}" + (f" with guard {guard!r}" if guard else " without distinguishing guards")
            )

    if machine.network is not None:
        device_ids = [d.id for d in machine.network.devices]
        if len(set(device_ids)) != len(device_ids):
            report.errors.append("$.network.devices: duplicate device id")
        for i, device in enumerate(machine.network.devices):
            for state_id in device.state_ids:
                if state_id not in state_ids:
                    report.errors.append(f"$.network.devices[{i}].state_ids: unknown state {state_id!r}")
        link_ids = [l.id for l in machine.network.links]
        if len(set(link_ids)) != len(link_ids):
            report.errors.append("$.network.links: duplicate link id")
        for i, link in enumerate(machine.network.links):
            for name in ("source", "target"):
                if getattr(link, name) not in device_ids:
                    report.errors.append(f"$.network.links[{i}].{name}: unknown device {getattr(link, name)!r}")

    element_ids = state_ids | transition_ids
    for i, uncertainty in enumerate(machine.uncertainties):
        if uncertainty.element_id not in element_ids:
            report.warnings.append(f"$.uncertainties[{i}].element_id: refers to unknown element {uncertainty.element_id!r}")

    if report.valid and initial_count == 1:
        unreachable = sorted(state_ids - _reachable(machine))
        if unreachable:
            report.warnings.append(f"$.states: unreachable from the initial state: {unreachable}")
    return report
