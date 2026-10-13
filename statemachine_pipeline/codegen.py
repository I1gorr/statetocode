"""Deterministic validation and reference execution for generated Python code."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from .models import StateMachine
from .validation import validate_state_machine
from .gemini import GeneratedCode


@dataclass
class CodeValidation:
    errors: list[str]
    warnings: list[str]

    @property
    def valid(self) -> bool:
        return not self.errors


def validate_generated_code(generated: GeneratedCode, machine: StateMachine) -> CodeValidation:
    result = CodeValidation([], [])
    if generated.target_language.lower() != "python":
        result.warnings.append("Only Python syntax and behavior checks are available")
        return result
    try:
        tree = ast.parse(generated.source_code)
    except SyntaxError as exc:
        result.errors.append(f"syntax: {exc}")
        return result
    if not any(isinstance(node, ast.ClassDef) for node in ast.walk(tree)):
        result.warnings.append("generated source has no class definition")
    for state in machine.states:
        if not re.search(rf"['\"]{re.escape(state.id)}['\"]", generated.source_code):
            result.errors.append(f"state {state.id!r} is not represented")
    for transition in machine.transitions:
        if not re.search(rf"['\"]{re.escape(transition.id)}['\"]", generated.source_code):
            result.errors.append(f"transition {transition.id}: transition id is not represented")
        for value, field in ((transition.source, "source"), (transition.target, "target")):
            if not re.search(rf"['\"]{re.escape(value)}['\"]", generated.source_code):
                result.errors.append(f"transition {transition.id}: {field} {value!r} is not represented")
        if transition.event and not re.search(rf"['\"]{re.escape(transition.event)}['\"]", generated.source_code):
            result.errors.append(f"transition {transition.id}: event {transition.event!r} is not represented")
    return result


def reference_step(machine: StateMachine, current_state: str, event: str) -> str:
    """Deterministic oracle used by tests and generated-code conformance checks."""
    validate_state_machine(machine).raise_for_errors()
    matches = [t for t in machine.transitions if t.source == current_state and t.event == event]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one transition for state={current_state!r}, event={event!r}")
    return matches[0].target
