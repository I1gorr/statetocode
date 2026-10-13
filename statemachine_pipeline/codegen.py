"""Deterministic Python code generation, reference semantics and conformance tests.

Runtime semantics (shared by ``reference_step`` and every generated machine):

* A transition's trigger is its ``event``; a transition without an event is
  triggered by its transition ID (documented fallback, never an invented name).
* ``submit(event, context)`` in a final state raises ``TerminalStateError``.
* Candidates are the transitions leaving the current state with that trigger.
  None -> ``InvalidEventError``. A guarded candidate is enabled only if its
  guard callable returns true; an unregistered guard raises
  ``GuardNotImplementedError`` (control flow is never guessed).
  No enabled candidate -> ``GuardRejectedError``; more than one ->
  ``NondeterministicTransitionError``.
* Exactly one enabled candidate fires: its action hook (if registered) runs,
  then the state changes. Unregistered actions are recorded in ``history``.
"""

from __future__ import annotations

import ast
from collections import deque
from dataclasses import dataclass, field
import json
import re
from typing import Any

from .models import StateMachine, Transition
from .validation import validate_state_machine
from .gemini import GeneratedCode

ERRORS = (
    "InvalidEventError", "GuardRejectedError", "TerminalStateError",
    "NondeterministicTransitionError", "GuardNotImplementedError",
)


@dataclass
class CodeValidation:
    errors: list[str]
    warnings: list[str]

    @property
    def valid(self) -> bool:
        return not self.errors


def trigger_of(transition: Transition) -> str:
    return transition.event if transition.event is not None else transition.id


def initial_state_of(machine: StateMachine) -> str:
    initial = machine.initial_state()
    if initial is None:
        if not machine.states:
            raise ValueError("state machine has no states")
        return machine.states[0].id
    return initial


class ReferenceError_(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def reference_step(machine: StateMachine, current_state: str, event: str,
                   guards: dict[str, bool] | None = None) -> str:
    """Deterministic oracle. Raises ``ValueError`` (with the error kind in ``.kind``)."""
    validate_state_machine(machine).raise_for_errors()
    try:
        return _step(machine, current_state, event, guards or {})[0]
    except ReferenceError_ as exc:
        error = ValueError(f"{exc.kind}: {exc}")
        error.kind = exc.kind  # type: ignore[attr-defined]
        raise error from None


def _step(machine: StateMachine, current: str, event: str, guards: dict[str, bool]) -> tuple[str, Transition]:
    state = machine.state(current)
    if state is not None and state.type == "final":
        raise ReferenceError_("TerminalStateError", f"state {current!r} is final")
    candidates = [t for t in machine.transitions if t.source == current and trigger_of(t) == event]
    if not candidates:
        raise ReferenceError_("InvalidEventError", f"event {event!r} is not accepted in state {current!r}")
    enabled = []
    for transition in candidates:
        if transition.guard is None:
            enabled.append(transition)
            continue
        if transition.guard not in guards:
            raise ReferenceError_("GuardNotImplementedError", f"guard {transition.guard!r} has no implementation")
        if guards[transition.guard]:
            enabled.append(transition)
    if not enabled:
        raise ReferenceError_("GuardRejectedError", f"all guards rejected event {event!r} in state {current!r}")
    if len(enabled) > 1:
        raise ReferenceError_("NondeterministicTransitionError",
                              f"transitions {[t.id for t in enabled]} are all enabled")
    return enabled[0].target, enabled[0]


# --------------------------------------------------------------------------
# Template generator
# --------------------------------------------------------------------------

def _identifier(text: str, fallback: str) -> str:
    name = re.sub(r"\W+", "_", text).strip("_")
    if not name or name[0].isdigit():
        name = f"_{name}" if name else fallback
    return name


def generate_python(machine: StateMachine) -> str:
    """Render a self-contained, dependency-free Python module for the machine."""
    validate_state_machine(machine).raise_for_errors()
    initial = initial_state_of(machine)
    finals = sorted(s.id for s in machine.states if s.type == "final")
    states = {s.id: s.label for s in machine.states}
    lines_states = "\n".join(f"    {s.id!r}: {s.label!r}," for s in machine.states)
    rows = []
    for t in machine.transitions:
        rows.append(
            "    TransitionSpec("
            f"id={t.id!r}, source={t.source!r}, target={t.target!r}, trigger={trigger_of(t)!r}, "
            f"guard={t.guard!r}, action={t.action!r}, kind={t.kind!r}, "
            f"network_event={(t.network.event_type if t.network else None)!r}, "
            f"protocol={(t.network.protocol if t.network else None)!r}),"
        )
    transitions_block = "\n".join(rows)
    doc_states = "\n".join(
        f"    {s.id}: {s.label or '(unlabelled)'}" + (f" [{s.type}]" if s.type != "normal" else "")
        for s in machine.states
    )
    doc_transitions = "\n".join(
        f"    {t.id}: {t.source} --{trigger_of(t)}"
        + (f" [{t.guard}]" if t.guard else "") + (f" / {t.action}" if t.action else "")
        + f"--> {t.target}" + (f"  (network: {t.network.event_type})" if t.network else "")
        + ("  (event missing in diagram: triggered by transition id)" if t.event is None else "")
        for t in machine.transitions
    ) or "    (none)"
    notes = []
    if machine.initial_state() is None:
        notes.append(f"No initial state was marked in the diagram; {initial!r} (first state) is used.")
    for u in machine.uncertainties:
        notes.append(f"Unconfirmed: {u.element_id}.{u.field}: {u.reason}")
    notes_block = "\n".join(f"    - {n}" for n in notes) or "    (none)"
    guards = sorted({t.guard for t in machine.transitions if t.guard})
    actions = sorted({t.action for t in machine.transitions if t.action})
    class_doc = f"State machine for diagram {machine.diagram_id!r}."

    return f'''"""{class_doc}

Generated deterministically from canonical JSON (schema {machine.schema_version}) by
statemachine_pipeline.codegen. Edit the diagram/JSON and regenerate instead of
editing this file.

States:
{doc_states}

Transitions:
{doc_transitions}

Notes:
{notes_block}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping


class StateMachineError(Exception):
    """Base class for runtime errors."""


class InvalidEventError(StateMachineError):
    """The event is not accepted in the current state."""


class GuardRejectedError(InvalidEventError):
    """The event is known in this state but every guard rejected it."""


class TerminalStateError(StateMachineError):
    """The machine is in a final state and accepts no events."""


class NondeterministicTransitionError(StateMachineError):
    """More than one transition is enabled for the event."""


class GuardNotImplementedError(StateMachineError):
    """A guard named in the diagram has no implementation registered."""


@dataclass(frozen=True)
class TransitionSpec:
    id: str
    source: str
    target: str
    trigger: str
    guard: str | None
    action: str | None
    kind: str
    network_event: str | None
    protocol: str | None


STATES: dict[str, str | None] = {{
{lines_states}
}}
INITIAL_STATE = {initial!r}
FINAL_STATES = frozenset({finals!r})
TRANSITIONS: tuple[TransitionSpec, ...] = (
{transitions_block}
)
GUARDS = {tuple(guards)!r}
ACTIONS = {tuple(actions)!r}

Guard = Callable[[Mapping[str, Any]], bool]
Action = Callable[[Mapping[str, Any]], None]


class StateMachine:
    """Executable state machine.

    Guards and actions named in the diagram are supplied by the caller:
    ``StateMachine(guards={{"ok": lambda ctx: ctx["ok"]}}, actions={{"log": print}})``.
    """

    def __init__(self, guards: Mapping[str, Guard] | None = None,
                 actions: Mapping[str, Action] | None = None,
                 initial_state: str = INITIAL_STATE) -> None:
        if initial_state not in STATES:
            raise ValueError(f"unknown initial state {{initial_state!r}}")
        self.guards: dict[str, Guard] = dict(guards or {{}})
        self.actions: dict[str, Action] = dict(actions or {{}})
        self.current_state = initial_state
        self.history: list[dict[str, Any]] = []

    @property
    def is_terminal(self) -> bool:
        return self.current_state in FINAL_STATES

    def allowed_events(self) -> list[str]:
        if self.is_terminal:
            return []
        return sorted({{t.trigger for t in TRANSITIONS if t.source == self.current_state}})

    def submit(self, event: str, context: Mapping[str, Any] | None = None) -> str:
        """Apply an event and return the new state."""
        context = context or {{}}
        if self.is_terminal:
            raise TerminalStateError(f"state {{self.current_state!r}} is final; event {{event!r}} rejected")
        candidates = [t for t in TRANSITIONS if t.source == self.current_state and t.trigger == event]
        if not candidates:
            raise InvalidEventError(f"event {{event!r}} is not accepted in state {{self.current_state!r}}")
        enabled = []
        for transition in candidates:
            if transition.guard is None:
                enabled.append(transition)
                continue
            guard = self.guards.get(transition.guard)
            if guard is None:
                raise GuardNotImplementedError(f"guard {{transition.guard!r}} has no implementation")
            if guard(context):
                enabled.append(transition)
        if not enabled:
            raise GuardRejectedError(f"all guards rejected {{event!r}} in state {{self.current_state!r}}")
        if len(enabled) > 1:
            raise NondeterministicTransitionError(
                f"transitions {{[t.id for t in enabled]}} are all enabled for {{event!r}}")
        transition = enabled[0]
        action_status = None
        if transition.action is not None:
            hook = self.actions.get(transition.action)
            if hook is None:
                action_status = "not_implemented"
            else:
                hook(context)
                action_status = "executed"
        self.history.append({{
            "transition": transition.id, "from": self.current_state, "event": event,
            "to": transition.target, "action": transition.action, "action_status": action_status,
            "network_event": transition.network_event,
        }})
        self.current_state = transition.target
        return self.current_state


if __name__ == "__main__":
    # Demo run: all guards true, fire each transition at most once.
    machine = StateMachine(guards={{name: (lambda ctx: True) for name in GUARDS}})
    print(f"start: {{machine.current_state}} ({{STATES[machine.current_state]}})")
    fired: set[str] = set()
    while True:
        options = [t for t in TRANSITIONS if t.source == machine.current_state and t.id not in fired]
        if machine.is_terminal or not options:
            break
        before, event = machine.current_state, options[0].trigger
        try:
            machine.submit(event)
        except StateMachineError as exc:
            print(f"{{event}}: {{type(exc).__name__}}: {{exc}}")
            break
        fired.add(machine.history[-1]["transition"])
        print(f"{{before}} --{{event}}--> {{machine.current_state}} ({{STATES[machine.current_state]}})")
'''


# --------------------------------------------------------------------------
# Test plan from the reference semantics
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Step:
    event: str
    guards: dict[str, bool]
    expected_state: str | None   # state after the step, or None when an error is expected
    expected_error: str | None


@dataclass
class PlannedTrace:
    start: str
    steps: list[Step]


@dataclass
class TestPlan:
    traces: list[PlannedTrace] = field(default_factory=list)
    covered_transitions: set[str] = field(default_factory=set)
    uncovered: list[str] = field(default_factory=list)
    started_outside_initial: list[str] = field(default_factory=list)

    def to_json(self) -> list[dict[str, Any]]:
        return [{"start": trace.start, "steps": [step.__dict__ for step in trace.steps]} for trace in self.traces]


def _guard_setting(machine: StateMachine, chosen: Transition) -> dict[str, bool]:
    """Guard values that enable ``chosen`` and disable its competitors."""
    values: dict[str, bool] = {}
    competitors = [t for t in machine.transitions
                   if t.source == chosen.source and trigger_of(t) == trigger_of(chosen) and t.id != chosen.id]
    for t in competitors:
        if t.guard is not None and t.guard != chosen.guard:
            values[t.guard] = False
    if chosen.guard is not None:
        values[chosen.guard] = True
    for t in machine.transitions:
        if t.guard is not None:
            values.setdefault(t.guard, False)
    return values


def build_test_plan(machine: StateMachine) -> TestPlan:
    """Traces that exercise every transition plus error behaviour.

    Transitions reachable from the initial state are reached by firing a path
    from it; transitions leaving unreachable states start the machine directly
    in their source state (``StateMachine(initial_state=...)``), so the
    generated code is still checked for them.
    """
    validate_state_machine(machine).raise_for_errors()
    plan = TestPlan()
    initial = initial_state_of(machine)

    def run(start: str, steps_spec: list[tuple[str, dict[str, bool]]]) -> PlannedTrace:
        state = start
        steps = []
        for event, guards in steps_spec:
            try:
                state, _ = _step(machine, state, event, guards)
                steps.append(Step(event, guards, state, None))
            except ReferenceError_ as exc:
                steps.append(Step(event, guards, None, exc.kind))
                break
        return PlannedTrace(start, steps)

    # Shortest firing path to every reachable state.
    paths: dict[str, list[tuple[str, dict[str, bool]]]] = {initial: []}
    queue = deque([initial])
    while queue:
        current = queue.popleft()
        state = machine.state(current)
        if state is not None and state.type == "final":
            continue
        for transition in machine.transitions:
            if transition.source != current:
                continue
            setting = _guard_setting(machine, transition)
            try:
                target, fired = _step(machine, current, trigger_of(transition), setting)
            except ReferenceError_:
                continue
            if fired.id == transition.id and target not in paths:
                paths[target] = paths[current] + [(trigger_of(transition), setting)]
                queue.append(target)

    def route(state_id: str) -> tuple[str, list[tuple[str, dict[str, bool]]]]:
        if state_id in paths:
            return initial, paths[state_id]
        return state_id, []

    for transition in machine.transitions:
        start, prefix = route(transition.source)
        if start != initial:
            plan.started_outside_initial.append(transition.id)
        trace = run(start, prefix + [(trigger_of(transition), _guard_setting(machine, transition))])
        plan.traces.append(trace)
        if trace.steps and trace.steps[-1].expected_error is None:
            plan.covered_transitions.add(transition.id)
        elif trace.steps:
            plan.uncovered.append(f"{transition.id} ({trace.steps[-1].expected_error})")
        if transition.guard is not None:
            rejected = dict(_guard_setting(machine, transition))
            rejected[transition.guard] = False
            plan.traces.append(run(start, prefix + [(trigger_of(transition), rejected)]))

    all_triggers = sorted({trigger_of(t) for t in machine.transitions})
    probe = "__invalid_event__"
    while probe in all_triggers:
        probe += "_"
    for state in machine.states:
        start, prefix = route(state.id)
        plan.traces.append(run(start, prefix + [(probe, {})]))
        if state.type == "final" and all_triggers:
            plan.traces.append(run(start, prefix + [(all_triggers[0], {})]))
    return plan


# --------------------------------------------------------------------------
# Static checks and sandboxed conformance run
# --------------------------------------------------------------------------

ALLOWED_IMPORTS = {"__future__", "dataclasses", "typing", "enum", "collections", "json", "abc", "functools", "re"}
FORBIDDEN_CALLS = {"eval", "exec", "compile", "open", "__import__", "input", "breakpoint", "globals", "vars"}


def static_check(source: str) -> list[str]:
    """Syntax plus a conservative allowlist: generated state machines need no I/O."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax: {exc}"]
    errors = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    errors.append(f"line {node.lineno}: import of {alias.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                errors.append(f"line {node.lineno}: import from {node.module!r} is not allowed")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            errors.append(f"line {node.lineno}: call to {node.func.id}() is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr not in {"__init__", "__name__", "__class__"}:
            errors.append(f"line {node.lineno}: dunder attribute access {node.attr!r} is not allowed")
    if not any(isinstance(n, ast.ClassDef) and n.name == "StateMachine" for n in tree.body):
        errors.append("generated source must define a top-level class StateMachine")
    return errors


@dataclass
class ConformanceReport:
    valid: bool
    errors: list[str]
    steps_checked: int
    traces: int
    covered_transitions: list[str]
    uncovered_transitions: list[str]
    mismatches: list[dict[str, Any]]
    started_outside_initial: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def verify_generated(source: str, machine: StateMachine, *, timeout: float = 10.0) -> ConformanceReport:
    """Static checks, then replay the reference test plan in the sandbox."""
    from .sandbox import run_conformance

    errors = static_check(source)
    plan = build_test_plan(machine)
    if errors:
        return ConformanceReport(False, errors, 0, len(plan.traces), sorted(plan.covered_transitions),
                                 plan.uncovered, [])
    outcome = run_conformance(source, plan.to_json(), timeout=timeout)
    if outcome.error is not None:
        return ConformanceReport(False, [outcome.error], 0, len(plan.traces), sorted(plan.covered_transitions),
                                 plan.uncovered, [])
    mismatches = []
    checked = 0
    for trace_index, (planned, actual) in enumerate(zip(plan.traces, outcome.results)):
        trace = planned.steps
        for step_index, (step, observed) in enumerate(zip(trace, actual)):
            checked += 1
            expected = {"state": step.expected_state, "error": step.expected_error}
            if observed != expected:
                mismatches.append({"trace": trace_index, "step": step_index, "event": step.event,
                                   "guards": step.guards, "expected": expected, "actual": observed})
        if len(actual) != len(trace):
            mismatches.append({"trace": trace_index, "expected_steps": len(trace), "actual_steps": len(actual)})
    return ConformanceReport(not mismatches, [], checked, len(plan.traces),
                             sorted(plan.covered_transitions), plan.uncovered, mismatches[:20],
                             plan.started_outside_initial)


def generate_tests(machine: StateMachine, module: str = "state_machine") -> str:
    """A pytest module replaying the reference plan against the generated module."""
    plan = build_test_plan(machine)
    payload = json.dumps(plan.to_json(), indent=1)
    return f'''"""Conformance tests generated from diagram {machine.diagram_id!r}.

Each trace was computed by the deterministic reference semantics in
statemachine_pipeline.codegen. Guard values are injected per step.
"""

import json

import pytest

import {module} as generated

TRACES = json.loads(r"""{payload}""")


def _machine(guard_values, start):
    guards = {{name: (lambda ctx, name=name: guard_values.get(name, False)) for name in generated.GUARDS}}
    return generated.StateMachine(guards=guards, initial_state=start)


@pytest.mark.parametrize("trace", TRACES, ids=[f"trace{{i}}" for i in range(len(TRACES))])
def test_trace(trace):
    guard_values = {{}}
    machine = _machine(guard_values, trace["start"])
    for step in trace["steps"]:
        guard_values.clear()
        guard_values.update(step["guards"])
        if step["expected_error"]:
            with pytest.raises(getattr(generated, step["expected_error"])):
                machine.submit(step["event"])
            return
        assert machine.submit(step["event"]) == step["expected_state"]


def test_final_states_reject_events():
    for state in generated.FINAL_STATES:
        machine = generated.StateMachine(initial_state=state)
        assert machine.allowed_events() == []
        with pytest.raises(generated.TerminalStateError):
            machine.submit("any-event")
'''


def validate_generated_code(generated: GeneratedCode, machine: StateMachine) -> CodeValidation:
    """Coverage check for LLM output (kept for the Gemini path), plus static checks."""
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
        for value, field_name in ((transition.source, "source"), (transition.target, "target")):
            if not re.search(rf"['\"]{re.escape(value)}['\"]", generated.source_code):
                result.errors.append(f"transition {transition.id}: {field_name} {value!r} is not represented")
        if transition.event and not re.search(rf"['\"]{re.escape(transition.event)}['\"]", generated.source_code):
            result.errors.append(f"transition {transition.id}: event {transition.event!r} is not represented")
    return result
