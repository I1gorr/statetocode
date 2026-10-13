"""State machine for diagram 'diagram'.

Generated deterministically from canonical JSON (schema 1.1) by
statemachine_pipeline.codegen. Edit the diagram/JSON and regenerate instead of
editing this file.

States:
    S0: Connected
    S1: Connecting
    S2: Established [initial]
    S3: Listen

Transitions:
    T0: S0 --connect / notify--> S0  (network: connect)
    T1: S0 --error--> S1  (network: failure)
    T2: S0 --T2--> S2  (event missing in diagram: triggered by transition id)
    T3: S2 --accept [n<3]--> S1  (network: accept)
    T4: S3 --SYN-ACK--> S0  (network: accept)

Notes:
    (none)
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


STATES: dict[str, str | None] = {
    'S0': 'Connected',
    'S1': 'Connecting',
    'S2': 'Established',
    'S3': 'Listen',
}
INITIAL_STATE = 'S2'
FINAL_STATES = frozenset([])
TRANSITIONS: tuple[TransitionSpec, ...] = (
    TransitionSpec(id='T0', source='S0', target='S0', trigger='connect', guard=None, action='notify', kind='network_event', network_event='connect', protocol=None),
    TransitionSpec(id='T1', source='S0', target='S1', trigger='error', guard=None, action=None, kind='network_event', network_event='failure', protocol=None),
    TransitionSpec(id='T2', source='S0', target='S2', trigger='T2', guard=None, action=None, kind='state_change', network_event=None, protocol=None),
    TransitionSpec(id='T3', source='S2', target='S1', trigger='accept', guard='n<3', action=None, kind='network_event', network_event='accept', protocol=None),
    TransitionSpec(id='T4', source='S3', target='S0', trigger='SYN-ACK', guard=None, action=None, kind='network_event', network_event='accept', protocol='TCP'),
)
GUARDS = ('n<3',)
ACTIONS = ('notify',)

Guard = Callable[[Mapping[str, Any]], bool]
Action = Callable[[Mapping[str, Any]], None]


class StateMachine:
    """Executable state machine.

    Guards and actions named in the diagram are supplied by the caller:
    ``StateMachine(guards={"ok": lambda ctx: ctx["ok"]}, actions={"log": print})``.
    """

    def __init__(self, guards: Mapping[str, Guard] | None = None,
                 actions: Mapping[str, Action] | None = None,
                 initial_state: str = INITIAL_STATE) -> None:
        if initial_state not in STATES:
            raise ValueError(f"unknown initial state {initial_state!r}")
        self.guards: dict[str, Guard] = dict(guards or {})
        self.actions: dict[str, Action] = dict(actions or {})
        self.current_state = initial_state
        self.history: list[dict[str, Any]] = []

    @property
    def is_terminal(self) -> bool:
        return self.current_state in FINAL_STATES

    def allowed_events(self) -> list[str]:
        if self.is_terminal:
            return []
        return sorted({t.trigger for t in TRANSITIONS if t.source == self.current_state})

    def submit(self, event: str, context: Mapping[str, Any] | None = None) -> str:
        """Apply an event and return the new state."""
        context = context or {}
        if self.is_terminal:
            raise TerminalStateError(f"state {self.current_state!r} is final; event {event!r} rejected")
        candidates = [t for t in TRANSITIONS if t.source == self.current_state and t.trigger == event]
        if not candidates:
            raise InvalidEventError(f"event {event!r} is not accepted in state {self.current_state!r}")
        enabled = []
        for transition in candidates:
            if transition.guard is None:
                enabled.append(transition)
                continue
            guard = self.guards.get(transition.guard)
            if guard is None:
                raise GuardNotImplementedError(f"guard {transition.guard!r} has no implementation")
            if guard(context):
                enabled.append(transition)
        if not enabled:
            raise GuardRejectedError(f"all guards rejected {event!r} in state {self.current_state!r}")
        if len(enabled) > 1:
            raise NondeterministicTransitionError(
                f"transitions {[t.id for t in enabled]} are all enabled for {event!r}")
        transition = enabled[0]
        action_status = None
        if transition.action is not None:
            hook = self.actions.get(transition.action)
            if hook is None:
                action_status = "not_implemented"
            else:
                hook(context)
                action_status = "executed"
        self.history.append({
            "transition": transition.id, "from": self.current_state, "event": event,
            "to": transition.target, "action": transition.action, "action_status": action_status,
            "network_event": transition.network_event,
        })
        self.current_state = transition.target
        return self.current_state


if __name__ == "__main__":
    # Demo run: all guards true, fire each transition at most once.
    machine = StateMachine(guards={name: (lambda ctx: True) for name in GUARDS})
    print(f"start: {machine.current_state} ({STATES[machine.current_state]})")
    fired: set[str] = set()
    while True:
        options = [t for t in TRANSITIONS if t.source == machine.current_state and t.id not in fired]
        if machine.is_terminal or not options:
            break
        before, event = machine.current_state, options[0].trigger
        try:
            machine.submit(event)
        except StateMachineError as exc:
            print(f"{event}: {type(exc).__name__}: {exc}")
            break
        fired.add(machine.history[-1]["transition"])
        print(f"{before} --{event}--> {machine.current_state} ({STATES[machine.current_state]})")
