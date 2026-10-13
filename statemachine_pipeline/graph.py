"""Editable graph operations over the canonical state-machine contract."""

from __future__ import annotations

from dataclasses import replace
from .models import Position, State, StateMachine, Transition
from .validation import validate_state_machine


class GraphEditor:
    """Copy-on-write editor with bounded undo/redo history."""

    def __init__(self, machine: StateMachine, *, history_limit: int = 100) -> None:
        self._current = machine
        self._undo: list[StateMachine] = []
        self._redo: list[StateMachine] = []
        self.history_limit = history_limit

    @property
    def machine(self) -> StateMachine:
        return self._current

    def _commit(self, candidate: StateMachine) -> StateMachine:
        report = validate_state_machine(candidate)
        report.raise_for_errors()
        self._undo.append(self._current)
        del self._undo[:-self.history_limit]
        self._current, self._redo = candidate, []
        return candidate

    def undo(self) -> StateMachine:
        if not self._undo:
            return self._current
        self._redo.append(self._current)
        self._current = self._undo.pop()
        return self._current

    def redo(self) -> StateMachine:
        if not self._redo:
            return self._current
        self._undo.append(self._current)
        self._current = self._redo.pop()
        return self._current

    def rename_state(self, state_id: str, label: str | None) -> StateMachine:
        states = tuple(replace(s, label=label) if s.id == state_id else s for s in self._current.states)
        return self._commit(replace(self._current, states=states))

    def move_state(self, state_id: str, x: float, y: float) -> StateMachine:
        states = tuple(
            replace(s, position=Position(x, y)) if s.id == state_id else s
            for s in self._current.states
        )
        return self._commit(replace(self._current, states=states))

    def add_state(self, label: str | None = None, state_type: str = "normal",
                  position: Position | None = None) -> StateMachine:
        ids = {s.id for s in self._current.states}
        index = 0
        while f"S{index}" in ids:
            index += 1
        state = State(f"S{index}", label, state_type, position)
        return self._commit(replace(self._current, states=self._current.states + (state,)))

    def delete_state(self, state_id: str) -> StateMachine:
        states = tuple(s for s in self._current.states if s.id != state_id)
        transitions = tuple(
            t for t in self._current.transitions
            if t.source != state_id and t.target != state_id
        )
        return self._commit(replace(self._current, states=states, transitions=transitions))

    def add_transition(self, source: str, target: str, *, event: str | None = None,
                       guard: str | None = None, action: str | None = None,
                       label: str | None = None, direction: str = "unknown") -> StateMachine:
        ids = {t.id for t in self._current.transitions}
        index = 0
        while f"T{index}" in ids:
            index += 1
        transition = Transition(f"T{index}", source, target, event, guard, action, label, None, direction)
        return self._commit(replace(self._current, transitions=self._current.transitions + (transition,)))

    def edit_transition(self, transition_id: str, **changes: object) -> StateMachine:
        allowed = {"source", "target", "event", "guard", "action", "label", "direction", "confidence"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError(f"unsupported transition fields: {sorted(unknown)}")
        transitions = tuple(
            replace(t, **changes) if t.id == transition_id else t
            for t in self._current.transitions
        )
        return self._commit(replace(self._current, transitions=transitions))

    def delete_transition(self, transition_id: str) -> StateMachine:
        return self._commit(replace(
            self._current,
            transitions=tuple(t for t in self._current.transitions if t.id != transition_id),
        ))
