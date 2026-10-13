"""Reference traces and isolated generated-code execution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from .codegen import initial_state_of, reference_step
from .models import StateMachine
from .sandbox import run_events


@dataclass(frozen=True)
class TraceResult:
    expected: tuple[str, ...]
    actual: tuple[str, ...] | None
    valid: bool
    error: str | None = None


def reference_trace(machine: StateMachine, events: list[str],
                    guards: dict[str, bool] | None = None) -> tuple[str, ...]:
    states = [initial_state_of(machine)]
    for event in events:
        states.append(reference_step(machine, states[-1], event, guards))
    return tuple(states)


def run_generated_trace(source: str, events: list[str], *, guards: dict[str, bool] | None = None,
                        timeout: float = 5.0) -> TraceResult:
    """Run generated code only inside the sandbox (separate interpreter, limits, audit hook)."""
    outcome = run_events(source, events, guards=guards, timeout=timeout)
    if outcome.error is not None:
        return TraceResult((), None, False, outcome.error)
    return TraceResult((), tuple(outcome.results), True)
