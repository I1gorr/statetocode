"""Reference traces and isolated generated-code execution helpers."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from .codegen import reference_step
from .models import StateMachine


@dataclass(frozen=True)
class TraceResult:
    expected: tuple[str, ...]
    actual: tuple[str, ...] | None
    valid: bool
    error: str | None = None


def reference_trace(machine: StateMachine, events: list[str]) -> tuple[str, ...]:
    initial = next((s.id for s in machine.states if s.type == "initial"), None)
    if initial is None:
        initial = machine.states[0].id
    states = [initial]
    for event in events:
        states.append(reference_step(machine, states[-1], event))
    return tuple(states)


def run_generated_trace(source: str, events: list[str], *, timeout: float = 5.0) -> TraceResult:
    """Run only in a temporary subprocess with a timeout and no inherited env."""
    payload = json.dumps(events)
    harness = (
        "import json,sys\n"
        "ns={}\n"
        "exec(compile(open(sys.argv[1]).read(), sys.argv[1], 'exec'), ns)\n"
        "cls=ns.get('StateMachine')\n"
        "if cls is None: raise RuntimeError('generated code must define StateMachine')\n"
        "machine=cls()\n"
        "states=[machine.current_state]\n"
        "for event in json.loads(sys.argv[2]):\n"
        "    machine.submit(event)\n"
        "    states.append(machine.current_state)\n"
        "print(json.dumps(states))\n"
    )
    with tempfile.TemporaryDirectory(prefix="state-machine-") as directory:
        source_path = Path(directory) / "generated.py"
        source_path.write_text(source, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-I", "-c", harness, str(source_path), payload],
                capture_output=True, text=True, timeout=timeout,
                cwd=directory, env={"PYTHONIOENCODING": "utf-8"},
                check=False,
            )
        except subprocess.TimeoutExpired:
            return TraceResult((), None, False, "generated code timed out")
        if result.returncode:
            return TraceResult((), None, False, result.stderr.strip() or "generated code failed")
        try:
            actual = tuple(json.loads(result.stdout))
        except (json.JSONDecodeError, TypeError) as exc:
            return TraceResult((), None, False, f"invalid trace output: {exc}")
        return TraceResult((), actual, True)
