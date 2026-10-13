"""Isolated execution of generated code.

Generated code never runs in this process. It runs in a separate Python
interpreter with:

* ``-I`` isolated mode (no user site-packages, no PYTHON* env vars, script dir
  not on ``sys.path``), an empty environment and a fresh temporary cwd;
* a wall-clock timeout plus POSIX resource limits (CPU seconds, address space,
  file size, process count) where the platform supports them;
* a ``sys.addaudithook`` installed *before* the generated module is executed,
  aborting on sockets, subprocesses, ``os.system``/``exec``/``spawn``, file
  writes/deletes and ``ctypes``;
* a static allowlist check (``codegen.static_check``) applied by callers first.

This is defence in depth for code produced by our own template or by an LLM;
it is not a hardened sandbox against a determined attacker (use a container
or VM for that).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

HARNESS = r'''
import json, sys

BLOCKED_PREFIXES = ("socket.", "subprocess.", "os.system", "os.exec", "os.spawn", "os.posix_spawn",
                    "os.fork", "os.remove", "os.unlink", "os.rmdir", "os.rename", "shutil.", "ctypes.",
                    "os.kill", "os.chmod", "os.chown", "urllib.", "http.", "ftplib.", "smtplib.", "webbrowser.")

def _audit(event, args):
    if event.startswith(BLOCKED_PREFIXES):
        raise RuntimeError(f"sandbox: blocked operation {event}")
    if event == "open" and len(args) > 1 and isinstance(args[1], str) and any(m in args[1] for m in "wax+"):
        raise RuntimeError("sandbox: file writes are blocked")

source_path, mode = sys.argv[1], sys.argv[2]
payload = json.loads(sys.stdin.read())
source = open(source_path, encoding="utf-8").read()
sys.addaudithook(_audit)
import types
module = types.ModuleType("generated_state_machine")
sys.modules[module.__name__] = module  # dataclasses resolves the defining module
namespace = module.__dict__
exec(compile(source, "generated_state_machine.py", "exec"), namespace)
cls = namespace.get("StateMachine")
if cls is None:
    raise RuntimeError("generated code must define StateMachine")

def make(guard_values):
    guard_names = namespace.get("GUARDS", ())
    guards = {name: (lambda ctx, name=name: bool(guard_values.get(name, False))) for name in guard_names}
    try:
        return cls(guards=guards)
    except TypeError:
        return cls()

if mode == "events":
    machine = make(payload.get("guards") or {})
    states = [machine.current_state]
    for event in payload["events"]:
        machine.submit(event)
        states.append(machine.current_state)
    print(json.dumps(states))
else:
    results = []
    for trace in payload:
        guard_values = {}
        machine = make(guard_values)
        observed = []
        for step in trace:
            guard_values.clear()
            guard_values.update(step["guards"])
            try:
                state = machine.submit(step["event"])
                observed.append({"state": machine.current_state, "error": None})
            except Exception as exc:  # report the error class name, compared against the oracle
                observed.append({"state": None, "error": type(exc).__name__})
                break
        results.append(observed)
    print(json.dumps(results))
'''


@dataclass
class SandboxOutcome:
    results: list[Any] = field(default_factory=list)
    error: str | None = None


def _limits() -> None:  # pragma: no cover - runs in the child
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
        resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    except (ImportError, ValueError, OSError):
        pass


def _run(source: str, mode: str, payload: Any, timeout: float) -> tuple[str | None, str | None]:
    with tempfile.TemporaryDirectory(prefix="statemachine-sandbox-") as directory:
        source_path = Path(directory) / "generated.py"
        source_path.write_text(source, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-c", HARNESS, str(source_path), mode],
                input=json.dumps(payload), capture_output=True, text=True, timeout=timeout,
                cwd=directory, env={"PYTHONIOENCODING": "utf-8", "PATH": os.defpath},
                preexec_fn=_limits if os.name == "posix" else None, check=False,
            )
        except subprocess.TimeoutExpired:
            return None, f"generated code timed out after {timeout:.0f}s"
        if completed.returncode:
            message = completed.stderr.strip().splitlines()
            return None, (message[-1] if message else f"generated code exited with {completed.returncode}")
        return completed.stdout, None


def run_conformance(source: str, traces: list[list[dict[str, Any]]], *, timeout: float = 10.0) -> SandboxOutcome:
    stdout, error = _run(source, "plan", traces, timeout)
    if error is not None:
        return SandboxOutcome([], error)
    try:
        return SandboxOutcome(json.loads(stdout or ""), None)
    except json.JSONDecodeError as exc:
        return SandboxOutcome([], f"invalid harness output: {exc}")


def run_events(source: str, events: list[str], *, guards: dict[str, bool] | None = None,
               timeout: float = 5.0) -> SandboxOutcome:
    stdout, error = _run(source, "events", {"events": events, "guards": guards or {}}, timeout)
    if error is not None:
        return SandboxOutcome([], error)
    try:
        return SandboxOutcome(json.loads(stdout or ""), None)
    except json.JSONDecodeError as exc:
        return SandboxOutcome([], f"invalid harness output: {exc}")
