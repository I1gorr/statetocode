"""Code generation, reference semantics, sandboxed conformance checks."""

import pytest

from statemachine_pipeline.codegen import (
    build_test_plan, generate_python, generate_tests, reference_step, static_check, verify_generated,
)
from statemachine_pipeline.models import State, StateMachine, Transition
from statemachine_pipeline.sandbox import run_conformance, run_events
from statemachine_pipeline.semantics import annotate_machine
from statemachine_pipeline.trace import reference_trace, run_generated_trace


def tcp() -> StateMachine:
    return annotate_machine(StateMachine("1.1", "tcp", (
        State("S0", "Disconnected", "initial"), State("S1", "Connecting"),
        State("S2", "Connected"), State("S3", "Closed", "final"),
    ), (
        Transition("T0", "S0", "S1", label="connect / send SYN"),
        Transition("T1", "S1", "S2", label="recv SYN-ACK [valid] / send ACK"),
        Transition("T2", "S1", "S0", label="timeout [retries>=3]"),
        Transition("T3", "S1", "S1", label="timeout [retries<3] / retransmit"),
        Transition("T4", "S2", "S3", label="close"),
        Transition("T5", "S2", "S0", label="linkDown"),
        Transition("T6", "S2", "S2"),  # unlabelled: triggered by its id
    )))


def test_reference_semantics():
    m = tcp()
    assert reference_step(m, "S0", "connect") == "S1"
    assert reference_step(m, "S1", "timeout", {"retries<3": True, "retries>=3": False}) == "S1"
    assert reference_step(m, "S1", "timeout", {"retries<3": False, "retries>=3": True}) == "S0"
    assert reference_step(m, "S2", "T6") == "S2"
    for state, event, guards, kind in (
        ("S0", "close", {}, "InvalidEventError"),
        ("S1", "timeout", {"retries<3": False, "retries>=3": False}, "GuardRejectedError"),
        ("S1", "timeout", {"retries<3": True, "retries>=3": True}, "NondeterministicTransitionError"),
        ("S1", "timeout", {}, "GuardNotImplementedError"),
        ("S3", "connect", {}, "TerminalStateError"),
    ):
        with pytest.raises(ValueError) as info:
            reference_step(m, state, event, guards)
        assert info.value.kind == kind


def test_template_code_passes_conformance_and_covers_every_transition():
    m = tcp()
    report = verify_generated(generate_python(m), m)
    assert report.valid, report.mismatches or report.errors
    assert set(report.covered_transitions) == {t.id for t in m.transitions}
    assert report.steps_checked > len(m.transitions)


def test_conformance_detects_wrong_behaviour():
    m = tcp()
    source = generate_python(m)
    mutants = [
        source.replace("id='T4', source='S2', target='S3'", "id='T4', source='S2', target='S0'"),
        source.replace("if self.is_terminal:\n            raise TerminalStateError", "if False:\n            raise TerminalStateError"),
        source.replace("if guard(context):", "if True:"),
    ]
    for mutant in mutants:
        assert mutant != source
        assert not verify_generated(mutant, m).valid


def test_static_check_and_sandbox_block_unsafe_code():
    assert any("import of 'os'" in e for e in static_check("import os\nclass StateMachine: pass"))
    assert any("open()" in e for e in static_check("class StateMachine:\n  def f(self): open('x', 'w')"))
    blocked = run_conformance(
        "import builtins\nGUARDS=()\nclass StateMachine:\n  current_state='S0'\n"
        "  def submit(self, e, c=None):\n    getattr(builtins, 'open')('escape.txt', 'w')\n",
        [{"start": "S0", "steps": [{"event": "x", "guards": {}}]}],
    )
    assert blocked.results == [[{"state": None, "error": "RuntimeError"}]]
    looping = run_events("class StateMachine:\n  def __init__(self, **k):\n    while True: pass\n", ["x"], timeout=2)
    assert looping.error and "timed out" in looping.error


def test_generated_tests_module_runs(tmp_path):
    import subprocess
    import sys
    m = tcp()
    (tmp_path / "state_machine.py").write_text(generate_python(m))
    (tmp_path / "test_state_machine.py").write_text(generate_tests(m, "state_machine"))
    result = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(tmp_path)],
                            capture_output=True, text=True, cwd=tmp_path, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr


def test_event_trace_matches_reference():
    m = tcp()
    events = ["connect", "recv SYN-ACK", "T6", "close"]
    guards = {"valid": True, "retries<3": False, "retries>=3": False}
    expected = reference_trace(m, events, guards)
    actual = run_generated_trace(generate_python(m), events, guards=guards)
    assert actual.valid and actual.actual == expected


def test_transitions_from_unreachable_states_are_still_exercised():
    m = StateMachine("1.1", "x", (State("S0", "A", "initial"), State("S1", "B"), State("S2", "C")),
                     (Transition("T0", "S0", "S1", "go"), Transition("T1", "S2", "S1", "never")))
    plan = build_test_plan(m)
    assert plan.covered_transitions == {"T0", "T1"}
    assert plan.started_outside_initial == ["T1"]
    report = verify_generated(generate_python(m), m)
    assert report.valid and set(report.covered_transitions) == {"T0", "T1"}


def test_machine_without_initial_state_uses_first_state_and_documents_it():
    m = StateMachine("1.1", "x", (State("S0", "A"), State("S1", "B")), (Transition("T0", "S0", "S1"),))
    source = generate_python(m)
    assert "No initial state was marked" in source
    assert verify_generated(source, m).valid
