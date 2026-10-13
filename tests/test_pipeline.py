import pytest

from statemachine_pipeline.codegen import reference_step, validate_generated_code
from statemachine_pipeline.gemini import GeneratedCode
from statemachine_pipeline.models import State, StateMachine, Transition
from statemachine_pipeline.reconstruct import Detection, reconstruct
from statemachine_pipeline.validation import validate_state_machine
from statemachine_pipeline.analysis import analyze
from statemachine_pipeline.graph import GraphEditor
from statemachine_pipeline.trace import reference_trace


def machine():
    return StateMachine(
        "1.0", "fixture",
        (State("S0", "Start", "initial"), State("S1", "Work"), State("S2", "End", "final")),
        (Transition("T0", "S0", "S1", event="go", direction="forward"),
         Transition("T1", "S1", "S1", event="retry", direction="loop"),
         Transition("T2", "S1", "S2", event="done", direction="forward")),
    )


def test_validates_self_loop_and_dangling_reference():
    report = validate_state_machine(machine())
    assert report.valid
    bad = StateMachine("1.0", "x", (State("S0", "x"),),
                       (Transition("T0", "S0", "missing", event="go"),))
    assert "$.transitions[0].target" in validate_state_machine(bad).errors[0]


def test_round_trip_is_deterministic():
    source = machine()
    assert StateMachine.from_dict(source.to_dict()).to_json() == source.to_json()


def test_reference_runtime_and_generated_coverage():
    source = machine()
    assert reference_step(source, "S0", "go") == "S1"
    assert reference_step(source, "S1", "retry") == "S1"
    code = GeneratedCode(
        "class Machine:\n  S0='S0'; S1='S1'; S2='S2'\n"
        "  # 'T0' 'T1' 'T2' 'go' 'retry' 'done'\n"
        "  def go(self): return 'S1'\n",
        "python", {}, [], [],
    )
    assert validate_generated_code(code, source).valid


def test_invalid_generated_syntax_is_rejected():
    result = validate_generated_code(GeneratedCode("class X(:", "python", {}, [], []), machine())
    assert not result.valid
    assert "syntax" in result.errors[0]


def test_reconstruction_preserves_parallel_transitions_and_warns_direction():
    detections = [
        Detection("state", (0, 0, 40, 40), 0.9, text="A"),
        Detection("state", (100, 0, 140, 40), 0.9, text="B"),
        Detection("arrow", (40, 15, 100, 25), 0.8, text="go"),
        Detection("arrow", (40, 25, 100, 35), 0.7, text="again"),
    ]
    result = reconstruct(detections)
    assert len(result.machine.transitions) == 2
    assert result.warnings


def test_gemini_requires_key():
    from statemachine_pipeline.gemini import GeminiGenerator
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiGenerator(api_key="").generate(machine())


def test_generation_cli_rejects_missing_input(tmp_path, capsys):
    from main import generate_command
    assert generate_command([str(tmp_path / "missing.json")]) == 1
    assert "Code generation failed" in capsys.readouterr().err


def test_graph_editor_undo_redo_and_analysis():
    editor = GraphEditor(machine())
    editor.rename_state("S1", "Processing")
    assert editor.machine.states[1].label == "Processing"
    editor.undo()
    assert editor.machine.states[1].label == "Work"
    editor.redo()
    assert editor.machine.states[1].label == "Processing"
    result = analyze(editor.machine)
    assert result.reachable_states == ("S0", "S1", "S2")
    assert result.branching_states == ("S1",)


def test_reference_trace():
    assert reference_trace(machine(), ["go", "retry", "done"]) == ("S0", "S1", "S1", "S2")


def test_editor_rejects_dangling_transition():
    editor = GraphEditor(machine())
    with pytest.raises(ValueError):
        editor.add_transition("S0", "missing", event="bad")


def test_reconstruction_uses_explicit_orientation_for_direction():
    detections = [
        Detection("state", (0, 0, 30, 30), 0.9),
        Detection("state", (100, 0, 130, 30), 0.9),
        Detection("arrow", (25, 10, 105, 20), 0.8, orientation=0),
    ]
    result = reconstruct(detections)
    assert result.machine.transitions[0].direction == "forward"
    assert result.machine.transitions[0].source == "S0"
    assert result.machine.transitions[0].target == "S1"


def test_arrowhead_detection_is_associated_with_shaft():
    detections = [
        Detection("state", (0, 0, 30, 30), 0.9),
        Detection("state", (100, 0, 130, 30), 0.9),
        Detection("arrow", (25, 10, 105, 20), 0.8),
        Detection("arrowhead", (98, 8, 108, 18), 0.9),
    ]
    result = reconstruct(detections)
    assert len(result.machine.transitions) == 1
    assert result.machine.transitions[0].direction == "forward"
