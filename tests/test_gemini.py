"""Gemini adapter contract handling, with the transport stubbed (no network)."""

import pytest

from statemachine_pipeline.gemini import GeminiGenerator
from statemachine_pipeline.models import State, StateMachine, Transition


def machine():
    return StateMachine("1.1", "x", (State("S0", "Idle", "initial"), State("S1", "Busy")),
                        (Transition("T0", "S0", "S1", "go"),))


def stub(monkeypatch, response):
    generator = GeminiGenerator(api_key="test-key")
    monkeypatch.setattr(generator, "_call", lambda *a, **k: response)
    return generator


def test_missing_key_is_a_clear_error():
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiGenerator(api_key="").explain(machine())


def test_suggestions_are_validated_and_never_applied(monkeypatch):
    m = machine()
    result = stub(monkeypatch, {"suggestions": [
        {"element_id": "T0", "field": "direction", "suggested_value": "reverse", "evidence": "head at Idle", "confidence": 0.8},
        {"element_id": "T0", "field": "direction", "suggested_value": "bidirectional", "evidence": "", "confidence": 0.9},
        {"element_id": "S1", "field": "add_transition", "suggested_value": "S0", "evidence": "second arrow", "confidence": 0.7},
        {"element_id": "T9", "field": "label", "suggested_value": "x", "confidence": 1},
        {"element_id": "T0", "field": "target", "suggested_value": "S7", "confidence": 1},
        {"field": "label"},
    ], "notes": "a single string note"}).suggest_corrections(m)
    assert [(s["element_id"], s["field"]) for s in result["suggestions"]] == [("T0", "direction"), ("S1", "add_transition")]
    assert len(result["rejected"]) == 4
    assert result["notes"] == ["a single string note"]
    assert m == machine()  # the machine itself is untouched


def test_explanation_drops_invented_elements(monkeypatch):
    result = stub(monkeypatch, {
        "summary": "s", "states": [{"id": "S0", "description": "idle"}, {"id": "S5", "description": "invented"}],
        "transitions": [{"id": "T0", "description": "go"}], "network_events": [],
        "execution_flows": "S0 -> S1", "ambiguities": [], "warnings": [],
    }).explain(machine())
    assert [s["id"] for s in result["states"]] == ["S0"]
    assert result["execution_flows"] == ["S0 -> S1"]
    assert any("do not exist" in w for w in result["warnings"])


def test_malformed_explanation_is_rejected(monkeypatch):
    with pytest.raises(ValueError, match="contract"):
        stub(monkeypatch, {"states": []}).explain(machine())


def test_llm_test_cases_are_checked_against_the_reference(monkeypatch):
    result = stub(monkeypatch, {"explanation": "e", "test_cases": [
        {"name": "ok", "events": ["go"], "expected_states": ["S1"]},
        {"name": "wrong", "events": ["go"], "expected_states": ["S0"]},
        {"name": "invalid", "events": ["stop"], "expected_error": "InvalidEventError"},
    ]}).explain_code(machine(), "source")
    assert [t["agrees_with_reference"] for t in result["test_cases"]] == [True, False, True]


def test_generated_code_parsing(monkeypatch):
    generated = stub(monkeypatch, {"source_code": "```python\nclass StateMachine: pass\n```", "target_language": "python",
                                   "assumptions": "one"}).generate(machine())
    assert generated.source_code == "class StateMachine: pass"
    assert generated.assumptions == ["one"]
