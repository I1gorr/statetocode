"""Canonical JSON contract, schema validation, semantics and graph rules."""

import json

from statemachine_pipeline.models import (
    NetworkDevice, NetworkEvent, NetworkLink, NetworkTopology, Position, SourceImage, State,
    StateMachine, Transition, Uncertainty,
)
from statemachine_pipeline.schema import schema_errors
from statemachine_pipeline.semantics import annotate_machine, classify_network_event, parse_label
from statemachine_pipeline.validation import validate_state_machine


def rich_machine() -> StateMachine:
    return StateMachine(
        "1.1", "tcp",
        (State("S0", "Disconnected", "initial", Position(10, 20), 0.9, (0, 0, 20, 40), "circle", 0.8, "ocr"),
         State("S1", "Connecting"), State("S2", "Connected"), State("S3", "Closed", "final", shape="double_circle")),
        (Transition("T0", "S0", "S1", "connect", None, "send SYN", "connect / send SYN", 0.9, "forward",
                    (Position(1, 2), Position(3, 4)), (5, 5, 30, 15), 0.7, "network_event",
                    NetworkEvent("connect", "TCP", "SYN"), "detector"),
         Transition("T1", "S1", "S2", "recv SYN-ACK", "valid"),
         Transition("T2", "S2", "S3", "close")),
        {"source": "image", "warnings": ["w"]},
        NetworkTopology((NetworkDevice("D0", "Client", "host", ("S0", "S1")), NetworkDevice("D1", "Server")),
                        (NetworkLink("L0", "D0", "D1", "Ethernet"),)),
        (Uncertainty("T1", "label", "low OCR confidence", ("a", "b")),),
        SourceImage("x.png", 640, 480, "abc"),
    )


def test_round_trip_preserves_every_field():
    machine = rich_machine()
    restored = StateMachine.from_dict(json.loads(machine.to_json()))
    assert restored == machine
    assert restored.to_json() == machine.to_json()


def test_version_1_0_documents_still_load_and_validate():
    legacy = {"schema_version": "1.0", "diagram_id": "old", "metadata": {"warnings": []},
              "states": [{"id": "S0", "label": "A", "type": "initial", "position": None, "confidence": None}],
              "transitions": [{"id": "T0", "source": "S0", "target": "S0", "event": "tick", "guard": None,
                               "action": None, "label": None, "confidence": None, "direction": "loop"}]}
    report = validate_state_machine(legacy)
    assert report.valid, report.errors
    machine = StateMachine.from_dict(legacy)
    assert machine.transitions[0].kind == "state_change" and machine.uncertainties == ()


def test_schema_rejects_unknown_fields_and_bad_values():
    document = json.loads(rich_machine().to_json())
    document["states"][0]["colour"] = "red"
    document["transitions"][0]["confidence"] = 3
    document["transitions"][0]["network"]["event_type"] = "teleport"
    errors = schema_errors(document)
    assert any("colour" in e for e in errors)
    assert any(e.startswith("$.transitions[0].confidence") for e in errors)
    assert any(e.startswith("$.transitions[0].network") for e in errors)
    assert not validate_state_machine(document).valid


def test_malformed_json_shapes_are_errors_not_crashes():
    for document in ({}, {"states": "x", "transitions": []}, {"states": [1], "transitions": [{}]},
                     {"schema_version": "1.1", "diagram_id": "x", "states": [{"id": "S0", "bbox": [1, 2]}], "transitions": []}):
        report = validate_state_machine(document)
        assert not report.valid


def test_semantic_validation_findings():
    machine = StateMachine("1.1", "x", (State("S0", "A", "initial"), State("S1", "B", "final"), State("S2", "C")),
                           (Transition("T0", "S0", "S1", "go"), Transition("T1", "S0", "S1", "go"),
                            Transition("T2", "S1", "S0", "back")))
    report = validate_state_machine(machine)
    assert report.valid
    text = "\n".join(report.warnings)
    assert "non-deterministic" in text
    assert "leaves final state" in text
    assert "unreachable" in text and "S2" in text


def test_network_topology_references_are_checked():
    machine = rich_machine()
    document = json.loads(machine.to_json())
    document["network"]["links"][0]["target"] = "D9"
    document["network"]["devices"][0]["state_ids"] = ["S9"]
    errors = validate_state_machine(document).errors
    assert any("unknown device 'D9'" in e for e in errors)
    assert any("unknown state 'S9'" in e for e in errors)


def test_two_initial_states_are_invalid():
    machine = StateMachine("1.1", "x", (State("S0", "A", "initial"), State("S1", "B", "initial")), ())
    assert not validate_state_machine(machine).valid


def test_parse_uml_labels():
    from dataclasses import astuple
    assert astuple(parse_label("connect [retries < 3] / open_socket")) == ("connect", "retries < 3", "open_socket", True)
    assert astuple(parse_label("timeout")) == ("timeout", None, None, True)
    assert astuple(parse_label("[ok]")) == (None, "ok", None, True)
    assert astuple(parse_label("/ log")) == (None, None, "log", True)
    assert parse_label("a [b").well_formed is False
    assert astuple(parse_label(None)) == (None, None, None, True)


def test_network_event_classification_never_invents_semantics():
    assert classify_network_event("SYN-ACK") == NetworkEvent("accept", "TCP", "SYN-ACK")
    assert classify_network_event("timeout").event_type == "timeout"
    assert classify_network_event("linkDown").event_type == "failure"
    assert classify_network_event("retry").event_type == "retry"
    assert classify_network_event("disconnect").event_type == "close"
    for plain in ("go", "submit", "pause", "tick", None):
        assert classify_network_event(plain) is None


def test_annotation_distinguishes_state_changes_from_network_events():
    machine = annotate_machine(StateMachine("1.1", "x", (State("S0", "Idle", "initial"), State("S1", "Online")), (
        Transition("T0", "S0", "S1", label="connect [auth] / log"),
        Transition("T1", "S1", "S0", label="go"),
        Transition("T2", "S1", "S1", label="weird [label"),
    )))
    t0, t1, t2 = machine.transitions
    assert (t0.event, t0.guard, t0.action, t0.kind) == ("connect", "auth", "log", "network_event")
    assert (t1.event, t1.kind, t1.network) == ("go", "state_change", None)
    assert t2.event is None and any(u.element_id == "T2" for u in machine.uncertainties)
