"""Evaluation metrics, corrections data and the synthetic generator."""

import json
from collections import Counter
from dataclasses import replace

from statemachine_pipeline.corrections import diff_machines, export_yolo, load_corrections, save_correction
from statemachine_pipeline.evaluate import compare, levenshtein, match_states, summarize
from statemachine_pipeline.models import State, StateMachine, Transition
from statemachine_pipeline.synthetic import generate_dataset
from statemachine_pipeline.validation import validate_state_machine


def truth() -> StateMachine:
    return StateMachine("1.1", "t", (
        State("S0", "Idle", "initial", bbox=(0, 0, 40, 40)),
        State("S1", "Busy", bbox=(100, 0, 140, 40)),
        State("S2", "Done", "final", bbox=(200, 0, 240, 40)),
    ), (Transition("T0", "S0", "S1", "go", label="go"), Transition("T1", "S1", "S2", "finish", label="finish"),
        Transition("T2", "S1", "S1", "tick", label="tick")))


def test_perfect_prediction_with_different_ids_scores_one():
    # Same diagram, states detected in another order: IDs differ, geometry matches.
    t = truth()
    rename = {"S0": "S2", "S1": "S0", "S2": "S1"}
    pred = StateMachine("1.1", "p",
                        tuple(replace(s, id=rename[s.id]) for s in t.states),
                        tuple(replace(x, source=rename[x.source], target=rename[x.target]) for x in t.transitions))
    summary = summarize(Counter(compare(pred, t)))
    assert summary["states"]["f1"] == 1.0
    assert summary["transitions"]["f1"] == 1.0
    assert summary["graph_exact_full"] == 1.0
    assert summary["endpoint_accuracy"] == 1.0


def test_reversed_edge_and_ocr_errors_are_measured():
    t = truth()
    pred = replace(t, transitions=(Transition("T0", "S1", "S0", "go", label="go"),
                                   Transition("T1", "S1", "S2", "finsh", label="finsh"),
                                   Transition("T2", "S1", "S1", "tick", label="tick")),
                   states=(replace(t.states[0], label="ldle"),) + t.states[1:])
    summary = summarize(Counter(compare(pred, t)))
    assert summary["transitions"]["precision"] == round(2 / 3, 4)
    assert summary["transitions_undirected"]["f1"] == 1.0
    assert summary["endpoint_accuracy"] == 0.5
    assert summary["state_ocr_exact"] == round(2 / 3, 4)
    assert summary["label_ocr_cer"] > 0
    assert summary["graph_exact_structure"] == 0.0


def test_missing_state_lowers_recall_not_precision():
    t = truth()
    pred = replace(t, states=t.states[:2], transitions=t.transitions[:1] + t.transitions[2:])
    summary = summarize(Counter(compare(pred, t)))
    assert summary["states"]["precision"] == 1.0 and summary["states"]["recall"] == round(2 / 3, 4)
    assert match_states(pred, t) == {"S0": "S0", "S1": "S1"}


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3 and levenshtein("", "ab") == 2


def test_corrections_round_trip_and_export(tmp_path):
    import cv2
    import numpy as np
    image = tmp_path / "d.png"
    cv2.imwrite(str(image), np.full((300, 300, 3), 255, np.uint8))
    predicted = truth()
    corrected = replace(predicted, transitions=predicted.transitions[:2] + (
        replace(predicted.transitions[2], event="tock", label="tock"),))
    path = tmp_path / "c.jsonl"
    record = save_correction(path, image=str(image), detections=[{"kind": "initial_marker", "bbox": [0, 50, 20, 60], "confidence": 0.9}],
                             machine=json.loads(corrected.to_json()), predicted=json.loads(predicted.to_json()))
    assert {"element": "T2", "change": "transition_event", "before": "tick", "after": "tock"} in record["changes"]
    save_correction(path, image=str(tmp_path / "held.png"), detections=[], machine=json.loads(corrected.to_json()), split="eval")
    assert len(load_corrections(path)) == 2
    stats = export_yolo(path, tmp_path / "yolo")
    assert stats["exported"] == 1 and stats["skipped_eval"] == 1
    labels = (tmp_path / "yolo" / "labels" / "train").glob("*.txt").__next__().read_text().split("\n")
    classes = Counter(line.split()[0] for line in labels if line)
    assert classes["0"] == 2 and classes["2"] == 1 and classes["4"] == 1  # states, final, initial marker


def test_diff_detects_structural_changes():
    a = truth()
    b = replace(a, states=a.states[:2], transitions=(replace(a.transitions[0], source="S1", target="S0"),))
    kinds = {c["change"] for c in diff_machines(a, b)}
    assert {"state_removed", "transition_removed", "transition_source", "transition_target"} <= kinds


def test_synthetic_generator_writes_consistent_ground_truth(tmp_path):
    totals = generate_dataset(tmp_path, count=6, seed=3)
    assert sum(totals.values()) == 6
    for graph in (tmp_path / "graphs").rglob("*.json"):
        machine = StateMachine.from_dict(json.loads(graph.read_text()))
        assert validate_state_machine(machine).valid
        assert sum(s.type == "initial" for s in machine.states) == 1
        label_file = tmp_path / "labels" / graph.parent.name / f"{graph.stem}.txt"
        lines = [l for l in label_file.read_text().splitlines() if l]
        state_boxes = sum(1 for l in lines if l.split()[0] in {"0", "2"})
        assert state_boxes == len(machine.states)
        assert sum(1 for l in lines if l.split()[0] == "1") == len(machine.transitions)
