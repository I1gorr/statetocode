"""Vision pipeline: preprocessing, arrow tracing, direction, association, labels."""

import math

import numpy as np
import pytest

from statemachine_pipeline.preprocess import ink_mask, load_gray, stroke_width
from statemachine_pipeline.reconstruct import reconstruct
from statemachine_pipeline.validation import validate_state_machine
from drawing import Canvas


def edges(machine):
    return sorted((t.source, t.target) for t in machine.transitions)


def label_of(machine, state_id):
    return next(s for s in machine.states if s.id == state_id)


def test_ink_mask_survives_uneven_lighting():
    canvas = Canvas()
    canvas.state(200, 240)
    gradient = np.linspace(0, 90, canvas.image.shape[1], dtype=np.float32)
    shaded = np.clip(canvas.image.astype(np.float32) - gradient[None, :, None], 0, 255).astype(np.uint8)
    mask = ink_mask(load_gray(shaded))
    # The shadowed right-hand side is background, not ink.
    assert mask[:, 500:].mean() < 0.01
    assert mask[240, 158:163].any()  # left outline of the state is ink
    assert 2 <= stroke_width(mask) <= 6


def test_straight_arrow_direction_and_endpoints(tmp_path):
    canvas = Canvas()
    a = canvas.state(150, 240)
    b = canvas.state(470, 240)
    canvas.arrow(b, a)  # drawn right -> left
    result = reconstruct(canvas.detections, "x", canvas.save(tmp_path / "d.png"), ocr=False)
    machine = result.machine
    left = min(machine.states, key=lambda s: s.position.x).id
    right = max(machine.states, key=lambda s: s.position.x).id
    assert edges(machine) == [(right, left)]
    assert machine.transitions[0].direction == "forward"


@pytest.mark.parametrize("use_head_detections", [True, False])
def test_reversed_arrow_is_not_mapped_by_box_order(tmp_path, use_head_detections):
    canvas = Canvas()
    a = canvas.state(150, 120)
    b = canvas.state(480, 360)
    canvas.arrow(b, a, detect_head=use_head_detections)
    detections = canvas.detections
    result = reconstruct(detections, "x", canvas.save(tmp_path / "d.png"), ocr=False)
    top = min(result.machine.states, key=lambda s: s.position.y).id
    bottom = max(result.machine.states, key=lambda s: s.position.y).id
    assert edges(result.machine) == [(bottom, top)]


def test_initial_marker_is_not_a_self_loop(tmp_path):
    canvas = Canvas()
    a = canvas.state(200, 240)
    b = canvas.state(480, 240)
    canvas.initial_marker(a)
    canvas.arrow(a, b)
    for detections in (canvas.detections, canvas.without("initial_marker")):
        machine = reconstruct(detections, "x", canvas.save(tmp_path / "d.png"), ocr=False).machine
        initial = [s.id for s in machine.states if s.type == "initial"]
        assert len(initial) == 1 and label_of(machine, initial[0]).position.x == 200
        assert all(t.source != t.target for t in machine.transitions)
        assert len(machine.transitions) == 1


def test_self_loop_and_curved_bidirectional_pair(tmp_path):
    canvas = Canvas()
    a = canvas.state(160, 260)
    b = canvas.state(470, 260)
    canvas.arrow(a, b, bend=0.25)
    canvas.arrow(b, a, bend=0.25)
    canvas.self_loop(b, -math.pi / 2)
    machine = reconstruct(canvas.detections, "x", canvas.save(tmp_path / "d.png"), ocr=False).machine
    left = min(machine.states, key=lambda s: s.position.x).id
    right = max(machine.states, key=lambda s: s.position.x).id
    assert edges(machine) == sorted([(left, right), (right, left), (right, right)])
    assert next(t for t in machine.transitions if t.source == t.target).direction == "loop"


def test_crossing_edges_are_traced_straight_through(tmp_path):
    canvas = Canvas(640, 520)
    tl = canvas.state(120, 110)
    br = canvas.state(520, 410)
    tr = canvas.state(520, 110)
    bl = canvas.state(120, 410)
    canvas.arrow(tl, br)
    canvas.arrow(tr, bl)
    machine = reconstruct(canvas.detections, "x", canvas.save(tmp_path / "d.png"), ocr=False).machine
    by_pos = {(round(s.position.x), round(s.position.y)): s.id for s in machine.states}
    expected = sorted([(by_pos[(120, 110)], by_pos[(520, 410)]), (by_pos[(520, 110)], by_pos[(120, 410)])])
    assert edges(machine) == expected


def test_final_state_from_double_outline(tmp_path):
    canvas = Canvas()
    a = canvas.state(160, 240)
    b = canvas.state(470, 240, final=True)
    canvas.arrow(a, b)
    detections = [d if d.kind != "final_state" else type(d)("state", d.bbox, d.confidence) for d in canvas.detections]
    machine = reconstruct(detections, "x", canvas.save(tmp_path / "d.png"), ocr=False).machine
    final = [s for s in machine.states if s.type == "final"]
    assert len(final) == 1 and final[0].position.x == 470
    assert any(u.element_id == final[0].id and u.field == "type" for u in machine.uncertainties)


def test_ambiguous_direction_is_reported_not_guessed(tmp_path):
    canvas = Canvas()
    a = canvas.state(160, 240)
    b = canvas.state(470, 240)
    canvas.arrow(a, b, head=False)
    machine = reconstruct(canvas.detections, "x", canvas.save(tmp_path / "d.png"), ocr=False).machine
    assert len(machine.transitions) == 1
    transition = machine.transitions[0]
    assert transition.direction == "unknown"
    assert any(u.element_id == transition.id and u.field == "direction" and len(u.candidates) == 2
               for u in machine.uncertainties)
    assert validate_state_machine(machine).valid


def test_dangling_arrow_and_missing_initial_are_surfaced(tmp_path):
    canvas = Canvas()
    a = canvas.state(160, 240)
    cv = __import__("cv2")
    cv.line(canvas.image, (200, 240), (330, 240), (0, 0, 0), 3)  # leaves a state, ends nowhere, no head
    machine = reconstruct(canvas.detections, "x", canvas.save(tmp_path / "d.png"), ocr=False)
    assert not machine.machine.transitions
    assert any("initial" in w for w in machine.warnings)


def test_label_is_assigned_to_the_nearest_arrow_path(tmp_path):
    pytest.importorskip("pytesseract")
    from statemachine_pipeline.ocr import ocr_available
    if not ocr_available():
        pytest.skip("tesseract is not installed")
    canvas = Canvas()
    a = canvas.state(150, 140, text="Idle")
    b = canvas.state(480, 140, text="Busy")
    c = canvas.state(315, 400, text="Done")
    canvas.arrow(a, b)
    canvas.arrow(b, c)
    canvas.text(280, 125, "start")
    canvas.text(420, 285, "finish")
    machine = reconstruct(canvas.detections, "x", canvas.save(tmp_path / "d.png")).machine
    names = {s.id: s.label for s in machine.states}
    assert sorted(names.values()) == ["Busy", "Done", "Idle"]
    by_names = {(names[t.source], names[t.target]): t for t in machine.transitions}
    assert by_names[("Idle", "Busy")].event == "start"
    assert by_names[("Busy", "Done")].event == "finish"
