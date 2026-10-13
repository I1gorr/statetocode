"""image -> detector -> graph -> JSON -> code -> traces, on rendered network-protocol diagrams.

Needs the trained detector (runs/detect/v2/weights/best.pt); skipped otherwise.
The diagrams are rendered by the synthetic generator with a fixed seed and are
not part of the training set.
"""

import json
import random
from pathlib import Path

import cv2
import pytest

from statemachine_pipeline.codegen import generate_python, verify_generated
from statemachine_pipeline.evaluate import compare, match_states, summarize
from statemachine_pipeline.models import StateMachine
from statemachine_pipeline.synthetic import render_sample
from statemachine_pipeline.validation import validate_state_machine

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "runs" / "detect" / "v2" / "weights" / "best.pt"
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="trained detector weights not available")


def network_samples(count: int, seed: int = 12345):
    rng = random.Random(seed)
    samples = []
    while len(samples) < count:
        sample = render_sample(rng, network_bias=1.0)
        if sample is not None:
            samples.append(sample)
    return samples


def test_network_diagrams_end_to_end(tmp_path):
    from collections import Counter
    from statemachine_pipeline.pipeline import run_pipeline

    total = Counter()
    for index, sample in enumerate(network_samples(8)):
        image = tmp_path / f"net_{index}.png"
        cv2.imwrite(str(image), sample.image)
        result = run_pipeline(image, MODEL)
        machine = result.machine
        assert result.validation.valid, result.validation.errors
        # JSON round trip is lossless.
        assert StateMachine.from_dict(json.loads(machine.to_json())) == machine
        # Generated code always matches the reconstructed graph's semantics.
        report = verify_generated(generate_python(machine), machine)
        assert report.valid, report.mismatches or report.errors
        total.update(compare(machine, sample.machine))
    summary = summarize(total)
    # Regression floor, well below measured held-out performance (see README).
    assert summary["states"]["f1"] >= 0.95
    assert summary["transitions_undirected"]["f1"] >= 0.75
    assert summary["initial_state_accuracy"] >= 0.75


def test_reconstructed_network_machine_executes_like_ground_truth(tmp_path):
    """Where the reconstruction is structurally exact, event traces through the
    generated code equal traces through the ground-truth machine."""
    from statemachine_pipeline.codegen import _step, ReferenceError_, initial_state_of, trigger_of
    from statemachine_pipeline.pipeline import run_pipeline
    from statemachine_pipeline.sandbox import run_events

    checked = 0
    for index, sample in enumerate(network_samples(12, seed=777)):
        image = tmp_path / f"net_{index}.png"
        cv2.imwrite(str(image), sample.image)
        machine = run_pipeline(image, MODEL).machine
        counts = compare(machine, sample.machine)
        if not counts.get("graph_exact_full"):
            continue
        mapping = match_states(machine, sample.machine)
        truth = sample.machine
        # Walk the ground truth with guard-free events, replay them on generated code.
        state, events = initial_state_of(truth), []
        for _ in range(6):
            options = [t for t in truth.transitions if t.source == state and t.guard is None and t.event]
            if not options or truth.state(state).type == "final":
                break
            try:
                state, _ = _step(truth, state, trigger_of(options[0]), {})
            except ReferenceError_:
                break
            events.append(trigger_of(options[0]))
        if not events:
            continue
        outcome = run_events(generate_python(machine), events)
        assert outcome.error is None, outcome.error
        assert mapping[outcome.results[-1]] == state
        checked += 1
    assert checked >= 1, "no structurally exact reconstruction to compare against"
