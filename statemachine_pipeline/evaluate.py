"""Graph-level evaluation of predicted state machines against ground truth.

States are matched by bounding-box IoU (optimal assignment), never by ID,
because IDs are just detection order. Transitions are compared through that
matching. All metrics are micro-averaged over the evaluated diagrams.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Callable

from .models import StateMachine


def levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _norm(text: str | None) -> str:
    return " ".join((text or "").split())


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match_states(pred: StateMachine, truth: StateMachine, threshold: float = 0.5) -> dict[str, str]:
    """Map predicted state id -> ground-truth state id (IoU >= threshold)."""
    pairs = []
    for p in pred.states:
        for t in truth.states:
            if p.bbox is not None and t.bbox is not None:
                score = _iou(p.bbox, t.bbox)
            elif p.position is not None and t.position is not None:
                score = max(0.0, 1.0 - math.dist((p.position.x, p.position.y), (t.position.x, t.position.y)) / 40.0)
            else:
                score = 0.0
            if score >= threshold:
                pairs.append((score, p.id, t.id))
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        pred_ids = sorted({p for _, p, _ in pairs})
        truth_ids = sorted({t for _, _, t in pairs})
        if not pairs:
            return {}
        cost = np.zeros((len(pred_ids), len(truth_ids)))
        for score, p, t in pairs:
            cost[pred_ids.index(p), truth_ids.index(t)] = -score
        rows, cols = linear_sum_assignment(cost)
        return {pred_ids[r]: truth_ids[c] for r, c in zip(rows, cols) if cost[r, c] < 0}
    except ImportError:
        mapping: dict[str, str] = {}
        for score, p, t in sorted(pairs, reverse=True):
            if p not in mapping and t not in mapping.values():
                mapping[p] = t
        return mapping


@dataclass
class Counts:
    values: Counter = field(default_factory=Counter)

    def add(self, other: dict[str, float]) -> None:
        self.values.update(other)

    def ratio(self, numerator: str, denominator: str) -> float | None:
        d = self.values[denominator]
        return round(self.values[numerator] / d, 4) if d else None

    def f1(self, tp: str, pred: str, gt: str) -> dict[str, float | None]:
        p = self.ratio(tp, pred)
        r = self.ratio(tp, gt)
        f = round(2 * p * r / (p + r), 4) if p is not None and r is not None and p + r else (0.0 if p is not None and r is not None else None)
        return {"precision": p, "recall": r, "f1": f}


def compare(pred: StateMachine, truth: StateMachine) -> dict[str, float]:
    """Raw counts for one diagram (summable across a dataset)."""
    c: Counter = Counter()
    mapping = match_states(pred, truth)
    truth_states = {s.id: s for s in truth.states}
    c["states_pred"] += len(pred.states)
    c["states_gt"] += len(truth.states)
    c["states_tp"] += len(mapping)
    for p in pred.states:
        if p.id not in mapping:
            continue
        t = truth_states[mapping[p.id]]
        c["state_type_total"] += 1
        c["state_type_correct"] += p.type == t.type
        if t.label is not None:
            c["state_ocr_total"] += 1
            c["state_ocr_exact"] += _norm(p.label) == _norm(t.label)
            c["state_ocr_char_errors"] += levenshtein(_norm(p.label), _norm(t.label))
            c["state_ocr_chars"] += len(_norm(t.label))
    truth_initial = truth.initial_state()
    pred_initial = pred.initial_state()
    c["initial_total"] += 1
    c["initial_correct"] += (pred_initial is not None and mapping.get(pred_initial) == truth_initial) or \
        (pred_initial is None and truth_initial is None)
    truth_finals = {s.id for s in truth.states if s.type == "final"}
    pred_finals = {mapping.get(s.id) for s in pred.states if s.type == "final"}
    c["final_gt"] += len(truth_finals)
    c["final_pred"] += sum(1 for s in pred.states if s.type == "final")
    c["final_tp"] += len(truth_finals & pred_finals)

    # Transitions through the state matching.
    def mapped(t) -> tuple[str, str] | None:
        if t.source in mapping and t.target in mapping:
            return mapping[t.source], mapping[t.target]
        return None

    truth_edges = Counter((t.source, t.target) for t in truth.transitions)
    pred_edges = Counter(e for e in (mapped(t) for t in pred.transitions) if e is not None)
    c["transitions_gt"] += len(truth.transitions)
    c["transitions_pred"] += len(pred.transitions)
    c["transitions_tp"] += sum((truth_edges & pred_edges).values())
    undirected_truth = Counter(tuple(sorted(e)) for e in truth_edges.elements())
    undirected_pred = Counter(tuple(sorted(e)) for e in pred_edges.elements())
    undirected_tp = sum((undirected_truth & undirected_pred).values())
    c["transitions_undirected_tp"] += undirected_tp
    # Endpoint accuracy: of transitions connecting the right pair of states,
    # how many also have the right source/target orientation.
    loops_tp = sum((Counter(e for e in truth_edges.elements() if e[0] == e[1]) & pred_edges).values())
    directed_non_loop = sum((truth_edges & pred_edges).values()) - loops_tp
    undirected_non_loop = undirected_tp - loops_tp
    c["endpoint_pairs"] += undirected_non_loop
    c["endpoint_correct"] += directed_non_loop

    # Label accuracy on directed matches (pair transitions with the same endpoints).
    available = {}
    for t in pred.transitions:
        key = mapped(t)
        if key is not None:
            available.setdefault(key, []).append(t)
    labels_ok = True
    for t in truth.transitions:
        candidates = available.get((t.source, t.target), [])
        if not candidates:
            labels_ok = False
            continue
        best = min(candidates, key=lambda p: levenshtein(_norm(p.label), _norm(t.label)))
        candidates.remove(best)
        if t.label is not None:
            c["label_total"] += 1
            c["label_exact"] += _norm(best.label) == _norm(t.label)
            c["label_char_errors"] += levenshtein(_norm(best.label), _norm(t.label))
            c["label_chars"] += len(_norm(t.label))
            c["event_total"] += 1
            c["event_exact"] += _norm(best.event) == _norm(t.event)
        if _norm(best.label) != _norm(t.label):
            labels_ok = False

    structural = (
        len(mapping) == len(truth.states) == len(pred.states)
        and truth_edges == pred_edges and sum(pred_edges.values()) == len(pred.transitions)
        and c["initial_correct"] == 1
        and truth_finals == pred_finals - {None}
    )
    c["diagrams"] += 1
    c["graph_exact_structure"] += structural
    c["graph_exact_full"] += structural and labels_ok and all(
        _norm(s.label) == _norm(truth_states[mapping[s.id]].label) for s in pred.states if s.id in mapping
    )
    return dict(c)


def summarize(counts: Counter) -> dict[str, Any]:
    c = Counts(counts)
    return {
        "diagrams": counts["diagrams"],
        "states": c.f1("states_tp", "states_pred", "states_gt"),
        "transitions": c.f1("transitions_tp", "transitions_pred", "transitions_gt"),
        "transitions_undirected": c.f1("transitions_undirected_tp", "transitions_pred", "transitions_gt"),
        "endpoint_accuracy": c.ratio("endpoint_correct", "endpoint_pairs"),
        "state_type_accuracy": c.ratio("state_type_correct", "state_type_total"),
        "initial_state_accuracy": c.ratio("initial_correct", "initial_total"),
        "final_states": c.f1("final_tp", "final_pred", "final_gt"),
        "state_ocr_exact": c.ratio("state_ocr_exact", "state_ocr_total"),
        "state_ocr_cer": c.ratio("state_ocr_char_errors", "state_ocr_chars"),
        "label_ocr_exact": c.ratio("label_exact", "label_total"),
        "label_ocr_cer": c.ratio("label_char_errors", "label_chars"),
        "event_exact": c.ratio("event_exact", "event_total"),
        "graph_exact_structure": c.ratio("graph_exact_structure", "diagrams"),
        "graph_exact_full": c.ratio("graph_exact_full", "diagrams"),
        "code_conformance": c.ratio("code_conformant", "code_checked"),
    }


def evaluate_split(
    images: list[Path], predict: Callable[[Path], StateMachine], truth_for: Callable[[Path], StateMachine],
    *, check_code: bool = False, progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    total: Counter = Counter()
    failures = []
    for index, image in enumerate(images):
        truth = truth_for(image)
        try:
            pred = predict(image)
        except Exception as exc:  # a crash on one image is a measured failure, not a skipped sample
            failures.append({"image": str(image), "error": f"{type(exc).__name__}: {exc}"})
            pred = StateMachine(truth.schema_version, truth.diagram_id, (), ())
        counts = compare(pred, truth)
        if check_code:
            from .codegen import generate_python, verify_generated
            total["code_checked"] += 1
            try:
                report = verify_generated(generate_python(pred), pred)
                total["code_conformant"] += report.valid
            except ValueError:
                pass  # invalid predicted graph: counted as non-conformant
        total.update(counts)
        if progress:
            progress(index + 1, len(images))
    summary = summarize(total)
    summary["failures"] = failures
    summary["counts"] = dict(total)
    return summary


def ground_truth_loader(dataset_root: Path) -> Callable[[Path], StateMachine]:
    def load(image: Path) -> StateMachine:
        graph = dataset_root / "graphs" / image.parent.name / f"{image.stem}.json"
        return StateMachine.from_dict(json.loads(graph.read_text(encoding="utf-8")))
    return load


def format_table(results: dict[str, dict[str, Any]]) -> str:
    """Markdown comparison table: one column per evaluated model."""
    rows = [
        ("State F1", lambda r: r["states"]["f1"]),
        ("Transition precision", lambda r: r["transitions"]["precision"]),
        ("Transition recall", lambda r: r["transitions"]["recall"]),
        ("Transition F1 (directed)", lambda r: r["transitions"]["f1"]),
        ("Transition F1 (undirected)", lambda r: r["transitions_undirected"]["f1"]),
        ("Endpoint accuracy", lambda r: r["endpoint_accuracy"]),
        ("Initial-state accuracy", lambda r: r["initial_state_accuracy"]),
        ("Final-state F1", lambda r: r["final_states"]["f1"]),
        ("State OCR exact", lambda r: r["state_ocr_exact"]),
        ("State OCR CER", lambda r: r["state_ocr_cer"]),
        ("Label OCR exact", lambda r: r["label_ocr_exact"]),
        ("Label OCR CER", lambda r: r["label_ocr_cer"]),
        ("Event exact", lambda r: r["event_exact"]),
        ("Graph exact (structure)", lambda r: r["graph_exact_structure"]),
        ("Graph exact (with labels)", lambda r: r["graph_exact_full"]),
        ("Generated-code conformance", lambda r: r["code_conformance"]),
    ]
    names = list(results)
    lines = ["| Metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for title, getter in rows:
        values = []
        for name in names:
            try:
                value = getter(results[name])
            except (KeyError, TypeError):
                value = None
            values.append("n/a" if value is None else f"{value:.3f}")
        lines.append(f"| {title} | " + " | ".join(values) + " |")
    lines.append("| Diagrams | " + " | ".join(str(results[n]["diagrams"]) for n in names) + " |")
    return "\n".join(lines)
