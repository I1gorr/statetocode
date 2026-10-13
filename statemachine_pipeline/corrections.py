"""Persist reviewed detections and produce honest evaluation reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_correction(path: str | Path, *, image: str, detections: list[dict[str, Any]],
                    machine: dict[str, Any], reviewer: str | None = None) -> None:
    record = {"image": image, "detections": detections, "machine": machine, "reviewer": reviewer}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def evaluation_report(predictions: list[dict[str, Any]], truth: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare exact state/transition IDs; no fabricated metrics."""
    truth_by_image = {item["image"]: item for item in truth}
    rows = []
    for prediction in predictions:
        expected = truth_by_image.get(prediction["image"])
        if expected is None:
            continue
        predicted_states = {s["id"] for s in prediction["machine"].get("states", [])}
        expected_states = {s["id"] for s in expected["machine"].get("states", [])}
        predicted_edges = {(t["source"], t["target"]) for t in prediction["machine"].get("transitions", [])}
        expected_edges = {(t["source"], t["target"]) for t in expected["machine"].get("transitions", [])}
        rows.append({
            "image": prediction["image"],
            "state_precision": len(predicted_states & expected_states) / len(predicted_states) if predicted_states else 0,
            "state_recall": len(predicted_states & expected_states) / len(expected_states) if expected_states else 0,
            "transition_precision": len(predicted_edges & expected_edges) / len(predicted_edges) if predicted_edges else 0,
            "transition_recall": len(predicted_edges & expected_edges) / len(expected_edges) if expected_edges else 0,
        })
    return {"samples": len(rows), "rows": rows}
