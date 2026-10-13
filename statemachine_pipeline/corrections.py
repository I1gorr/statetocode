"""User corrections as training/evaluation data.

Every correction stores the image, the machine the pipeline predicted, the
machine after the user's edits, a structured diff, and the raw detections.
Records marked ``split="eval"`` are held out: ``export_yolo`` never turns them
into training labels, so they can measure improvement honestly.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import shutil
from pathlib import Path
from typing import Any

from .models import StateMachine


def diff_machines(predicted: StateMachine, corrected: StateMachine) -> list[dict[str, Any]]:
    """Field-level changes, keyed by element ID (the editor keeps IDs stable)."""
    changes: list[dict[str, Any]] = []
    before_states = {s.id: s for s in predicted.states}
    after_states = {s.id: s for s in corrected.states}
    for state_id in sorted(before_states.keys() - after_states.keys()):
        changes.append({"element": state_id, "change": "state_removed"})
    for state_id in sorted(after_states.keys() - before_states.keys()):
        changes.append({"element": state_id, "change": "state_added"})
    for state_id in sorted(before_states.keys() & after_states.keys()):
        for name in ("label", "type", "bbox"):
            old, new = getattr(before_states[state_id], name), getattr(after_states[state_id], name)
            if old != new:
                changes.append({"element": state_id, "change": f"state_{name}", "before": old, "after": new})
    before_t = {t.id: t for t in predicted.transitions}
    after_t = {t.id: t for t in corrected.transitions}
    for tid in sorted(before_t.keys() - after_t.keys()):
        changes.append({"element": tid, "change": "transition_removed"})
    for tid in sorted(after_t.keys() - before_t.keys()):
        t = after_t[tid]
        changes.append({"element": tid, "change": "transition_added", "after": f"{t.source}->{t.target}"})
    for tid in sorted(before_t.keys() & after_t.keys()):
        for name in ("source", "target", "event", "guard", "action", "label", "direction"):
            old, new = getattr(before_t[tid], name), getattr(after_t[tid], name)
            if old != new:
                changes.append({"element": tid, "change": f"transition_{name}", "before": old, "after": new})
    return changes


def save_correction(path: str | Path, *, image: str, detections: list[dict[str, Any]],
                    machine: dict[str, Any], reviewer: str | None = None,
                    predicted: dict[str, Any] | None = None, split: str = "train") -> dict[str, Any]:
    """Append one correction record (JSON Lines) and return it."""
    if split not in {"train", "eval"}:
        raise ValueError("split must be 'train' or 'eval'")
    corrected = StateMachine.from_dict(machine)
    changes = diff_machines(StateMachine.from_dict(predicted), corrected) if predicted else []
    record = {
        "image": image, "detections": detections, "machine": machine, "predicted": predicted,
        "changes": changes, "reviewer": reviewer, "split": split,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def load_corrections(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        return []
    records = []
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{target}:{line_number}: malformed correction record ({exc})") from exc
    # A later correction of the same image supersedes earlier ones.
    latest: dict[str, dict[str, Any]] = {}
    for record in records:
        latest[record["image"]] = record
    return list(latest.values())


def _yolo_line(cls: int, box, width: int, height: int) -> str | None:
    x1, y1, x2, y2 = max(0.0, box[0]), max(0.0, box[1]), min(float(width), box[2]), min(float(height), box[3])
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return f"{cls} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}"


def export_yolo(corrections_path: str | Path, output: str | Path, *, exclude_dirs: tuple[str, ...] = ()) -> dict[str, int]:
    """Turn ``split == "train"`` corrections into YOLO labels.

    States, final states, label boxes and arrow extents come from the corrected
    JSON. Arrowhead boxes are approximated around the arrow tip, and initial
    markers reuse the detector box that touched the corrected initial state;
    both are documented approximations.
    """
    from .synthetic import CLASS_ID
    import cv2

    output = Path(output)
    (output / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output / "labels" / "train").mkdir(parents=True, exist_ok=True)
    stats = {"exported": 0, "skipped_eval": 0, "skipped_missing": 0, "skipped_heldout": 0}
    for record in load_corrections(corrections_path):
        if record.get("split", "train") != "train":
            stats["skipped_eval"] += 1
            continue
        image = Path(record["image"])
        if any(part in str(image.resolve()) for part in exclude_dirs):
            stats["skipped_heldout"] += 1
            continue
        pixels = cv2.imread(str(image))
        if pixels is None:
            stats["skipped_missing"] += 1
            continue
        height, width = pixels.shape[:2]
        machine = StateMachine.from_dict(record["machine"])
        lines = []
        for state in machine.states:
            if state.bbox is not None:
                lines.append(_yolo_line(CLASS_ID["final_state" if state.type == "final" else "state"], state.bbox, width, height))
        for transition in machine.transitions:
            if transition.label_bbox is not None:
                lines.append(_yolo_line(CLASS_ID["label"], transition.label_bbox, width, height))
            if len(transition.points) >= 2:
                xs = [p.x for p in transition.points]
                ys = [p.y for p in transition.points]
                lines.append(_yolo_line(CLASS_ID["arrow"], (min(xs) - 6, min(ys) - 6, max(xs) + 6, max(ys) + 6), width, height))
                tip = transition.points[-1]
                lines.append(_yolo_line(CLASS_ID["arrowhead"], (tip.x - 10, tip.y - 10, tip.x + 10, tip.y + 10), width, height))
        for detection in record.get("detections", []):
            if detection.get("kind") == "initial_marker" and machine.initial_state() is not None:
                lines.append(_yolo_line(CLASS_ID["initial_marker"], detection["bbox"], width, height))
        name = f"correction_{stats['exported']:05d}_{image.stem}"
        shutil.copyfile(image, output / "images" / "train" / f"{name}{image.suffix}")
        (output / "labels" / "train" / f"{name}.txt").write_text(
            "\n".join(line for line in lines if line) + "\n", encoding="utf-8")
        stats["exported"] += 1
    return stats


def evaluation_report(predictions: list[dict[str, Any]], truth: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare predicted and true machines per image using IoU state matching."""
    from collections import Counter
    from .evaluate import compare, summarize

    truth_by_image = {item["image"]: item for item in truth}
    total: Counter = Counter()
    rows = []
    for prediction in predictions:
        expected = truth_by_image.get(prediction["image"])
        if expected is None:
            continue
        counts = compare(StateMachine.from_dict(prediction["machine"]), StateMachine.from_dict(expected["machine"]))
        total.update(counts)
        rows.append({"image": prediction["image"], **summarize(Counter(counts))})
    return {"samples": len(rows), "rows": rows, "summary": summarize(total) if rows else None}
