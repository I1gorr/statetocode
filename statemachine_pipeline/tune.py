"""Tune association thresholds on a validation split (never on the test split).

Detections are computed once per image; only the deterministic reconstruction
is re-run per configuration (OCR off, since it does not affect association).
Coordinate descent keeps the search cheap and reproducible.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Callable

from .arrows import DEFAULT_CONFIG, load_config
from .detector import detect
from .evaluate import compare, ground_truth_loader, summarize
from .reconstruct import reconstruct

SEARCH_SPACE: dict[str, tuple[float, ...]] = {
    "head_ratio": (1.8, 2.2, 2.6),
    "direction_margin": (0.15, 0.25, 0.35),
    "endpoint_tolerance": (10.0, 14.0, 20.0),
    "overlap_reject": (0.3, 0.4, 0.55),
    "straight_bias": (0.002, 0.004, 0.008),
}


def objective(summary: dict[str, Any]) -> float:
    return (summary["transitions"]["f1"] or 0.0) + 0.5 * (summary["graph_exact_structure"] or 0.0) \
        + 0.25 * (summary["initial_state_accuracy"] or 0.0)


def tune(images: list[Path], dataset_root: Path, model: Path, *, passes: int = 2,
         log: Callable[[str], None] = print) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    truth_for = ground_truth_loader(dataset_root)
    cache = [(image, detect(image, model), truth_for(image)) for image in images]

    def score(config: dict[str, float]) -> tuple[float, dict[str, Any]]:
        total: Counter = Counter()
        for image, detections, truth in cache:
            machine = reconstruct(detections, image.stem, image, ocr=False, config=config).machine
            total.update(compare(machine, truth))
        summary = summarize(total)
        return objective(summary), summary

    best = load_config()
    best_score, best_summary = score(best)
    baseline_summary = best_summary
    log(f"start: objective={best_score:.4f}")
    for _ in range(passes):
        improved = False
        for name, values in SEARCH_SPACE.items():
            for value in values:
                if value == best[name]:
                    continue
                candidate = {**best, name: value}
                value_score, summary = score(candidate)
                if value_score > best_score + 1e-6:
                    best, best_score, best_summary, improved = candidate, value_score, summary, True
                    log(f"  {name}={value}: objective={value_score:.4f} (improved)")
        if not improved:
            break
    return best, baseline_summary, best_summary


def save_config(config: dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    changed = {k: v for k, v in config.items() if DEFAULT_CONFIG.get(k) != v}
    path.write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
