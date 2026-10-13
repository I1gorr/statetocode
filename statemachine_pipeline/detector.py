"""Optional Ultralytics adapter using the existing state/arrow class convention."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from .reconstruct import Detection

# 2-class models (state, arrow) remain supported; newer models add the rest.
STATE_MACHINE_CLASSES = {
    "state", "final_state", "initial", "final", "arrow", "transition",
    "arrowhead", "initial_marker", "label",
}


def detect(
    image_path: str | Path,
    model_path: str | Path,
    *,
    confidence: float = 0.4,
    diagram_type: str = "state-machine",
) -> list[Detection]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed in the active Linux Python environment. "
            "Create/activate .venv with `python3 -m venv .venv && source .venv/bin/activate`, "
            "then run `python -m pip install -r requirements.txt`."
        ) from exc
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {model_path}")
    model = _load_model(str(model_path.resolve()), model_path.stat().st_mtime)
    names = model.names if hasattr(model, "names") else {0: "state", 1: "arrow"}
    class_names = {
        str(name).lower()
        for name in (names.values() if hasattr(names, "values") else names)
    }
    required = {"state", "arrow"} if diagram_type == "state-machine" else {"place", "transition", "arc"}
    if not required.issubset(class_names):
        raise RuntimeError(
            f"YOLO model classes {sorted(class_names)!r} do not contain the required "
            f"{sorted(required)} classes for {diagram_type}; use matching trained weights"
        )
    supported = (
        STATE_MACHINE_CLASSES if diagram_type == "state-machine"
        else {"place", "petri_place", "transition", "petri_transition", "arc", "petri_arc"}
    )
    detections: list[Detection] = []
    for result in model(str(image_path), conf=confidence, verbose=False):
        for box in result.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            kind = str(names.get(cls, cls)).lower()
            if kind not in supported:
                continue
            detections.append(Detection(kind, (x1, y1, x2, y2), float(box.conf[0])))
    return detections


@lru_cache(maxsize=4)
def _load_model(path: str, _mtime: float):
    from ultralytics import YOLO
    return YOLO(path)
