"""Optional Ultralytics adapter using the existing state/arrow class convention."""

from __future__ import annotations

from pathlib import Path
from .reconstruct import Detection


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
    model = YOLO(str(model_path))
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
    detections: list[Detection] = []
    for result in model(str(image_path), conf=confidence, verbose=False):
        for box in result.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            kind = str(names.get(cls, cls)).lower()
            supported = (
                {"state", "arrow", "transition", "arrowhead", "initial", "final"}
                if diagram_type == "state-machine"
                else {"place", "petri_place", "transition", "petri_transition", "arc", "petri_arc"}
            )
            if kind not in supported:
                continue
            detections.append(Detection(kind, (x1, y1, x2, y2), float(box.conf[0])))
    return detections
