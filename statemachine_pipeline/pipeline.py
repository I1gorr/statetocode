"""Image -> canonical JSON in one call (detector + reconstruction + validation)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .detector import detect
from .reconstruct import Detection, Reconstruction, reconstruct
from .validation import ValidationReport, validate_state_machine


@dataclass
class PipelineResult:
    detections: list[Detection]
    reconstruction: Reconstruction
    validation: ValidationReport

    @property
    def machine(self):
        return self.reconstruction.machine


def run_pipeline(image_path: str | Path, model_path: str | Path, *, confidence: float = 0.4,
                 ocr: bool = True, diagram_id: str | None = None) -> PipelineResult:
    image_path = Path(image_path)
    detections = detect(image_path, model_path, confidence=confidence)
    result = reconstruct(detections, diagram_id or image_path.stem, image_path, ocr=ocr)
    return PipelineResult(detections, result, validate_state_machine(result.machine))
