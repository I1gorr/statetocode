"""JSON Schema (draft 2020-12) for the canonical state-machine contract."""

from __future__ import annotations

from typing import Any

_NULLABLE_STRING = {"type": ["string", "null"]}
_CONFIDENCE = {"type": ["number", "null"], "minimum": 0, "maximum": 1}
_POSITION = {
    "type": "object",
    "required": ["x", "y"],
    "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
    "additionalProperties": False,
}
_BBOX = {
    "type": ["array", "null"],
    "items": {"type": "number"},
    "minItems": 4,
    "maxItems": 4,
}
_PROVENANCE = {"enum": ["detector", "ocr", "user", "llm", "synthetic", "unknown"]}

STATE_MACHINE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://statetocode.local/state-machine-1.1.json",
    "title": "Canonical state machine",
    "type": "object",
    "required": ["schema_version", "diagram_id", "states", "transitions"],
    "properties": {
        "schema_version": {"enum": ["1.0", "1.1"]},
        "diagram_id": {"type": "string", "minLength": 1},
        "metadata": {
            "type": "object",
            "properties": {"warnings": {"type": "array", "items": {"type": "string"}}},
        },
        "states": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "label": _NULLABLE_STRING,
                    "type": {"enum": ["initial", "normal", "final"]},
                    "position": {"anyOf": [_POSITION, {"type": "null"}]},
                    "confidence": _CONFIDENCE,
                    "bbox": _BBOX,
                    "shape": {"enum": ["circle", "double_circle", "rounded_rect", "unknown"]},
                    "label_confidence": _CONFIDENCE,
                    "provenance": _PROVENANCE,
                },
                "additionalProperties": False,
            },
        },
        "transitions": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "source", "target"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "source": {"type": "string", "minLength": 1},
                    "target": {"type": "string", "minLength": 1},
                    "event": _NULLABLE_STRING,
                    "guard": _NULLABLE_STRING,
                    "action": _NULLABLE_STRING,
                    "label": _NULLABLE_STRING,
                    "confidence": _CONFIDENCE,
                    "direction": {"enum": ["forward", "backward", "loop", "unknown"]},
                    "points": {"type": "array", "items": _POSITION},
                    "label_bbox": _BBOX,
                    "label_confidence": _CONFIDENCE,
                    "kind": {"enum": ["state_change", "network_event", "internal"]},
                    "network": {
                        "anyOf": [
                            {"type": "null"},
                            {
                                "type": "object",
                                "required": ["event_type"],
                                "properties": {
                                    "event_type": {"enum": [
                                        "connect", "accept", "send", "receive", "ack", "timeout",
                                        "failure", "retry", "close", "reset", "other",
                                    ]},
                                    "protocol": _NULLABLE_STRING,
                                    "packet": _NULLABLE_STRING,
                                },
                                "additionalProperties": False,
                            },
                        ]
                    },
                    "provenance": _PROVENANCE,
                },
                "additionalProperties": False,
            },
        },
        "network": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {
                        "devices": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["id"],
                                "properties": {
                                    "id": {"type": "string", "minLength": 1},
                                    "label": _NULLABLE_STRING,
                                    "role": _NULLABLE_STRING,
                                    "state_ids": {"type": "array", "items": {"type": "string"}},
                                    "position": {"anyOf": [_POSITION, {"type": "null"}]},
                                },
                                "additionalProperties": False,
                            },
                        },
                        "links": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["id", "source", "target"],
                                "properties": {
                                    "id": {"type": "string", "minLength": 1},
                                    "source": {"type": "string"},
                                    "target": {"type": "string"},
                                    "medium": _NULLABLE_STRING,
                                    "label": _NULLABLE_STRING,
                                },
                                "additionalProperties": False,
                            },
                        },
                    },
                    "additionalProperties": False,
                },
            ]
        },
        "uncertainties": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["element_id", "field", "reason"],
                "properties": {
                    "element_id": {"type": "string"},
                    "field": {"type": "string"},
                    "reason": {"type": "string"},
                    "candidates": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": False,
            },
        },
        "source_image": {
            "anyOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "required": ["path", "width", "height"],
                    "properties": {
                        "path": {"type": "string"},
                        "width": {"type": "integer", "minimum": 1},
                        "height": {"type": "integer", "minimum": 1},
                        "sha256": _NULLABLE_STRING,
                    },
                    "additionalProperties": False,
                },
            ]
        },
    },
    "additionalProperties": False,
}


def schema_errors(document: Any) -> list[str]:
    """Return JSON-path-style schema violations (empty when valid).

    ``jsonschema`` is preferred; if it is unavailable the caller still gets the
    hand-written structural checks in ``validation.py``.
    """
    try:
        import jsonschema
    except ImportError:
        return []
    validator = jsonschema.Draft202012Validator(STATE_MACHINE_SCHEMA)
    errors = []
    for error in sorted(validator.iter_errors(document), key=lambda e: list(e.absolute_path)):
        path = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}" for part in error.absolute_path
        )
        errors.append(f"{path}: {error.message}")
    return errors
