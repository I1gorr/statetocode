"""Gemini adapter: explanations, ambiguity suggestions, optional code generation.

Gemini never edits the graph. Every response is parsed against a JSON
contract; suggestions are returned for the user to accept or reject, and
generated code is accepted only after deterministic conformance checks
(``codegen.verify_generated``). Graph correctness never depends on the LLM.
The SDK is imported lazily so the rest of the pipeline works without it.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any
from .models import StateMachine
from .validation import validate_state_machine

SYSTEM_PROMPT_VERSION = "state-machine-codegen-2"
SUGGESTION_FIELDS = {"label", "event", "guard", "action", "source", "target", "direction", "type", "add_transition"}
ALLOWED_VALUES = {
    "direction": {"forward", "loop", "unknown", "reverse"},
    "type": {"initial", "normal", "final"},
}


def _as_list(value: Any) -> list[str]:
    """LLMs sometimes return a string where a list was requested."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _load_dotenv() -> None:
    """Load local .env values when python-dotenv is unavailable.

    Existing environment variables always win, and this deliberately supports
    only the simple KEY=VALUE form used by this project.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        env_path = os.path.join(os.getcwd(), ".env")
        if not os.path.isfile(env_path):
            env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
        if not os.path.isfile(env_path):
            return
        with open(env_path, encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                name, value = name.strip(), value.strip()
                if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
                    value = value[1:-1]
                os.environ.setdefault(name, value)
    else:
        load_dotenv()


@dataclass(frozen=True)
class GeneratedCode:
    source_code: str
    target_language: str
    state_mapping: dict[str, str]
    assumptions: list[str]
    warnings: list[str]


def _strip_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


class GeminiGenerator:
    def __init__(self, *, model: str | None = None, api_key: str | None = None) -> None:
        _load_dotenv()
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        try:
            self.max_retries = max(0, int(os.getenv("GEMINI_MAX_RETRIES", "2")))
        except ValueError:
            self.max_retries = 2
        try:
            self.timeout = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "60"))
        except ValueError:
            self.timeout = 60.0
        # An explicit empty key is useful for tests and must not fall back to
        # an ambient credential unexpectedly.
        self.api_key = os.getenv("GEMINI_API_KEY") if api_key is None else api_key

    # -- transport -----------------------------------------------------------

    def _call(self, contents: Any, system_instruction: str) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        try:
            from google import genai
            from google.genai import errors as genai_errors
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed in the active Python environment. "
                "Activate .venv first: `source .venv/bin/activate`, then run "
                "`python -m pip install -r requirements.txt`."
            ) from exc
        client = genai.Client(api_key=self.api_key)
        config = {
            "system_instruction": system_instruction,
            "response_mime_type": "application/json",
            "temperature": 0,
        }
        response = None
        for attempt in range(self.max_retries + 1):
            try:
                response = client.models.generate_content(model=self.model, contents=contents, config=config)
                break
            except genai_errors.APIError as exc:
                retryable = exc.code in {429, 500, 502, 503, 504}
                if not retryable or attempt >= self.max_retries:
                    raise RuntimeError(f"Gemini API request failed ({exc.code}): {exc.message}") from exc
                time.sleep(2 ** attempt)
        raw = getattr(response, "text", None)
        if not raw:
            raise RuntimeError("Gemini returned an empty response")
        try:
            value = json.loads(_strip_fence(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Gemini response was not valid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError("Gemini response must be a JSON object")
        return value

    @staticmethod
    def _image_part(image_bytes: bytes, mime_type: str) -> Any:
        from google.genai import types
        return types.Part.from_bytes(data=image_bytes, mime_type=mime_type)

    # -- code generation -----------------------------------------------------

    def generate(self, machine: StateMachine, *, target_language: str = "python", requirements: str = "") -> GeneratedCode:
        validate_state_machine(machine).raise_for_errors()
        value = self._call(self._prompt(machine, target_language, requirements), self._system_instruction())
        try:
            return GeneratedCode(
                source_code=_strip_fence(str(value["source_code"])),
                target_language=str(value["target_language"]),
                state_mapping=dict(value.get("state_mapping", {})),
                assumptions=_as_list(value.get("assumptions")),
                warnings=_as_list(value.get("warnings")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Gemini response did not match the code-generation contract: {exc}") from exc

    @staticmethod
    def _system_instruction() -> str:
        return (
            f"{SYSTEM_PROMPT_VERSION}. Treat the supplied canonical JSON as the only source of truth. "
            "Implement every transition exactly: source, target, trigger, guard, action. The trigger of a "
            "transition is its event; when the event is null the trigger is the transition id (document it; "
            "never invent an event name). For Python, define module constants GUARDS (tuple of guard names) "
            "and a class StateMachine(guards=None, actions=None, initial_state=<initial>) with attribute "
            "current_state holding state ids and submit(event, context=None) returning the new state id. "
            "Semantics: in a final state raise TerminalStateError; no transition for the trigger -> "
            "InvalidEventError; guard callables decide enabling and a missing guard implementation raises "
            "GuardNotImplementedError; none enabled -> GuardRejectedError (subclass of InvalidEventError); "
            "more than one enabled -> NondeterministicTransitionError; run the action hook (if registered) "
            "then change state. Use only the Python standard library modules dataclasses/typing/enum; no I/O, "
            "no imports of os/sys/socket/subprocess. Return JSON only with source_code, target_language, "
            "state_mapping, assumptions, and warnings."
        )

    @staticmethod
    def _prompt(machine: StateMachine, language: str, requirements: str) -> str:
        return json.dumps({
            "canonical_state_machine": machine.to_dict(),
            "target_language": language,
            "runtime_requirements": "submit(event, context), current_state, explicit error classes",
            "user_requirements": requirements,
        }, sort_keys=True)

    # -- explanations ------------------------------------------------------

    def explain(self, machine: StateMachine, *, question: str = "") -> dict[str, Any]:
        """Structured explanation grounded in the JSON and deterministic analysis."""
        report = validate_state_machine(machine)
        report.raise_for_errors()
        from .analysis import analyze
        analysis = analyze(machine)
        prompt = json.dumps({
            "canonical_state_machine": machine.to_dict(),
            "deterministic_analysis": {
                "reachable_states": analysis.reachable_states,
                "unreachable_states": analysis.unreachable_states,
                "dead_ends": analysis.dead_ends,
                "cycles": analysis.cycles,
                "validation_warnings": report.warnings,
            },
            "question": question,
            "required_fields": {
                "summary": "string",
                "states": "[{id, description}] one per state",
                "transitions": "[{id, description}] one per transition",
                "network_events": "[{transition_id, description}] only for transitions with kind network_event",
                "execution_flows": "[string] representative paths from the initial state",
                "ambiguities": "[string] unclear or missing information, never resolved by guessing",
                "warnings": "[string]",
                "answer": "string answer to the question, or empty",
            },
        }, sort_keys=True)
        value = self._call(prompt, (
            "Explain the supplied canonical state machine for a reader. Use only the JSON and the "
            "deterministic analysis. Never add, remove or change states or transitions, never invent "
            "events, guards, actions or protocol behaviour that are not in the JSON; when a label is null "
            "say it is unlabelled. Distinguish states (conditions the system is in), transitions (changes "
            "between states), network events (packets/timeouts/failures that trigger transitions) and "
            "network links (physical connections, only if present under network). Return JSON only."
        ))
        known_states = {s.id for s in machine.states}
        known_transitions = {t.id for t in machine.transitions}
        try:
            result = {
                "summary": str(value["summary"]),
                "states": [{"id": str(s["id"]), "description": str(s["description"])}
                           for s in value.get("states", []) if str(s.get("id")) in known_states],
                "transitions": [{"id": str(t["id"]), "description": str(t["description"])}
                                for t in value.get("transitions", []) if str(t.get("id")) in known_transitions],
                "network_events": [
                    {"transition_id": str(n["transition_id"]), "description": str(n["description"])}
                    for n in value.get("network_events", []) if str(n.get("transition_id")) in known_transitions
                ],
                "execution_flows": _as_list(value.get("execution_flows")),
                "ambiguities": _as_list(value.get("ambiguities")),
                "warnings": _as_list(value.get("warnings")),
                "answer": str(value.get("answer", "")),
            }
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"Gemini explanation did not match its JSON contract: {exc}") from exc
        dropped = len(value.get("states", [])) - len(result["states"]) + len(value.get("transitions", [])) - len(result["transitions"])
        if dropped > 0:
            result["warnings"].append(f"{dropped} described element(s) do not exist in the diagram and were dropped")
        return result

    def suggest_corrections(self, machine: StateMachine, image_bytes: bytes | None = None,
                            mime_type: str = "image/png") -> dict[str, Any]:
        """Ask for evidence-based corrections of uncertain detections. Never applied automatically."""
        validate_state_machine(machine).raise_for_errors()
        prompt = json.dumps({
            "canonical_state_machine": machine.to_dict(),
            "uncertainties": [u.__dict__ for u in machine.uncertainties],
            "instructions": (
                "Compare the JSON with the diagram image. Propose corrections only where the image clearly "
                "shows something different (misread label, reversed arrow, wrong endpoint, missing initial "
                "or final marker, missing arrow). Each suggestion: element_id (existing state or transition "
                f"id), field (one of {sorted(SUGGESTION_FIELDS)}), suggested_value (string or null), evidence "
                "(what in the image supports it), confidence (0..1). Allowed values: direction is 'reverse' "
                "(swap source and target), 'forward', 'loop' or 'unknown'; type is initial/normal/final; "
                "source and target are state ids. For an arrow missing from the JSON use field "
                "'add_transition' with element_id = source state id and suggested_value = target state id. "
                "If the image is unclear, do not suggest."
            ),
            "required_fields": {"suggestions": "[...]", "notes": "[string]"},
        }, sort_keys=True, default=list)
        contents: Any = prompt
        if image_bytes:
            contents = [self._image_part(image_bytes, mime_type), prompt]
        value = self._call(contents, (
            "You review an automatically reconstructed state diagram. You only suggest; a human decides. "
            "Do not invent labels you cannot read. Return JSON only."
        ))
        states = {s.id for s in machine.states}
        transitions = {t.id for t in machine.transitions}
        accepted, rejected = [], []
        for raw in value.get("suggestions", []) or []:
            try:
                suggestion = {
                    "element_id": str(raw["element_id"]),
                    "field": str(raw["field"]),
                    "suggested_value": None if raw.get("suggested_value") is None else str(raw["suggested_value"]),
                    "evidence": str(raw.get("evidence", "")),
                    "confidence": max(0.0, min(1.0, float(raw.get("confidence", 0.0)))),
                }
            except (KeyError, TypeError, ValueError):
                rejected.append({"suggestion": raw, "reason": "malformed"})
                continue
            field_name, value_ = suggestion["field"], suggestion["suggested_value"]
            element_ok = suggestion["element_id"] in states | transitions
            field_ok = field_name in SUGGESTION_FIELDS
            # The suggestion must be applicable as-is: right element kind, legal value.
            if field_name in {"source", "target", "add_transition"} and value_ not in states:
                field_ok = False
            if field_name in {"add_transition", "type"} and suggestion["element_id"] not in states:
                field_ok = False
            if field_name in {"source", "target", "direction", "event", "guard", "action"} and \
                    suggestion["element_id"] not in transitions:
                field_ok = False
            if field_name in ALLOWED_VALUES and value_ not in ALLOWED_VALUES[field_name]:
                field_ok = False
            if not (element_ok and field_ok):
                rejected.append({"suggestion": suggestion,
                                 "reason": "unknown element, field, value or state reference"})
                continue
            accepted.append(suggestion)
        return {"suggestions": accepted, "rejected": rejected,
                "notes": _as_list(value.get("notes"))}

    def explain_code(self, machine: StateMachine, source_code: str) -> dict[str, Any]:
        """Explain generated code and propose test cases; tests are checked against the oracle."""
        from .codegen import ReferenceError_, _step, initial_state_of

        value = self._call(json.dumps({
            "canonical_state_machine": machine.to_dict(),
            "source_code": source_code,
            "required_fields": {
                "explanation": "string",
                "test_cases": "[{name, events: [string], guards: {name: bool}, expected_states: [state ids after each event] or expected_error: string}]",
            },
        }, sort_keys=True), (
            "Explain how the code implements the state machine and propose concise test cases using only "
            "triggers present in the JSON. Return JSON only."
        ))
        checked = []
        for case in value.get("test_cases", []) or []:
            try:
                events = [str(e) for e in case.get("events", [])]
                guards = {str(k): bool(v) for k, v in (case.get("guards") or {}).items()}
            except (TypeError, AttributeError):
                continue
            state = initial_state_of(machine)
            observed: list[str] = []
            error = None
            for event in events:
                try:
                    state, _ = _step(machine, state, event, guards)
                    observed.append(state)
                except ReferenceError_ as exc:
                    error = exc.kind
                    break
            expected_states = [str(s) for s in case.get("expected_states", []) or []]
            expected_error = case.get("expected_error")
            agrees = (observed == expected_states and error is None and not expected_error) or \
                     (error is not None and expected_error is not None and str(expected_error).endswith(error))
            checked.append({"name": str(case.get("name", "")), "events": events, "guards": guards,
                            "expected_states": expected_states, "expected_error": expected_error,
                            "reference_states": observed, "reference_error": error,
                            "agrees_with_reference": agrees})
        return {"explanation": str(value.get("explanation", "")), "test_cases": checked}
