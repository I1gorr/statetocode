"""Gemini code-generation adapter. The SDK is optional until generation is requested."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any
from .models import StateMachine
from .validation import validate_state_machine

SYSTEM_PROMPT_VERSION = "state-machine-codegen-1"


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


class GeminiGenerator:
    def __init__(self, *, model: str | None = None, api_key: str | None = None) -> None:
        _load_dotenv()
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        try:
            self.max_retries = max(0, int(os.getenv("GEMINI_MAX_RETRIES", "2")))
        except ValueError:
            self.max_retries = 2
        # An explicit empty key is useful for tests and must not fall back to
        # an ambient credential unexpectedly.
        self.api_key = os.getenv("GEMINI_API_KEY") if api_key is None else api_key

    def generate(self, machine: StateMachine, *, target_language: str = "python", requirements: str = "") -> GeneratedCode:
        report = validate_state_machine(machine)
        report.raise_for_errors()
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
        prompt = self._prompt(machine, target_language, requirements)
        config = {
            "system_instruction": self._system_instruction(),
            "response_mime_type": "application/json",
            "temperature": 0,
        }
        for attempt in range(self.max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
                break
            except genai_errors.APIError as exc:
                retryable = exc.code in {429, 500, 502, 503, 504}
                if not retryable or attempt >= self.max_retries:
                    raise RuntimeError(
                        f"Gemini API request failed ({exc.code}): {exc.message}"
                    ) from exc
                time.sleep(2 ** attempt)
        raw = getattr(response, "text", None)
        if not raw:
            raise RuntimeError("Gemini returned an empty response")
        try:
            value: dict[str, Any] = json.loads(raw)
            source_code = str(value["source_code"]).strip()
            if source_code.startswith("```"):
                lines = source_code.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                source_code = "\n".join(lines).strip()
            return GeneratedCode(
                source_code=source_code,
                target_language=str(value["target_language"]),
                state_mapping=dict(value.get("state_mapping", {})),
                assumptions=list(value.get("assumptions", [])),
                warnings=list(value.get("warnings", [])),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Gemini response did not match the code-generation contract: {exc}") from exc

    def explain(self, machine: StateMachine, *, question: str = "") -> dict[str, Any]:
        """Return a structured explanation without allowing Gemini to alter the graph."""
        validate_state_machine(machine).raise_for_errors()
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        try:
            from google import genai
            from google.genai import errors as genai_errors
        except ImportError as exc:
            raise RuntimeError("Install google-genai in the active environment") from exc
        prompt = json.dumps({
            "canonical_state_machine": machine.to_dict(),
            "question": question,
            "required_fields": ["summary", "execution_flows", "ambiguities", "warnings"],
        }, sort_keys=True)
        try:
            response = genai.Client(api_key=self.api_key).models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "system_instruction": (
                        "Explain the supplied canonical JSON only. Never add or change states or transitions. "
                        "Return JSON with summary, execution_flows, ambiguities, and warnings."
                    ),
                    "response_mime_type": "application/json",
                    "temperature": 0,
                },
            )
        except genai_errors.APIError as exc:
            raise RuntimeError(f"Gemini API request failed ({exc.code}): {exc.message}") from exc
        try:
            value = json.loads(getattr(response, "text", ""))
            return {
                "summary": str(value["summary"]),
                "execution_flows": list(value.get("execution_flows", [])),
                "ambiguities": list(value.get("ambiguities", [])),
                "warnings": list(value.get("warnings", [])),
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Gemini explanation did not match its JSON contract: {exc}") from exc

    @staticmethod
    def _system_instruction() -> str:
        return (
            f"{SYSTEM_PROMPT_VERSION}. Treat the supplied canonical JSON as the only source of truth. "
            "Implement every valid transition and preserve source, target, direction, events, guards, and actions. "
            "For a transition whose event is null, accept its transition id (and document that fallback) rather "
            "than inventing an event name. Do not invent ambiguous behavior. Return JSON only with source_code, "
            "target_language, state_mapping, assumptions, and warnings. Generate complete runnable code with an "
            "explicit current-state variable, event dispatch, guard/action hooks, initial-state initialization, "
            "final-state handling, self-loop support, parallel-transition preservation, and explicit invalid-event errors."
        )

    @staticmethod
    def _prompt(machine: StateMachine, language: str, requirements: str) -> str:
        return json.dumps({
            "canonical_state_machine": machine.to_dict(),
            "target_language": language,
            "runtime_requirements": "submit(event), current_state, explicit invalid-transition errors",
            "user_requirements": requirements,
        }, sort_keys=True)
