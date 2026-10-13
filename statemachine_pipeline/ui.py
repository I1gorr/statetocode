"""Local web editor: upload a diagram, inspect detections, edit the graph, generate code.

Dependency-free (``http.server`` + static HTML/JS). The canonical JSON is the
single source of truth: every graph edit produces new JSON, which is validated
server-side; generated code is re-derived from JSON and marked stale on edits.
Binds to 127.0.0.1 by default; it is a single-user development tool.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .models import SCHEMA_VERSION, StateMachine
from .network import NetworkGraph, validate_network_graph
from .validation import validate_state_machine

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_FILES = {"index.html", "app.js", "app.css", "network.html"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_JSON_BYTES = 5 * 1024 * 1024
IMAGE_SIGNATURES = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"RIFF", b"BM")


def element_issues(machine: StateMachine, messages: list[str]) -> dict[str, list[str]]:
    """Map validation messages ($.states[i] / $.transitions[i]) to element IDs."""
    issues: dict[str, list[str]] = {}
    for message in messages:
        match = re.match(r"\$\.(states|transitions)\[(\d+)\]", message)
        if not match:
            continue
        collection = machine.states if match.group(1) == "states" else machine.transitions
        index = int(match.group(2))
        if index < len(collection):
            issues.setdefault(collection[index].id, []).append(message)
    return issues


def validation_payload(document: dict[str, Any]) -> dict[str, Any]:
    report = validate_state_machine(document)
    payload: dict[str, Any] = report.to_dict()
    if report.valid:
        machine = StateMachine.from_dict(document)
        payload["element_errors"] = element_issues(machine, report.errors)
        payload["element_warnings"] = element_issues(machine, report.warnings)
        from .analysis import analyze
        analysis = analyze(machine)
        payload["analysis"] = {
            "reachable_states": analysis.reachable_states,
            "unreachable_states": analysis.unreachable_states,
            "dead_ends": analysis.dead_ends,
            "cycles": analysis.cycles,
        }
    else:
        payload["element_errors"] = {}
        payload["element_warnings"] = {}
    return payload


class EditorSession:
    """Mutable server-side state for one editor instance."""

    def __init__(self, path: Path | None, model_path: Path | None, upload_dir: Path, corrections: Path) -> None:
        self.path = path
        self.model_path = model_path
        self.upload_dir = upload_dir
        self.corrections = corrections
        self.lock = threading.Lock()
        self.image: Path | None = None
        self.detections: list[dict[str, Any]] = []
        self.predicted: dict[str, Any] | None = None
        self.document: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION, "diagram_id": "untitled", "states": [], "transitions": [],
            "metadata": {"source": "user", "warnings": []},
        }
        if path is not None and path.is_file():
            self.document = json.loads(path.read_text(encoding="utf-8"))
            source = (self.document.get("source_image") or {}).get("path")
            if source and Path(source).is_file():
                self.image = Path(source)

    def save(self, document: dict[str, Any]) -> None:
        self.document = document
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False)
            self.path.write_text(payload + "\n", encoding="utf-8")


def serve(
    path: str | Path | None, host: str = "127.0.0.1", port: int = 8000,
    *, diagram_type: str = "auto", model_path: str | Path | None = None,
    upload_dir: str | Path | None = None, corrections_path: str | Path | None = None,
    ready: threading.Event | None = None, server_holder: list | None = None,
) -> None:
    root = Path(__file__).resolve().parent.parent
    session = EditorSession(
        Path(path) if path else None,
        Path(model_path) if model_path else None,
        Path(upload_dir) if upload_dir else root / "data" / "uploads",
        Path(corrections_path) if corrections_path else root / "data" / "corrections" / "corrections.jsonl",
    )

    def network_mode() -> bool:
        if diagram_type == "network":
            return True
        if diagram_type == "state-machine":
            return False
        return "nodes" in session.document

    class Handler(BaseHTTPRequestHandler):
        server_version = "StateDiagramEditor/1.1"

        def _send(self, code: int, body: bytes | str, content_type: str) -> None:
            encoded = body.encode() if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(encoded)

        def _json(self, code: int, value: Any) -> None:
            self._send(code, json.dumps(value, ensure_ascii=False), "application/json; charset=utf-8")

        def _body(self, limit: int) -> bytes:
            length = int(self.headers.get("Content-Length", "0") or 0)
            if length > limit:
                raise ValueError(f"request body exceeds {limit // (1024 * 1024)} MB")
            return self.rfile.read(length)

        def _json_body(self) -> dict[str, Any]:
            value = json.loads(self._body(MAX_JSON_BYTES) or b"{}")
            if not isinstance(value, dict):
                raise ValueError("expected a JSON object")
            return value

        def do_GET(self) -> None:
            route = urlparse(self.path).path
            if route in {"/", "/index.html"}:
                page = "network.html" if network_mode() else "index.html"
                self._send(200, (STATIC_DIR / page).read_bytes(), "text/html; charset=utf-8")
            elif route.startswith("/static/") and route[len("/static/"):] in STATIC_FILES:
                name = route[len("/static/"):]
                content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
                self._send(200, (STATIC_DIR / name).read_bytes(), content_type)
            elif route == "/api/machine":
                self._json(200, session.document)
            elif route == "/api/session":
                self._json(200, {
                    "has_image": session.image is not None,
                    "detections": session.detections,
                    "model": str(session.model_path) if session.model_path else None,
                    "file": str(session.path) if session.path else None,
                    "has_prediction": session.predicted is not None,
                })
            elif route == "/api/image":
                if session.image is None or not session.image.is_file():
                    self._json(404, {"errors": ["no image loaded"]})
                    return
                content_type = mimetypes.guess_type(session.image.name)[0] or "image/png"
                self._send(200, session.image.read_bytes(), content_type)
            else:
                self._json(404, {"errors": ["not found"]})

        def do_POST(self) -> None:
            route = urlparse(self.path).path
            handlers = {
                "/api/machine": self._post_machine,
                "/api/validate": self._post_validate,
                "/api/upload": self._post_upload,
                "/api/generate": self._post_generate,
                "/api/explain": self._post_explain,
                "/api/suggest": self._post_suggest,
                "/api/explain-code": self._post_explain_code,
                "/api/correction": self._post_correction,
            }
            handler = handlers.get(route)
            if handler is None:
                self._json(404, {"errors": ["not found"]})
                return
            try:
                handler()
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                self._json(400, {"errors": [str(exc)]})
            except RuntimeError as exc:  # missing API key, model failures, unavailable tools
                self._json(503, {"errors": [str(exc)]})
            except OSError as exc:
                self._json(500, {"errors": [f"I/O error: {exc}"]})

        # -- handlers --------------------------------------------------

        def _post_machine(self) -> None:
            body = self._json_body()
            if "nodes" in body:
                errors = validate_network_graph(body)
                if errors:
                    self._json(400, {"errors": errors})
                    return
                graph = NetworkGraph.from_dict(body)
                with session.lock:
                    session.save(json.loads(graph.to_json()))
                self._json(200, json.loads(graph.to_json()))
                return
            report = validation_payload(body)
            if not report["valid"]:
                self._json(400, report)
                return
            canonical = json.loads(StateMachine.from_dict(body).to_json())
            with session.lock:
                session.save(canonical)
            self._json(200, canonical)

        def _post_validate(self) -> None:
            self._json(200, validation_payload(self._json_body()))

        def _post_upload(self) -> None:
            data = self._body(MAX_UPLOAD_BYTES)
            if not data or not data.startswith(IMAGE_SIGNATURES):
                raise ValueError("upload must be a PNG, JPEG, WebP or BMP image")
            import cv2
            import numpy as np
            pixels = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if pixels is None:
                raise ValueError("the uploaded file could not be decoded as an image")
            if session.model_path is None or not session.model_path.exists():
                raise RuntimeError("no trained detector weights found; pass --model to `main.py serve`")
            digest = hashlib.sha256(data).hexdigest()[:16]
            name = re.sub(r"[^A-Za-z0-9_.-]", "_", self.headers.get("X-Filename", "diagram.png"))[-60:]
            suffix = Path(name).suffix.lower() if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"} else ".png"
            session.upload_dir.mkdir(parents=True, exist_ok=True)
            target = session.upload_dir / f"{digest}{suffix}"
            target.write_bytes(data)
            from .pipeline import run_pipeline
            result = run_pipeline(target, session.model_path, diagram_id=Path(name).stem or digest)
            document = json.loads(result.machine.to_json())
            detections = [{"kind": d.kind, "bbox": [round(v, 1) for v in d.bbox], "confidence": round(d.confidence, 3)}
                          for d in result.detections]
            with session.lock:
                session.image = target
                session.detections = detections
                session.predicted = document
                session.save(document)
            self._json(200, {"machine": document, "detections": detections,
                             "validation": validation_payload(document)})

        def _machine_from(self, body: dict[str, Any]) -> StateMachine:
            document = body.get("machine", session.document)
            report = validate_state_machine(document)
            report.raise_for_errors()
            return StateMachine.from_dict(document)

        def _post_generate(self) -> None:
            body = self._json_body()
            machine = self._machine_from(body)
            from .codegen import generate_python, generate_tests, verify_generated
            backend = body.get("backend", "template")
            assumptions: list[str] = []
            warnings: list[str] = []
            if backend == "gemini":
                from .gemini import GeminiGenerator
                generated = GeminiGenerator().generate(machine)
                source = generated.source_code
                assumptions, warnings = generated.assumptions, generated.warnings
            elif backend == "template":
                source = generate_python(machine)
            else:
                raise ValueError(f"unknown backend {backend!r}")
            report = verify_generated(source, machine)
            self._json(200, {
                "backend": backend, "source": source,
                "tests": generate_tests(machine, "state_machine") if report.valid else None,
                "conformance": report.to_dict(), "assumptions": assumptions, "warnings": warnings,
                "machine_hash": hashlib.sha256(machine.to_json().encode()).hexdigest(),
            })

        def _post_explain(self) -> None:
            body = self._json_body()
            from .gemini import GeminiGenerator
            self._json(200, GeminiGenerator().explain(self._machine_from(body), question=str(body.get("question", ""))))

        def _post_suggest(self) -> None:
            body = self._json_body()
            from .gemini import GeminiGenerator
            data = session.image.read_bytes() if session.image and session.image.is_file() else None
            mime = mimetypes.guess_type(session.image.name)[0] if session.image else "image/png"
            self._json(200, GeminiGenerator().suggest_corrections(self._machine_from(body), data, mime or "image/png"))

        def _post_explain_code(self) -> None:
            body = self._json_body()
            from .gemini import GeminiGenerator
            self._json(200, GeminiGenerator().explain_code(self._machine_from(body), str(body["source"])))

        def _post_correction(self) -> None:
            body = self._json_body()
            machine = self._machine_from(body)
            if session.image is None:
                raise ValueError("corrections need an uploaded image")
            from .corrections import save_correction
            record = save_correction(
                session.corrections, image=str(session.image), detections=session.detections,
                machine=json.loads(machine.to_json()), predicted=session.predicted,
                reviewer=body.get("reviewer") or None, split=str(body.get("split", "train")),
            )
            self._json(200, {"saved": str(session.corrections), "changes": record["changes"], "split": record["split"]})

        def log_message(self, *_: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    if server_holder is not None:
        server_holder.append(server)
    if ready is not None:
        ready.set()
    try:
        server.serve_forever()
    finally:
        server.server_close()
