"""Editor HTTP API: JSON <-> graph <-> code stay consistent; bad input is rejected."""

import json
import threading
import urllib.error
import urllib.request

import pytest

from statemachine_pipeline.ui import serve


@pytest.fixture
def server(tmp_path):
    document = {
        "schema_version": "1.1", "diagram_id": "ui", "metadata": {"warnings": []},
        "states": [{"id": "S0", "label": "Idle", "type": "initial"}, {"id": "S1", "label": "Busy"}],
        "transitions": [{"id": "T0", "source": "S0", "target": "S1", "event": "go"}],
    }
    path = tmp_path / "machine.json"
    path.write_text(json.dumps(document))
    ready, holder = threading.Event(), []
    thread = threading.Thread(target=serve, args=(path, "127.0.0.1", 0), kwargs={
        "ready": ready, "server_holder": holder, "model_path": tmp_path / "missing.pt",
        "upload_dir": tmp_path / "uploads", "corrections_path": tmp_path / "corrections.jsonl",
    }, daemon=True)
    thread.start()
    ready.wait(5)
    base = f"http://127.0.0.1:{holder[0].server_address[1]}"
    yield base, path
    holder[0].shutdown()


def call(base, route, body=None, raw=None, headers=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    request = urllib.request.Request(base + route, data=data, headers=headers or {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_pages_and_static_files(server):
    base, _ = server
    status, body = call(base, "/")
    assert status == 200 and b"State Diagram Studio" in body
    assert call(base, "/static/app.js")[0] == 200
    assert call(base, "/static/../ui.py")[0] == 404


def test_edit_validate_save_reload_round_trip(server):
    base, path = server
    _, body = call(base, "/api/machine")
    machine = json.loads(body)
    machine["transitions"].append({"id": "T1", "source": "S1", "target": "S1", "event": "tick", "direction": "loop"})
    machine["states"][1]["label"] = "Working"
    status, saved = call(base, "/api/machine", machine)
    assert status == 200
    on_disk = json.loads(path.read_text())
    assert on_disk == json.loads(saved)
    assert [t["id"] for t in on_disk["transitions"]] == ["T0", "T1"]
    assert on_disk["states"][1]["label"] == "Working"
    _, reloaded = call(base, "/api/machine")
    assert json.loads(reloaded) == on_disk


def test_invalid_graph_is_rejected_with_element_errors(server):
    base, path = server
    before = path.read_text()
    bad = json.loads(before)
    bad["transitions"][0]["target"] = "S9"
    status, body = call(base, "/api/machine", bad)
    assert status == 400 and any("unknown state 'S9'" in e for e in json.loads(body)["errors"])
    assert path.read_text() == before  # nothing written
    status, body = call(base, "/api/validate", {"schema_version": "1.1", "diagram_id": "x", "states": [
        {"id": "S0", "type": "initial"}, {"id": "S1", "type": "initial"}], "transitions": []})
    assert json.loads(body)["valid"] is False
    assert call(base, "/api/machine", raw=b"{not json")[0] == 400


def test_generate_reflects_current_json(server):
    base, _ = server
    _, body = call(base, "/api/machine")
    machine = json.loads(body)
    machine["transitions"].append({"id": "T1", "source": "S1", "target": "S0", "event": "reset"})
    status, body = call(base, "/api/generate", {"machine": machine, "backend": "template"})
    result = json.loads(body)
    assert status == 200 and result["conformance"]["valid"]
    assert "trigger='reset'" in result["source"] and result["tests"]
    assert set(result["conformance"]["covered_transitions"]) == {"T0", "T1"}


def test_upload_validation_and_missing_model(server):
    base, _ = server
    status, body = call(base, "/api/upload", raw=b"hello", headers={"X-Filename": "x.png"})
    assert status == 400 and b"PNG" in body
    import cv2
    import numpy as np
    ok, png = cv2.imencode(".png", np.full((50, 50, 3), 255, np.uint8))
    status, body = call(base, "/api/upload", raw=png.tobytes(), headers={"X-Filename": "x.png"})
    assert status == 503 and b"weights" in body


def test_gemini_routes_fail_cleanly_without_key(server, monkeypatch):
    base, _ = server
    monkeypatch.setenv("GEMINI_API_KEY", "")
    status, body = call(base, "/api/explain", {})
    assert status == 503 and b"GEMINI_API_KEY" in body


def test_corrections_need_an_image(server):
    base, _ = server
    status, body = call(base, "/api/correction", {"split": "train"})
    assert status == 400 and b"image" in body
