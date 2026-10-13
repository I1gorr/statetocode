"""Editor graph operations (static/app.js `Ops`), run under Node.

The editor keeps graph, JSON and code in sync by routing every edit through
these pure functions; the resulting JSON must stay valid for the server.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from statemachine_pipeline.validation import validate_state_machine

APP = Path(__file__).resolve().parent.parent / "statemachine_pipeline" / "static" / "app.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")

SCRIPT = r"""
const Ops = require(process.argv[1]);
const out = {};
let m = Ops.emptyMachine(), a, b, c, t0, t1, loop;
[m, a] = Ops.addState(m, {x: 0, y: 0}, 'Idle');
[m, b] = Ops.addState(m, {x: 200, y: 0}, 'Busy');
[m, c] = Ops.addState(m, {x: 100, y: 200}, 'Done');
m = Ops.updateState(m, c, {type: 'final'});
[m, t0] = Ops.addTransition(m, a, b, {event: 'go'});
[m, t1] = Ops.addTransition(m, b, a, {event: 'back'});
[m, loop] = Ops.addTransition(m, b, b, {event: 'tick'});
out.ids = [a, b, c, t0, t1, loop];
out.types = m.states.map(s => s.type);
out.loopDirection = m.transitions.find(t => t.id === loop).direction;
const history = new Ops.History();
history.push(m);
const reversed = Ops.reverseTransition(m, t0);
out.reversed = [reversed.transitions[0].source, reversed.transitions[0].target];
const undone = history.undo(reversed);
out.undone = [undone.transitions[0].source, undone.transitions[0].target];
out.redone = history.redo(undone).transitions[0].source;
m = Ops.updateState(m, b, {type: 'initial'});
out.initials = m.states.filter(s => s.type === 'initial').map(s => s.id);
m.uncertainties = [{element_id: t1, field: 'direction', reason: 'r', candidates: []}];
const geometry = Ops.edgeGeometry(m, () => 30);
out.parallelDistinct = geometry[t0].d !== geometry[t1].d;
out.loopIsCurve = geometry[loop].d.includes('C');
const deleted = Ops.deleteState(m, a);
out.afterDelete = {states: deleted.states.map(s => s.id), transitions: deleted.transitions.map(t => t.id), uncertainties: deleted.uncertainties.length};
let n; [n] = Ops.addDevice(m, 'Client', [a, b]); let link; [n] = Ops.addDevice(n, 'Server', []); [n, link] = Ops.addLink(n, 'D0', 'D1', 'Ethernet');
out.network = n.network;
out.removedDevice = Ops.deleteDevice(n, 'D1').network.links.length;
out.linkErrors = [['D0', 'D0'], ['D1', 'D0'], ['D0', 'D9']].map(([x, y]) => { try { Ops.addLink(n, x, y); return 'ok'; } catch (e) { return e.message; } });
out.standalonePosition = n.network.devices[1].position !== null;
out.layout = Ops.autoLayout(m).states.map(s => s.position);
out.label = Ops.transitionText({event: 'connect', guard: 'n<3', action: 'log', label: 'x'});
out.final = m;
console.log(JSON.stringify(out));
"""


def run_ops():
    result = subprocess.run(["node", "-e", SCRIPT, str(APP)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_editor_operations():
    out = run_ops()
    assert out["ids"] == ["S0", "S1", "S2", "T0", "T1", "T2"]
    assert out["types"] == ["initial", "normal", "final"]
    assert out["loopDirection"] == "loop"
    assert out["reversed"] == ["S1", "S0"]
    assert out["undone"] == ["S0", "S1"] and out["redone"] == "S1"
    assert out["initials"] == ["S1"]  # setting a new initial state demotes the old one
    assert out["parallelDistinct"] and out["loopIsCurve"]
    assert out["afterDelete"] == {"states": ["S1", "S2"], "transitions": ["T2"], "uncertainties": 0}
    assert out["network"]["links"] == [{"id": "L0", "source": "D0", "target": "D1", "medium": "Ethernet", "label": None}]
    assert out["removedDevice"] == 0
    assert out["linkErrors"] == ["A device cannot be linked to itself", "D1 and D0 are already linked", "Choose two existing devices"]
    assert out["standalonePosition"]
    assert len({(p["x"], p["y"]) for p in out["layout"]}) == 3
    assert out["label"] == "connect [n<3] / log"


def test_edited_json_is_valid_for_the_server():
    final = run_ops()["final"]
    report = validate_state_machine(final)
    assert report.valid, report.errors
