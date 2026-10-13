# State diagram to code

The reusable pipeline is in `statemachine_pipeline/`. It uses the versioned JSON contract in
`models.py` between detection, reconstruction, validation, Gemini, and code checks.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env  # set GEMINI_API_KEY only for generation
python main.py images/test1.png --model yolo11n.pt --output outputs/diagram.json
python main.py generate outputs/diagram.json \
  --output outputs/state_machine.py \
  --report outputs/generation-report.json
python main.py analyze outputs/diagram.json
python main.py edit outputs/diagram.json
python main.py explain outputs/diagram.json --output outputs/explanation.json
python -m pytest -q
```

Use the Linux `.venv/bin/python` and `.venv/bin/pip`; do not use the legacy
`venv/Scripts/python.exe` directory, which contains Windows executables and is
not part of the Linux setup.

YOLO class names follow `dataset.yaml` (`state`, `arrow`). Labels and arrow semantics not
provided by the detector remain `null`/`unknown`; reconstruction warnings are stored in
`metadata.warnings` rather than silently inventing behavior. Gemini generation is optional and
requires `GEMINI_API_KEY`. Generate code from the validated JSON with:

Petri-net diagrams use a separate canonical contract with `places`, `transitions`, and
weighted `arcs`, plus an initial marking. Use a model trained with `place`, `transition`,
and `arc` classes:

```bash
python main.py images/petri.png --diagram-type petri-net \
  --model runs/petri/detect/weights/best.pt \
  --output outputs/petri.json
```

Network diagrams made from components drawn as circles are supported without reusing
the state-machine weights. OpenCV detects circular components and connecting strokes,
and optional Tesseract OCR reads labels inside circles:

```bash
python main.py images/network.png --diagram-type network \
  --output outputs/network.json
python main.py edit outputs/network.json
```

Generate deterministic synthetic network diagrams plus ground-truth JSON for testing:

```bash
python generate_network_diagrams.py --output outputs/network_dataset --count 20
python main.py edit outputs/network_dataset/network_0000.json --diagram-type network
```

The editor chooses the state-machine or network editor from the selected
`--diagram-type`; `auto` (the default) infers it from the JSON contract. Use
`--diagram-type state-machine` for state-machine JSON when an explicit selection is
preferred.

The network contract is `network-1.0` with `nodes` and `edges`; low-confidence or
unreadable labels are retained as `null` and surfaced in `metadata.warnings` for
correction in the editor. This baseline is intentionally separate from YOLO training:
collect representative hand-drawn images and measure circle/edge precision and recall
before training a `component_circle`/`connector` detector.

Petri-net execution is deterministic and token based:

```python
from statemachine_pipeline.petrinet import enabled_transitions, fire
```

```bash
export GEMINI_API_KEY=your-key
python main.py generate outputs/diagram.json \
  --output outputs/state_machine.py \
  --report outputs/generation-report.json
```

Gemini receives only the canonical JSON and a strict runtime contract. A transition with no
detected event uses its transition ID as the runtime input fallback and is recorded as an
assumption; Gemini must not invent a semantic event name. Responses are parsed as JSON, then
Python syntax and state/transition coverage checks run before source is written. Arbitrary
generated code is not executed by the CLI.

The dependency-free editor runs locally at `http://127.0.0.1:8000`, supports moving
and renaming states, adding transitions, auto-layout, undo/redo, and validated JSON
save-back. After editing, regenerate code because the JSON contract is the source of
truth. Verify a generated Python trace with:

```bash
python main.py verify outputs/diagram.json outputs/state_machine.py T0 T6
```

Ask Gemini for a structured explanation, execution flows, and ambiguity report:

```bash
python main.py explain outputs/diagram.json --output outputs/explanation.json
```
