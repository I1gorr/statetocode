# State diagram → code

Turns an image of a state-transition diagram (including computer-network protocol
diagrams) into a validated canonical JSON graph, lets you correct it in a web
editor, and generates Python code that is checked against the graph's exact
semantics in a sandbox.

```
image ──► YOLO detector ──► stroke tracing + OCR ──► canonical JSON (schema 1.1) ──► validation
              (states, arrows,     (endpoints, direction,        │   ▲                │
               arrowheads, labels,  labels → event/guard/action,  ▼   │ edits          ▼
               initial markers)     network events, uncertainty)  web editor ◄── Gemini (explain /
                                                                   │                  suggest; never edits)
                                                                   ▼
                                               template / Gemini code ──► sandboxed conformance tests
                                                                   │
                                     corrections (JSONL) ──► YOLO labels ──► fine-tuning ──► evaluation
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate          # fish: source .venv/bin/activate.fish
python -m pip install -r requirements.txt
sudo pacman -S tesseract tesseract-data-eng   # or apt install tesseract-ocr; OCR is optional
cp .env.example .env               # GEMINI_API_KEY only needed for explain/suggest/Gemini codegen
```

Use the Linux `.venv`; the legacy `venv/Scripts` directory contains Windows executables.

The trained detector is `runs/detect/v2/weights/best.pt` (classes `state`, `arrow`,
`final_state`, `arrowhead`, `initial_marker`, `label`). The CLI picks it by default and
falls back to the original 2-class model `runs/detect/train/weights/best.pt`.

## Usage

```bash
python main.py images/test1.png -o outputs/diagram.json        # image -> canonical JSON
python main.py analyze outputs/diagram.json                     # reachability, cycles, dead ends
python main.py generate outputs/diagram.json \
    --output outputs/state_machine.py --tests outputs/test_state_machine.py \
    --report outputs/generation-report.json                     # template code + conformance check
python main.py generate outputs/diagram.json --backend gemini   # Gemini code, same checks
python main.py verify outputs/diagram.json outputs/state_machine.py              # full conformance
python main.py verify outputs/diagram.json outputs/state_machine.py connect ACK  # one event trace
python main.py explain outputs/diagram.json --question "What happens on timeout?"
python main.py suggest outputs/diagram.json                     # Gemini correction suggestions
python main.py serve                                            # editor at http://127.0.0.1:8000
python main.py edit outputs/diagram.json                        # editor bound to a JSON file
python -m pytest -q
```

Network-topology diagrams (circles joined by strokes) still use the separate
`network-1.0` contract: `python main.py IMAGE --diagram-type network`. Petri nets:
`--diagram-type petri-net` with a Petri-net model.

### Editor (`python main.py serve`)

Upload an image and the detections are shown over it, next to the reconstructed graph:

* **Graph**: zoom (wheel, `+`/`−`, Fit), pan (drag the background), drag states,
  double-click to add a state, **Connect** to draw a transition (same state twice →
  self-loop), Delete key, Ctrl+Z / Ctrl+Y, auto layout.
* **Inspector**: rename, change state type/shape, edit source/target, reverse direction,
  confirm an uncertain direction, edit event/guard/action/raw label, mark a transition as
  a network event (type, protocol, packet), add/remove network devices and links, mark
  uncertainties as reviewed.
* Uncertain detections are amber and dashed, invalid elements red; the **Issues** tab
  lists validation errors, uncertainties and detector warnings (click to select).
* **JSON**: the canonical JSON, editable; it is validated server-side before it replaces
  the graph. Import a JSON file.
* **Code**: generate (template or Gemini) and verify; the tab shows a *stale* badge
  as soon as the graph changes after generation.
* **Explain**: Gemini explanation and image-based correction suggestions, each accepted
  or rejected by you.
* **Export**: JSON, SVG of the graph, generated code, generated tests.
* **Save correction…**: stores image + prediction + your corrected graph as training
  data or as held-out evaluation data.

## How it works

### Canonical JSON (`statemachine_pipeline/models.py`, `schema.py`)

Schema `1.1` (JSON Schema in `schema.py`; `1.0` files still load). It distinguishes:

| Concept | Where |
|---|---|
| State (a condition the system is in) | `states[]`: id, label, type (initial/normal/final), position, bbox, shape, confidences, provenance |
| Transition (a change between states) | `transitions[]`: source, target, event, guard, action, raw label, direction, traced path `points`, `label_bbox`, confidence, provenance |
| Network event (what triggers a transition) | `transitions[].kind = "network_event"` + `network: {event_type, protocol, packet}` |
| Network connection (physical/logical link) | `network.devices[]` (with the states they own) and `network.links[]` |
| Uncertainty | `uncertainties[]`: element, field, reason, candidate interpretations |
| Provenance | `source_image` (path, size, sha256), per-element `provenance`, `metadata.warnings` |

`validate_state_machine` runs the schema first and then graph rules: dangling
references, duplicate ids, at most one initial state, non-determinism, transitions out of
final states, unreachable states, unknown devices/links. Every editor save and every code
generation goes through it. `StateMachine.from_dict(json) == machine` is tested for all
fields.

### Vision (`detector.py`, `reconstruct.py`, `arrows.py`, `ocr.py`, `preprocess.py`)

1. **Detection**: YOLO11n finds states, final states, arrows, arrowheads, initial
   markers and text labels.
2. **Ink mask**: background division plus Otsu, robust to shadows and paper gradients.
3. **Association by stroke tracing** (not nearest centre). Each state region is padded
   until its outline ink ends, and a *port* is where ink touches the ring outside it. From
   every port, a direction-following tracer walks the stroke. It prefers going straight,
   so it passes through crossing edges, follows curves, bridges small gaps and text boxes,
   and stops when it enters a state (edge), returns to its own state after leaving its box
   (self-loop) or runs out of ink (free end). Arrows are traced from both ends; paths that
   mostly re-trace accepted arrows are dropped as crossing mis-traces.
4. **Direction**: detected arrowheads on the arrow's own end segment are the main
   evidence. As a fallback, the ink cross-section near each end is compared with the shaft
   width. Each detected arrowhead can be the head of only one arrow; this resolves shallow
   crossings between parallel arcs. Weak evidence → `direction: "unknown"` plus an
   uncertainty listing both readings.
5. **Initial / final**: a free-ended arrow whose head is at a state (or a detected
   initial-marker box) marks the initial state. Final states come from the detector
   class or from a double outline in the radial ink profile.
6. **Labels**: Tesseract reads state interiors and label boxes after removing outline
   fragments. Each label goes to the transition whose *traced path* is nearest; ties
   become an uncertainty. `semantics.py` parses `event [guard] / action` and classifies
   network events from an explicit keyword table (SYN/ACK/FIN, timeout, retry,
   connect, disconnect, failure, send/receive). Unknown words stay plain events, so no
   protocol behaviour is invented.

### Code generation and verification (`codegen.py`, `sandbox.py`)

The template generator writes a dependency-free `StateMachine` class with these semantics:

* An event-less transition is triggered by its ID.
* Guards are injected callables; a missing guard implementation raises
  `GuardNotImplementedError`.
* Errors are explicit: `InvalidEventError`, `GuardRejectedError`, `TerminalStateError`
  for final states, and `NondeterministicTransitionError`.
* Action hooks run before the state changes; network-event metadata is kept per
  transition.

The same semantics are implemented by a deterministic reference (`reference_step`). The
test plan fires every transition: from the initial state when it is reachable, otherwise
starting in its source state. It also adds guard-rejection and invalid-event probes.

The plan runs against generated code in a separate interpreter. That interpreter has:

* `-I -S` isolation, an empty environment and a temporary working directory;
* a timeout plus CPU, memory, file-size and process limits;
* an audit hook that blocks sockets, subprocesses, file writes, deletes and `ctypes`.

Before anything runs, an AST allowlist rejects imports such as `os`, `socket` and
`subprocess`, and calls to `open`/`eval`/`exec`. Gemini-generated code must pass the same
checks to be accepted. `--tests` writes the plan as a standalone pytest module. This is
defence in depth for a local tool, not a hardened sandbox; use a container for untrusted
code at scale.

### Gemini (`gemini.py`)

The model comes from `GEMINI_MODEL` in `.env` and the key from `GEMINI_API_KEY`; nothing
is hard-coded. Gemini receives the validated JSON plus the deterministic analysis, and
for suggestions also the image. Every response is parsed against a JSON contract:

* Explanations of states or transitions that don't exist are dropped.
* Suggestions are checked against the element kind and the allowed values, then shown for
  you to accept or reject; none are applied automatically.
* LLM-proposed test cases are replayed on the reference semantics and marked as agreeing
  or not.

A missing key, an API error or malformed output gives a clear error (HTTP 503/400 in the
editor), and the rest of the pipeline keeps working.

## Training and the correction loop

```bash
python generate_dataset.py --output dataset_v2 --count 2400 --seed 7     # images + YOLO labels + ground-truth JSON
yolo detect train data=dataset_v2/dataset.yaml model=yolo11n.pt imgsz=640 epochs=60 fliplr=0.0 \
     project=runs/detect name=v2                                         # ≈15 min on an RTX 4070 laptop GPU
python main.py tune --model runs/detect/v2/weights/best.pt               # association thresholds, val split only
python main.py evaluate --model runs/detect/v2/weights/best.pt --split test --check-code
python main.py compare outputs/eval_*.json

# after reviewing diagrams in the editor and using "Save correction…":
python main.py export-corrections            # train-split corrections -> YOLO labels (+ merged dataset yaml)
yolo detect train data=data/corrections/yolo/dataset_with_corrections.yaml model=runs/detect/v2/weights/best.pt epochs=20
python main.py evaluate --corrections data/corrections/corrections.jsonl --model <new best.pt>   # held-out corrections only
```

Corrections are JSONL records: image, raw detections, predicted graph, corrected graph
and a field-level diff. Records saved as *held-out evaluation* are never exported for
training. Exported labels for states, final states, text and arrows come from the
corrected JSON. Arrowhead boxes are approximated around the arrow tip, and initial
markers reuse the detector's box. Both approximations are documented in
`corrections.py`.

## Evaluation

The synthetic generator (`statemachine_pipeline/synthetic.py`) renders:

* labelled circle and rounded-rectangle states, double-circle finals, initial markers
  (plain or UML dot);
* straight, curved, crossing and bidirectional transitions and self-loops;
* UML `event [guard] / action` labels with generic or network-protocol vocabularies;
* blur, noise and JPEG artefacts.

It writes the exact ground-truth JSON alongside each image. States are matched to ground
truth by box IoU (Hungarian assignment), never by ID. Thresholds were set on the
`val` split only. `test` (360 diagrams) was never used for training or tuning.

Held-out test results (`python main.py compare outputs/eval_*.json`):

RESULTS_TABLE

* **A**: the original pipeline: 2-class model, box geometry, nearest-centre association, no OCR.
* **B**: the original 2-class model with the new tracing, OCR and semantics (pipeline effect only).
* **C**: the retrained 6-class model with the new pipeline.

Endpoint accuracy is the share of transitions linking the right pair of states that also
have the right direction. *Graph exact* means every state, transition, direction, initial
and final state is right; *with labels* also requires every OCR'd name and label to be
character-exact. Code conformance is the share of predicted graphs whose generated code
passes the full conformance plan; a diagram with no detected states counts as a failure.

## Tests

`python -m pytest -q` covers:

* contract/schema/round-trip and semantic validation;
* label parsing and network-event classification;
* vision on drawn fixtures: direction, reversed arrows, initial marker vs. self-loop,
  curved bidirectional pairs, crossings, double-ring finals, ambiguous and dangling arrows,
  OCR label assignment;
* reference semantics, conformance and mutation detection, sandbox blocking;
* editor HTTP API synchronisation and error handling;
* Gemini contract handling (stubbed);
* evaluation metrics, corrections export and generator consistency;
* end-to-end image → JSON → code → traces on rendered network-protocol diagrams (needs
  the v2 weights).

## Limitations

* **Synthetic training data only.** The detector and the thresholds have only been
  measured on rendered diagrams. Photos of hand-drawn diagrams are out of distribution:
  expect lower detection quality and much lower OCR accuracy, since Tesseract reads
  handwriting poorly. The correction loop exists to close this gap, but no real
  corrections have been collected or trained on yet, so no improvement from it is
  claimed.
* Shallow crossings between arcs, arrows passing under text, and very small self-loops
  are the main association errors. They are reported as uncertainties when detected, but
  not always detected.
* Label OCR is the weakest stage (see the table). Misread guards change the semantics, so
  review guards before generating code.
* Code generation targets Python. Gemini can emit other languages, but only Python is
  executed and conformance-checked.
* The sandbox is process-level isolation, not a container.
