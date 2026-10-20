# State Diagram → Code

**Computer-vision pipeline that turns a picture of a state-transition diagram
(including computer-network protocol diagrams such as a TCP handshake) into a validated
graph, lets a human correct it in a web editor, and generates Python code that is
automatically tested against the diagram.**

> One-line pitch: *upload a photo or screenshot of a state diagram → get a machine-readable
> graph → fix mistakes in an editor → get working, tested code.*

## Contents

1. [Problem and motivation](#1-problem-and-motivation)
2. [System overview](#2-system-overview)
3. [Tech stack](#3-tech-stack)
4. [Computer-vision pipeline in detail](#4-computer-vision-pipeline-in-detail)
5. [Canonical JSON (data model)](#5-canonical-json-data-model)
6. [Code generation and verification](#6-code-generation-and-verification)
7. [Role of the LLM (Gemini)](#7-role-of-the-llm-gemini)
8. [Training data and model training](#8-training-data-and-model-training)
9. [Evaluation and results](#9-evaluation-and-results)
10. [Web editor and correction loop](#10-web-editor-and-correction-loop)
11. [Setup and usage](#11-setup-and-usage)
12. [Project structure](#12-project-structure)
13. [Testing](#13-testing)
14. [Limitations (be upfront about these)](#14-limitations)
15. [Questions the professor may ask, with answers](#15-likely-questions-and-answers)

Also see `DEMO.md` for a step-by-step live-demo script.

---

## 1. Problem and motivation

State machines (UML state diagrams, protocol diagrams, UI flows) are usually drawn as
images: whiteboard photos, slides, PDFs, textbook figures. To use one in software, someone
re-types it by hand as code, which is slow and easy to get wrong.

The task is hard for a computer because the output is a **graph**, not a label:

* the correct **states** must be found and **named**;
* every **arrow** must be linked to the right **source** and **target** state, including
  arrows that cross, curve, loop back to the same state, or run in both directions
  between two states;
* every arrow needs a **label** that is split into `event [guard] / action`;
* the **initial** and **final** states must be identified.

A single wrong arrow endpoint or direction gives code with wrong behaviour, so the goal
is not just "detect shapes" but **reconstruct the correct graph, and say when unsure**.

## 2. System overview

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

Key design principle: **learned components do perception, deterministic components do
reasoning.** A neural network finds *what is where*; classical image processing decides
*what connects to what*; deterministic validators and tests decide whether the result and
the generated code are acceptable. The LLM only explains and suggests.

| Stage | Module | Technique | Learned? |
|---|---|---|---|
| Object detection | `detector.py` | YOLO11n | yes |
| Ink segmentation | `preprocess.py` | background division + Otsu threshold | no |
| Arrow tracing / association | `arrows.py`, `reconstruct.py` | direction-following stroke tracer on the ink mask | no |
| Arrow direction | `arrows.py` | arrowhead detection + cross-section width ("headness") | partly |
| Text reading | `ocr.py` | Tesseract OCR with glyph clean-up | yes (off-the-shelf) |
| Label meaning | `semantics.py` | parser + keyword table | no |
| Validation | `validation.py`, `schema.py` | JSON Schema + graph rules | no |
| Code generation | `codegen.py` | template (default) or Gemini | no / LLM |
| Verification | `sandbox.py`, `trace.py` | conformance tests in isolated process | no |

## 3. Tech stack

* **Python 3** (whole project)
* **Ultralytics YOLO11n**: object detector (PyTorch under the hood)
* **OpenCV + NumPy**: image processing (morphology, thresholding, distance transform, CLAHE)
* **SciPy**: Hungarian algorithm (`linear_sum_assignment`) for matching during evaluation
* **Tesseract / pytesseract**: OCR (optional; the pipeline degrades gracefully without it)
* **jsonschema**: formal schema for the data model
* **Pillow**: rendering of the synthetic training diagrams
* **google-genai**: Gemini API (optional; explain / suggest / alternative code backend)
* **`http.server` + vanilla HTML/JS**: web editor, with no web framework
* **pytest**: 71 tests

## 4. Computer-vision pipeline in detail

### 4.1 Detection (YOLO11n)

A single-stage object detector finds six classes: `state`, `arrow`, `final_state`,
`arrowhead`, `initial_marker`, `label`. It returns a bounding box and a confidence for
each. YOLO11n ("nano") was chosen because it is small and fast and runs on a laptop GPU
or even a CPU. The images are simple line drawings, so a bigger model is not needed.

**Important:** YOLO only gives **boxes**. A bounding box of a curved or diagonal arrow
covers a large area and contains other objects, so it cannot tell where the arrow starts,
ends, or which way it points. That is why the next stage exists.

### 4.2 Ink mask (`preprocess.py: ink_mask`)

To analyse the actual strokes we need a clean black/white "ink" image that works for
photos with shadows too:

1. Estimate the background with a large **morphological closing** (removes thin dark
   lines, keeps the paper shading).
2. **Divide** the image by the background → shadows and paper gradients flatten out.
3. Light Gaussian blur, then **Otsu thresholding** to choose the ink/paper threshold
   automatically.
4. Small morphological opening removes speckle noise.

`stroke_width` then estimates pen thickness from the **distance transform** (the median
distance at the stroke centre-line is half the width). The tracer uses this so it adapts
to thick and thin lines. `preprocess_image` also provides **CLAHE** contrast enhancement
(on the L channel of LAB) for faint images.

### 4.3 Associating arrows to states by stroke tracing (`arrows.py`)

This is the core of the project and the biggest improvement over the first version.

**The naive approach (the original version):** link each end of an arrow's box to the
*nearest state centre*. It fails on exactly the interesting cases: it invents self-loops,
swaps directions, and connects the wrong states when edges cross.

**Our approach: follow the pen.**

1. **State regions.** Each detected state box is expanded until the outline ink of that
   state ends, then removed from the mask. What remains are free strokes.
2. **Ports.** A *port* is a place where ink touches a thin ring just outside a state, so
   it is a point where an arrow leaves or enters that state.
3. **Tracer.** From each port, a tracer walks along the stroke in small steps (2 px). At
   each step it looks ahead within ±50° and picks the direction with enough ink, with a
   **small penalty for turning** (`straight_bias`). Effects:
   * it **goes straight through crossings** instead of turning onto the other edge;
   * it **follows curves** (turning is allowed, just penalised);
   * it **bridges small gaps** (up to ~14 px + stroke width), such as a label on the
     arrow or a hand-drawn break;
   * it **stops** when it enters another state (→ an edge), returns to its own state
     after leaving its box (→ a **self-loop**), or runs out of ink (→ a free end).
4. **Both ends.** Every arrow is traced from both ends. Traces describing the same arrow
   (endpoints within 14 px) are merged. If only one end could be traced the result is
   marked as uncertain. Candidate paths that mostly re-trace already accepted arrows are
   rejected (this removes mis-traces at crossings).

### 4.4 Arrow direction

Direction decides source versus target, so it needs evidence:

* **Main evidence:** a detected `arrowhead` box located at the arrow's own end segment.
* **Fallback ("headness"):** an arrowhead is *wider* than the shaft. The code measures the
  ink cross-section width near each end of the traced path, divided by the shaft width.
  About 1.0 for a plain end, ≥ ~2 for a head (`head_ratio = 2.2`). The end with clearly
  higher headness (`direction_margin`) is the head.
* **Exclusivity:** each detected arrowhead can belong to **only one** arrow. This resolves
  shallow crossings between two parallel curved arrows.
* **Weak evidence → `direction: "unknown"`** plus an *uncertainty* listing both readings.
  The system never silently guesses.

### 4.5 Initial and final states

* **Initial:** an arrow with a free end whose head touches a state is the initial marker
  (e.g. the "entry arrow" or UML black dot). A detected `initial_marker` box adjacent to a
  state is used as a fallback. Several candidates → highest confidence wins, others are
  reported.
* **Final:** from the `final_state` class, or from the **double ring** found by sampling a
  **radial ink profile** (the fraction of 90 angles with ink at each relative radius; two
  rings with a gap between them mean a double outline).
* **Shape:** circle vs rounded rectangle is decided by whether the diagonals have ink near
  the box corners (rectangles do, circles do not).

### 4.6 Reading text (OCR)

* State names: the interior of each state is cropped, upscaled (×2–×3), binarised (Otsu),
  and the **outline fragments are removed** (components touching the border or spanning
  most of the crop), because Tesseract otherwise reads the circle as `|` or quotes.
* Transition labels: each detected `label` box is assigned to the transition whose
  **traced path is nearest** (not the nearest centre). If the label is about equally close
  to two transitions, it is stored as an uncertainty.
* Low OCR confidence (< 0.6) is recorded as an uncertainty so the user reviews it.
* If the detector has no `label` class, text boxes are found by connected components.

### 4.7 Label semantics (`semantics.py`)

Labels follow the UML pattern `event [guard] / action`, for example
`timeout [retries<3] / retransmit`. They are parsed with a small grammar. Words from an
explicit keyword table (SYN, ACK, FIN, timeout, retry, connect, disconnect, failure,
send, receive…) mark the transition as a **network event**. Unknown words stay plain
events: no protocol behaviour is invented.

### 4.8 Honesty about uncertainty

Every ambiguous decision creates an `uncertainty` record with the element, the field, the
reason and candidate interpretations. In the editor, uncertain elements are amber and
dashed. Dangling arrows do **not** create fake transitions; they raise a warning that a
state may be missing.

## 5. Canonical JSON (data model)

Everything (editor, validator, code generator, evaluation) uses one schema, version
**1.1** (`models.py`, formal JSON Schema in `schema.py`; 1.0 files still load).

| Concept | Where |
|---|---|
| State (a condition the system is in) | `states[]`: id, label, type (initial/normal/final), position, bbox, shape, confidences, provenance |
| Transition (a change between states) | `transitions[]`: source, target, event, guard, action, raw label, direction, traced path `points`, `label_bbox`, confidence, provenance |
| Network event (what triggers a transition) | `transitions[].kind = "network_event"` + `network: {event_type, protocol, packet}` |
| Network connection (physical/logical link) | `network.devices[]` (with the states they own) and `network.links[]` |
| Uncertainty | `uncertainties[]`: element, field, reason, candidate interpretations |
| Provenance | `source_image` (path, size, sha256), per-element `provenance`, `metadata.warnings` |

`validate_state_machine` first checks the JSON Schema, then graph rules: dangling
references, duplicate ids, at most one initial state, non-determinism, transitions out of
final states, unreachable states, unknown devices/links. Every editor save and every code
generation goes through it. Round-trip `StateMachine.from_dict(json) == machine` is tested.

**Why one canonical JSON?** It is the single source of truth. The graph view, the JSON
tab and the generated code are all *views* of it, so they cannot drift apart.

## 6. Code generation and verification

**Template generator (default, deterministic).** `codegen.py` writes a dependency-free
`StateMachine` class with defined semantics:

* An event-less transition is triggered by its ID.
* Guards are injected callables; a missing guard implementation raises
  `GuardNotImplementedError`.
* Explicit errors: `InvalidEventError`, `GuardRejectedError`, `TerminalStateError`
  (final states), `NondeterministicTransitionError`.
* Action hooks run *before* the state changes; network-event metadata is kept per
  transition.

**Conformance testing.** The same semantics are implemented separately as a **reference
model** (`reference_step`). A test plan is built from the graph: it fires **every
transition** (from the initial state if reachable, otherwise starting at the source
state) and adds guard-rejection and invalid-event probes. The generated code must produce
exactly the same state sequence and the same errors as the reference. The plan can be
exported as a standalone pytest module (`--tests`). Mutation tests check that deliberately
broken code is caught.

**Sandbox (`sandbox.py`).** Generated code, including Gemini's, runs in a separate Python
interpreter with:

* `-I -S` isolation, an empty environment and a temporary working directory;
* a timeout plus CPU, memory, file-size and process limits;
* an **audit hook** that blocks sockets, subprocesses, file writes/deletes and `ctypes`;
* before running, an **AST allowlist** that rejects imports such as `os`, `socket`,
  `subprocess` and calls to `open`/`eval`/`exec`.

This is defence in depth for a local tool, **not** a hardened container.

## 7. Role of the LLM (Gemini)

Optional (needs `GEMINI_API_KEY`; model name from `GEMINI_MODEL`). Gemini is used only for:

* **Explain**: natural-language answers about the diagram ("What happens on timeout?").
* **Suggest**: compares the image with the graph and proposes corrections.
* **Alternative code backend** (`--backend gemini`) and **proposed test cases**.

Safeguards: responses must parse against a JSON contract; explanations of non-existent
states/transitions are dropped; suggestions are validated and shown for **Accept/Reject**
(never auto-applied); Gemini code must pass the same AST check, sandbox and conformance
plan; LLM test cases are replayed on the reference model and marked agree/disagree. If the
key is missing or the API fails, the rest of the system works unchanged.

## 8. Training data and model training

**Synthetic data (`synthetic.py`, `generate_dataset.py`).** Labelling thousands of real
diagrams by hand is impractical, so diagrams are **rendered programmatically**. Each
sample gives the image, YOLO labels **and** the exact ground-truth JSON, so we can
evaluate not only detection but the whole reconstructed graph. Variety includes:

* circle and rounded-rectangle states, double-circle final states, initial markers
  (plain arrow or UML dot);
* straight, curved, crossing and bidirectional transitions, and self-loops;
* UML `event [guard] / action` labels using generic or network-protocol vocabularies;
* line-width jitter, blur, noise and JPEG artefacts.

| Split | Images | Use |
|---|---|---|
| train | 1680 | YOLO training |
| val | 360 | YOLO validation, threshold tuning |
| test | 360 | **final results only, never used for training or tuning** |

(2400 images total, `--seed 7`, 70/15/15 split.)

**Training (YOLO11n, transfer learning).** Starting from pretrained `yolo11n.pt` weights,
60 epochs (early stopping patience 15), batch 16, image size 640, `fliplr=0.0`. About 9
minutes on a laptop GPU. Horizontal-flip augmentation is **disabled** because mirroring a
diagram mirrors its text, which does not occur in real diagrams.

Detector quality on the validation split at the final epoch (from `results.csv`):
precision 0.997, recall 0.988, **mAP@50 0.995**, mAP@50-95 0.933.
Detection is therefore nearly solved on this data. The remaining errors are mainly in
association and OCR (see below), which is why those got their own methods.

**Threshold tuning (`tune.py`).** The tracer's thresholds (`head_ratio`,
`direction_margin`, `endpoint_tolerance`, `overlap_reject`, `straight_bias`) are tuned by
**coordinate descent** on the **validation split only** (objective: transition F1 +
0.5·structure-exact + 0.25·initial-state accuracy). Detections are computed once and only
the deterministic reconstruction is re-run, so tuning is fast and reproducible. The result
is saved to `config/association.json`.

## 9. Evaluation and results

**Method (`evaluate.py`).** Predicted states are matched to ground truth by **bounding-box
IoU ≥ 0.5 using the Hungarian algorithm** (optimal one-to-one assignment), *never by ID*,
because IDs are just detection order. Transitions are then compared through that
matching. Metrics are micro-averaged over the 360 held-out test diagrams. Text accuracy
uses exact match and **CER** (character error rate, from Levenshtein edit distance).

| Metric | A: original | B: new pipeline, v1 model | C: new pipeline, v2 model |
|---|---|---|---|
| State F1 | 0.877 | 0.877 | **1.000** |
| Transition precision | 0.282 | 0.696 | **0.950** |
| Transition recall | 0.370 | 0.505 | **0.908** |
| Transition F1 (directed) | 0.320 | 0.585 | **0.928** |
| Transition F1 (undirected) | 0.486 | 0.717 | **0.957** |
| Endpoint accuracy | 0.614 | 0.793 | **0.964** |
| Initial-state accuracy | 0.000 | 0.519 | **0.983** |
| Final-state F1 | n/a | 0.279 | **0.997** |
| State OCR exact | 0.000 | 0.835 | 0.847 |
| State OCR CER | 1.000 | 0.058 | 0.052 |
| Label OCR exact | 0.000 | 0.214 | 0.654 |
| Label OCR CER | 1.000 | 0.617 | 0.228 |
| Event exact | 0.000 | 0.254 | 0.747 |
| Graph exact (structure) | 0.000 | 0.089 | **0.719** |
| Graph exact (with labels) | 0.000 | 0.019 | 0.269 |
| Generated-code conformance | 0.994 | 0.994 | **1.000** |
| Diagrams | 360 | 360 | 360 |

* **A** = original pipeline: 2-class model, box geometry, nearest-centre association, no OCR.
* **B** = original 2-class model + the new stroke tracing, OCR and semantics (**isolates
  the effect of the new pipeline**).
* **C** = retrained 6-class model + new pipeline (**adds the effect of better detection**).

This A/B/C **ablation** is the main evidence: B vs A shows the algorithm helps
(transition F1 0.32 → 0.59); C vs B shows the extra classes (arrowheads, initial markers,
final states, labels) help further (→ 0.93).

Definitions: *endpoint accuracy* = share of transitions with the right pair of states
**and** the right direction; *graph exact (structure)* = every state, transition,
direction, initial and final state is right; *with labels* additionally requires every
name and label to be character-exact; *code conformance* = share of predicted graphs
whose generated code passes the full conformance plan (a diagram with no detected states
counts as a failure).

**How to read the table honestly:** structure is strong (0.93 transition F1, 0.72 whole
graphs exactly right). The weakest part is **OCR of transition labels** (0.65 exact), and
requiring *everything* including every character to be exact gives 0.27 per diagram. That
is why the editor exists and why uncertain OCR is flagged.

Reproduce: `python main.py evaluate --split test --check-code` then
`python main.py compare outputs/eval_*.json`.

## 10. Web editor and correction loop

`python main.py serve` → http://127.0.0.1:8000 (stdlib `http.server`, no framework,
127.0.0.1 only, upload size and image-signature checks).

* **Source panel:** detections drawn over the image, traced arrow paths in orange.
* **Graph panel:** zoom/pan, drag states, double-click to add a state, **Connect** to
  draw a transition (same state twice → self-loop), Delete, Ctrl+Z / Ctrl+Y, auto layout.
* **Inspector:** rename, change type/shape, edit source/target, reverse direction, confirm
  an uncertain direction, edit event/guard/action, mark a network event, edit network
  devices and links.
* **Issues tab:** validation errors, uncertainties and warnings; click to select. Amber
  dashed = uncertain, red = invalid.
* **JSON tab:** editable; validated server-side before it replaces the graph.
* **Code tab:** generate (template or Gemini) and verify; a **stale** badge appears as soon
  as the graph changes after generation.
* **Explain tab:** Gemini explanation and suggestions (accept/reject).
* **Export:** JSON, SVG, generated code, generated tests.
* **Save correction…:** stores image + prediction + your corrected graph.

**Correction loop (human-in-the-loop learning).** Corrections are saved as JSONL records
(image, raw detections, predicted graph, corrected graph, field-level diff).
`export-corrections` converts train-split corrections into YOLO labels for fine-tuning.
Records saved as *held-out evaluation* are **never** used for training, only to measure
whether a retrained model really improved.

```bash
python main.py export-corrections
yolo detect train data=data/corrections/yolo/dataset_with_corrections.yaml \
     model=runs/detect/v2/weights/best.pt epochs=20
python main.py evaluate --corrections data/corrections/corrections.jsonl --model <new best.pt>
```

Network-topology diagrams (circles joined by lines) use a separate `network-1.0` contract
(`--diagram-type network`, OpenCV circle/stroke baseline, `network_cv.py`). A Petri-net
mode also exists (`petrinet.py`, needs a separately trained model).

## 11. Setup and usage

```bash
cd statetocode
python3 -m venv .venv
source .venv/bin/activate          # fish: source .venv/bin/activate.fish
python -m pip install -r requirements.txt
sudo pacman -S tesseract tesseract-data-eng   # or: apt install tesseract-ocr   (OCR is optional)
cp .env.example .env               # set GEMINI_API_KEY only if you want Gemini features
```

The trained detector is expected at `runs/detect/v2/weights/best.pt` (the `runs/`,
`*.pt`, `dataset*/` and `test_images/` folders are git-ignored; regenerate with the
commands in §8). The CLI falls back to the older 2-class model `runs/detect/train/weights/best.pt`.

```bash
python main.py images/test1.png -o outputs/diagram.json        # image -> canonical JSON
python main.py analyze outputs/diagram.json                     # reachability, cycles, dead ends
python main.py generate outputs/diagram.json \
    --output outputs/state_machine.py --tests outputs/test_state_machine.py \
    --report outputs/generation-report.json                     # template code + conformance check
python main.py generate outputs/diagram.json --backend gemini   # Gemini code, same checks
python main.py verify outputs/diagram.json outputs/state_machine.py              # full conformance
python main.py verify outputs/diagram.json outputs/state_machine.py connect ACK  # replay one event trace
python main.py explain outputs/diagram.json --question "What happens on timeout?"
python main.py suggest outputs/diagram.json                     # Gemini correction suggestions
python main.py serve                                            # editor at http://127.0.0.1:8000
python main.py edit outputs/diagram.json                        # editor bound to a JSON file
python main.py tune --model runs/detect/v2/weights/best.pt      # tune thresholds on val split
python main.py evaluate --split test --check-code               # metrics on held-out data
python main.py compare outputs/eval_*.json                      # side-by-side result table
python make_test_images.py                                      # 35 demo/test images
python -m pytest -q
```

Training from scratch:

```bash
python generate_dataset.py --output dataset_v2 --count 2400 --seed 7
yolo detect train data=dataset_v2/dataset.yaml model=yolo11n.pt imgsz=640 epochs=60 fliplr=0.0 \
     project=runs/detect name=v2
```

Using the generated code (guards are plain functions you supply):

```python
from tcp_machine import StateMachine, InvalidEventError
m = StateMachine(guards={"retries< 3": lambda ctx: ctx["retries"] < 3,
                         "retries>=3": lambda ctx: ctx["retries"] >= 3})
m.submit("connect")                    # 'S1'
m.submit("timeout", {"retries": 1})    # stays in SYN_SENT (retransmit)
m.submit("timeout", {"retries": 3})    # back to CLOSED
m.submit("ACK")                        # raises InvalidEventError
```

## 12. Project structure

```
statetocode/
├── main.py                      CLI: detect, analyze, generate, verify, explain, suggest,
│                                serve/edit, evaluate, compare, tune, export-corrections
├── generate_dataset.py          renders the synthetic YOLO dataset + ground-truth JSON
├── make_test_images.py          35 held-out demo images (generic, protocol, hard cases, photo-like)
├── generate_network_diagrams.py network-topology sample generator
├── statemachine_pipeline/
│   ├── detector.py              YOLO adapter (loads model, filters classes)
│   ├── preprocess.py            ink mask, stroke width, CLAHE
│   ├── arrows.py                stroke tracer, ports, direction ("headness"), config
│   ├── reconstruct.py           detections + image → StateMachine (states, edges, labels, initial/final)
│   ├── ocr.py                   Tesseract wrapper, glyph clean-up, text-box finding
│   ├── semantics.py             event [guard] / action parser, network-event classifier
│   ├── models.py, schema.py     canonical data model and JSON Schema
│   ├── validation.py, analysis.py   graph rules; reachability / cycles / dead ends
│   ├── codegen.py               template code generator + reference semantics + test plan
│   ├── sandbox.py, trace.py     AST allowlist, isolated execution, trace replay
│   ├── gemini.py                Gemini explain / suggest / code backend with contract checks
│   ├── synthetic.py             diagram renderer with exact ground truth
│   ├── evaluate.py, tune.py     metrics (Hungarian matching, CER) and threshold tuning
│   ├── corrections.py           correction JSONL → YOLO labels
│   ├── ui.py + static/          web editor server and front-end
│   ├── network.py, network_cv.py   network-topology contract and OpenCV baseline
│   └── petrinet.py              Petri-net model
├── tests/                       71 pytest tests
├── config/association.json      tuned tracer thresholds (written by `tune`)
├── legacy/                      the first, box-only version (kept for comparison)
├── DEMO.md                      live-demo script
└── outputs/                     evaluation logs, comparison.md, sample outputs
```

## 13. Testing

`python -m pytest -q` → **71 passed** (about 25 s). Covers:

* contract, schema, round-trip and semantic validation;
* label parsing and network-event classification;
* vision on drawn fixtures: direction, reversed arrows, initial marker vs. self-loop,
  curved bidirectional pairs, crossings, double-ring finals, ambiguous and dangling arrows,
  OCR label assignment;
* reference semantics, conformance, **mutation detection**, sandbox blocking;
* editor HTTP API synchronisation and error handling;
* Gemini contract handling (stubbed, so no key needed);
* evaluation metrics, corrections export and generator consistency;
* end-to-end image → JSON → code → traces on rendered network-protocol diagrams (needs the
  v2 weights).

## 14. Limitations

State these before being asked:

* **Synthetic training data only.** All measured numbers are on rendered diagrams.
  Photos or hand-drawn diagrams are out of distribution: expect lower detection quality
  and much lower OCR accuracy (Tesseract reads handwriting poorly). The correction loop
  exists to close this gap, but **no real corrections have been collected or trained on
  yet**, so no improvement from it is claimed.
* Remaining association errors: shallow crossings between arcs, arrows passing under
  text, very small self-loops. They are flagged as uncertain when detected, but not always.
* **Label OCR is the weakest stage.** A misread guard changes the semantics, so review
  guards before using the generated code.
* Code generation is verified for **Python only**. Gemini can emit other languages but
  only Python is executed and conformance-checked.
* The sandbox is process-level isolation, not a container.
* Conformance tests check that the code matches the *graph*, not that the graph matches
  the *drawing*; the latter is what the detection metrics and human review are for.

## 15. Likely questions and answers

**Q: Why not just give the image to an LLM (GPT/Gemini) and ask for code?**
Arrow endpoints and directions must be exactly right, and an LLM's answer cannot be
verified or measured on our data. Here perception is deterministic and measured on 360
held-out diagrams, the output is validated against a schema, and the code is tested.
The LLM is limited to explaining and suggesting, and a human accepts or rejects each
suggestion.

**Q: Why does YOLO alone not solve it?**
YOLO returns axis-aligned boxes. For an arrow, the box tells nothing about where the
line starts or ends or which way it points, and boxes of crossing arrows overlap. A graph
needs exact connectivity, so we trace the actual pixels of the strokes.

**Q: What was wrong with "link arrow ends to the nearest state"?**
It ignores the drawing: crossing edges get connected to the wrong states, self-loops are
invented, direction is guessed. Our ablation shows it: transition F1 was 0.32 with that
method versus 0.59 with tracing on the same detector (B), and 0.93 with the retrained
detector (C).

**Q: How does the tracer handle crossing arrows?**
It prefers to continue straight (turning is penalised), so at a crossing it passes
through rather than switching onto the other stroke. Each arrow is traced from both ends
and the results are cross-checked. Paths that mostly re-trace accepted arrows are
discarded. Test image: `crossing_edges.png`.

**Q: How do you know the arrow direction?**
First, a detected arrowhead near that end of the arrow. If that is missing, the ink
cross-section is measured near both ends and compared with the shaft width: the arrowhead
end is about twice as wide. Each arrowhead belongs to only one arrow. If the evidence is
weak, direction is set to `unknown` and both options are shown to the user.

**Q: How do you detect initial and final states?**
Initial: an arrow with a free end pointing into a state (or a detected initial-marker
box). Final: the `final_state` class, or a double ring found from the radial ink profile.

**Q: Why train on synthetic data? Is that valid?**
It gives unlimited, perfectly labelled data, including exact graph truth that lets us
evaluate full reconstruction. The limitation is the domain gap to real photos; we say so,
and we built the correction loop to adapt to real images. We do not claim real-world
accuracy.

**Q: Why is `fliplr` (horizontal flip) set to 0?**
Flipping produces mirrored text, which never occurs in real diagrams, so it would teach
the detector (especially the `label` class) unrealistic examples. It is a design choice
we made, not something we ablated.

**Q: How did you avoid overfitting / data leakage?**
Three disjoint splits: train 1680, val 360 (used for early stopping and tuning of tracer
thresholds), test 360 (used only for final reporting). Held-out corrections are never
exported as training data.

**Q: Why match states with the Hungarian algorithm and IoU?**
Output IDs are just detection order, so comparing by ID would be meaningless. We need the
best one-to-one pairing between predicted and true states; Hungarian (SciPy
`linear_sum_assignment`) finds the globally optimal assignment on IoU, with IoU ≥ 0.5 as
the match criterion.

**Q: What is CER?**
Character error rate: Levenshtein edit distance between the OCR text and the true text,
divided by the true length. 0 is perfect; the state-name CER is 0.052, the label CER 0.228.

**Q: Why are the exact-with-labels numbers (0.27) so much lower than structure (0.72)?**
That metric requires every character of every name, event and guard to be exact on the
whole diagram, and Tesseract makes occasional errors on small text such as `retries<3`.
One wrong character fails the whole diagram. The editor lets a user fix these quickly,
and uncertain OCR is flagged.

**Q: Why use OpenCV classical methods in a "deep learning" project?**
Each tool is used for what it is good at. A network is strong at "what is this object",
but it does not give the precise connectivity of thin strokes. Classical image processing
with explicit rules is precise, explainable and easy to debug, with a few tunable
parameters (13 in `DEFAULT_CONFIG`, 5 of them tuned). It also needs no extra labelled data.

**Q: How do you ensure the generated code is correct?**
We build a test plan from the graph, one test per transition plus negative cases, and
compare the code to an independent reference implementation of the same semantics. Mutated
(broken) code is tested to ensure the checker catches it. Result on the test set: 100%
conformance of generated code for the predicted graphs (360/360).

**Q: Is it safe to run generated code?**
It is run in a separate locked-down interpreter (isolated mode, empty env, temp dir,
CPU/memory/time limits, audit hook blocking files, sockets, subprocesses), after a static
AST check that rejects dangerous imports and calls. This is defence in depth for a local
tool, not a container. For untrusted code at scale we would use a container.

**Q: What does the LLM do and what does it not do?**
It explains the diagram, suggests corrections and can write alternative code. It never
detects arrows, validates the graph, or accepts code. Its output must parse against a
contract and is checked by the deterministic parts.

**Q: What would you do next?**
Collect real hand-drawn and photographed diagrams, fix them in the editor to build a
real-image training set, fine-tune the detector, try a stronger OCR (handwriting-capable
model) for labels, and support more target languages with the same conformance check.

**Q: What did you contribute vs. use off-the-shelf?**
Off-the-shelf: YOLO11n (Ultralytics), Tesseract, OpenCV primitives, Gemini API.
Our own: the synthetic diagram generator, the stroke-tracing association and direction
algorithm, the canonical schema and validator, the label semantics parser, code generator
with reference semantics and sandboxed conformance testing, the evaluation framework, the
threshold tuner, the editor and the correction loop.
