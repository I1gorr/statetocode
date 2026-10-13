# Demo guide: State Diagram → Code

A script for showing the project live. It takes about **15 minutes**, or **5** if you only
do Parts 1 and 4. Every output quoted below was produced by the current code on the
images in `test_images/`.

**One-line pitch:** *Upload a picture of a state diagram (including network protocols
like a TCP handshake). The system reconstructs the states and transitions, lets you fix
anything it got wrong, and generates Python code that is automatically tested against the
diagram.*

---

## 0. Before the demo (5 minutes, do this once)

```bash
cd statetocode
source .venv/bin/activate.fish        # bash/zsh: source .venv/bin/activate
python make_test_images.py            # creates test_images/ (35 images)
python -m pytest -q                   # expect: 71 passed
python main.py serve                  # editor at http://127.0.0.1:8000
```

Checklist:

- [ ] Browser open at **http://127.0.0.1:8000**, window at least 1400 px wide (three panels side by side).
- [ ] A second terminal open in `statetocode/` with the venv active, for the CLI parts.
- [ ] `.env` contains `GEMINI_API_KEY` if you want to show Part 5. Without it, everything
      else still works and Gemini buttons show a clear "GEMINI_API_KEY is not configured".
- [ ] Optional warm-up: upload any image once so the detector is loaded. The first
      upload takes a few seconds longer.

Images used in this demo:

| Image | Why |
|---|---|
| `test_images/hard_cases/tcp_handshake.png` | main example: network events, guards, actions, retry loop |
| `test_images/hard_cases/crossing_edges.png` | crossing arrows are not confused |
| `test_images/hard_cases/arrow_without_head.png` | the system admits uncertainty instead of guessing |
| `test_images/hard_cases/bidirectional_and_loop.png` | two-way arcs and a self-loop |
| `test_images/photo_like/net_01_photo.jpg` | honest limits on photo-like input |

---

## Part 1: Image → graph (3 min)

1. In the editor click **Upload image** and pick `test_images/hard_cases/tcp_handshake.png`.
2. The status bar says **"Detected 5 states and 6 transitions"**.

**Show on screen**

- **Left panel (source image).** Coloured boxes are what the neural network (YOLO) found:
  states, arrows, arrowheads, text labels, and the initial-state marker. Orange lines are
  the arrow paths the system *traced*. Toggle **Detections** in the toolbar to compare
  with the raw image.
- **Middle panel (reconstructed graph).** Same diagram, rebuilt from the data:
  - `CLOSED` has the initial marker (dot plus arrow).
  - `DONE` has a double ring (final state).
  - **Green** transitions are *network events* (connect, timeout, SYN-ACK, close, ACK).
  - The retry **self-loop** sits on `SYN_SENT`.
- Click the transition `SYN_SENT → SYN_SENT`. The Inspector shows the label split into
  parts:
  - **Event** `timeout`
  - **Guard** `retries< 3`
  - **Action** `retransmit`
  - **Kind** network event, type `timeout`

**What to say**

> "The original version linked every arrow to the *nearest* state centre. That produced
> fake self-loops and wrong directions. Now the system follows each pen stroke from the
> state outline to wherever it ends. That's how it knows where an arrow really starts and
> stops, and that the arrow into CLOSED comes from nowhere, so it marks the initial state
> rather than a transition."

> "It also understands *what* a transition means. The label is parsed into
> event / guard / action, and network vocabulary such as SYN, ACK, timeout or retry is
> tagged as a network event. Unknown words stay plain events, so it never invents
> protocol behaviour."

---

## Part 2: It handles hard cases and admits uncertainty (3 min)

Upload each of these in turn:

| Image | What to point out |
|---|---|
| `crossing_edges.png` | Two diagonals cross in the middle. The result is `Idle→Busy` and `Wait→Done`, **not** `Idle→Done`: the tracer goes straight through crossings. |
| `bidirectional_and_loop.png` | Two curved arrows between the same states, one each way (`open`, `close`), plus a self-loop `tick`. |
| `arrow_without_head.png` | A line with no arrowhead. The edge is drawn **amber and dashed with a "?"**. The **Issues** tab (red badge) says *"arrowhead evidence is ambiguous"* and lists both possible directions. |
| `dangling_arrow.png` | A stroke ends in empty space. **No transition is invented**; Issues says a state may be missing. |
| `no_initial_marker.png` | Issues warns that no initial state is marked. |

**What to say**

> "When the image is ambiguous the system doesn't guess. It marks the element for review,
> lists the candidate interpretations, and the user decides."

---

## Part 3: Correct it in the editor (3 min)

Go back to `tcp_handshake.png` and show that every change keeps the graph, the JSON and
the code in sync.

1. **Fix an OCR slip.** Click the self-loop and change Guard `retries< 3` to `retries<3`.
2. **Reverse an arrow.** Click `CLOSED → SYN_SENT`, click **Reverse direction**, then press
   **Ctrl+Z** to undo it (Ctrl+Y redoes).
3. **Add a state.** Double-click empty canvas, then name it `LISTEN` in the Inspector.
4. **Add a transition.** Click **Connect**, click `CLOSED`, then click `LISTEN`. Set Event to
   `listen`. Clicking the same state twice makes a self-loop.
5. **Move and zoom.** Drag states; use the mouse wheel to zoom and drag the background to
   pan; **Fit** recentres; **Auto layout** rearranges.
6. **Show the JSON.** Open the **JSON** tab: your edits are already in it. Change a state
   name in the text and click **Apply JSON**; the graph updates. Then type a broken
   reference (e.g. a transition target `"S99"`) and **Apply JSON**: it's rejected with
   *"unknown state 'S99'"*.
7. **Network topology** (bottom of the Inspector tab):
   - Add devices `Client` and `Server`.
   - Under *States of this device*, select the client's states.
   - Under **Connect devices**, choose Client → Server with Medium `Ethernet` and click
     **Connect devices**. Or click **Link devices on canvas** and then the two device
     boxes.
   - Linking the same pair again is refused.

**What to say**

> "There is one source of truth: the canonical JSON. The graph view, the JSON tab and
> the generated code are all views of it. Invalid edits are rejected before they reach the
> file."

---

## Part 4: Generate code that is proven to match the diagram (3 min)

In the editor: open the **Code** tab and click **Generate & verify**.

- The report says **"Verified: … steps in … traces match the reference semantics;
  N/N transitions exercised."**
- Make any edit in the graph. The Code tab shows a **stale** badge until you regenerate.
- **Export…** offers the JSON, an SVG of the graph, the code and the generated tests.

Then, in the terminal (outputs below are real):

```bash
python main.py test_images/hard_cases/tcp_handshake.png -o outputs/demo/tcp.json
python main.py generate outputs/demo/tcp.json \
    --output outputs/demo/tcp_machine.py --tests outputs/demo/test_tcp_machine.py
# Generated python code: outputs/demo/tcp_machine.py, 38 conformance steps passed

python outputs/demo/tcp_machine.py
# start: S0 (CLOSED)
# S0 --connect--> S1 (SYN_SENT)
# S1 --SYN-ACK--> S2 (ESTAB)
# S2 --close--> S4 (FIN_WAIT)
# S4 --ACK--> S3 (DONE)

cd outputs/demo && python -m pytest -q test_tcp_machine.py && cd ../..
# 15 passed

python main.py verify outputs/demo/tcp.json outputs/demo/tcp_machine.py connect SYN-ACK close ACK
# "expected": ["S0","S1","S2","S4","S3"], "actual": ["S0","S1","S2","S4","S3"], "valid": true
```

Use the generated class from Python. Guards are plain functions you supply:

```python
# run from outputs/demo/
from tcp_machine import StateMachine, InvalidEventError

m = StateMachine(
    guards={"retries< 3": lambda ctx: ctx["retries"] < 3,
            "retries>=3": lambda ctx: ctx["retries"] >= 3},
    actions={"send SYN": lambda ctx: print("  -> sending SYN")},
)
m.submit("connect")                    # prints "-> sending SYN", returns 'S1'
m.submit("timeout", {"retries": 1})    # 'S1'  (retransmit, stays in SYN_SENT)
m.submit("timeout", {"retries": 3})    # 'S0'  (gives up, back to CLOSED)
m.submit("ACK")                        # InvalidEventError: event 'ACK' is not accepted in state 'S0'
```

**What to say**

> "The generated code has explicit errors for invalid events, rejected guards, final
> states and ambiguous transitions. Before the code is accepted, a test plan built from
> the diagram fires *every* transition and checks the errors. The tests run in a separate,
> locked-down Python process with no file writes, no network and a time limit. Code from
> Gemini has to pass the same tests."

---

## Part 5: Gemini explains and suggests, but never decides (2 min, needs API key)

In the editor, open the **Explain** tab.

1. Type the question *"What happens when the connection times out?"* and click
   **Explain diagram**. Real answer on this diagram:

   > "When the connection times out in the SYN_SENT state, the system checks the retry
   > count. If retries are less than 3, it performs a retransmit action and remains in the
   > SYN_SENT state. If retries are 3 or greater, the system transitions back to the
   > CLOSED state."

2. Click **Suggest corrections**. Gemini compares the image with the graph. Each
   suggestion has **Accept** / **Reject** buttons; nothing changes until you accept.
   Suggestions with impossible values are discarded automatically.
3. In the **Code** tab, **Explain code** proposes test cases. Each one is marked ✔ or ✘
   depending on whether it agrees with the deterministic reference semantics.

CLI equivalent: `python main.py explain outputs/demo/tcp.json --question "..."`.

**What to say**

> "The LLM explains things and makes suggestions. It never does arrow detection, graph
> validation or code acceptance; those are deterministic and tested."

---

## Part 6: Results and honest limits (1–2 min)

Held-out test set: 360 diagrams never used for training or tuning.

| Metric | Original pipeline | New pipeline + retrained model |
|---|---|---|
| Transition F1 | 0.320 | **0.928** |
| Arrow direction correct | 0.614 | **0.964** |
| Initial state correct | 0.000 | **0.983** |
| Final-state F1 | n/a (never detected) | **0.997** |
| Whole graph exactly right (ignoring text) | 0.000 | **0.719** |
| Transition labels read exactly | 0.000 | 0.654 |
| Generated code passes its tests | 0.994 | **1.000** |

Show the limits honestly: upload `test_images/photo_like/net_01_photo.jpg` (skewed,
shaded, low contrast). The structure is mostly right, but state names are misread (for
example `Seenngd nn`). Then say:

> "Training data is synthetic. Shapes transfer much better than text. To close that gap
> there's a feedback loop: **Save correction…** stores the image, the prediction and your
> fix. `python main.py export-corrections` turns them into training labels for
> fine-tuning, and corrections marked *held-out* are used only to measure whether the
> retrained model is actually better."

---

## Optional: network topology diagrams (1 min)

```bash
python main.py test_images/network_topology/network_0000.png --diagram-type network -o outputs/demo/topology.json
python main.py edit outputs/demo/topology.json
```

This opens the topology editor (components and connections) for diagrams drawn as
circles joined by lines.

---

## Likely questions

**Why not just send the image to an LLM?**
Arrow endpoints and directions must be exactly right for the code to be correct, and an
LLM's answer can't be verified. Here, detection and validation are deterministic and
measured; the LLM only explains and suggests.

**How does it know which state an arrow belongs to?**
It traces the drawn stroke pixel by pixel from where the ink touches a state's outline.
It prefers going straight (so it passes through crossings) and follows curves. Direction
comes from detected arrowheads, and each arrowhead can belong to only one arrow.

**What if it's wrong?**
Uncertain elements are amber; the Issues tab lists them with the alternatives. You fix
them in the editor and can save the fix as training data.

**Is running generated code safe?**
It runs in a separate interpreter with resource limits, an empty environment, a
temporary directory and an audit hook blocking files, sockets and subprocesses. A static
check also rejects imports such as `os`, `socket` or `subprocess` before it runs. It's
defence in depth for a local tool, not a container.

**Which languages?**
The deterministic generator produces Python. Gemini can produce other languages, but
only Python is executed and verified.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Detection failed: YOLO weights not found` | Check that `runs/detect/v2/weights/best.pt` exists, or pass `--model`. |
| Labels are all empty | Install Tesseract (`sudo pacman -S tesseract tesseract-data-eng` or `apt install tesseract-ocr`). The status/Issues will say OCR is unavailable. |
| Gemini buttons show an error | Set `GEMINI_API_KEY` in `.env` and restart `python main.py serve`. |
| Port 8000 busy | `python main.py serve --port 8765` |
| Layout cramped | Untick **Image** in the toolbar to hide the source panel, or widen the window. |
| `test_images/` missing | `python make_test_images.py` |

## Command cheat-sheet

```bash
python main.py serve                                   # editor
python main.py IMAGE -o out.json                       # image -> JSON
python main.py analyze out.json                        # reachability, cycles, network events
python main.py generate out.json --tests test_out.py   # code + tests, verified
python main.py verify out.json code.py EVENT ...       # replay events
python main.py explain out.json --question "..."       # Gemini explanation
python main.py suggest out.json                        # Gemini correction suggestions
python main.py evaluate --split test                   # metrics on held-out data
python make_test_images.py                             # regenerate demo images
```
