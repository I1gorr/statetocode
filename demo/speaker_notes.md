# Speaker notes

Talking points for each card of the presentation, in order.

## Problem: A diagram is an image. Software needs a graph.
- The output is a graph, not a label. Every arrow needs the correct source, target and direction.
- We are doing structured reconstruction, not plain object detection.
- A wrong guess produces wrong code with no warning, so uncertainty is part of the design.

## System: Five stages take an image to tested code
- Walk the five boxes left to right. Only box 2 is learned.
- A missing Gemini key or missing Tesseract does not break the rest of the system.

## System: Learned parts perceive. Deterministic parts reason.
- Arrow endpoints and directions must be exactly right for the code to be right, and an LLM answer cannot be checked.
- Here perception is measured on 360 held-out diagrams, the output is validated, and the code is tested.

## Detection: YOLO11n finds the parts, but only as boxes
- YOLO only gives axis-aligned boxes. For a curved arrow the box says nothing about endpoints or direction.
- That is why the next stages work on the actual pixels.

## Detection: The detector is nearly solved on this data
- mAP@50 of 0.992 on 360 unseen diagrams, with 5,108 labelled objects.
- mAP@50-95 is lower at 0.934 because it demands tighter box placement. The Classes card shows where.

## Strokes: First, a clean ink mask, even from a shadowed photo
- This stage is classical OpenCV with no learning. Dividing by the estimated background is what makes shadowed photos work.
- Otsu picks the threshold automatically, so there is no manual tuning per image.

## Strokes: Follow the pen, do not guess the nearest state
- This is the main contribution. The first version linked each arrow end to the nearest state centre, which invents self-loops, swaps directions and links wrong states when edges cross.
- Because turning is penalised, the tracer keeps going straight at a crossing instead of switching edges.
- Demo image: crossing_edges.png.

## Meaning: The arrowhead is the evidence for direction
- Direction decides source versus target, so it needs evidence.
- Exclusivity separates shallow crossings of two nearly parallel curved arrows.
- Demo image: arrow_without_head.png.

## Meaning: Start and end states, names and labels
- An arrow that starts in empty space and ends at a state is the entry arrow, not a transition.
- Shape: a rounded rectangle versus a circle is decided by whether the box diagonals carry ink near the corners.
- We never invent protocol behaviour: unknown words stay plain events.

## Meaning: When unsure, the system says so
- A system that silently guesses produces wrong code with no warning. Ours hands the decision to the user.
- Demo images: arrow_without_head.png, dangling_arrow.png and no_initial_marker.png.

## Code: One canonical JSON is the source of truth
- The graph view, the JSON tab and the generated code are all views of one file, so they cannot drift apart.
- Validation runs on every editor save and every code generation.

## Code: Generated code is checked against an independent reference
- The plan can be exported as a standalone pytest module.
- A diagram with no detected states counts as a failure in the conformance figure.

## Code: Sandboxed execution, and a limited role for Gemini
- Be upfront that the sandbox is process-level isolation. For untrusted code at scale we would use a container.
- Suggestions are never auto-applied. LLM test cases are replayed on the reference model and marked as agreeing or not.

## Data and scoring: Synthetic diagrams with exact ground truth
- Hand-labelling thousands of real diagrams is impractical, so diagrams are rendered by code, each with its exact true graph.
- That lets us evaluate the whole reconstructed graph, not only detection.

## Data and scoring: We score the whole graph, not just the boxes
- Hungarian matching gives the best one-to-one pairing between predicted and true states.
- Metrics are summed over all 360 diagrams before dividing (micro-averaged).

## Precision and recall: Raise the threshold: precision up, recall down
- Set the threshold to 0.90: recall falls from 0.99 to 0.70 while precision only rises from 0.987 to 0.998.
- The pipeline uses 0.40. The best-F1 point is 0.63, with F1 0.991 against 0.989, so F1 is flat across a wide range.
- A missed arrowhead makes direction uncertain, and the system flags it for review instead of guessing.

## Precision and recall: Every class at the chosen threshold
- Arrowhead and initial marker have the lowest recall: small objects that look like parts of arrows.
- State and final state stay at 1.000 across a wide range of thresholds.

## Classes: Arrowheads are the hard class
- States and final states score 0.995 on both measures: large and clean shapes.
- Arrowheads drop from 0.982 to 0.832 under the strict measure. They are a few pixels across, so a small shift lowers the overlap a lot. The detector finds them but cannot place the box to the pixel.

## Classes: The confusion matrix is nearly diagonal
- Choose Errors only. 14 initial markers were called arrows (both are short arrows), 9 arrows were called arrowheads, and 17 arrowheads and 4 arrows were missed.
- The errors make sense for line drawings: they are between classes that look alike.

## Training: Training converged without overfitting
- Validation loss stays close to training loss and mAP plateaus. All 60 epochs ran, since the early-stopping patience of 15 never triggered.
- Switch to the original model: it starts at precision 0.024 with recall 0.984, an untrained classifier firing on nearly everything.

## Graph results: Detector accuracy is not graph accuracy
- The same detector gives 0.32 with nearest-centre linking and 0.585 with stroke tracing. Retraining the detector then raises it to 0.928.
- Lesson: a component metric such as mAP can look saturated while the task metric still has most of its headroom. Measure the task.

## Graph results: Transition F1 on 360 held-out diagrams
- A to B changes only the algorithm: 0.320 to 0.585. B to C changes only the detector: 0.585 to 0.928.
- Other numbers for C: state F1 1.000, endpoint accuracy 0.964, initial-state accuracy 0.983 (was 0.000), final-state F1 0.997, whole graph structurally exact 0.719 (was 0.000).

## Graph results: Where the transitions went
- Pipeline A over-predicted: 1,660 transitions for 1,268 real ones. B was cautious at 920. C predicted 1,212 and got 1,151 right.
- In C, 36 transitions had the right pair of states but the wrong direction, 81 were missed or linked wrongly, and 25 were spurious.

## Graph results: Precision and recall by hand
- Precision is true positives over everything predicted. Recall is true positives over everything real.
- A transition counts as correct only if the pair of states and the direction are both right.

## Graph results: Structure is strong, text is the weak spot
- 72 percent of whole graphs are structurally exact. Requiring every character of every name and label to be exact gives 27 percent, because one wrong character fails the diagram.
- State names have a character error rate of 0.052. Transition labels 0.228: small text on or near lines, with brackets and slashes.

## Editor: A web editor, and a loop that learns from fixes
- The editor is a small server on Python's standard library with plain HTML and JavaScript, bound to localhost.
- Records saved as held-out evaluation are never exported for training. They only measure whether a retrained model really improved.
- No real corrections have been collected yet, so no improvement from this loop is claimed.

## Wrap-up: What the numbers do not claim
- Say these before you are asked. The biggest is the domain gap: every measured number is on rendered diagrams.
- Next steps: collect and fix real hand-drawn and photographed diagrams, fine-tune the detector, try a handwriting-capable OCR model, support more target languages with the same conformance check.
