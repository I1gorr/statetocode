"""Build a folder of diagram images to try the pipeline and editor on.

    python make_test_images.py                # writes test_images/
    python make_test_images.py --output my_images --count 12 --seed 99

Everything is generated with fixed seeds that differ from the training data
(dataset_v2 uses seed 7), so these images were never seen by the detector.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from statemachine_pipeline.synthetic import render_sample

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "tests"))
from drawing import Canvas  # noqa: E402  (shared fixture drawing helper)


def synthetic(folder: Path, count: int, seed: int, network_bias: float, prefix: str) -> list[str]:
    folder.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    names = []
    while len(names) < count:
        sample = render_sample(rng, network_bias=network_bias)
        if sample is None:
            continue
        name = f"{prefix}_{len(names) + 1:02d}"
        cv2.imwrite(str(folder / f"{name}.png"), sample.image)
        truth = replace(sample.machine, diagram_id=name)
        (folder / f"{name}.truth.json").write_text(truth.to_json() + "\n", encoding="utf-8")
        names.append(name)
    return names


def label(canvas: Canvas, x: int, y: int, text: str, size: int = 15) -> None:
    """Transition label in the font the detector was trained on (DejaVu Sans)."""
    from PIL import Image, ImageDraw
    from statemachine_pipeline.synthetic import _font
    pil = Image.fromarray(canvas.image)
    ImageDraw.Draw(pil).text((x, y), text, fill=(0, 0, 0), font=_font(size))
    canvas.image = np.array(pil)


def hard_cases(folder: Path) -> dict[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    cases: dict[str, str] = {}

    c = Canvas(640, 520)
    tl, br, tr, bl = c.state(120, 110, text="Idle"), c.state(520, 410, text="Busy"), c.state(520, 110, text="Wait"), c.state(120, 410, text="Done")
    c.initial_marker(tl)
    c.arrow(tl, br); c.arrow(tr, bl); c.arrow(tl, tr)
    c.save(folder / "crossing_edges.png")
    cases["crossing_edges.png"] = "Two diagonals cross in the middle: expect Idle->Busy and Wait->Done (not Idle->Done), plus Idle->Wait."

    c = Canvas()
    a, b = c.state(160, 260, text="Closed"), c.state(470, 260, text="Open")
    c.initial_marker(a)
    c.arrow(a, b, bend=0.25); c.arrow(b, a, bend=0.25); c.self_loop(b, -math.pi / 2)
    # Closed->Open bows downward, Open->Closed upward; each label sits beside its own arc.
    label(c, 295, 302, "open"); label(c, 292, 198, "close"); label(c, 505, 172, "tick")
    c.save(folder / "bidirectional_and_loop.png")
    cases["bidirectional_and_loop.png"] = "Curved Closed<->Open pair (one arrow each way) and a self-loop on Open."

    c = Canvas()
    a, b = c.state(160, 240, text="A"), c.state(470, 240, text="B")
    c.initial_marker(a)
    c.arrow(a, b, head=False)
    c.save(folder / "arrow_without_head.png")
    cases["arrow_without_head.png"] = "Line with no arrowhead: direction must come out 'unknown' and be listed in Issues for you to fix."

    c = Canvas()
    a = c.state(180, 240, text="Start")
    c.initial_marker(a)
    cv2.line(c.image, (222, 240), (360, 240), (0, 0, 0), 3)
    c.save(folder / "dangling_arrow.png")
    cases["dangling_arrow.png"] = "A stroke leaves Start and ends in empty space: flagged as a possibly missing state, no transition invented."

    c = Canvas()
    a, b, f = c.state(130, 240, text="Run"), c.state(330, 240, text="Stop"), c.state(530, 240, final=True, text="End")
    c.arrow(a, b); c.arrow(b, f)
    c.save(folder / "no_initial_marker.png")
    cases["no_initial_marker.png"] = "No initial marker: a warning asks you to mark the initial state; End is a final (double ring)."

    # A larger, hand-laid-out TCP-like handshake with guards and actions.
    c = Canvas(900, 620)
    closed = c.state(160, 120, r=48, text="CLOSED")
    syn_sent = c.state(450, 120, r=50, text="SYN_SENT")
    established = c.state(780, 120, r=55, text="ESTAB")
    fin_wait = c.state(780, 450, r=50, text="FIN_WAIT")
    timed = c.state(450, 450, r=50, final=True, text="DONE")
    c.initial_marker(closed, angle=math.pi)
    c.arrow(closed, syn_sent); label(c, 225, 92, "connect / send SYN")
    c.arrow(syn_sent, established); label(c, 545, 92, "SYN-ACK / send ACK")
    c.arrow(syn_sent, closed, bend=-0.3); label(c, 245, 160, "timeout [retries>=3]")
    c.self_loop(syn_sent, -math.pi / 2); label(c, 520, 18, "timeout [retries<3] / retransmit")
    c.arrow(established, fin_wait); label(c, 650, 280, "close / send FIN")
    c.arrow(fin_wait, timed); label(c, 600, 425, "ACK")
    c.save(folder / "tcp_handshake.png")
    cases["tcp_handshake.png"] = ("Hand-laid-out TCP-style handshake: network events (connect, SYN-ACK, timeout, close, ACK), "
                                  "guards [retries<3]/[retries>=3], actions, a retry self-loop and a final DONE state.")
    return cases


def photo_like(source_folder: Path, folder: Path, names: list[str], seed: int) -> list[str]:
    """Perspective, uneven lighting and low contrast, as from a phone camera.
    Geometry changes, so no ground truth is written for these."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    out = []
    for index, name in enumerate(names):
        image = cv2.imread(str(source_folder / f"{name}.png"))
        h, w = image.shape[:2]
        jitter = lambda: rng.uniform(-0.06, 0.06) * w  # noqa: E731
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[jitter(), jitter()], [w + jitter(), jitter()], [w + jitter(), h + jitter()], [jitter(), h + jitter()]])
        warped = cv2.warpPerspective(image, cv2.getPerspectiveTransform(src, dst), (w, h), borderValue=(235, 235, 235))
        gradient = np.linspace(rng.uniform(0.6, 0.8), 1.0, w, dtype=np.float32)[None, :, None]
        if index % 2:
            gradient = gradient[:, ::-1]
        shaded = warped.astype(np.float32) * gradient
        shaded = shaded * rng.uniform(0.75, 0.9) + rng.uniform(20, 45)  # lower contrast, lifted blacks
        shaded = np.clip(shaded + rng.normal(0, 4, shaded.shape), 0, 255).astype(np.uint8)
        target = f"{name}_photo.jpg"
        cv2.imwrite(str(folder / target), shaded, [cv2.IMWRITE_JPEG_QUALITY, 80])
        out.append(target)
    return out


def write_readme(output: Path, sections: dict[str, list[tuple[str, str]]]) -> None:
    lines = [
        "# Test images",
        "",
        "Generated by `python make_test_images.py` (fixed seeds, none of these were used for training).",
        "",
        "Open the editor with `python main.py serve`, then **Upload image**; or run the CLI:",
        "",
        "```bash",
        "python main.py test_images/state_machines/network_protocols/net_01.png -o outputs/net_01.json",
        "python main.py generate outputs/net_01.json --tests outputs/test_net_01.py",
        "python main.py test_images/network_topology/network_0000.png --diagram-type network -o outputs/topology.json",
        "```",
        "",
        "Files ending in `.truth.json` are the exact graphs the images were drawn from. Compare with",
        "`python main.py edit test_images/.../X.truth.json` or score a prediction in Python with",
        "`statemachine_pipeline.evaluate.compare(predicted, truth)`.",
        "",
    ]
    for title, items in sections.items():
        lines.append(f"## {title}")
        lines.append("")
        for name, note in items:
            lines.append(f"- `{name}`" + (f": {note}" if note else ""))
        lines.append("")
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "test_images")
    parser.add_argument("--count", type=int, default=8, help="diagrams per synthetic category")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    out = args.output
    if out.exists():
        shutil.rmtree(out)
    sm = out / "state_machines"
    generic = synthetic(sm / "generic", args.count, args.seed, 0.0, "generic")
    network = synthetic(sm / "network_protocols", args.count, args.seed + 1, 1.0, "net")
    cases = hard_cases(out / "hard_cases")
    photos = photo_like(sm / "network_protocols", out / "photo_like", network[:4], args.seed) + \
        photo_like(sm / "generic", out / "photo_like", generic[:4], args.seed + 1)

    from generate_network_diagrams import generate as generate_topology
    generate_topology(out / "network_topology", count=4, seed=args.seed)

    real = out / "real"
    real.mkdir(parents=True, exist_ok=True)
    if (ROOT / "images" / "test1.png").exists():
        shutil.copyfile(ROOT / "images" / "test1.png", real / "test1.png")

    write_readme(out, {
        "state_machines/generic": [(f"{n}.png", "generic states/events, with ground truth") for n in generic],
        "state_machines/network_protocols": [(f"{n}.png", "network-protocol vocabulary (connect, SYN, timeout...), with ground truth") for n in network],
        "hard_cases": list(cases.items()),
        "photo_like": [(n, "perspective + shading + low contrast; expect lower accuracy (no ground truth)") for n in photos],
        "network_topology": [(p.name, "circles joined by lines; use --diagram-type network")
                             for p in sorted((out / "network_topology").glob("*.png"))],
        "real": [("test1.png", "the project's original sample (2 states, initial arrow, one transition)")],
    })
    total = sum(1 for _ in out.rglob("*.png")) + sum(1 for _ in out.rglob("*.jpg"))
    print(f"Wrote {total} images to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
