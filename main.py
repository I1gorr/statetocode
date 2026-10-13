"""CLI: diagram image -> canonical JSON -> verified code, plus editor, evaluation and training data.

    python main.py IMAGE [--model M] [--output out.json]        detect + reconstruct
    python main.py generate JSON [--backend template|gemini]     generate + verify code
    python main.py verify JSON SOURCE [EVENTS...]                 replay events / full conformance
    python main.py analyze JSON                                   reachability, cycles, dead ends
    python main.py explain JSON                                   Gemini explanation (needs key)
    python main.py suggest JSON [--image IMG]                     Gemini correction suggestions
    python main.py serve [JSON]                                   web editor (upload, edit, generate)
    python main.py edit JSON                                      alias of serve with a file
    python main.py evaluate --model M [--split test]              graph-level metrics on a split
    python main.py compare A.json B.json ...                      metric table from evaluate outputs
    python main.py tune --model M                                 tune association on the val split
    python main.py export-corrections                             corrections -> YOLO training labels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from statemachine_pipeline.analysis import analyze
from statemachine_pipeline.models import StateMachine
from statemachine_pipeline.network import validate_network_graph
from statemachine_pipeline.network_cv import detect_network
from statemachine_pipeline.validation import validate_state_machine

ROOT = Path(__file__).resolve().parent
MODEL_CANDIDATES = (
    ROOT / "runs/detect/v2/weights/best.pt",
    ROOT / "runs/detect/train/weights/best.pt",
)


def default_model() -> Path:
    return next((path for path in MODEL_CANDIDATES if path.exists()), MODEL_CANDIDATES[0])


def load_machine(path: Path) -> StateMachine:
    return StateMachine.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _write(path: Path | None, payload: str) -> None:
    if path is None:
        print(payload)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload + "\n", encoding="utf-8")


def detect_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Convert a state diagram image to canonical JSON")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=None, help="YOLO weights (default: newest trained model)")
    parser.add_argument("--output", "-o", type=Path)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--diagram-type", choices=("state-machine", "petri-net", "network"), default="state-machine")
    parser.add_argument("--no-ocr", action="store_true", help="skip Tesseract label reading")
    args = parser.parse_args(argv)
    model = args.model or default_model()
    try:
        if args.diagram_type == "network":
            graph = detect_network(args.image)
            errors = validate_network_graph(graph)
            _write(args.output, graph.to_json())
            for warning in graph.metadata.get("warnings", []):
                print(f"Warning: {warning}", file=sys.stderr)
            return 2 if errors else 0
        if args.diagram_type == "petri-net":
            from statemachine_pipeline.detector import detect
            from statemachine_pipeline.petrinet import reconstruct_petri, validate_petri_net
            detections = detect(args.image, model, confidence=args.confidence, diagram_type="petri-net")
            net = reconstruct_petri(detections, diagram_id=args.image.stem)
            report = validate_petri_net(net)
            _write(args.output, net.to_json())
            for warning in report.warnings + list(net.metadata.get("warnings", [])):
                print(f"Warning: {warning}", file=sys.stderr)
            return 0 if report.valid else 2
        from statemachine_pipeline.pipeline import run_pipeline
        result = run_pipeline(args.image, model, confidence=args.confidence, ocr=not args.no_ocr)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Detection failed: {exc}", file=sys.stderr)
        return 1
    _write(args.output, result.machine.to_json())
    for warning in result.validation.warnings:
        print(f"Warning: {warning}", file=sys.stderr)
    for uncertainty in result.machine.uncertainties:
        print(f"Review: {uncertainty.element_id}.{uncertainty.field}: {uncertainty.reason}", file=sys.stderr)
    for error in result.validation.errors:
        print(f"Error: {error}", file=sys.stderr)
    return 0 if result.validation.valid else 2


def generate_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python main.py generate",
        description="Generate and verify executable code from canonical state-machine JSON",
    )
    parser.add_argument("json_input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/state_machine.py"))
    parser.add_argument("--tests", type=Path, help="also write a pytest module for the generated code")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--backend", choices=("template", "gemini"), default="template")
    parser.add_argument("--language", default="python")
    parser.add_argument("--requirements", default="")
    parser.add_argument("--model", help="Gemini model (gemini backend)")
    args = parser.parse_args(argv)

    try:
        from statemachine_pipeline.codegen import generate_python, generate_tests, verify_generated

        machine = load_machine(args.json_input)
        validate_state_machine(machine).raise_for_errors()
        assumptions: list[str] = []
        warnings: list[str] = []
        backend_model = None
        if args.backend == "template":
            if args.language.lower() != "python":
                raise ValueError("the template backend generates Python; use --backend gemini for other languages")
            source = generate_python(machine)
            language = "python"
        else:
            from statemachine_pipeline.gemini import GeminiGenerator
            generator = GeminiGenerator(model=args.model)
            backend_model = generator.model
            generated = generator.generate(machine, target_language=args.language, requirements=args.requirements)
            source, language = generated.source_code, generated.target_language
            assumptions, warnings = generated.assumptions, generated.warnings
        conformance = None
        if language.lower() == "python":
            conformance = verify_generated(source, machine)
            if not conformance.valid:
                print("Generated code was rejected:", file=sys.stderr)
                for error in conformance.errors:
                    print(f"- {error}", file=sys.stderr)
                for mismatch in conformance.mismatches[:10]:
                    print(f"- {json.dumps(mismatch)}", file=sys.stderr)
                return 2
        else:
            warnings.append("only Python output can be executed and conformance-checked; review this code manually")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(source.rstrip() + "\n", encoding="utf-8")
        if args.tests and language.lower() == "python":
            args.tests.parent.mkdir(parents=True, exist_ok=True)
            args.tests.write_text(generate_tests(machine, args.output.stem), encoding="utf-8")
        report = {
            "backend": args.backend, "model": backend_model, "target_language": language,
            "source_code": str(args.output), "tests": str(args.tests) if args.tests else None,
            "conformance": conformance.to_dict() if conformance else None,
            "assumptions": assumptions, "warnings": warnings,
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        checked = f", {conformance.steps_checked} conformance steps passed" if conformance else ""
        print(f"Generated {language} code: {args.output}{checked}")
        for warning in warnings:
            print(f"Warning: {warning}", file=sys.stderr)
        return 0
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Code generation failed: {exc}", file=sys.stderr)
        return 1


def analyze_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python main.py analyze")
    parser.add_argument("json_input", type=Path)
    args = parser.parse_args(argv)
    machine = load_machine(args.json_input)
    report = validate_state_machine(machine)
    result = analyze(machine)
    print(json.dumps({
        "valid": report.valid,
        "errors": report.errors,
        "warnings": report.warnings,
        "initial_states": result.initial_states,
        "reachable_states": result.reachable_states,
        "unreachable_states": result.unreachable_states,
        "dead_ends": result.dead_ends,
        "cycles": result.cycles,
        "branching_states": result.branching_states,
        "paths_to_final": result.paths_to_final,
        "network_transitions": [
            {"id": t.id, "event": t.event, "event_type": t.network.event_type}
            for t in machine.transitions if t.network
        ],
    }, indent=2, sort_keys=True))
    return 0


def verify_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python main.py verify")
    parser.add_argument("json_input", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("events", nargs="*", help="events (or transition IDs for unlabelled transitions)")
    args = parser.parse_args(argv)
    from statemachine_pipeline.codegen import verify_generated
    from statemachine_pipeline.trace import reference_trace, run_generated_trace

    machine = load_machine(args.json_input)
    source = args.source.read_text(encoding="utf-8")
    if not args.events:
        report = verify_generated(source, machine)
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0 if report.valid else 2
    try:
        expected = reference_trace(machine, args.events)
    except ValueError as exc:
        print(f"Reference trace rejected the events: {exc}", file=sys.stderr)
        return 2
    result = run_generated_trace(source, args.events)
    if result.actual is None:
        print(f"Generated trace failed: {result.error}", file=sys.stderr)
        return 2
    valid = tuple(expected) == tuple(result.actual)
    print(json.dumps({"expected": expected, "actual": result.actual, "valid": valid}, indent=2))
    return 0 if valid else 2


def serve_command(argv: list[str], *, require_file: bool = False) -> int:
    parser = argparse.ArgumentParser(prog="python main.py serve")
    parser.add_argument("json_input", type=Path, nargs="?" if not require_file else None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--diagram-type", choices=("auto", "state-machine", "network"), default="auto")
    args = parser.parse_args(argv)
    from statemachine_pipeline.ui import serve
    print(f"Open http://{args.host}:{args.port}/")
    serve(args.json_input, args.host, args.port, diagram_type=args.diagram_type,
          model_path=args.model or default_model())
    return 0


def explain_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python main.py explain")
    parser.add_argument("json_input", type=Path)
    parser.add_argument("--question", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model")
    args = parser.parse_args(argv)
    try:
        from statemachine_pipeline.gemini import GeminiGenerator
        explanation = GeminiGenerator(model=args.model).explain(load_machine(args.json_input), question=args.question)
        _write(args.output, json.dumps(explanation, indent=2, sort_keys=True))
        return 0
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Explanation failed: {exc}", file=sys.stderr)
        return 1


def suggest_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python main.py suggest")
    parser.add_argument("json_input", type=Path)
    parser.add_argument("--image", type=Path, help="diagram image (default: source_image in the JSON)")
    parser.add_argument("--model")
    args = parser.parse_args(argv)
    try:
        from statemachine_pipeline.gemini import GeminiGenerator
        machine = load_machine(args.json_input)
        image = args.image or (Path(machine.source_image.path) if machine.source_image else None)
        data = image.read_bytes() if image and image.is_file() else None
        mime = "image/jpeg" if image and image.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
        result = GeminiGenerator(model=args.model).suggest_corrections(machine, data, mime)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Suggestion failed: {exc}", file=sys.stderr)
        return 1


def evaluate_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python main.py evaluate")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset_v2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--check-code", action="store_true", help="generate + conformance-test code per diagram")
    parser.add_argument("--corrections", type=Path, help="evaluate on held-out (split=eval) user corrections instead")
    parser.add_argument("--legacy", action="store_true",
                        help="baseline: original box-geometry association (nearest state centre), no OCR")
    parser.add_argument("--name", help="column name for the results (default: model run name)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    from statemachine_pipeline.evaluate import evaluate_split, format_table, ground_truth_loader
    from statemachine_pipeline.pipeline import run_pipeline

    model = args.model or default_model()
    if args.corrections:
        from statemachine_pipeline.corrections import load_corrections
        records = [r for r in load_corrections(args.corrections) if r.get("split") == "eval"]
        truth = {Path(r["image"]).resolve(): StateMachine.from_dict(r["machine"]) for r in records}
        images = sorted(truth)
        truth_for = lambda image: truth[image.resolve()]  # noqa: E731
        if not images:
            print("No held-out corrections (split=eval) to evaluate.", file=sys.stderr)
            return 1
    else:
        images = sorted((args.dataset / "images" / args.split).glob("*.png"))
        truth_for = ground_truth_loader(args.dataset)
    if args.limit:
        images = images[:args.limit]

    def progress(done: int, total: int) -> None:
        if done % 25 == 0 or done == total:
            print(f"  {done}/{total}", file=sys.stderr)

    if args.legacy:
        from statemachine_pipeline.detector import detect
        from statemachine_pipeline.reconstruct import _reconstruct_from_boxes

        def predict(image: Path) -> StateMachine:
            return _reconstruct_from_boxes(detect(image, model), image.stem, image).machine
    else:
        def predict(image: Path) -> StateMachine:
            return run_pipeline(image, model, ocr=not args.no_ocr).machine

    summary = evaluate_split(images, predict, truth_for, check_code=args.check_code, progress=progress)
    summary["model"] = str(model)
    summary["split"] = str(args.corrections or args.split)
    summary["pipeline"] = "legacy-box-association" if args.legacy else "stroke-tracing" + ("" if args.no_ocr else "+ocr")
    name = args.name or (model.parent.parent.name if model.parent.name == "weights" else model.stem)
    print(format_table({name: summary}))
    output = args.output or ROOT / "outputs" / f"eval_{name}.json"
    _write(output, json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved {output}", file=sys.stderr)
    return 0


def compare_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python main.py compare")
    parser.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args(argv)
    from statemachine_pipeline.evaluate import format_table
    results = {}
    for path in args.results:
        data = json.loads(path.read_text(encoding="utf-8"))
        results[path.stem.removeprefix("eval_")] = data
    print(format_table(results))
    return 0


def tune_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python main.py tune")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset_v2")
    parser.add_argument("--split", default="val", help="never the test split")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--output", type=Path, default=ROOT / "config" / "association.json")
    args = parser.parse_args(argv)
    if args.split == "test":
        print("Refusing to tune on the held-out test split.", file=sys.stderr)
        return 2
    from statemachine_pipeline.tune import save_config, tune
    images = sorted((args.dataset / "images" / args.split).glob("*.png"))[:args.limit]
    config, before, after = tune(images, args.dataset, args.model or default_model(),
                                 log=lambda m: print(m, file=sys.stderr))
    save_config(config, args.output)
    print(json.dumps({"config": config, "before": {k: before[k] for k in ("transitions", "graph_exact_structure", "initial_state_accuracy")},
                      "after": {k: after[k] for k in ("transitions", "graph_exact_structure", "initial_state_accuracy")}},
                     indent=2, sort_keys=True))
    return 0


def export_corrections_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python main.py export-corrections")
    parser.add_argument("--corrections", type=Path, default=ROOT / "data" / "corrections" / "corrections.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "corrections" / "yolo")
    parser.add_argument("--base-dataset", type=Path, default=ROOT / "dataset_v2")
    args = parser.parse_args(argv)
    from statemachine_pipeline.corrections import export_yolo
    from statemachine_pipeline.synthetic import YOLO_CLASSES
    stats = export_yolo(args.corrections, args.output,
                        exclude_dirs=(str((args.base_dataset / "images" / "test").resolve()),))
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(YOLO_CLASSES))
    yaml_path = args.output / "dataset_with_corrections.yaml"
    yaml_path.write_text(
        f"train:\n  - {(args.base_dataset / 'images' / 'train').resolve()}\n  - {(args.output / 'images' / 'train').resolve()}\n"
        f"val: {(args.base_dataset / 'images' / 'val').resolve()}\n"
        f"test: {(args.base_dataset / 'images' / 'test').resolve()}\n\nnames:\n{names}\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2))
    print(f"Fine-tune with: yolo detect train data={yaml_path} model={default_model()} epochs=20 imgsz=640")
    return 0


COMMANDS = {
    "generate": generate_command,
    "analyze": analyze_command,
    "verify": verify_command,
    "edit": lambda argv: serve_command(argv, require_file=True),
    "serve": serve_command,
    "explain": explain_command,
    "suggest": suggest_command,
    "evaluate": evaluate_command,
    "compare": compare_command,
    "tune": tune_command,
    "export-corrections": export_corrections_command,
}


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] in COMMANDS:
        return COMMANDS[sys.argv[1]](sys.argv[2:])
    return detect_command(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
