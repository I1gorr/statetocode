"""CLI for diagram detection and Gemini-backed code generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from statemachine_pipeline.detector import detect
from statemachine_pipeline.codegen import validate_generated_code
from statemachine_pipeline.gemini import GeminiGenerator
from statemachine_pipeline.analysis import analyze
from statemachine_pipeline.trace import reference_trace, run_generated_trace
from statemachine_pipeline.ui import serve
from statemachine_pipeline.petrinet import reconstruct_petri, validate_petri_net
from statemachine_pipeline.models import StateMachine
from statemachine_pipeline.reconstruct import reconstruct
from statemachine_pipeline.validation import validate_state_machine
from statemachine_pipeline.network import NetworkGraph, validate_network_graph
from statemachine_pipeline.network_cv import detect_network


def detect_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Convert a state diagram image to canonical JSON")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("yolo11n.pt"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--diagram-type", choices=("state-machine", "petri-net", "network"), default="state-machine")
    args = parser.parse_args(argv)
    if args.diagram_type == "network":
        graph = detect_network(args.image)
        errors = validate_network_graph(graph)
        payload = graph.to_json()
        if args.output:
            args.output.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
        if graph.metadata.get("warnings"):
            print("Warnings:", *graph.metadata["warnings"], sep="\n- ")
        return 2 if errors else 0
    detections = detect(args.image, args.model, confidence=args.confidence, diagram_type=args.diagram_type)
    if args.diagram_type == "petri-net":
        net = reconstruct_petri(detections, diagram_id=args.image.stem)
        report = validate_petri_net(net)
        payload = net.to_json()
        if args.output:
            args.output.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
        if report.warnings or net.metadata.get("warnings"):
            print("Warnings:", *(report.warnings + list(net.metadata.get("warnings", []))), sep="\n- ")
        return 0 if report.valid else 2
    result = reconstruct(detections, image_path=args.image)
    report = validate_state_machine(result.machine)
    payload = result.machine.to_json()
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    if report.warnings:
        print("Warnings:", *report.warnings, sep="\n- ")
    return 0 if report.valid else 2


def generate_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python main.py generate",
        description="Generate and validate executable code from canonical state-machine JSON",
    )
    parser.add_argument("json_input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/state_machine.py"))
    parser.add_argument("--report", type=Path)
    parser.add_argument("--language", default="python")
    parser.add_argument("--requirements", default="")
    parser.add_argument("--model")
    args = parser.parse_args(argv)

    try:
        from statemachine_pipeline.models import StateMachine

        machine = StateMachine.from_dict(json.loads(args.json_input.read_text(encoding="utf-8")))
        generator = GeminiGenerator(model=args.model)
        generated = generator.generate(
            machine,
            target_language=args.language,
            requirements=args.requirements,
        )
        validation = validate_generated_code(generated, machine)
        if not validation.valid:
            print("Generated code was rejected:", file=sys.stderr)
            for error in validation.errors:
                print(f"- {error}", file=sys.stderr)
            return 2
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(generated.source_code + "\n", encoding="utf-8")
        report = {
            "target_language": generated.target_language,
            "model": generator.model,
            "source_code": str(args.output),
            "state_mapping": generated.state_mapping,
            "assumptions": generated.assumptions,
            "warnings": generated.warnings + validation.warnings,
        }
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Generated validated {generated.target_language} code: {args.output}")
        for warning in report["warnings"]:
            print(f"Warning: {warning}", file=sys.stderr)
        return 0
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Code generation failed: {exc}", file=sys.stderr)
        return 1


def load_machine(path: Path) -> StateMachine:
    return StateMachine.from_dict(json.loads(path.read_text(encoding="utf-8")))


def analyze_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python main.py analyze")
    parser.add_argument("json_input", type=Path)
    args = parser.parse_args(argv)
    result = analyze(load_machine(args.json_input))
    print(json.dumps({
        "initial_states": result.initial_states,
        "reachable_states": result.reachable_states,
        "unreachable_states": result.unreachable_states,
        "dead_ends": result.dead_ends,
        "cycles": result.cycles,
        "branching_states": result.branching_states,
        "paths_to_final": result.paths_to_final,
    }, indent=2, sort_keys=True))
    return 0


def verify_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python main.py verify")
    parser.add_argument("json_input", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("events", nargs="*", help="events or transition IDs")
    args = parser.parse_args(argv)
    machine = load_machine(args.json_input)
    expected = reference_trace(machine, args.events)
    result = run_generated_trace(args.source.read_text(encoding="utf-8"), args.events)
    if result.actual is None:
        print(f"Generated trace failed: {result.error}", file=sys.stderr)
        return 2
    result = result.__class__(expected, result.actual, expected == result.actual, result.error)
    print(json.dumps({"expected": result.expected, "actual": result.actual, "valid": result.valid}, indent=2))
    return 0 if result.valid else 2


def edit_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python main.py edit")
    parser.add_argument("json_input", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--diagram-type", choices=("auto", "state-machine", "network"), default="auto")
    args = parser.parse_args(argv)
    print(f"Open http://{args.host}:{args.port}/")
    serve(args.json_input, args.host, args.port, diagram_type=args.diagram_type)
    return 0


def explain_command(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python main.py explain")
    parser.add_argument("json_input", type=Path)
    parser.add_argument("--question", default="")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model")
    args = parser.parse_args(argv)
    try:
        explanation = GeminiGenerator(model=args.model).explain(
            load_machine(args.json_input), question=args.question
        )
        payload = json.dumps(explanation, indent=2, sort_keys=True)
        if args.output:
            args.output.write_text(payload + "\n", encoding="utf-8")
        else:
            print(payload)
        return 0
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f"Explanation failed: {exc}", file=sys.stderr)
        return 1


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        return generate_command(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        return analyze_command(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        return verify_command(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "edit":
        return edit_command(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == "explain":
        return explain_command(sys.argv[2:])
    return detect_command(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
