"""Deterministic graph analysis for the editor and reports."""

from collections import defaultdict, deque
from dataclasses import dataclass
from .models import StateMachine
from .validation import validate_state_machine


@dataclass(frozen=True)
class GraphAnalysis:
    initial_states: tuple[str, ...]
    reachable_states: tuple[str, ...]
    unreachable_states: tuple[str, ...]
    dead_ends: tuple[str, ...]
    cycles: tuple[tuple[str, ...], ...]
    branching_states: tuple[str, ...]
    paths_to_final: dict[str, tuple[tuple[str, ...], ...]]


def analyze(machine: StateMachine) -> GraphAnalysis:
    validate_state_machine(machine).raise_for_errors()
    state_ids = {s.id for s in machine.states}
    adjacency: dict[str, list[str]] = defaultdict(list)
    for transition in machine.transitions:
        adjacency[transition.source].append(transition.target)
    initial = tuple(s.id for s in machine.states if s.type == "initial")
    roots = initial or tuple(s.id for s in machine.states[:1])
    reachable: set[str] = set()
    queue = deque(roots)
    while queue:
        current = queue.popleft()
        if current in reachable:
            continue
        reachable.add(current)
        queue.extend(adjacency[current])
    final = {s.id for s in machine.states if s.type == "final"}
    dead_ends = tuple(sorted(s for s in state_ids if not adjacency[s] and s not in final))
    branching = tuple(sorted(s for s in state_ids if len(adjacency[s]) > 1))
    cycles: set[tuple[str, ...]] = set()

    def visit(node: str, path: tuple[str, ...]) -> None:
        if node in path:
            cycle = path[path.index(node):]
            cycles.add(tuple(sorted(cycle)))
            return
        for target in adjacency[node]:
            visit(target, path + (node,))
    for root in roots:
        visit(root, ())

    paths: dict[str, tuple[tuple[str, ...], ...]] = {}
    for start in state_ids:
        found: list[tuple[str, ...]] = []
        stack: list[tuple[str, tuple[str, ...]]] = [(start, (start,))]
        while stack:
            node, path = stack.pop()
            if node in final:
                found.append(path)
                continue
            if len(path) > len(state_ids) + 1:
                continue
            for target in adjacency[node]:
                if target not in path:
                    stack.append((target, path + (target,)))
        paths[start] = tuple(sorted(found))
    return GraphAnalysis(
        initial, tuple(sorted(reachable)), tuple(sorted(state_ids - reachable)),
        dead_ends, tuple(sorted(cycles)), branching, paths,
    )
