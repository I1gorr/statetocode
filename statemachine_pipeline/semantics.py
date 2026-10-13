"""Deterministic transition-label semantics (no LLM involved).

Labels follow the UML convention ``event [guard] / action``; every part is
optional. Network events are recognised only from an explicit keyword table so
that protocol behaviour is never invented: an unmatched event stays a plain
``state_change`` transition.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re

from .models import NetworkEvent, StateMachine, Transition, Uncertainty


@dataclass(frozen=True)
class ParsedLabel:
    event: str | None
    guard: str | None
    action: str | None
    well_formed: bool


_LABEL = re.compile(
    r"""^\s*
    (?P<event>[^\[\]/]*?)\s*
    (?:\[\s*(?P<guard>[^\[\]]*?)\s*\])?\s*
    (?:/\s*(?P<action>.*?))?\s*$""",
    re.VERBOSE,
)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = " ".join(value.split())
    return value or None


def parse_label(text: str | None) -> ParsedLabel:
    """Split ``event [guard] / action``; malformed text is flagged, not guessed."""
    if text is None or not text.strip():
        return ParsedLabel(None, None, None, True)
    match = _LABEL.match(text)
    if match is None or text.count("[") != text.count("]") or text.count("/") > 1:
        return ParsedLabel(None, None, None, False)
    return ParsedLabel(
        _clean(match.group("event")),
        _clean(match.group("guard")),
        _clean(match.group("action")),
        True,
    )


# Ordered: the first matching rule wins. Each entry is (pattern, event_type, protocol).
_NETWORK_RULES: tuple[tuple[re.Pattern[str], str, str | None], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), event_type, protocol)
    for pattern, event_type, protocol in (
        (r"\bsyn[\s_-]*ack\b", "accept", "TCP"),
        (r"\bsyn\b", "connect", "TCP"),
        (r"\bfin\b", "close", "TCP"),
        (r"\brst\b|\breset\b", "reset", None),
        (r"\back\b|\backnowledg", "ack", None),
        (r"\btime[\s_-]?out\b|\bexpired?\b", "timeout", None),
        (r"\bretr(y|ies|ansmit)\b|\breconnect", "retry", None),
        (r"\bdisconnect|\bclose|\bteardown|\bhang[\s_-]?up", "close", None),
        (r"\bfail|\berror\b|\bunreachable|\blink[\s_-]?down|\bdrop", "failure", None),
        (r"\bconnect|\bdial\b|\bopen\b|\bhandshake|\blink[\s_-]?up", "connect", None),
        (r"\baccept|\blisten", "accept", None),
        (r"\bsend\b|\bsent\b|\btransmit|\btx\b|\bpush\b", "send", None),
        (r"\brecei?v|\brecv\b|\brx\b|\bdata\b|\bpacket\b", "receive", None),
        (r"\bdhcp", "other", "DHCP"),
        (r"\bhttp", "other", "HTTP"),
    )
)


def classify_network_event(event: str | None) -> NetworkEvent | None:
    """Return network semantics for an event name, or ``None`` when not a network event."""
    if not event:
        return None
    normalised = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", event).replace("_", " ")
    for pattern, event_type, protocol in _NETWORK_RULES:
        match = pattern.search(normalised)
        if match:
            # TCP control flags name the packet itself (SYN, SYN-ACK, FIN).
            packet = re.sub(r"[\s_]+", "-", match.group(0)).upper() if protocol == "TCP" else None
            return NetworkEvent(event_type, protocol, packet)  # type: ignore[arg-type]
    return None


def annotate_transition(transition: Transition) -> tuple[Transition, list[Uncertainty]]:
    """Fill event/guard/action from the raw label and classify network events.

    Existing non-null event/guard/action values (e.g. user edits) are kept.
    """
    uncertainties: list[Uncertainty] = []
    parsed = parse_label(transition.label)
    if not parsed.well_formed:
        uncertainties.append(Uncertainty(
            transition.id, "label",
            f"label {transition.label!r} does not follow 'event [guard] / action'; review event fields",
        ))
    event = transition.event if transition.event is not None and transition.event != transition.label else parsed.event
    guard = transition.guard if transition.guard is not None else parsed.guard
    action = transition.action if transition.action is not None else parsed.action
    if not parsed.well_formed and transition.event == transition.label:
        event = None
    network = transition.network or classify_network_event(event)
    kind = transition.kind
    if network is not None and kind == "state_change":
        kind = "network_event"
    return replace(transition, event=event, guard=guard, action=action,
                   network=network, kind=kind), uncertainties


def annotate_machine(machine: StateMachine) -> StateMachine:
    transitions = []
    uncertainties = list(machine.uncertainties)
    for transition in machine.transitions:
        annotated, found = annotate_transition(transition)
        transitions.append(annotated)
        uncertainties.extend(u for u in found if u not in uncertainties)
    return replace(machine, transitions=tuple(transitions), uncertainties=tuple(uncertainties))
