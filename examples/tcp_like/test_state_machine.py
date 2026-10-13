"""Conformance tests generated from diagram 'diagram'.

Each trace was computed by the deterministic reference semantics in
statemachine_pipeline.codegen. Guard values are injected per step.
"""

import json

import pytest

import state_machine as generated

TRACES = json.loads(r"""[
 {
  "start": "S0",
  "steps": [
   {
    "event": "connect",
    "guards": {
     "n<3": false
    },
    "expected_state": "S0",
    "expected_error": null
   }
  ]
 },
 {
  "start": "S0",
  "steps": [
   {
    "event": "error",
    "guards": {
     "n<3": false
    },
    "expected_state": "S1",
    "expected_error": null
   }
  ]
 },
 {
  "start": "S0",
  "steps": [
   {
    "event": "T2",
    "guards": {
     "n<3": false
    },
    "expected_state": "S2",
    "expected_error": null
   }
  ]
 },
 {
  "start": "S2",
  "steps": [
   {
    "event": "accept",
    "guards": {
     "n<3": true
    },
    "expected_state": "S1",
    "expected_error": null
   }
  ]
 },
 {
  "start": "S2",
  "steps": [
   {
    "event": "accept",
    "guards": {
     "n<3": false
    },
    "expected_state": null,
    "expected_error": "GuardRejectedError"
   }
  ]
 },
 {
  "start": "S3",
  "steps": [
   {
    "event": "SYN-ACK",
    "guards": {
     "n<3": false
    },
    "expected_state": "S0",
    "expected_error": null
   }
  ]
 },
 {
  "start": "S0",
  "steps": [
   {
    "event": "__invalid_event__",
    "guards": {},
    "expected_state": null,
    "expected_error": "InvalidEventError"
   }
  ]
 },
 {
  "start": "S2",
  "steps": [
   {
    "event": "accept",
    "guards": {
     "n<3": true
    },
    "expected_state": "S1",
    "expected_error": null
   },
   {
    "event": "__invalid_event__",
    "guards": {},
    "expected_state": null,
    "expected_error": "InvalidEventError"
   }
  ]
 },
 {
  "start": "S2",
  "steps": [
   {
    "event": "__invalid_event__",
    "guards": {},
    "expected_state": null,
    "expected_error": "InvalidEventError"
   }
  ]
 },
 {
  "start": "S3",
  "steps": [
   {
    "event": "__invalid_event__",
    "guards": {},
    "expected_state": null,
    "expected_error": "InvalidEventError"
   }
  ]
 }
]""")


def _machine(guard_values, start):
    guards = {name: (lambda ctx, name=name: guard_values.get(name, False)) for name in generated.GUARDS}
    return generated.StateMachine(guards=guards, initial_state=start)


@pytest.mark.parametrize("trace", TRACES, ids=[f"trace{i}" for i in range(len(TRACES))])
def test_trace(trace):
    guard_values = {}
    machine = _machine(guard_values, trace["start"])
    for step in trace["steps"]:
        guard_values.clear()
        guard_values.update(step["guards"])
        if step["expected_error"]:
            with pytest.raises(getattr(generated, step["expected_error"])):
                machine.submit(step["event"])
            return
        assert machine.submit(step["event"]) == step["expected_state"]


def test_final_states_reject_events():
    for state in generated.FINAL_STATES:
        machine = generated.StateMachine(initial_state=state)
        assert machine.allowed_events() == []
        with pytest.raises(generated.TerminalStateError):
            machine.submit("any-event")
