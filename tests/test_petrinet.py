from statemachine_pipeline.petrinet import (
    PetriArc, PetriNet, PetriPlace, PetriTransition,
    enabled_transitions, fire, validate_petri_net,
)


def net():
    return PetriNet(
        "petri-net-1.0", "fixture",
        (PetriPlace("P0", "ready", initial_tokens=1), PetriPlace("P1", "done")),
        (PetriTransition("PT0", "finish"),),
        (PetriArc("A0", "P0", "PT0"), PetriArc("A1", "PT0", "P1")),
    )


def test_petri_validation_and_token_firing():
    machine = net()
    assert validate_petri_net(machine).valid
    marking = {"P0": 1, "P1": 0}
    assert enabled_transitions(machine, marking) == ("PT0",)
    assert fire(machine, marking, "PT0") == {"P0": 0, "P1": 1}


def test_petri_disabled_transition_is_rejected():
    machine = net()
    try:
        fire(machine, {"P0": 0, "P1": 0}, "PT0")
    except ValueError as exc:
        assert "not enabled" in str(exc)
    else:
        raise AssertionError("disabled transition should fail")


def test_petri_rejects_dangling_arc():
    report = validate_petri_net(PetriNet(
        "petri-net-1.0", "x", (PetriPlace("P0"),), (PetriTransition("PT0"),),
        (PetriArc("A0", "P0", "missing"),),
    ))
    assert not report.valid
