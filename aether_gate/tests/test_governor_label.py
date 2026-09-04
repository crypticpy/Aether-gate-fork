#
# Aether-gate — U1: the governor's `state_label`, the few plain words a switch shows.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""test_governor.py is at its line budget; the label's own tests live here and
borrow its snapshot and governor builders."""
from aether_gate.tests.test_governor import gov, snap
def test_the_state_label_is_a_few_plain_words_never_the_machine_narrating():
    """U1: the switch prints `state_label`, so it must read like a person --
    listening / trying X / kept / put back / holding X -- and never a tool
    key or a state name; the sentence stays in `why`."""
    g = gov()
    assert g.status()["state_label"] == "off"       # nothing ticked yet
    g.tick(snap(0.0))
    assert g.state == "measuring" and g.status()["state_label"] == "listening"
    assert g.why.startswith("30 dB of ADC headroom")     # R7: what was ruled out
    hot = [{"hz": 1200.0, "db": 20.0}]
    act = g.tick(snap(1.0, carriers=hot, coherence=0.5))[0]
    assert act["label"] == "trying a null on a carrier"
    assert g.status()["state_label"] == "trying a null on a carrier"
    g.applied(act, 1.0, before=3.0)
    assert g.status()["state_label"] == "trying a null on a carrier"
    held = {"held": True, "tool": "null", "depth_db": 12.0, "configured": True,
            "hz": 1200.0, "target": "signal"}
    g.tick(snap(2.0, objective=3.0, carriers=hot, coherence=0.5, squeeze=held))
    assert g.state == "settling" and g.status()["state_label"] == "trying a null on a carrier"
    assert "measuring what a squeeze did" in g.why
    g.tick(snap(1.0 + g.settle_s + 0.1, objective=4.0, carriers=hot, coherence=0.5, squeeze=held))
    assert g.status()["state_label"] == "kept" and "kept a squeeze: +1.0 dB" in g.why
    g.tick(snap(1.0 + g.settle_s + 1.0, objective=4.0, carriers=hot, coherence=0.5, squeeze=held))
    assert g.status()["state_label"] == "holding a squeeze"
    assert "holding" not in g.why       # R8: state_label already says that
    assert "already squeezing +1200 Hz" in g.why        # R7: the remainder
    for key in ("nb", "dig", "guard", "mode"):      # keys, not words a person uses
        assert key not in g.status()["state_label"].split()
    g.auto = False
    g.tick(snap(20.0))
    assert g.status()["state_label"] == "off"

