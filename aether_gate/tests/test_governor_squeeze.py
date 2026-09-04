#
# Aether-gate — AUTO CLEAN banks a squeeze only while the squeeze is HOLDING.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Live, on a 50 Hz comb 33.7 dB over the floor: the governor proposed
squeeze=comb, read the talker two seconds later, found +1.1 dB -- the
objective's own noise -- and banked it. From then on the AUTO CLEAN banner
said "squeezing the comb with a notch" and the CHAIN's squeeze row said
"no comb found", because the comb detector had never found one and the
squeeze had never taken hold.

So: a squeeze is scored on `held`, not on the talker's delta; the detector
gets to SQUEEZE_SETTLE_S to say so; a squeeze that HELD and stops is put
back after SQUEEZE_GRACE_S; and the rule never says "already squeezing"
about a target that is not holding.

Written on hand-made snapshots like test_governor.py, whose `snap`/`gov`/
`run` helpers this file reuses; test_governor.py is at its line budget.

Run:  .venv/bin/python -m pytest aether_gate/tests/test_governor_squeeze.py -q
"""
from aether_gate.core import governor as G

from .test_governor import gov, run, snap

M20, M40 = 14_175_000, 7_150_000
HUM = {"mains_hz": 50.0, "harmonics": 3, "hum_db": 33.7, "coherence": 0.02,
       "band_hz": M20}
LOOKING = {"held": False, "tool": None, "depth_db": None, "target": "comb",
           "hz": None, "configured": True, "reason": "no comb found"}
NOTCHING = {"held": True, "tool": "notch", "depth_db": 14.0, "target": "comb",
            "hz": None, "configured": True, "reason": None}


def _proposed_a_comb(g=None, t=0.0):
    """The governor asks for the comb and the write lands; the squeeze is
    configured from here on, and still looking for a comb."""
    g = gov() if g is None else g
    acts = run(g, snap(t, **HUM))
    assert acts[0]["params"] == {"squeeze": "comb"} and acts[0]["kind"] == "mains"
    return g


def test_a_comb_that_never_took_hold_is_never_banked_on_the_talkers_noise():
    """Mutation: scoring the squeeze on the objective alone, which is what it
    did. +1.1 dB with nothing notched is the band, not the tool."""
    g = _proposed_a_comb()
    assert g.tick(snap(3.0, objective=11.1, squeeze=LOOKING, **HUM)) == []
    assert g.state == "settling" and "take hold" in g.why
    assert g.holding == {}
    acts = g.tick(snap(G.SQUEEZE_SETTLE_S + 0.1, objective=11.1,
                       squeeze=LOOKING, **HUM))
    assert acts[0]["revert"] is True and acts[0]["params"] == {"squeeze": ""}
    assert g.holding == {} and g.events[-1]["result"] == "undone"
    assert g.events[-1]["why"].endswith("; put back: the squeeze found no comb to notch")
    assert ("mains", "squeeze") in g.backoff


def test_a_refused_target_says_its_own_reason_when_the_deadline_runs_out():
    """The detector is not the only way a squeeze holds nothing: "too weak"
    and "outside the passband" are the squeeze's own words and are what the
    operator gets back."""
    g = _proposed_a_comb()
    refused = dict(LOOKING, reason="outside the passband")
    g.tick(snap(G.SQUEEZE_SETTLE_S + 0.1, objective=11.1, squeeze=refused, **HUM))
    assert g.events[-1]["why"].endswith("; put back: the squeeze never took hold "
                                        "(outside the passband)")


def test_a_comb_that_takes_hold_inside_the_deadline_is_kept():
    """...and the fix does not stop AUTO CLEAN keeping a squeeze that works:
    held, with the talker no worse, is exactly what it is there for."""
    g = _proposed_a_comb()
    assert g.tick(snap(3.0, objective=11.1, squeeze=NOTCHING, **HUM)) == []
    assert list(g.holding) == ["squeeze"]
    assert g.events[-1]["result"] == "kept" and g.events[-1]["delta_db"] == 1.1
    assert g.status()["holding"][0]["tool"] == "squeeze"


def _holding_the_comb(t=3.0):
    g = _proposed_a_comb()
    g.tick(snap(t, objective=11.1, squeeze=NOTCHING, **HUM))
    assert list(g.holding) == ["squeeze"]
    return g


def test_a_held_comb_that_goes_away_is_put_back_after_the_grace():
    """Mutation: holding it for ever. A retune inside the band moves the teeth
    out of the passband and the squeeze lets go -- the banner would go on
    naming a notch that is not notching anything."""
    lost = dict(LOOKING, reason="outside the passband")
    g = _holding_the_comb()
    assert g.tick(snap(5.0, objective=11.1, squeeze=lost, **HUM)) == []
    assert list(g.holding) == ["squeeze"]            # one refresh is not gone
    assert g.tick(snap(5.0 + G.SQUEEZE_GRACE_S - 0.1, objective=11.1,
                       squeeze=lost, **HUM)) == []
    acts = g.tick(snap(5.0 + G.SQUEEZE_GRACE_S + 0.1, objective=11.1,
                       squeeze=lost, **HUM))
    assert acts[0]["revert"] is True and acts[0]["params"] == {"squeeze": ""}
    assert acts[0]["label"] == "putting it back"
    assert g.holding == {} and ("mains", "squeeze") in g.backoff
    e = g.events[-1]
    assert e["result"] == "undone" and e["tool"] == "squeeze"
    assert e["why"] == "put back: the squeeze lost its comb (outside the passband)"
    assert e["wall"] is not None                     # R9: the timeline shows it


def test_a_held_row_carries_its_wall_clock_stamp():
    """Mutation: `since` alone. It is the governor's uptime, and the app was
    drawing a held null as "20671 d" old from it; `since_wall` is the epoch
    stamp an age can actually be read from, beside `since` the way every
    event carries `wall` beside `t` (R9)."""
    g = _holding_the_comb()
    h = g.holding["squeeze"]
    assert h["since_wall"] is not None and h["since_wall"] != h["since"]
    row = g.status()["holding"][0]
    assert row["since_wall"] == h["since_wall"]


def test_a_comb_that_comes_straight_back_is_left_alone():
    """Mutation: putting it back on the first not-held refresh. `held` is a
    per-refresh judgement and the teeth come and go with the retune."""
    g = _holding_the_comb()
    g.tick(snap(5.0, objective=11.1, squeeze=LOOKING, **HUM))
    g.tick(snap(6.0, objective=11.1, squeeze=NOTCHING, **HUM))
    assert g.tick(snap(20.0, objective=11.1, squeeze=NOTCHING, **HUM)) == []
    assert list(g.holding) == ["squeeze"]


def test_the_band_change_gets_there_first_and_they_do_not_fight():
    """Mutation: putting the lost squeeze back as well as the band's own
    revert. Two reverts of one tool is two writes for one move."""
    g = _holding_the_comb()
    g.tick(snap(5.0, objective=11.1, squeeze=LOOKING, **HUM))
    acts = g.tick(snap(6.0, objective=11.1, squeeze=LOOKING,
                       **dict(HUM, band_hz=M40)))
    assert [a["tool"] for a in acts] == ["squeeze"]
    assert g.events[-1]["why"] == "put back the squeeze: band changed"
    assert g.tick(snap(20.0, objective=11.1, squeeze=LOOKING,
                       **dict(HUM, band_hz=M40))) == []


def test_the_rule_says_it_is_still_looking_not_already_squeezing():
    """R7/R3: "already squeezing the mains comb" over a row reading "no comb
    found" is the banner's lie in one line. While it is searching, say that."""
    g = _holding_the_comb()
    g.tick(snap(5.0, objective=11.1, squeeze=LOOKING, **HUM))
    out = {o["tool"]: o["why"] for o in g.status()["ruled_out"]}
    assert out["squeeze"] == "the squeeze is still looking for a comb"
    assert "the squeeze is still looking for a comb" in g.why
    # ...and once it IS holding, the rule says so in the old words
    g.tick(snap(6.0, objective=11.1, squeeze=NOTCHING, **HUM))
    out = {o["tool"]: o["why"] for o in g.status()["ruled_out"]}
    assert out["squeeze"] == "already squeezing the mains comb"


def test_a_signal_target_that_is_not_holding_is_not_already_squeezing_either():
    hot = [{"hz": 1200.0, "db": 20.0}]
    weak = {"held": False, "tool": None, "depth_db": None, "target": "signal",
            "hz": 1200.0, "configured": True, "reason": "too weak"}
    g = gov()
    g.holding["squeeze"] = {"tool": "squeeze", "params": {"squeeze": 1200},
                            "kind": "carrier", "why": "", "since": 0.0,
                            "delta_db": 1.0, "scorer": "snr",
                            "undo": {"squeeze": ""}}
    assert g.tick(snap(carriers=hot, coherence=0.6, squeeze=weak)) == []
    out = {o["tool"]: o["why"] for o in g.status()["ruled_out"]}
    assert out["squeeze"] == "the squeeze has not taken hold (too weak)"


def test_after_the_put_back_the_rule_can_ask_for_the_comb_again():
    """Once it is back and the pair's silence is up, the mains comb is a
    proposal like any other -- the put-back is not a life sentence."""
    lost = dict(LOOKING, reason="outside the passband")
    g = _holding_the_comb()
    g.tick(snap(5.0, objective=11.1, squeeze=lost, **HUM))
    acts = g.tick(snap(9.0, objective=11.1, squeeze=lost, **HUM))
    g.applied(acts[0], 9.0)                     # the release lands: no target now
    g.tick(snap(10.0, objective=11.1, **HUM))   # ...and is ours, not the operator's
    t = 9.0 + G.BACKOFF_S
    assert g.tick(snap(t - 1.0, objective=11.1, **HUM)) == []
    assert "backing off" in g.why
    assert g.tick(snap(t + 1.0, objective=11.1, **HUM))[0]["params"] == {"squeeze": "comb"}
