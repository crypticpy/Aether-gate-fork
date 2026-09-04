#
# Aether-gate — G4: AUTO CLEAN knows the loops (site tools vs span tools).
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The band is a loop of its own. What the SITE earns -- the front-end guard,
the blanker -- carries across a band change untouched; what the SPAN earns --
a null on the floor, a squeeze on a carrier -- is PUT BACK, because what kept
it was a measurement of a signal that is not in the window any more.

Written on hand-made snapshots like test_governor.py, whose `snap`/`gov`/`run`
helpers this file reuses; test_governor.py is at its 800-line budget.

Each test names the mutation it catches in its own body.

Run:  .venv/bin/python -m pytest aether_gate/tests/test_governor_bands.py -q
"""
from aether_gate.adapters import diversity_governor as adapt
from aether_gate.core import governor as G

from .test_governor import gov, run, snap

M20, M40 = 14_175_000, 7_150_000


def _holding_a_squeeze(t=0.0, band=M20):
    """A governor holding a squeeze on a carrier and a blanker on impulses:
    one span tool and one site tool, both kept."""
    g = gov()
    s = snap(t=t, band_hz=band, impulses_per_s=20.0, impulse_db=14.0,
             carriers=[{"hz": 900.0, "db": 20.0}])
    run(g, s)                                   # the blanker goes on
    run(g, dict(s, t=t + 5.0, nb={"on": True, "db": 11.0, "auto": "off"}))
    s = dict(s, t=t + 10.0, nb={"on": True, "db": 11.0, "auto": "off"})
    run(g, s)                                   # ...then the squeeze
    run(g, dict(s, t=t + 20.0, squeeze={"held": True, "tool": "notch",
                                        "depth_db": 20.0, "target": "signal",
                                        "hz": 900.0, "configured": True}))
    return g, dict(s, t=t + 25.0,
                   squeeze={"held": True, "tool": "notch", "depth_db": 20.0,
                            "target": "signal", "hz": 900.0, "configured": True})


def test_the_setup_holds_one_site_tool_and_one_span_tool():
    g, _s = _holding_a_squeeze()
    assert set(g.holding) == {"nb", "squeeze"}


def test_a_band_change_puts_the_span_tool_back_and_keeps_the_site_tool():
    """Mutation: releasing instead of reverting (what `auto off` does). The
    notch would stay in the chain on the new band with nobody holding it, and
    nothing would ever take it out."""
    g, s = _holding_a_squeeze()
    acts = g.tick(dict(s, t=100.0, band_hz=M40))
    assert [a["tool"] for a in acts] == ["squeeze"]
    a = acts[0]
    assert a["revert"] is True and a["params"] == {"squeeze": ""}
    assert set(g.holding) == {"nb"}             # the site's tool is untouched


def test_the_event_says_it_in_the_operators_words():
    """Mutation: an event with no reason, or the release wording. The CHAIN
    window's list is the only place this is ever explained."""
    g, s = _holding_a_squeeze()
    g.tick(dict(s, t=100.0, band_hz=M40))
    last = g.status()["events"][-1]
    assert last["why"] == "put back the squeeze: band changed"
    assert last["result"] == "undone" and last["tool"] == "squeeze"


def test_the_null_is_a_span_tool_too():
    """Mutation: SPAN_TOOLS holding only the squeeze. A null solved against a
    20 m floor is not a null on 40 m."""
    g = gov()
    s = snap(band_hz=M20, mode="off", coherence=0.9)
    run(g, s)
    run(g, dict(s, t=5.0, mode="null"))
    assert "mode" in g.holding
    acts = g.tick(dict(s, t=100.0, mode="null", band_hz=M40))
    assert [a["tool"] for a in acts] == ["mode"]
    assert acts[0]["params"] == {"mode": "off"} and acts[0]["revert"] is True
    assert g.status()["events"][-1]["why"] == "put back the null: band changed"


def test_a_centre_move_inside_the_band_changes_nothing():
    """Mutation: keying off slice_hz instead of band_hz. Every 2 kHz nudge
    would put the squeeze back, and AUTO CLEAN would never hold anything."""
    g, s = _holding_a_squeeze()
    assert g.tick(dict(s, t=100.0, slice_hz=14_290_000.0, band_hz=M20)) == []
    assert set(g.holding) == {"nb", "squeeze"}


def test_the_first_snapshot_seeds_the_band_and_puts_nothing_back():
    """Mutation: treating band None -> 20 m as a change. AUTO CLEAN would
    revert whatever it held the first time it ever saw a band."""
    g, s = _holding_a_squeeze(band=None)
    assert set(g.holding) == {"nb", "squeeze"}
    assert g.tick(dict(s, t=100.0, band_hz=M20)) != []      # None -> 20 m IS a move
    g2 = gov()
    assert g2.tick(snap(t=1.0, band_hz=M20)) == []          # ...but the first read is not
    assert g2._band == M20


def test_the_span_backoffs_go_and_the_site_backoffs_stay():
    """Mutation: keeping every backoff. Five minutes of silence about a
    squeeze measured on 20 m says nothing about 40 m."""
    g, s = _holding_a_squeeze()
    g.backoff = {("carrier", "squeeze"): 400.0, ("floor", "mode"): 400.0,
                 ("impulse", "nb"): 400.0, ("neighbour", "guard"): 400.0}
    g.tick(dict(s, t=100.0, band_hz=M40))
    assert set(g.backoff) == {("impulse", "nb"), ("neighbour", "guard")}


def test_the_dig_backoff_goes_and_the_pairs_already_dug_stay():
    """Mutation: leaving the dig's backoff alone. Live it held DIG OUT off
    for half an hour after a band change while the operator waited -- the
    (kind, tool) key has no frequency in it. `_dug` does, so it stays."""
    g, s = _holding_a_squeeze()
    g.backoff[("weak", "dig")] = 2000.0
    g._dug.add((7185000, "3"))
    g.tick(dict(s, t=100.0, band_hz=M40))
    assert ("weak", "dig") not in g.backoff
    assert g._dug == {(7185000, "3")}


def test_a_pending_span_move_is_put_back_before_it_is_scored():
    """Mutation: scoring it anyway. The `before` reading was taken on the old
    band, so the delta would be the band change, not the tool."""
    g = gov()
    s = snap(band_hz=M20, mode="off", coherence=0.9)
    acts = g.tick(s)
    g.applied(acts[0], s["t"], before=s["objective"])
    assert g.pending is not None
    acts = g.tick(dict(s, t=0.5, mode="null", band_hz=M40))
    assert [a["tool"] for a in acts] == ["mode"] and acts[0]["revert"] is True
    assert g.pending is None and g.holding == {}


def test_the_spread_is_one_bands_spread():
    """Mutation: carrying the objective ring across. The margin every undo is
    judged by would be half the spread of two different bands."""
    g, s = _holding_a_squeeze()
    for k in range(G.SPREAD_N):
        run(g, dict(s, t=30.0 + k, objective=10.0 + k))
    assert g.spread_db() > 0.0
    g.tick(dict(s, t=100.0, band_hz=M40))
    assert g.spread_db() == 0.0


def test_the_dig_is_not_put_back_by_a_band_change():
    """Mutation: putting "dig" in SPAN_TOOLS. The dig has no undo -- it owns
    its own revert (G5) -- so this would try to write None as a setting."""
    assert "dig" not in G.SPAN_TOOLS and "dig" not in G.SITE_TOOLS
    assert set(G.SITE_TOOLS) | set(G.SPAN_TOOLS) | {"dig"} == set(G.RULES)


# --- the adapter half -------------------------------------------------------

class _Adapter:
    _slice_hz = 14_250_000.0

    def __init__(self, band_hz=M20):
        self.band_hz = band_hz

    def diversity_status(self, slice_id=None):
        return {"available": True, "mode": "track", "band_hz": self.band_hz,
                "retuned_at": 1_700_000_000.0, "band_changed_at": None,
                "nb": {"enabled": False, "threshold_db": 9.0},
                "noise_profile": {}, "squeeze": {}, "talker": None, "focus": None}

    def filter_status(self):
        return {"available": True, "low_hz": 300.0, "high_hz": 2700.0}


def test_the_snapshot_carries_the_band_from_diversity_status():
    """Mutation: forgetting to plumb band_hz through. The policy would see
    None for ever and no band change would ever reach it."""
    r = adapt.GovernorRunner(_Adapter(), clock=lambda: 5.0)
    assert r.snapshot()["band_hz"] == M20
    r.a.band_hz = M40
    assert r.snapshot()["band_hz"] == M40
