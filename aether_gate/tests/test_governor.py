#
# Aether-gate — B25 AUTO CLEAN, on hand-written snapshots (no hardware).
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""core/governor.py is pure -- no adapter, no clock, no thread -- so every rule
here is driven with a made-up snapshot and an explicit `t`, the same way B23's
guard is driven in test_frontend_guard.py.

The adapter half (adapters/diversity_governor.py) is covered against a fake
adapter that answers the five status calls with dicts: enough to prove the
snapshot's shapes, the writes' targets, and that the runner never needs its
thread to be tested. Nothing here opens a device, starts a thread or names a
talker, so no site log is written (conftest.py redirects it regardless).

Run:  .venv/bin/python -m pytest aether_gate/tests/test_governor.py -q
"""
from aether_gate.adapters import diversity_governor as adapt
from aether_gate.core import governor as G


def snap(t=0.0, **kw):
    """A quiet, healthy band: no rule fires on this one."""
    s = {
        "t": t, "available": True, "objective": 10.0,
        "mode": "track", "focus": None, "talking": False, "coherence": 0.1,
        "squeeze": {"held": False, "tool": None, "depth_db": None,
                    "target": "signal", "hz": None, "configured": False},
        "nb": {"on": False, "db": 9.5, "auto": "off"},
        "impulses_per_s": 0.0, "impulse_db": None,
        "mains_hz": None, "harmonics": 0, "carriers": [],
        "frontend_available": True, "guard": False, "headroom_db": 30.0,
        "dig_running": False, "dig_gain_db": None,
    }
    for k, v in kw.items():
        if isinstance(v, dict) and isinstance(s.get(k), dict):
            s[k] = dict(s[k], **v)
        else:
            s[k] = v
    return s


def gov(**kw):
    g = G.Governor(**kw)
    g.auto = True
    return g


def run(g, s):
    """One tick, applying whatever it proposed with a flat objective."""
    acts = g.tick(s)
    for a in acts:
        g.applied(a, s["t"], before=s["objective"])
    return acts


# --- the thresholds are quoted, not invented --------------------------------

def test_every_borrowed_threshold_still_matches_its_source():
    from aether_gate.adapters.diversity_state import _DiversityState
    from aether_gate.adapters.frontend_guard import HEADROOM_LOW_DB
    from aether_gate.core.squeeze import MIN_LEVEL_DB, NULL_ENTER_COHERENCE
    assert G.NULL_COHERENCE == NULL_ENTER_COHERENCE
    assert G.NULLABLE_COHERENCE == _DiversityState.NULLABLE_COHERENCE
    assert G.CARRIER_MIN_DB == MIN_LEVEL_DB
    assert G.HEADROOM_LOW_DB == HEADROOM_LOW_DB


def test_blank_threshold_matches_the_site_pages_own_recommendation():
    # adapters/noise_kinds.py: min(30, max(6, round((impulse_db - 3) * 2) / 2))
    for db in (0.0, 8.0, 12.7, 40.0, 99.0):
        want = min(30.0, max(6.0, round((db - 3.0) * 2) / 2))
        assert G.blank_threshold_db(db) == want


# --- default off ------------------------------------------------------------

def test_off_by_default_and_proposes_nothing():
    g = G.Governor()
    assert g.auto is False
    assert g.tick(snap(impulses_per_s=9.0, impulse_db=14.0)) == []
    assert g.status()["holding"] == []
    assert g.status()["state"] == "idle"


def test_turning_auto_off_releases_what_it_held_without_reverting():
    g = gov()
    run(g, snap(0.0, impulses_per_s=4.0, impulse_db=12.7))
    run(g, snap(3.0, nb={"on": True, "db": 9.5}))
    assert list(g.holding) == ["nb"]
    g.auto = False
    assert g.tick(snap(4.0, nb={"on": True, "db": 9.5})) == []      # no revert
    assert g.holding == {}
    assert g.status()["events"][-1]["result"] == "released"


# --- kind -> tool -----------------------------------------------------------

def test_impulses_turn_the_blanker_on_at_the_recommended_threshold():
    g = gov()
    acts = g.tick(snap(impulses_per_s=4.0, impulse_db=12.7))
    assert [a["tool"] for a in acts] == ["nb"]
    assert acts[0]["params"] == {"nb": True, "nb_db": 9.5}
    assert acts[0]["kind"] == "impulse"


def test_the_blanker_is_left_alone_while_the_arm_owns_it():
    g = gov()
    assert g.tick(snap(impulses_per_s=9.0, impulse_db=14.0,
                       nb={"auto": "auto"})) == []


def test_a_directional_floor_nulls_but_only_from_off_or_manual():
    assert gov().tick(snap(mode="manual", coherence=0.6))[0]["params"] == {"mode": "null"}
    assert gov().tick(snap(mode="track", coherence=0.6)) == []
    assert gov().tick(snap(mode="off", coherence=0.6, focus=7)) == []
    assert gov().tick(snap(mode="off", coherence=0.39)) == []       # not directional


def test_a_carrier_in_the_passband_is_squeezed_strongest_first():
    acts = gov().tick(snap(carriers=[{"hz": 1200.0, "db": 9.0},
                                     {"hz": -800.0, "db": 21.0},
                                     {"hz": 400.0, "db": 3.0}]))    # under MIN_LEVEL
    assert len(acts) == 1
    assert acts[0]["params"] == {"squeeze": -800}
    assert acts[0]["kind"] == "carrier"


def test_coherence_says_which_tool_the_squeeze_will_reach_for():
    hot = [{"hz": 1200.0, "db": 20.0}]
    assert "a null" in gov().tick(snap(carriers=hot, coherence=0.5))[0]["why"]
    assert "a notch" in gov().tick(snap(carriers=hot, coherence=0.49))[0]["why"]


def test_coherent_hum_squeezes_the_comb_and_incoherent_hum_is_left():
    acts = gov().tick(snap(mains_hz=60.0, harmonics=2, coherence=0.6))
    assert acts[0]["params"] == {"squeeze": "comb"}
    assert acts[0]["kind"] == "mains"
    assert gov().tick(snap(mains_hz=60.0, harmonics=2, coherence=0.2)) == []


def test_no_second_squeeze_on_a_tone_a_deep_null_already_took_out():
    held = {"held": True, "tool": "null", "depth_db": 12.0, "configured": True,
            "target": "signal", "hz": 1200.0}
    g = gov()
    g.holding["squeeze"] = {"tool": "squeeze", "params": {"squeeze": 1200},
                            "kind": "carrier", "why": "", "since": 0.0,
                            "delta_db": 1.0}
    assert g.tick(snap(mains_hz=60.0, harmonics=2, coherence=0.6, squeeze=held)) == []
    # ...but a shallow one has not taken it out, and the comb is proposed
    shallow = dict(held, depth_db=1.0)
    assert g.tick(snap(1.0, mains_hz=60.0, harmonics=2, coherence=0.6,
                       squeeze=shallow))[0]["params"] == {"squeeze": "comb"}


def test_low_headroom_hands_it_to_the_front_end_guard():
    acts = gov().tick(snap(headroom_db=1.5))
    assert acts[0]["tool"] == "guard" and acts[0]["params"] == {"guard": True}
    assert gov().tick(snap(headroom_db=1.5, guard=True)) == []      # already on
    assert gov().tick(snap(headroom_db=1.5, frontend_available=False)) == []


def test_the_operators_own_squeeze_target_is_never_taken_over():
    theirs = {"configured": True, "target": "signal", "hz": 300.0}
    assert gov().tick(snap(carriers=[{"hz": 1200.0, "db": 20.0}],
                           squeeze=theirs)) == []


# --- the blanker's hysteresis -----------------------------------------------

def test_the_blanker_goes_off_only_after_thirty_quiet_seconds_and_only_if_ours():
    g = gov()
    run(g, snap(0.0, impulses_per_s=4.0, impulse_db=12.7))
    run(g, snap(3.0, nb={"on": True}))                   # settled, kept, held
    assert list(g.holding) == ["nb"]
    assert g.tick(snap(4.0, nb={"on": True})) == []      # quiet: clock starts
    assert g.tick(snap(30.0, nb={"on": True})) == []     # 26 s is not 30
    acts = g.tick(snap(34.1, nb={"on": True}))
    assert acts[0]["params"] == {"nb": False}
    # a blanker the governor never turned on is not turned off for the operator
    g2 = gov()
    g2.tick(snap(0.0, nb={"on": True}))
    assert g2.tick(snap(100.0, nb={"on": True})) == []


# --- undo, backoff, and one at a time ---------------------------------------

def _objective_ring(g, t0=0.0, value=10.0, n=G.SPREAD_N):
    for i in range(n):
        g.tick(snap(t0 + i * 0.01, objective=value))


def test_a_move_that_cost_more_than_the_margin_is_put_back_and_backs_off():
    g = gov()
    run(g, snap(0.0, impulses_per_s=4.0, impulse_db=12.7))
    assert g.state == "settling"
    assert g.tick(snap(1.0, objective=10.0, nb={"on": True})) == []   # still settling
    acts = g.tick(snap(3.0, objective=4.0, nb={"on": True}))          # -6 dB
    assert len(acts) == 1 and acts[0]["revert"] is True
    assert acts[0]["params"] == {"nb": False, "nb_db": 9.5}
    assert g.holding == {}
    assert g.events[-1]["result"] == "undone"
    assert g.events[-1]["delta_db"] == -6.0
    # and it is not asked for again inside the backoff
    g.applied(acts[0], 3.0)
    assert g.tick(snap(200.0, impulses_per_s=4.0, impulse_db=12.7)) == []
    assert g.tick(snap(400.0, impulses_per_s=4.0, impulse_db=12.7))[0]["tool"] == "nb"


def test_a_move_inside_the_margin_is_kept():
    g = gov()
    run(g, snap(0.0, impulses_per_s=4.0, impulse_db=12.7))
    assert g.tick(snap(3.0, objective=9.8, nb={"on": True})) == []
    assert list(g.holding) == ["nb"]
    assert g.events[-1]["result"] == "kept"
    assert g.status()["holding"][0]["tool"] == "nb"


def test_only_one_action_is_ever_in_flight():
    g = gov()
    # impulses AND a carrier AND low headroom, all at once
    s = snap(0.0, impulses_per_s=4.0, impulse_db=12.7, headroom_db=1.0,
             carriers=[{"hz": 1200.0, "db": 20.0}])
    acts = run(g, s)
    assert len(acts) == 1
    assert acts[0]["tool"] == "guard"                    # the chain's order: front end
    assert g.tick(dict(s, t=0.5)) == []                  # settling: nothing else
    assert g.tick(dict(s, t=1.0)) == []


def test_the_chain_order_is_guard_then_nb_then_mode_then_squeeze_then_dig():
    assert G.RULES == ("guard", "nb", "mode", "squeeze", "dig")


# --- the operator always wins ------------------------------------------------

def test_a_knob_the_operator_moved_is_not_touched_for_a_minute():
    g = gov()
    g.tick(snap(0.0))                                    # seed the observed state
    g.tick(snap(1.0, nb={"on": True}))                   # operator turned it on
    g.tick(snap(2.0, nb={"on": False}))                  # ...and off again
    assert g._operator_at["nb"] == 2.0
    assert g.tick(snap(3.0, impulses_per_s=4.0, impulse_db=12.7)) == []
    assert g.tick(snap(61.0, impulses_per_s=4.0, impulse_db=12.7)) == []
    assert g.tick(snap(63.0, impulses_per_s=4.0, impulse_db=12.7))[0]["tool"] == "nb"


def test_an_operator_write_releases_a_tool_the_governor_was_holding():
    g = gov()
    run(g, snap(0.0, impulses_per_s=4.0, impulse_db=12.7))
    g.tick(snap(3.0, nb={"on": True}))                   # kept and held
    assert list(g.holding) == ["nb"]
    g.tick(snap(10.0, nb={"on": True, "db": 20.0}))      # operator moved nb_db
    assert g.holding == {}
    assert g.events[-1]["result"] == "released"
    assert "operator" in g.events[-1]["why"]


def test_our_own_write_landing_is_not_read_as_the_operator():
    g = gov()
    run(g, snap(0.0, impulses_per_s=4.0, impulse_db=12.7))
    g.tick(snap(0.5, impulses_per_s=4.0, nb={"on": True}))   # the write shows up
    assert "nb" not in g._operator_at


# --- the dig hand-off --------------------------------------------------------

def test_a_weak_talker_on_a_steady_band_is_handed_to_the_dig():
    g = gov()
    _objective_ring(g, 0.0, 3.0)
    acts = g.tick(snap(1.0, objective=3.0, talking=True))
    assert acts[0]["tool"] == "dig" and acts[0]["params"] == {"seconds": 60}
    assert acts[0]["undo"] is None


def test_the_dig_is_not_started_on_a_jumpy_band_or_while_one_is_running():
    g = gov()
    for i, j in enumerate((3.0, 9.0, 3.0, 9.0, 3.0, 9.0)):
        g.tick(snap(i * 0.01, objective=j, talking=True))
    assert g.tick(snap(1.0, objective=3.0, talking=True)) == []
    g2 = gov()
    _objective_ring(g2, 0.0, 3.0)
    assert g2.tick(snap(1.0, objective=3.0, talking=True, dig_running=True)) == []


def test_the_dig_is_scored_but_never_reverted():
    g = gov()
    _objective_ring(g, 0.0, 3.0)
    run(g, snap(1.0, objective=3.0, talking=True))
    assert g.tick(snap(2.0, objective=3.0, talking=True, dig_running=True)) == []
    assert g.tick(snap(70.0, objective=0.5, talking=True,
                       dig_gain_db=2.4)) == []          # objective fell: still kept
    assert g.events[-1]["result"] == "kept"
    assert g.events[-1]["delta_db"] == 2.4


# --- housekeeping ------------------------------------------------------------

def test_the_events_list_is_capped_at_fifty():
    g = gov()
    for i in range(80):
        g.events.append({"t": float(i), "tool": "nb", "kind": "impulse",
                         "params": {}, "undo": None, "why": "", "before": None,
                         "result": "kept", "delta_db": 0.0})
    st = g.status()
    assert len(st["events"]) == G.MAX_EVENTS == 50
    assert st["events"][0]["t"] == 30.0


def test_nothing_is_proposed_before_there_is_an_objective_to_score_against():
    g = gov()
    assert g.tick(snap(objective=None, impulses_per_s=9.0, impulse_db=14.0)) == []
    assert g.status()["state"] == "measuring"
    assert g.tick({"t": 0.0, "available": False}) == []


def test_a_write_that_threw_backs_the_pair_off_and_says_so():
    g = gov()
    acts = g.tick(snap(impulses_per_s=4.0, impulse_db=12.7))
    g.failed(acts[0], "RuntimeError: no stream", 0.0)
    assert g.pending is None
    assert g.status()["events"][-1]["result"] == "error"
    assert g.status()["backoff"][0]["tool"] == "nb"


def test_the_status_shape_the_app_reads():
    g = gov()
    st = g.status()
    assert set(st) == {"auto", "state", "why", "settle_s", "margin_db",
                       "spread_db", "holding", "pending", "events", "backoff"}
    assert st["auto"] is True and st["pending"] is None
    assert isinstance(st["why"], str) and st["why"]


# --- the adapter half --------------------------------------------------------

class FakeAdapter:
    """The five status calls and the three writes, and nothing else."""

    def __init__(self, **over):
        self._slice_hz = 7185000.0
        self.frontend = type("FE", (), {"enabled": False})()
        self.writes = []
        self.div = {
            "available": True, "mode": "track", "focus": None, "talking": True,
            "talk_mod": 1.0, "noise_coherence": 0.62,
            "snr_db": {"a": 8.1, "b": 1.2, "out": 8.6},
            "passband": {"flatness": 0.97},
            "nb": {"enabled": False, "threshold_db": 9.5, "blanked_pct": 0.2,
                   "auto": {"mode": "off"}},
            "squeeze": {"held": False, "tool": None, "depth_db": None,
                        "target": "comb", "hz": None, "since": None},
            "noise_profile": {"impulses_per_s": 4.0, "impulse_db": 12.7,
                              "mains_hz": 60.0, "harmonics": 2},
        }
        self.div.update(over)

    def diversity_status(self, slice_id=None):
        return self.div

    def filter_status(self):
        return {"low_hz": 100, "high_hz": 2900, "_sign": -1.0}

    def diversity_finder(self):
        return {"candidates": [
            {"hz": 7184000.0, "kind": "carrier", "snr_db": 18.0, "score": 0.9,
             "width_hz": 300.0},                       # -1000 Hz: in the LSB passband
            {"hz": 7190000.0, "kind": "carrier", "snr_db": 30.0, "score": 0.9,
             "width_hz": 300.0},                       # +5000 Hz: outside it
        ]}

    def frontend_status(self):
        return {"available": True, "guard": False, "headroom_db": 28.8}

    def diversity_dig(self, **kw):
        if kw:
            self.writes.append(("dig", kw))
        return {"running": False, "gain_db": 0.0}

    def set_diversity(self, **kw):
        self.writes.append(("diversity", kw))


def test_the_snapshot_reads_the_gates_own_shapes():
    r = adapt.GovernorRunner(FakeAdapter(), clock=lambda: 0.0)
    s = r.snapshot()
    assert s["available"] and s["objective"] is not None
    assert s["coherence"] == 0.62 and s["mode"] == "track"
    assert s["nb"] == {"on": False, "db": 9.5, "auto": "off"}
    assert s["impulses_per_s"] == 4.0 and s["mains_hz"] == 60.0
    assert s["squeeze"]["configured"] is False
    assert s["carriers"] == [{"hz": -1000.0, "db": 18.0}]     # the out-of-band one is gone
    assert s["frontend_available"] and s["headroom_db"] == 28.8


def test_a_configured_comb_target_reads_as_the_operators():
    a = FakeAdapter()
    a.div["squeeze"] = dict(a.div["squeeze"], since=123.0)
    assert adapt.GovernorRunner(a).snapshot()["squeeze"]["configured"] is True


def test_the_runner_writes_through_the_adapters_public_setters():
    a = FakeAdapter()
    r = adapt.GovernorRunner(a, clock=lambda: 0.0)
    r.gov.auto = True
    r.tick(0.0)
    # the chain's order: the impulses this fake is showing come before the
    # carrier it is also showing, and only one move goes out per tick
    assert a.writes == [("diversity", {"nb": True, "nb_db": 9.5})]
    r._write("squeeze", {"squeeze": -1000})
    r._write("mode", {"mode": "null"})
    r._write("guard", {"guard": True})
    r._write("dig", {"seconds": 60})
    assert a.writes[1:] == [("diversity", {"squeeze": -1000}),
                            ("diversity", {"mode": "null"}),
                            ("dig", {"seconds": 60})]
    assert a.frontend.enabled is True


def test_the_runner_is_inert_while_auto_is_off():
    a = FakeAdapter()
    r = adapt.GovernorRunner(a, clock=lambda: 0.0)
    r.tick(0.0)
    assert a.writes == []
    st = r.status()
    assert st["auto"] is False and st["running"] is False and st["available"] is True


def test_a_setter_that_throws_is_reported_not_raised():
    class Angry(FakeAdapter):
        def set_diversity(self, **kw):
            raise RuntimeError("no stream")

    r = adapt.GovernorRunner(Angry(), clock=lambda: 0.0)
    r.gov.auto = True
    st = r.tick(0.0)
    assert st["events"][-1]["result"] == "error"
    assert "no stream" in st["events"][-1]["why"]


def test_a_status_call_that_throws_leaves_the_governor_measuring():
    class Broken(FakeAdapter):
        def diversity_status(self, slice_id=None):
            raise RuntimeError("gone")

    r = adapt.GovernorRunner(Broken(), clock=lambda: 0.0)
    r.gov.auto = True
    assert r.tick(0.0)["state"] == "measuring"
