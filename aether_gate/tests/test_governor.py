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
import time

from aether_gate.adapters import diversity_governor as adapt
from aether_gate.core import governor as G
from aether_gate.core import governor_proxy as P

WALL0 = 1_700_000_000.0         # R9: the wall clock the fake `t` starts from


def snap(t=0.0, **kw):
    """A quiet, healthy band: no rule fires on this one."""
    s = {
        "t": t, "wall": WALL0 + t, "available": True, "objective": 10.0,
        "mode": "track", "focus": None, "talking": False, "coherence": 0.1,
        "squeeze": {"held": False, "tool": None, "depth_db": None,
                    "target": "signal", "hz": None, "configured": False},
        "nb": {"on": False, "db": 9.5, "auto": "off"},
        "impulses_per_s": 0.0, "impulse_db": None,
        "mains_hz": None, "harmonics": 0, "hum_db": None, "carriers": [],
        "frontend_available": True, "guard": False, "headroom_db": 30.0,
        "clips_1s": 0, "blanked_pct": 0.0, "floor_db": -100.0,
        "slice_hz": 7185000.0, "talker": None,
        "dig_running": False, "dig_gain_db": None, "dig_note": None,
        "dig_unsteady": False, "dig_verdict": None, "dig_cancelled": False,
        "dig_age_s": None,
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


def test_hum_gets_the_comb_whether_or_not_the_loops_agree_on_it():
    """R6: a comb notch is spectral: no direction needed. Live, a 60 Hz comb
    11.8 dB over the floor at coherence 0.02 got nothing at all."""
    for coh in (0.6, 0.02):
        acts = gov().tick(snap(mains_hz=60.0, harmonics=2, hum_db=11.8,
                               coherence=coh))
        assert acts[0]["params"] == {"squeeze": "comb"} and acts[0]["kind"] == "mains"


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


def test_no_objective_no_longer_means_no_move_but_still_means_no_stream():
    g = gov()
    acts = g.tick(snap(objective=None, impulses_per_s=9.0, impulse_db=14.0))
    assert acts[0]["tool"] == "nb" and acts[0]["scorer"] == "proxy:blanking"
    assert g.status()["objective_source"] == "none"
    assert gov().tick({"t": 0.0, "available": False}) == []


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
    assert set(st) == {"auto", "state", "why", "state_label", "settle_s", "margin_db",
                       "spread_db", "objective_source", "holding", "pending",
                       "events", "backoff", "ruled_out"}
    assert st["objective_source"] == "snr"
    assert st["auto"] is True and st["pending"] is None
    assert isinstance(st["why"], str) and st["why"]


# --- no talker: the proxy objectives -----------------------------------------

def test_the_proxy_thresholds_are_borrowed_from_the_same_places_too():
    from aether_gate.core.digout import BLANK_FREE_PCT
    from aether_gate.core.squeeze import MIN_LEVEL_DB
    assert P.NULL_DEPTH_KEEP_DB == MIN_LEVEL_DB
    assert P.BLANKED_MAX_PCT == BLANK_FREE_PCT
    assert P.NOTCH_DEPTH_KEEP_DB > P.NULL_DEPTH_KEEP_DB
    assert P.HEADROOM_LOW_DB == G.HEADROOM_LOW_DB


def _squeeze_move(g, depth_db, tool="null", floor_after=-100.2):
    """Propose a squeeze with nobody talking, then settle it at `depth_db`."""
    run(g, snap(0.0, objective=None, coherence=0.6,
                carriers=[{"hz": 1200.0, "db": 20.0}]))
    held = {"held": True, "tool": tool, "depth_db": depth_db, "configured": True,
            "hz": 1200.0}
    return g.tick(snap(3.0, objective=None, coherence=0.6, squeeze=held,
                       floor_db=floor_after))


def test_a_squeeze_with_no_talker_is_kept_on_its_own_measured_depth():
    g = gov()
    assert _squeeze_move(g, 14.2) == []
    assert list(g.holding) == ["squeeze"]
    e = g.events[-1]
    assert e["result"] == "kept" and e["scorer"] == "proxy:depth"
    assert "14.2 dB of null depth" in e["why"] and "no talker to score by" in e["why"]
    assert g.status()["holding"][0]["scorer"] == "proxy:depth"


def test_a_squeeze_is_given_time_to_take_hold_and_scored_the_moment_it_does():
    # on air the comb squeeze was scored at 2 s, before its detector had even
    # read the band, and put back as "no null held on the mains"
    assert G.SQUEEZE_SETTLE_S == 8.0
    g = gov()
    run(g, snap(0.0, objective=None, coherence=0.6,
                carriers=[{"hz": 1200.0, "db": 20.0}]))
    not_yet = {"held": False, "tool": None, "depth_db": None, "configured": True,
               "hz": 1200.0, "reason": None}
    assert g.tick(snap(3.0, objective=None, coherence=0.6, squeeze=not_yet)) == []
    assert g.state == "settling" and "take hold" in g.why
    held = {"held": True, "tool": "null", "depth_db": 12.0, "configured": True,
            "hz": 1200.0, "reason": None}
    assert g.tick(snap(5.0, objective=None, coherence=0.6, squeeze=held)) == []
    assert g.events[-1]["result"] == "kept" and "12.0 dB of null depth" in g.events[-1]["why"]


def test_a_squeeze_that_never_takes_hold_is_put_back_with_its_own_reason():
    g = gov()
    run(g, snap(0.0, objective=None, coherence=0.6,
                carriers=[{"hz": 1200.0, "db": 20.0}]))
    refused = {"held": False, "tool": None, "depth_db": None, "configured": True,
               "hz": 1200.0, "reason": "too weak"}
    assert g.tick(snap(7.9, objective=None, coherence=0.6, squeeze=refused)) == []
    acts = g.tick(snap(8.1, objective=None, coherence=0.6, squeeze=refused))
    assert acts[0]["revert"] is True
    assert g.events[-1]["result"] == "undone"
    assert "too weak on the carrier" in g.events[-1]["why"]


def test_a_shallow_null_is_put_back_and_a_notch_is_asked_for_more():
    g = gov()
    acts = _squeeze_move(g, 3.1)                     # under NULL_DEPTH_KEEP_DB
    assert acts[0]["revert"] is True and acts[0]["params"] == {"squeeze": ""}
    assert g.holding == {} and g.events[-1]["result"] == "undone"
    assert "3.1 dB of null depth" in g.events[-1]["why"]
    # 8 dB holds a null but not a notch: a designed response is asked for more
    assert _squeeze_move(gov(), 8.0, tool="null") == []
    assert _squeeze_move(gov(), 8.0, tool="notch")[0]["revert"] is True
    assert _squeeze_move(gov(), 12.0, tool="notch") == []


def test_a_deep_null_that_lifted_the_passband_floor_still_goes_back():
    g = gov()
    acts = _squeeze_move(g, 14.2, floor_after=-98.5)   # the floor rose 1.5 dB
    assert acts[0]["revert"] is True
    assert "floor rose 1.5 dB" in g.events[-1]["why"]
    assert _squeeze_move(gov(), 14.2, floor_after=-99.2) == []      # 0.8 dB is free


def test_the_blanker_with_no_talker_is_scored_on_what_it_eats():
    g = gov()
    run(g, snap(0.0, objective=None, impulses_per_s=4.0, impulse_db=12.7))
    assert g.pending["scorer"] == "proxy:blanking"
    assert g.tick(snap(3.0, objective=None, nb={"on": True}, blanked_pct=3.0,
                       impulses_per_s=0.4, floor_db=-100.1)) == []
    assert g.events[-1]["result"] == "kept"
    assert "impulses -3.6/s" in g.events[-1]["why"]
    # over BLANKED_MAX_PCT it is gating the signal, not the noise
    g2 = gov()
    run(g2, snap(0.0, objective=None, impulses_per_s=4.0, impulse_db=12.7))
    acts = g2.tick(snap(3.0, objective=None, nb={"on": True}, blanked_pct=9.0))
    assert acts[0]["revert"] is True and acts[0]["params"] == {"nb": False, "nb_db": 9.5}


def test_an_idle_null_with_no_talker_has_to_take_a_decibel_off_the_floor():
    g = gov()
    run(g, snap(0.0, objective=None, mode="manual", coherence=0.6))
    assert g.pending["scorer"] == "proxy:floor"
    assert g.tick(snap(3.0, objective=None, mode="null", coherence=0.6,
                       floor_db=-101.6)) == []
    assert g.events[-1]["result"] == "kept"
    assert "1.6 dB off the passband floor" in g.events[-1]["why"]
    g2 = gov()
    run(g2, snap(0.0, objective=None, mode="manual", coherence=0.6))
    acts = g2.tick(snap(3.0, objective=None, mode="null", coherence=0.6,
                        floor_db=-100.4))            # only 0.4 dB: not worth a mode
    assert acts[0]["revert"] is True and acts[0]["params"] == {"mode": "manual"}


def _guard_move(g, clips_after, hr_after, clips_before=14, hr_before=1.5):
    """Propose the guard with nobody talking, then settle it. The guard steps
    the LNA on its own loop, so it gets GUARD_SETTLE_S and not SETTLE_S."""
    run(g, snap(0.0, objective=None, headroom_db=hr_before, clips_1s=clips_before))
    assert g.pending["scorer"] == "proxy:clips"
    assert g.tick(snap(9.9, objective=None, guard=True, headroom_db=hr_after,
                       clips_1s=clips_after)) == []          # still settling at 9.9 s
    assert g.state == "settling"
    return g.tick(snap(10.1, objective=None, guard=True, headroom_db=hr_after,
                       clips_1s=clips_after))


def test_the_guard_gets_ten_seconds_because_it_steps_the_lna_itself():
    assert G.GUARD_SETTLE_S == 10.0
    g = gov()
    assert _guard_move(g, 0, 8.0) == []
    assert g.events[-1]["result"] == "kept"
    assert "clips 14/s -> 0" in g.events[-1]["why"]


def test_the_guard_is_kept_when_either_of_its_numbers_moved_the_right_way():
    # the clips fell but the headroom did not come back
    g = gov()
    assert _guard_move(g, 3, 1.5) == []
    assert "clips 14/s -> 3" in g.events[-1]["why"]
    # the clips held but the headroom did
    g2 = gov()
    assert _guard_move(g2, 14, 5.7) == []
    assert "headroom +4.2 dB" in g2.events[-1]["why"]
    # no clipping at all after the step, whatever the headroom says
    g3 = gov()
    assert _guard_move(g3, 0, 1.4) == []
    assert g3.events[-1]["result"] == "kept"


def test_the_guard_goes_back_only_when_neither_number_moved():
    g = gov()
    acts = _guard_move(g, 18, 1.4)          # more clips, less headroom
    assert acts[0]["revert"] is True and acts[0]["params"] == {"guard": False}
    assert "clips 14/s -> 18" in g.events[-1]["why"]
    assert "headroom -0.1 dB" in g.events[-1]["why"]


def test_a_talker_turning_up_switches_the_scorer_back_and_re_litigates_nothing():
    g = gov()
    run(g, snap(0.0, objective=None, headroom_db=1.5, clips_1s=40))
    g.tick(snap(10.1, objective=None, guard=True, headroom_db=8.0, clips_1s=0))
    assert g.status()["objective_source"] == "none"
    assert g.status()["holding"][0]["scorer"] == "proxy:clips"
    acts = run(g, snap(20.0, objective=2.0, guard=True, headroom_db=8.0,
                       impulses_per_s=4.0, impulse_db=12.7))
    assert g.status()["objective_source"] == "snr"
    assert acts[0]["tool"] == "nb" and acts[0]["scorer"] == "snr"
    # the guard kept under a proxy is still held, still says what kept it, and
    # was never re-scored against the objective that has just turned up
    held = [h for h in g.status()["holding"] if h["tool"] == "guard"][0]
    assert held["scorer"] == "proxy:clips" and held["delta_db"] is None


def test_every_event_and_holding_row_names_the_scorer_that_judged_it():
    g = gov()
    run(g, snap(0.0, impulses_per_s=4.0, impulse_db=12.7))       # objective present
    g.tick(snap(3.0, objective=9.8, nb={"on": True}))
    assert [e["scorer"] for e in g.status()["events"]] == ["snr"]
    assert g.status()["holding"][0]["scorer"] == "snr"


# --- the dig: once per talker, and only on what the dig stands behind --------

def test_the_dig_is_never_started_without_a_talker_to_score_it():
    g = gov()
    for i in range(G.SPREAD_N):
        g.tick(snap(i * 0.01, objective=None))
    assert g.tick(snap(1.0, objective=None, talking=False)) == []
    assert g.tick(snap(2.0, objective=None, talking=True)) == []   # no objective
    assert g.status()["objective_source"] == "none"


def test_a_dig_hand_off_happens_once_for_a_frequency_and_talker():
    g = gov()
    _objective_ring(g, 0.0, 3.0)
    acts = run(g, snap(1.0, objective=3.0, talking=True))
    assert acts[0]["tool"] == "dig" and acts[0]["key"] == (7185000, None)
    g.tick(snap(70.0, objective=3.0, talking=True, dig_gain_db=1.2))
    assert g.holding["dig"]["delta_db"] == 1.2
    _objective_ring(g, 3000.0, 3.0)                  # the backoff is long gone
    assert g.tick(snap(3600.0, objective=3.0, talking=True)) == []
    # a different talker, or a different frequency, is a different question
    assert g.tick(snap(3601.0, objective=3.0, talking=True,
                       talker="G0ABC"))[0]["tool"] == "dig"
    assert g.tick(snap(3602.0, objective=3.0, talking=True,
                       slice_hz=14074000.0))[0]["tool"] == "dig"


def test_the_dig_backs_off_for_half_an_hour_not_ten_minutes():
    assert G.DIG_BACKOFF_S == 1800.0
    g = gov()
    _objective_ring(g, 0.0, 3.0)
    run(g, snap(1.0, objective=3.0, talking=True))
    assert g.backoff[("weak", "dig")] == 1.0 + G.DIG_BACKOFF_S


def test_a_tentative_dig_scores_nothing_and_backs_the_pair_off_again():
    tentative = "the band swung 5.4 dB while sampling; results are tentative"
    g = gov()
    _objective_ring(g, 0.0, 3.0)
    run(g, snap(1.0, objective=3.0, talking=True))
    g.tick(snap(70.0, objective=3.0, talking=True, dig_gain_db=28.61,
                dig_unsteady=True, dig_note=tentative))
    e = g.events[-1]
    assert e["result"] == "kept" and e["delta_db"] == 0.0    # 28.61 was the artefact
    assert "tentative" in e["why"] and "not +28.6 dB" in e["why"]
    assert g.backoff[("weak", "dig")] == 70.0 + G.DIG_BACKOFF_S
    # ...and while the dig's own last word is that, nothing else is handed over
    _objective_ring(g, 4000.0, 3.0)
    assert g.tick(snap(4200.0, objective=3.0, talking=True, talker="G0ABC",
                       dig_note=tentative, dig_age_s=60.0)) == []
    assert g.tick(snap(4201.0, objective=3.0, talking=True, talker="G0ABC",
                       dig_unsteady=True, dig_age_s=60.0)) == []


def test_the_tentative_block_runs_out_on_the_digs_own_clock():
    """The gate keeps the last run's note for ever, so an hour-old "tentative"
    must not be a permanent lockout -- it is DIG_BACKOFF_S from the dig's own
    end, which the adapter hands over as an AGE because the two clocks differ."""
    tentative = "the band swung 5.4 dB while sampling; results are tentative"
    inside = dict(objective=3.0, talking=True, talker="G0ABC",
                  dig_note=tentative, dig_unsteady=True)
    g = gov()
    _objective_ring(g, 0.0, 3.0)
    # one second short of the window: still blocked
    assert g.tick(snap(1.0, dig_age_s=G.DIG_BACKOFF_S - 1.0, **inside)) == []
    # ...and one second past it, the same note no longer blocks anything
    assert g.tick(snap(2.0, dig_age_s=G.DIG_BACKOFF_S + 1.0,
                       **inside))[0]["tool"] == "dig"


def test_an_untimestamped_note_is_aged_from_when_the_governor_first_saw_it():
    """No `ends` in the dig status (it was cancelled before it ever ran, say):
    the clock starts at the tick that first saw this note, not at zero."""
    inside = dict(objective=3.0, talking=True, talker="G0ABC",
                  dig_note="results are tentative", dig_age_s=None)
    g = gov()
    _objective_ring(g, 0.0, 3.0)
    assert g.tick(snap(100.0, **inside)) == []              # first sight: t = 100
    assert g._note_at[1] == 100.0
    assert g.tick(snap(100.0 + G.DIG_BACKOFF_S - 1.0, **inside)) == []
    assert g.tick(snap(100.0 + G.DIG_BACKOFF_S + 1.0, **inside))[0]["tool"] == "dig"
    # a NEW note restarts the window rather than inheriting the old one's age
    g2 = gov()
    _objective_ring(g2, 0.0, 3.0)
    g2.tick(snap(100.0, **inside))
    assert g2.tick(snap(5000.0, **dict(inside, dig_note="also tentative"))) == []


def test_a_dig_that_found_nothing_counts_as_tried_and_says_so():
    g = gov()
    _objective_ring(g, 0.0, 3.0)
    run(g, snap(1.0, objective=3.0, talking=True))
    g.tick(snap(70.0, objective=3.0, talking=True, dig_gain_db=0.0))
    e = g.events[-1]
    assert e["delta_db"] == 0.0 and "found nothing here" in e["why"]
    _objective_ring(g, 4000.0, 3.0)
    assert g.tick(snap(4200.0, objective=3.0, talking=True)) == []


def test_a_cancelled_or_worse_dig_banks_nothing_either():
    for over, want in ((dict(dig_cancelled=True), "cancelled"),
                       (dict(dig_verdict="worse"), "worse")):
        g = gov()
        _objective_ring(g, 0.0, 3.0)
        run(g, snap(1.0, objective=3.0, talking=True))
        g.tick(snap(70.0, objective=3.0, talking=True, dig_gain_db=4.0, **over))
        assert g.events[-1]["delta_db"] == 0.0
        assert want in g.events[-1]["why"]


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
        return {"available": True, "guard": False, "headroom_db": 28.8,
                "clips_1s": 0}

    def diversity_spatial(self):
        return {"available": True, "start_hz": 7184000.0, "step_hz": 500.0,
                "passband_hz": [7185000.0, 7186000.0],
                # bins 2..4 are the passband; the loud one outside it must not
                # move the median, and neither must the carrier inside it
                "level_db": [-40.0, -41.0, -99.0, -98.0, -97.0, -20.0]}

    def diversity_dig(self, **kw):
        if kw:
            self.writes.append(("dig", kw))
        return {"running": False, "gain_db": 0.0,
                "ends": getattr(self, "dig_ends", None)}

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


def test_the_snapshot_carries_what_the_proxies_score_by():
    s = adapt.GovernorRunner(FakeAdapter(), clock=lambda: 0.0).snapshot()
    assert s["floor_db"] == -98.0          # median of the three passband bins
    assert s["blanked_pct"] == 0.2 and s["clips_1s"] == 0
    assert s["slice_hz"] == 7185000.0 and s["talker"] is None
    assert s["dig_unsteady"] is False and s["dig_note"] is None
    assert s["dig_verdict"] is None and s["dig_cancelled"] is False
    assert s["dig_age_s"] is None            # this fake has never run one


def test_the_snapshot_turns_the_digs_wall_clock_stamp_into_an_age():
    import time as _t
    a = FakeAdapter()
    a.dig_ends = _t.time() - 90.0
    s = adapt.GovernorRunner(a, clock=lambda: 0.0).snapshot()
    assert 89.0 < s["dig_age_s"] < 95.0       # a duration, not either clock


def test_the_floor_is_the_median_of_the_passband_bins_and_nothing_else():
    strip = {"start_hz": 0.0, "step_hz": 100.0,
             "level_db": [-10.0, -90.0, -91.0, -92.0, -10.0]}
    assert P.inband_floor_db(dict(strip, passband_hz=[100.0, 300.0])) == -91.0
    assert P.inband_floor_db(strip) == -90.0            # no passband: the whole strip
    assert P.inband_floor_db({}) is None
    assert P.inband_floor_db({"level_db": [], "start_hz": 0.0, "step_hz": 1.0}) is None


def test_auto_off_stops_a_dig_the_governor_itself_started():
    a = FakeAdapter()
    r = adapt.GovernorRunner(a, clock=lambda: 0.0)
    r.gov.auto = True
    r.gov.pending = {"tool": "dig", "kind": "weak", "t": 0.0, "why": "",
                     "params": {"seconds": 60}, "undo": None, "before": None,
                     "result": "pending", "delta_db": None, "scorer": "snr"}
    r.stop()
    assert a.writes == [("dig", {"cancel": True})]
    # a dig the OPERATOR started is not ours to stop
    a2 = FakeAdapter()
    r2 = adapt.GovernorRunner(a2, clock=lambda: 0.0)
    r2.gov.auto = True
    r2.stop()
    assert a2.writes == []


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


def test_stop_does_not_block_the_http_thread_on_a_slow_tick():
    """stop() runs on the HTTP thread for GET /diversity/set?auto=off. The
    tick thread's join used to wait up to 5 s for it, so
    that one GET could stall for 5 s if a tick was slow or wedged inside an
    adapter call. Mutation: reverting the join's timeout in stop() back to
    something >= ~1 s."""
    class SlowAdapter(FakeAdapter):
        def diversity_status(self, slice_id=None):
            time.sleep(3.0)
            return self.div

    a = SlowAdapter()
    r = adapt.GovernorRunner(a)
    r.set_auto(True)
    time.sleep(1.3)                  # into the first tick's 3 s status() sleep
    t0 = time.monotonic()
    r.stop()
    assert time.monotonic() - t0 < 1.0
    assert r.status()["running"] is False       # reported not-running immediately
    time.sleep(2.5)                  # let the blocked tick actually finish
    assert r.status()["running"] is False        # still not running once it does
