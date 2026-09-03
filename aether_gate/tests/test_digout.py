#
# Aether-gate — the "dig this out" search, against a made-up landscape.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""No radio, no thread, no clock: the search hands out ops and we hand back
numbers off a function of the settings, so "does coordinate descent actually
find the good corner, and does it put back what it broke" is a unit test.

Run:  python -m pytest aether_gate/tests/test_digout.py
"""
import pytest

from aether_gate.core import digout

SNAP = {"post": True, "subband": True, "mrc": False, "nb": True, "nb_db": 11.0,
        "width": (100.0, 2900.0), "contour": False, "anf": False,
        "auto_eq": False, "apf": False, "agc": "med"}


def drive(search, landscape, snapshot=None, t0=0.0):
    """Play the runner: set what it says, measure the landscape, feed it back.

    Returns (trace, settings-as-left, clock) where trace is every op seen.
    """
    snapshot = SNAP if snapshot is None else snapshot
    cur, trace, t = dict(snapshot), [], t0
    search.begin(snapshot, t)
    cur = dict(search.snapshot)
    for _ in range(10000):
        op = search.next_op(t)
        if op["op"] == "done":
            trace.append(("done",))
            break
        if op["op"] == "set":
            cur[op["knob"]] = op["to"]
            trace.append(("set", op["knob"], op["to"], op["revert"]))
            continue
        t += op["settle_s"]
        v = landscape(cur)
        trace.append(("measure", op.get("why"), v))
        search.feed(v, t)
    else:                                     # pragma: no cover - a runaway plan
        pytest.fail("the search never said done")
    return trace, cur, t


def flat(_cur):
    return 10.0


# ---- the objective ---------------------------------------------------------

def test_the_objective_is_snr_plus_voice_plus_flatness_minus_blanking():
    div = {"snr_db": {"out": 12.0}, "talking": True, "talk_mod": 2.0,
           "passband": {"flatness": 1.0}, "nb": {"blanked_pct": 0.5}}
    assert digout.objective(div) == pytest.approx(12.0 + 3.0 + 2.0)
    # a CW candidate has no syllabic content: that term is simply not scored
    assert digout.objective(div, kind="cw") == pytest.approx(12.0 + 2.0)
    # nobody talking, so no voice term either
    assert digout.objective(dict(div, talking=False)) == pytest.approx(14.0)


def test_the_blanker_cannot_win_by_gating_the_signal():
    div = {"snr_db": {"out": 12.0}, "passband": {"flatness": 0.0},
           "nb": {"blanked_pct": 30.0}}
    assert digout.objective(div) == pytest.approx(10.0)
    assert digout.objective(dict(div, nb={"blanked_pct": 5.0})) == pytest.approx(12.0)


def test_the_objective_falls_back_to_post_v2_snr_and_then_gives_up():
    assert digout.objective({"post": {"snr_out_db": 8.0}}) == pytest.approx(8.0)
    assert digout.objective({"snr_db": {"out": None}, "post": {}}) is None
    assert digout.objective({}) is None
    assert digout.objective({"snr_db": {"out": float("nan")}}) is None


def test_finder_kind_reads_the_candidate_on_this_frequency_only():
    f = {"candidates": [
        {"hz": 3962000.0, "score": 1.0, "kind": "cw", "width_hz": 2700.0},
        {"hz": 3900000.0, "score": 0.9, "kind": "voice", "width_hz": 2700.0}]}
    assert digout.finder_kind(f, 3962100.0) == "cw"
    assert digout.finder_kind(f, 3900200.0) == "voice"
    assert digout.finder_kind(f, 3800000.0) is None       # nothing near enough
    assert digout.finder_kind(f, None) is None
    assert digout.finder_kind({"candidates": []}, 3962000.0) is None
    weak = {"candidates": [{"hz": 3962000.0, "score": 0.05, "kind": "cw"}]}
    assert digout.finder_kind(weak, 3962000.0) is None


# ---- the plan --------------------------------------------------------------

def test_a_sixty_second_run_tries_fewer_knobs_than_a_five_minute_one():
    planned = {}
    for seconds in (60, 180, 300):
        s = digout.DigSearch(seconds)
        s.begin(SNAP, 0.0)
        planned[seconds] = [t["knob"] for t in s._plan]
        assert s.trials_planned == len(planned[seconds])
    assert len(planned[60]) == 8 < len(planned[180]) == 28 < len(planned[300]) == 48
    # the short run is the head of the long one: the biggest levers first
    assert planned[300][:8] == planned[60]
    assert planned[60][:4] == ["post", "subband", "mrc", "nb"]
    assert "nb_db" not in planned[60]         # the fiddly one only gets a long run


def test_the_cw_plan_reaches_for_the_peak_filter_not_the_voice_shaping():
    s = digout.DigSearch(180, kind="cw")
    s.begin(SNAP, 0.0)
    knobs = [t["knob"] for t in s._plan[:8]]
    assert "apf" in knobs and "auto_eq" not in knobs
    widths = [t["to"] for t in s._plan if t["knob"] == "width"]
    assert widths and all(hi - lo < 600.0 for lo, hi in widths)


def test_a_knob_the_snapshot_never_reported_is_never_tried():
    s = digout.DigSearch(300)
    s.begin({"post": False, "subband": False}, 0.0)
    assert {t["knob"] for t in s._plan} == {"post", "subband"}


# ---- the search -----------------------------------------------------------

def test_it_finds_the_corner_the_landscape_likes():
    def landscape(cur):
        return 10.0 + (4.0 if cur["post"] == "v2" else 0.0) \
                    + (2.0 if cur["subband"] is False else 0.0)

    s = digout.DigSearch(300)
    _trace, left, _t = drive(s, landscape)
    r = s.report()
    assert r["best"]["post"] == "v2" and r["best"]["subband"] is False
    assert left["post"] == "v2" and left["subband"] is False
    assert r["gain_db"] == pytest.approx(6.0)
    assert r["objective_before"] == 10.0 and r["objective_after"] == 16.0
    kept = [st["knob"] for st in r["steps"] if st["kept"]]
    assert kept == ["post", "subband"]
    assert r["changed"] == {"post": "v2", "subband": False}


def test_a_step_that_measures_worse_is_put_straight_back():
    def landscape(cur):
        return 10.0 if cur == SNAP else 4.0        # any change is a disaster

    s = digout.DigSearch(180)
    trace, left, _t = drive(s, landscape)
    r = s.report()
    assert r["steps"] and not any(st["kept"] for st in r["steps"])
    assert r["gain_db"] == 0.0
    assert left == s.snapshot and r["changed"] == {}
    # every trial was followed by a revert set and a re-measure of the incumbent
    reverts = [op for op in trace if op[0] == "set" and op[3]]
    assert len(reverts) == len(r["steps"])
    assert ("measure", "revert", 10.0) in trace


def test_a_tie_goes_to_the_setting_the_operator_already_had():
    s = digout.DigSearch(60)
    _trace, left, _t = drive(s, flat)
    assert left == s.snapshot
    assert all(st["delta_db"] == 0.0 and not st["kept"] for st in s.steps)


class Jitter:
    """A landscape that likes post v2 by `lift` dB, with one baseline read
    knocked sideways by `noise` — a band moving under the measurement."""

    def __init__(self, noise, lift=1.0):
        self.n, self.noise, self.lift = 0, noise, lift

    def __call__(self, cur):
        self.n += 1
        base = 10.0 + (self.lift if cur["post"] == "v2" else 0.0)
        return base + (self.noise if self.n == 2 else 0.0)


def test_the_margin_is_half_the_baseline_spread():
    """A jittery band has to be beaten by more than its own jitter."""
    quiet = digout.DigSearch(60)
    drive(quiet, Jitter(0.0))
    assert quiet.margin_db == digout.MIN_MARGIN_DB
    assert quiet.baseline_spread_db == 0.0 and quiet.unsteady is False
    assert quiet.note() is None
    assert quiet.steps[0]["knob"] == "post" and quiet.steps[0]["kept"]

    noisy = digout.DigSearch(60)
    drive(noisy, Jitter(3.0))
    assert noisy.baseline_spread_db == pytest.approx(3.0)
    assert noisy.margin_db == pytest.approx(1.5)      # half the range
    assert not noisy.steps[0]["kept"]          # +1 dB does not clear 1.5 dB


def test_an_unsteady_band_caps_the_margin_and_says_so():
    """The 05:58 run on 80 m: three baseline reads 7.7 dB apart. The old
    rule made the margin 7.7 and threw away a +4.8 dB win; the margin is now
    capped, so post v2 is kept and the operator is told the ground moved."""
    s = digout.DigSearch(60)
    drive(s, Jitter(7.7, lift=4.79))
    assert s.baseline_spread_db == pytest.approx(7.7)
    assert s.margin_db == digout.MARGIN_MAX_DB == 2.0
    assert s.unsteady is True
    assert s.note() == "the band swung 7.7 dB while sampling; results are tentative"
    assert s.steps[0]["knob"] == "post" and s.steps[0]["kept"]
    assert s.gain_db == pytest.approx(4.79)
    r = s.report()
    assert r["unsteady"] is True and r["note"] and r["baseline_spread_db"] == 7.7


def test_the_margin_is_never_wider_than_a_setting_could_clear():
    for spread in (0.0, 2.0, 7.7, 40.0):
        s = digout.DigSearch(60)
        drive(s, Jitter(spread, lift=0.0))
        assert digout.MIN_MARGIN_DB <= s.margin_db <= digout.MARGIN_MAX_DB


def test_the_incumbent_follows_the_band_between_trials():
    """The band lifts 5 dB half way through; a later revert re-measures, so
    the next candidate is judged against the band as it is now, not as it was."""
    class Drift:
        def __init__(self):
            self.n = 0

        def __call__(self, cur):
            self.n += 1
            return 10.0 + (5.0 if self.n > 6 else 0.0)

    s = digout.DigSearch(180)
    drive(s, Drift())
    assert s.incumbent == pytest.approx(15.0)
    assert not any(st["kept"] for st in s.steps)   # the lift is not credited to a knob
    r = s.report()
    # the band really did come up 5 dB, and the report says so - but the
    # operator is holding none of it, so the gain is zero
    assert r["objective_before"] == 10.0 and r["objective_after"] == 15.0
    assert r["gain_db"] == 0.0 and r["changed"] == {}


def test_the_run_stops_when_the_clock_runs_out_not_when_the_plan_does():
    s = digout.DigSearch(60, hold_s=3.0)
    _trace, _left, t = drive(s, flat)
    assert t <= 60.0
    assert s.phase == "done"
    assert s.trials_done <= s.trials_planned


def test_a_chain_that_stops_answering_ends_the_run():
    s = digout.DigSearch(180)
    s.begin(SNAP, 0.0)
    s.next_op(0.0)
    s.feed(None, 3.0)
    assert s.phase == "done"
    assert s.next_op(3.0) == {"op": "done"}


def test_feeding_when_nothing_was_asked_for_is_a_programming_error():
    s = digout.DigSearch(60)
    s.begin(SNAP, 0.0)
    with pytest.raises(RuntimeError):
        s.feed(1.0, 0.0)


def test_next_op_repeats_the_measure_until_it_is_fed():
    s = digout.DigSearch(60)
    s.begin(SNAP, 0.0)
    first = s.next_op(0.0)
    assert first["op"] == "measure"
    assert s.next_op(1.0) == first


# ---- the report the app draws ---------------------------------------------

def test_the_report_carries_everything_the_panel_needs():
    s = digout.DigSearch(60)
    assert s.report()["phase"] == "idle"
    drive(s, lambda cur: 10.0 + (4.0 if cur["post"] == "v2" else 0.0))
    r = s.report(48.0)
    for key in ("gain_db", "steps", "best", "started", "ends", "elapsed_s",
                "phase", "objective_before", "objective_after", "margin_db",
                "trials_planned", "trials_done", "changed", "kind", "seconds",
                "measured_best_db", "measured_best", "baseline_spread_db",
                "unsteady", "note"):
        assert key in r, key
    assert r["phase"] == "done" and r["started"] == 0.0 and r["ends"] == 60.0
    assert r["elapsed_s"] == 48.0
    step = r["steps"][0]
    assert set(step) == {"knob", "from", "to", "delta_db", "kept", "at_s"}
    assert step["knob"] == "post" and step["from"] is True and step["to"] == "v2"
    assert step["delta_db"] == pytest.approx(4.0) and step["kept"] is True


def test_nothing_kept_is_a_zero_gain_and_the_near_miss_is_still_reported():
    """8 trials, 8 reverts, changed {} - the gain the operator is holding is
    zero. What nearly won is still on the report so the panel can say "post
    v2 measured +1.5 dB but did not clear the 2.0 dB margin"."""
    s = digout.DigSearch(60)
    _trace, left, _t = drive(s, Jitter(7.7, lift=1.5))
    r = s.report()
    assert r["margin_db"] == 2.0                    # an unsteady band, capped
    assert left == s.snapshot and r["changed"] == {}
    assert not any(st["kept"] for st in r["steps"])
    assert r["gain_db"] == 0.0
    assert r["measured_best_db"] == pytest.approx(1.5)
    assert r["measured_best"]["knob"] == "post"
    assert r["measured_best"]["to"] == "v2"
    assert r["measured_best"]["kept"] is False


def test_the_gain_is_the_sum_of_the_kept_steps():
    def landscape(cur):
        return (10.0 + (4.0 if cur["post"] == "v2" else 0.0)
                + (2.0 if cur["subband"] is False else 0.0))

    s = digout.DigSearch(300)
    drive(s, landscape)
    r = s.report()
    kept = [st["delta_db"] for st in r["steps"] if st["kept"]]
    assert kept == [4.0, 2.0]
    assert r["gain_db"] == pytest.approx(sum(kept)) == 6.0
    assert r["measured_best_db"] == 4.0


def test_an_empty_run_reports_no_near_miss_rather_than_crashing():
    s = digout.DigSearch(60)
    assert s.measured_best_db == 0.0 and s.measured_best is None
    assert s.gain_db == 0.0 and s.report()["note"] is None


def test_set_kwargs_speak_the_adapters_own_language():
    assert digout.set_kwargs("post", "v2") == {"post": "v2"}
    assert digout.set_kwargs("nb", "auto") == {"nb": "auto"}
    assert digout.set_kwargs("width", (300.0, 2700.0)) == {"low_hz": 300.0,
                                                            "high_hz": 2700.0}
    assert digout.set_kwargs("agc", "slow") == {"agc": "slow"}
