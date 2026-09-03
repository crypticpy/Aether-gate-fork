#
# Aether-gate — the "dig this out" runner: snapshot, search, put it back.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""A fake adapter that remembers every call and a fake clock that never
sleeps, so the whole minute happens in a millisecond. What matters here is
not which knob wins — test_digout.py covers that — but that the runner reads
the settings before it touches them, puts every one of them back on worse,
on cancel and on a driver that throws, and only ever runs one at a time.

Run:  python -m pytest aether_gate/tests/test_diversity_dig.py
"""
import queue
import threading
import time

import pytest

from aether_gate.adapters.diversity_dig import DigRunner, read_snapshot

START = {"post": True, "subband": True, "mrc": False, "nb": True, "nb_db": 11.0,
         "low_hz": 100.0, "high_hz": 2900.0, "contour": False, "anf": False,
         "apf": False, "auto_eq": False, "agc": "med"}


class FakeAdapter:
    """Just the four calls the runner is allowed to make."""

    _slice_hz = 3962000.0

    def __init__(self, score=None, kind="voice", raise_on=None):
        self.calls = []
        self.score = score or (lambda s: 10.0)
        self.kind = kind
        self.raise_on = raise_on            # a set_diversity kwarg that explodes
        self.s = dict(START)
        self.available = True
        self.offset = 0.0                   # the band moving under a read

    # ---- what the runner reads ----
    def knobs(self):
        d = dict(self.s)
        d["width"] = (d.pop("low_hz"), d.pop("high_hz"))
        return d

    def diversity_status(self, slice_id=None):
        self.calls.append(("diversity_status", {}))
        if not self.available:
            return {"available": False}
        return {"available": True, "channels": 2, "talking": False,
                "snr_db": {"a": 1.0, "b": 1.0,
                           "out": self.score(self.knobs()) + self.offset},
                "passband": {"flatness": 0.0},
                "post": {"enabled": self.s["post"] is not False,
                         "version": 2 if self.s["post"] == "v2" else 1},
                "subband": {"enabled": self.s["subband"]},
                "mrc": {"enabled": self.s["mrc"]},
                "nb": {"enabled": self.s["nb"] is not False,
                       "threshold_db": self.s["nb_db"], "blanked_pct": 0.0,
                       "auto": {"mode": "auto" if self.s["nb"] == "auto" else "on"}},
                "talker": {"id": 7}, "focus": None}

    def filter_status(self):
        self.calls.append(("filter_status", {}))
        return {"available": True, "low_hz": self.s["low_hz"],
                "high_hz": self.s["high_hz"],
                "contour": {"enabled": self.s["contour"]},
                "anf": {"enabled": self.s["anf"]},
                "apf": {"enabled": self.s["apf"]},
                "auto_eq": {"enabled": self.s["auto_eq"]},
                "agc": {"mode": self.s["agc"]}}

    def diversity_finder(self):
        self.calls.append(("diversity_finder", {}))
        return {"available": True, "candidates": [
            {"hz": self._slice_hz, "score": 1.0, "kind": self.kind,
             "width_hz": 2700.0}]}

    # ---- what the runner writes ----
    def set_diversity(self, **kw):
        self.calls.append(("set_diversity", dict(kw)))
        if self.raise_on is not None and self.raise_on in kw:
            raise RuntimeError("the driver stopped answering")
        self.s.update(kw)
        return self.diversity_status()

    def filter_set(self, **kw):
        self.calls.append(("filter_set", dict(kw)))
        self.s.update(kw)
        return self.filter_status()


class Clock:
    """A clock that only moves when the runner sleeps."""

    def __init__(self):
        self.t = 1000.0
        self.holds = 0

    def sleep(self, seconds):
        self.holds += 1
        self.t += float(seconds)

    def now(self):
        return self.t


class Bumpy(Clock):
    """A clock that lifts the band by `offset` dB for the read after hold
    number `at` — one unsteady baseline sample, on purpose."""

    def __init__(self, adapter, at, offset):
        super().__init__()
        self.a, self.at, self.offset = adapter, at, offset

    def sleep(self, seconds):
        super().sleep(seconds)
        self.a.offset = self.offset if self.holds == self.at else 0.0


class Pacer(Clock):
    """The same, but the test lets each hold through one at a time."""

    def __init__(self):
        super().__init__()
        self.at, self.go = queue.Queue(), queue.Queue()

    def sleep(self, seconds):
        self.at.put(seconds)
        self.go.get(timeout=5.0)
        super().sleep(seconds)

    def release(self, n=1):
        """Wait until the thread is parked at a hold, then let it run on."""
        for _ in range(n):
            self.at.get(timeout=5.0)
            self.go.put(1)


def _runner(adapter, clock=None, wall=None):
    c = clock or Clock()
    return DigRunner(adapter, clock=c.now, sleep=c.sleep,
                     wall=wall or (lambda: 1_700_000_000.0)), c


def _finish(r, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        st = r.status()
        if not st["running"]:
            return st
        time.sleep(0.005)
    pytest.fail("the dig never finished")


def _likes_v2(s):
    return 10.0 + (4.0 if s["post"] == "v2" else 0.0)


# ---- reading the settings before touching them -----------------------------

def test_the_snapshot_is_every_knob_the_two_status_dicts_report():
    a = FakeAdapter()
    snap = read_snapshot(a.diversity_status(), a.filter_status())
    assert snap == {"post": True, "subband": True, "mrc": False, "nb": True,
                    "nb_db": 11.0, "width": (100.0, 2900.0), "contour": False,
                    "anf": False, "apf": False, "auto_eq": False, "agc": "med"}


def test_a_section_the_adapter_does_not_report_is_left_out():
    assert read_snapshot({"available": True}, {}) == {}
    assert read_snapshot({"post": {"enabled": True, "version": 2}}, {}) == {"post": "v2"}
    assert read_snapshot({}, {"agc": {"mode": "nonsense"}}) == {}


# ---- a whole run -----------------------------------------------------------

def test_a_run_keeps_what_helped_and_says_so():
    a = FakeAdapter(score=_likes_v2)
    r, _c = _runner(a)
    started = r.start(60)
    assert started["phase"] != "idle" and started["started"] is not None
    st = _finish(r)
    assert st["phase"] == "done" and st["gain_db"] == pytest.approx(4.0)
    assert st["best"]["post"] == "v2" and a.s["post"] == "v2"
    assert st["kind"] == "voice" and st["talker_id"] == 7
    assert st["snapshot"]["post"] is True
    assert st["verdict"] is None and st["record"] is None and st["error"] is None


def test_worse_puts_every_knob_back_where_it_was():
    a = FakeAdapter(score=_likes_v2)
    r, _c = _runner(a)
    r.start(180)
    _finish(r)
    assert a.s["post"] == "v2"                  # the search left it on
    st = r.verdict("WORSE ")
    assert st["verdict"] == "worse"
    assert a.s == START                         # every knob is back
    rec = st["record"]
    assert rec["kind"] == "dig" and rec["verdict"] == "worse"
    assert rec["gain_db"] == pytest.approx(4.0) and rec["seconds"] == 180
    assert rec["objective_before"] == 10.0 and rec["objective_after"] == 14.0
    assert rec["talker_id"] == 7 and rec["signal"] == "voice"
    assert rec["measured_best_db"] == pytest.approx(4.0)
    assert rec["margin_db"] == 0.5 and rec["unsteady"] is False
    assert rec["note"] is None
    assert set(rec) >= {"kind", "t", "gain_db", "verdict", "best",
                        "objective_before", "objective_after", "seconds",
                        "measured_best_db", "margin_db", "unsteady", "note"}


def test_better_and_keep_both_leave_the_settings_alone():
    for word in ("better", "keep"):
        a = FakeAdapter(score=_likes_v2)
        r, _c = _runner(a)
        r.start(60)
        _finish(r)
        st = r.verdict(word)
        assert st["verdict"] == word and a.s["post"] == "v2"
        assert st["record"]["verdict"] == word


def test_a_verdict_is_a_word_from_the_list_given_once_after_the_run():
    a = FakeAdapter()
    r, _c = _runner(a)
    with pytest.raises(RuntimeError):
        r.verdict("better")                     # nothing has run yet
    r.start(60)
    _finish(r)
    with pytest.raises(ValueError):
        r.verdict("louder")
    r.verdict("keep")
    with pytest.raises(RuntimeError):
        r.verdict("better")


def test_only_one_dig_at_a_time():
    a = FakeAdapter()
    p = Pacer()
    r, _c = _runner(a, clock=p)
    r.start(300)
    p.at.get(timeout=5.0)                       # parked at the first hold
    with pytest.raises(RuntimeError):
        r.start(60)
    p.go.put(1)
    r.cancel()


def test_the_seconds_are_the_three_the_button_offers():
    r, _c = _runner(FakeAdapter())
    for bad in (10, 90, 600, 0):
        with pytest.raises(ValueError):
            r.start(bad)


def test_an_adapter_with_no_pair_running_refuses_to_start():
    a = FakeAdapter()
    a.available = False
    r, _c = _runner(a)
    with pytest.raises(RuntimeError):
        r.start(60)


# ---- the phases the app draws ---------------------------------------------

def test_the_phases_go_idle_sampling_searching_done():
    a = FakeAdapter(score=_likes_v2)
    p = Pacer()
    r, _c = _runner(a, clock=p)
    assert r.status()["phase"] == "idle" and r.status()["running"] is False
    r.start(300)
    for _ in range(3):                          # the three baseline reads
        p.at.get(timeout=5.0)
        assert r.status()["phase"] == "sampling"
        p.go.put(1)
    p.at.get(timeout=5.0)                       # parked on the first candidate
    assert r.status()["phase"] == "searching"
    assert r.status()["remaining_s"] < 300.0
    p.go.put(1)
    r.cancel()
    assert r.status()["phase"] == "done"


def test_nothing_is_set_before_the_baseline_is_measured():
    a = FakeAdapter()
    p = Pacer()
    r, _c = _runner(a, clock=p)
    r.start(300)
    for _ in range(3):
        p.release()
    p.at.get(timeout=5.0)
    writes = [c for c in a.calls if c[0] in ("set_diversity", "filter_set")]
    assert len(writes) == 1                     # exactly the first candidate
    p.go.put(1)
    r.cancel()


def test_one_knob_moves_at_a_time():
    a = FakeAdapter(score=_likes_v2)
    r, _c = _runner(a)
    r.start(300)
    _finish(r)
    for name, kw in a.calls:
        if name in ("set_diversity", "filter_set"):
            assert len(kw) == 1 or set(kw) == {"low_hz", "high_hz"}, kw


# ---- putting it back -------------------------------------------------------

def test_cancel_stops_the_run_and_restores_the_snapshot():
    a = FakeAdapter(score=_likes_v2)
    p = Pacer()
    r, _c = _runner(a, clock=p)
    r.start(300)
    for _ in range(5):                          # baselines, then a kept step
        p.release()
    p.at.get(timeout=5.0)
    assert a.s["post"] == "v2"                  # mid-run it really is changed
    p.go.put(1)
    r.cancel()
    st = _finish(r)
    assert st["cancelled"] is True and st["phase"] == "done"
    assert a.s == START


def test_cancel_with_nothing_running_is_harmless():
    r, _c = _runner(FakeAdapter())
    assert r.cancel()["phase"] == "idle"


def test_a_driver_that_throws_mid_run_restores_and_reports_it():
    a = FakeAdapter(raise_on="subband")
    r, _c = _runner(a)
    r.start(300)
    st = _finish(r)
    assert st["phase"] == "done"
    assert "the driver stopped answering" in st["error"]
    assert a.s == START


def test_a_chain_that_stops_answering_ends_the_run_without_a_traceback():
    a = FakeAdapter()
    r, _c = _runner(a)
    r.start(60)
    a.available = False                         # the pair drops out mid-run
    st = _finish(r)
    assert st["phase"] == "done" and st["error"] is None


def test_the_clock_the_app_shows_is_wall_time():
    a = FakeAdapter()
    r, _c = _runner(a, wall=lambda: 1_700_000_000.0)
    st = r.start(180)
    assert st["started"] == 1_700_000_000.0
    assert st["ends"] == 1_700_000_180.0
    st = _finish(r)
    assert 0.0 <= st["elapsed_s"] <= 180.0
    assert st["record"] is None


def test_the_holds_are_never_shorter_than_the_chain_takes_to_settle():
    a = FakeAdapter()
    c = Clock()
    slept = []
    r = DigRunner(a, clock=c.now, sleep=lambda s: (slept.append(s), c.sleep(s))[1],
                  wall=lambda: 0.0)
    r.start(60)
    _finish(r)
    assert slept and min(slept) >= 2.5


def test_the_runner_never_reaches_past_the_adapters_public_calls():
    a = FakeAdapter(score=_likes_v2)
    r, _c = _runner(a)
    r.start(60)
    _finish(r)
    r.verdict("better")
    assert {name for name, _ in a.calls} <= {"diversity_status", "filter_status",
                                             "diversity_finder", "set_diversity",
                                             "filter_set"}


def test_threads_do_not_outlive_the_run():
    a = FakeAdapter()
    r, _c = _runner(a)
    r.start(60)
    _finish(r)
    assert not any(t.name == "diversity-dig" and t.is_alive()
                   for t in threading.enumerate())


# ---- the band moving under the measurement ---------------------------------

def test_an_unsteady_band_caps_the_margin_and_reaches_the_app_as_a_note():
    """The 05:58 run on 80 m, end to end: the three baseline reads land
    7.7 dB apart. The margin is capped at 2.0 rather than 7.7, so post v2's
    +4.8 dB is kept instead of thrown away, and the operator is told the
    ground was moving while we measured it."""
    a = FakeAdapter(score=lambda s: 10.0 + (4.79 if s["post"] == "v2" else 0.0))
    c = Bumpy(a, at=2, offset=7.7)
    r = DigRunner(a, clock=c.now, sleep=c.sleep, wall=lambda: 1_700_000_000.0)
    r.start(60)
    st = _finish(r)
    assert st["baseline_spread_db"] == pytest.approx(7.7)
    assert st["margin_db"] == 2.0
    assert st["unsteady"] is True
    assert st["note"] == "the band swung 7.7 dB while sampling; results are tentative"
    assert st["gain_db"] == pytest.approx(4.79)
    assert st["changed"] == {"post": "v2"} and a.s["post"] == "v2"
    rec = r.verdict("better")["record"]
    assert rec["unsteady"] is True and rec["note"] == st["note"]
    assert rec["measured_best_db"] == pytest.approx(4.79)


def test_a_steady_band_says_nothing_and_takes_the_floor_margin():
    a = FakeAdapter(score=_likes_v2)
    r, _c = _runner(a)
    r.start(60)
    st = _finish(r)
    assert st["unsteady"] is False and st["note"] is None
    assert st["margin_db"] == 0.5 and st["baseline_spread_db"] == 0.0


def test_a_run_that_keeps_nothing_holds_no_gain():
    """Whatever the band did while we watched, the gain is what the operator
    is holding: nothing kept, nothing changed, 0.0 dB."""
    a = FakeAdapter(score=lambda s: 10.0)
    c = Bumpy(a, at=2, offset=7.7)          # unsteady, so the margin is 2.0
    r = DigRunner(a, clock=c.now, sleep=c.sleep, wall=lambda: 0.0)
    r.start(60)
    st = _finish(r)
    assert st["gain_db"] == 0.0 and st["changed"] == {}
    assert not any(step["kept"] for step in st["steps"])
    assert a.s == START
