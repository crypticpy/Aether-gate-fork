#
# Aether-gate — G5: DIG OUT stops when the dial moves ("moved").
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""A dig scores one signal for a minute. If the operator tunes away, every read
after that is the new station being judged by the old one's answers, and every
knob it keeps was kept for somebody else. So the dial moving off the run --
across a band, or further than the passband it is working in -- ends the run
with the verdict `moved`, and puts every knob back exactly as `worse` does.

The fake adapter and the fake clock are test_diversity_dig.py's.

Each test names the mutation it catches in its own body.

Run:  .venv/bin/python -m pytest aether_gate/tests/test_dig_moved.py -q
"""
from aether_gate.adapters.diversity_dig import MOVED
from aether_gate.core import governor_proxy as proxy

from .test_diversity_dig import START, Clock, FakeAdapter, _finish, _runner


class Mover(Clock):
    """A clock that moves the dial to `hz` on hold number `at`."""

    def __init__(self, adapter, hz, at=2):
        super().__init__()
        self.a, self.hz, self.at = adapter, hz, at

    def sleep(self, seconds):
        super().sleep(seconds)
        if self.holds == self.at:
            self.a._slice_hz = self.hz


def _run_with(clock_for, seconds=60):
    a = FakeAdapter()
    a._slice_hz = 3_962_000.0                    # 80 m, where the run starts
    c = clock_for(a)
    r, _c = _runner(a, clock=c)
    r.start(seconds)
    return a, r, _finish(r)


def _knobs_back(a):
    """Every snapshotted knob is where the run found it."""
    return {k: a.s[k] for k in START} == START


def test_a_band_change_ends_the_run_and_puts_every_knob_back():
    """Mutation: no move check at all (the pre-G5 loop). The search would go
    on turning knobs on 40 m and keep whatever scored well there, having
    measured it against a talker on 80 m."""
    a, r, st = _run_with(lambda a: Mover(a, 7_150_000.0))
    assert st["verdict"] == MOVED and st["running"] is False
    assert st["cancelled"] is False              # a move is not a cancel
    assert _knobs_back(a)
    assert st["changed"] == {}


def test_the_record_says_which_bands():
    """Mutation: a verdict with no reason. The app has to be able to say why
    a run the operator started stopped by itself."""
    _a, _r, st = _run_with(lambda a: Mover(a, 7_150_000.0))
    rec = st["record"]
    assert rec["verdict"] == MOVED
    assert rec["why"] == "the band changed: 80 m -> 40 m"
    assert rec["hz"] == 3_962_000.0 and rec["band_hz"] == 3_750_000


def test_a_same_band_move_wider_than_the_passband_ends_it_too():
    """Mutation: only checking the band. Tuning 5 kHz down 80 m is a
    different QSO, and the run would go on scoring it."""
    a, _r, st = _run_with(lambda a: Mover(a, 3_967_000.0))    # +5 kHz, 2.8 kHz wide
    assert st["verdict"] == MOVED
    assert st["record"]["why"] == "the dial moved 5.0 kHz, further than the 2800 Hz passband"
    assert _knobs_back(a)


def test_a_move_inside_the_passband_is_not_a_move():
    """Mutation: any move ending the run. The operator nudges the dial while
    a dig is running and the minute they paid for would be thrown away."""
    a, _r, st = _run_with(lambda a: Mover(a, 3_962_800.0))    # +800 Hz
    assert st["verdict"] is None and st["record"] is None
    assert st["trials_done"] == st["trials_planned"] > 0       # it ran the lot
    assert a._slice_hz == 3_962_800.0                          # ...and the dial did move


def test_the_operator_still_owns_the_verdict_when_nothing_moved():
    """Mutation: the gate writing a verdict on every run. `moved` is the only
    verdict the gate says, and only when the dial says it."""
    a = FakeAdapter()
    r, _c = _runner(a)
    r.start(60)
    st = _finish(r)
    assert st["verdict"] is None
    assert r.verdict("keep")["verdict"] == "keep"


def test_a_moved_run_banks_nothing_in_the_governor():
    """Mutation: leaving `moved` out of dig_delta_db. AUTO CLEAN would bank
    the gain of a run whose settings were all put back."""
    delta, note = proxy.dig_delta_db({"dig_verdict": MOVED, "dig_gain_db": 3.4})
    assert delta == 0.0 and "nothing banked" in note
