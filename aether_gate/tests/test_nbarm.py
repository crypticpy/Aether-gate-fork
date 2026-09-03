#
# Aether-gate — the blanker arms itself, no hardware.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""NbArm against synthetic NoiseProfile.status() dicts: a rising impulse
rate that must earn the hold before it arms, a threshold that tracks the
measured excess but does not dither on every tenth of a dB, a falling
rate that must earn its own longer hold before it lets go, a mains comb
that must never arm the blanker by itself, and the operator's manual
on/off always winning over whatever auto would have decided.

Run:  python -m pytest aether_gate/tests/test_nbarm.py
"""
import pytest

from aether_gate.core.nbarm import (
    ARM_HOLD_S, ARM_RATE, DISARM_HOLD_S, DISARM_RATE, MARGIN_DB,
    NB_DB_MAX, NB_DB_MIN, NbArm,
)


def _status(rate=0.0, db=None, mains_hz=None):
    """A NoiseProfile.status() shaped dict with only the fields NbArm reads
    actually varied; the rest are plausible filler."""
    return {
        "mains_hz": mains_hz,
        "hum_db": 20.0 if mains_hz else 0.0,
        "harmonics": 4 if mains_hz else 0,
        "impulses_per_s": rate,
        "impulse_db": db,
        "periodic": [],
        "seconds": 2.0,
        "window_s": 2.0,
        "impulse_window_s": 4.0,
    }


def _arm(nb, rate=12.0, db=14.0, t0=0.0):
    """Feed a steady impulsive rate across the arm hold; returns the last
    Decision, which must be armed once this returns."""
    d = None
    t = t0
    while t <= t0 + ARM_HOLD_S:
        d = nb.update(_status(rate=rate, db=db), t)
        t += 1.0
    assert d.nb_on, d
    return d


def test_rising_impulses_arm_after_the_hold_and_not_before():
    nb = NbArm(mode="auto")
    st = _status(rate=12.0, db=14.0)
    t = 0.0
    while t < ARM_HOLD_S:
        d = nb.update(st, t)
        assert d.nb_on is False, d
        t += 1.0
    d = nb.update(st, ARM_HOLD_S)
    assert d.nb_on is True
    assert d.changed is True
    assert d.threshold is not None


def test_a_rate_below_arm_rate_never_arms_no_matter_how_long():
    nb = NbArm(mode="auto")
    st = _status(rate=ARM_RATE - 0.5, db=10.0)
    for t in range(0, 60, 5):
        d = nb.update(st, float(t))
        assert d.nb_on is False, d


def test_threshold_tracks_the_excess_with_the_margin_and_clamps():
    nb = NbArm(mode="auto")
    d = _arm(nb, db=14.0)
    expected = round((14.0 - MARGIN_DB) * 2) / 2
    assert d.threshold == expected

    hot = nb.update(_status(rate=12.0, db=50.0), ARM_HOLD_S + 1.0)
    assert hot.threshold == NB_DB_MAX

    cold = nb.update(_status(rate=12.0, db=1.0), ARM_HOLD_S + 2.0)
    assert cold.threshold == NB_DB_MIN


def test_small_excess_changes_do_not_move_the_threshold():
    nb = NbArm(mode="auto")
    first = _arm(nb, db=14.0)
    d = nb.update(_status(rate=12.0, db=first.threshold + MARGIN_DB + 1.0),
                  ARM_HOLD_S + 1.0)
    assert d.threshold == first.threshold, (first.threshold, d.threshold)
    assert d.nb_on is True and d.changed is False


def test_disarm_only_after_the_hold():
    nb = NbArm(mode="auto")
    _arm(nb, db=14.0)
    quiet = _status(rate=0.0, db=None)
    below_start = ARM_HOLD_S + 1.0
    nb.update(quiet, below_start)                          # below_since starts here
    still_on = nb.update(quiet, below_start + DISARM_HOLD_S - 1.0)
    assert still_on.nb_on is True, still_on
    now_off = nb.update(quiet, below_start + DISARM_HOLD_S)
    assert now_off.nb_on is False
    assert now_off.changed is True
    assert now_off.threshold is None
    assert "stopped" in now_off.reason


def test_a_dropout_inside_a_burst_does_not_disarm():
    nb = NbArm(mode="auto")
    _arm(nb, db=14.0)
    t = ARM_HOLD_S + 1.0
    nb.update(_status(rate=0.0, db=None), t)                # a gap in the burst
    t += DISARM_HOLD_S / 2.0
    d = nb.update(_status(rate=12.0, db=14.0), t)            # impulses resume
    assert d.nb_on is True
    t += DISARM_HOLD_S - 1.0                                 # short of a fresh hold
    d = nb.update(_status(rate=0.0, db=None), t)
    assert d.nb_on is True, d


def test_a_mains_comb_alone_never_arms_it():
    nb = NbArm(mode="auto")
    st = _status(rate=0.0, db=None, mains_hz=60.0)
    for t in range(0, 120, 5):
        d = nb.update(st, float(t))
        assert d.nb_on is False, d


def test_manual_on_wins_and_reports_regardless_of_the_profile():
    nb = NbArm(mode="on")
    d = nb.update(_status(rate=0.0, db=None), 0.0)
    assert d.nb_on is True
    assert d.threshold is None
    assert d.reason == "manual: on"


def test_manual_off_wins_and_reports_regardless_of_the_profile():
    nb = NbArm(mode="off")
    d = nb.update(_status(rate=50.0, db=30.0), 0.0)
    assert d.nb_on is False
    assert d.threshold is None
    assert d.reason == "manual: off"


def test_manual_off_overrides_an_already_armed_auto_blanker():
    nb = NbArm(mode="auto")
    _arm(nb, db=14.0)
    nb.set_mode("off")
    d = nb.update(_status(rate=12.0, db=14.0), ARM_HOLD_S + 1.0)
    assert d.nb_on is False
    assert d.reason == "manual: off"


def test_returning_to_auto_re_earns_the_hold_rather_than_reusing_a_stale_one():
    nb = NbArm(mode="auto")
    _arm(nb, db=14.0)
    nb.set_mode("off")
    nb.update(_status(rate=12.0, db=14.0), ARM_HOLD_S + 100.0)
    nb.set_mode("auto")
    d = nb.update(_status(rate=12.0, db=14.0), ARM_HOLD_S + 100.0)
    assert d.nb_on is False, d       # the hold starts over, not instantly armed


def test_status_returns_the_control_ports_shape():
    nb = NbArm(mode="auto")
    _arm(nb, db=14.0)
    s = nb.status()
    assert set(s) == {"mode", "armed", "threshold", "reason", "since_s"}
    assert s["mode"] == "auto"
    assert s["armed"] is True
    assert s["threshold"] == round((14.0 - MARGIN_DB) * 2) / 2
    assert isinstance(s["reason"], str) and s["reason"]
    assert s["since_s"] >= 0.0


def test_status_before_any_update_is_off_and_unarmed():
    nb = NbArm(mode="auto")
    s = nb.status()
    assert s["armed"] is False
    assert s["threshold"] is None


def test_set_mode_rejects_junk():
    nb = NbArm()
    with pytest.raises(ValueError):
        nb.set_mode("blink")
    for m in ("auto", "on", "off"):
        nb.set_mode(m)          # none of these raise
    assert nb.mode == "off"


def test_disarm_rate_sits_below_arm_rate_so_the_hysteresis_actually_exists():
    assert DISARM_RATE < ARM_RATE
