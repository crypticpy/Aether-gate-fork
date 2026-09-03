#
# Aether-gate — B23 front-end linearity guard, on a fake clock (no hardware).
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""adapters/frontend_guard.py is pure — no Soapy, no numpy, no thread — so
every case here drives it with hand-fed readings and an explicit `t`. The
adapter-level wiring (peak-per-block into headroom, the writeSetting that
turns a decision into a real rfgain_sel write) has no synthetic-block harness
in this test suite to hook into: test_soapy_recovery.py and test_resolution.py
both replicate `_read_loop`'s logic by hand rather than running the real
reader thread against a fake `_sdr.readStream`, and no existing fixture feeds
manufactured IQ through that thread. So this file covers the policy only, and
the report for this task says so rather than inventing a first such harness.

Run:  .venv/bin/python -m pytest aether_gate/tests/test_frontend_guard.py -q
"""
from aether_gate.adapters.frontend_guard import (
    CLEAR_S, HOLD_S, LOW_TICKS_REQUIRED, FrontEndGuard,
)


def _guard(floor="0", max_state="9", enabled=True):
    return FrontEndGuard(floor_state=floor, max_state=max_state, enabled=enabled)


# --- clipping / low headroom steps up ---------------------------------------

def test_clipping_steps_up_once_not_twice_within_the_hold():
    g = _guard()
    t = 0.0
    # first low tick: counted, no step yet
    assert g.tick(t=t, headroom_db=0.0, clip_count=3, lna_state="0") is None
    assert g.state != "stepping_up"
    # second consecutive low tick: steps
    t += 0.05
    new = g.tick(t=t, headroom_db=0.0, clip_count=3, lna_state="0")
    assert new == "1"
    assert g.state == "stepping_up"
    assert g.events[-1]["reason"] == "clipping"
    assert g.events[-1] == {"t": t, "from": "0", "to": "1",
                             "reason": "clipping", "headroom_db": 0.0}
    # still clipping, but inside the hold window: must not step again
    t += HOLD_S * 0.5
    again = g.tick(t=t, headroom_db=0.0, clip_count=5, lna_state="1")
    assert again is None
    assert g.state == "holding"


def test_low_headroom_without_clips_needs_two_consecutive_ticks_too():
    g = _guard()
    for i in range(LOW_TICKS_REQUIRED - 1):
        assert g.tick(t=float(i), headroom_db=1.5, clip_count=0, lna_state="0") is None
    new = g.tick(t=float(LOW_TICKS_REQUIRED - 1), headroom_db=1.5, clip_count=0, lna_state="0")
    assert new == "1"
    assert g.events[-1]["reason"] == "headroom 1.5 dB"


def test_a_good_tick_between_two_low_ones_resets_the_streak():
    g = _guard()
    assert g.tick(t=0.0, headroom_db=0.5, clip_count=0, lna_state="0") is None
    assert g.tick(t=1.0, headroom_db=20.0, clip_count=0, lna_state="0") is None  # clears the streak
    assert g.tick(t=2.0, headroom_db=0.5, clip_count=0, lna_state="0") is None   # streak restarts at 1
    assert g.tick(t=3.0, headroom_db=0.5, clip_count=0, lna_state="0") == "1"


# --- headroom hysteresis / step down -----------------------------------------

def test_14db_for_40s_does_nothing():
    g = _guard(floor="0")
    g.tick(t=0.0, headroom_db=20.0, clip_count=0, lna_state="2")  # past any hold
    t = HOLD_S + 1.0
    for _ in range(40):
        assert g.tick(t=t, headroom_db=14.0, clip_count=0, lna_state="2") is None
        t += 1.0
    assert g.state != "stepping_down"


def test_16db_for_29s_does_nothing_16db_for_30s_steps_down():
    g = _guard(floor="0")
    base = HOLD_S + 1.0
    g.tick(t=base, headroom_db=16.0, clip_count=0, lna_state="2")  # clear_since starts here
    t = base
    for _ in range(29):
        t += 1.0
        assert g.tick(t=t, headroom_db=16.0, clip_count=0, lna_state="2") is None
    t += 1.0  # 30 s since clear_since
    new = g.tick(t=t, headroom_db=16.0, clip_count=0, lna_state="2")
    assert new == "1"
    assert g.events[-1]["reason"] == "clear for 30 s"


def test_a_dip_under_15db_resets_the_clear_clock():
    g = _guard(floor="0")
    base = HOLD_S + 1.0
    g.tick(t=base, headroom_db=16.0, clip_count=0, lna_state="2")
    t = base + 25.0
    assert g.tick(t=t, headroom_db=16.0, clip_count=0, lna_state="2") is None
    t += 1.0
    assert g.tick(t=t, headroom_db=10.0, clip_count=0, lna_state="2") is None  # resets
    t += CLEAR_S - 1.0
    # would have been >=30s since the ORIGINAL clear_since, but not since the reset
    assert g.tick(t=t, headroom_db=16.0, clip_count=0, lna_state="2") is None


# --- floor and ceiling are hard bounds ---------------------------------------

def test_never_below_floor():
    g = _guard(floor="2")
    base = HOLD_S + 1.0
    g.tick(t=base, headroom_db=16.0, clip_count=0, lna_state="2")
    t = base
    for _ in range(31):
        t += 1.0
        r = g.tick(t=t, headroom_db=16.0, clip_count=0, lna_state="2")
        assert r is None
    assert g.lna_state == "2"


def test_never_above_max():
    g = _guard(max_state="9")
    assert g.tick(t=0.0, headroom_db=0.0, clip_count=1, lna_state="9") is None
    r = g.tick(t=0.1, headroom_db=0.0, clip_count=1, lna_state="9")
    assert r is None
    assert g.lna_state == "9"


# --- disabled guard --------------------------------------------------------

def test_disabled_guard_never_steps_even_when_conditions_scream_for_it():
    g = _guard(enabled=False)
    t = 0.0
    for _ in range(100):
        assert g.tick(t=t, headroom_db=0.0, clip_count=50, lna_state="0") is None
        t += 1.0
    assert len(g.events) == 0
    assert g.state == "idle"


def test_reenabling_after_disabled_starts_the_streak_fresh():
    g = _guard(enabled=False)
    g.tick(t=0.0, headroom_db=0.0, clip_count=1, lna_state="0")
    g.tick(t=1.0, headroom_db=0.0, clip_count=1, lna_state="0")
    g.enabled = True
    # first enabled tick is only streak count 1, not immediately a step
    assert g.tick(t=2.0, headroom_db=0.0, clip_count=1, lna_state="0") is None
    assert g.tick(t=2.1, headroom_db=0.0, clip_count=1, lna_state="0") == "1"


# --- events + status shape ---------------------------------------------------

def test_events_are_capped_and_shaped():
    # A caller always feeds back the device's own last-reported state (see
    # tick()'s docstring), so this drives the guard the way soapy.py would:
    # each step's "to" becomes the next tick's lna_state. Each step costs
    # LOW_TICKS_REQUIRED ticks (the streak resets to 0 after every step), so
    # 45 ticks is comfortably more than the 20-event cap needs to be proven.
    g = _guard(max_state="99")
    t = 0.0
    for _ in range(45):
        t += HOLD_S + 0.01
        g.tick(t=t, headroom_db=0.0, clip_count=1, lna_state=g.lna_state)
    assert len(g.events) == 20               # capped, not merely "happens to be under"
    for ev in g.events:
        assert set(ev.keys()) == {"t", "from", "to", "reason", "headroom_db"}


def test_snapshot_keys_present_in_every_state():
    expected = {"guard", "floor_state", "max_state", "lna_state", "state",
                "hold_until", "events"}
    for enabled in (True, False):
        g = _guard(enabled=enabled)
        assert set(g.snapshot().keys()) == expected
        g.tick(t=0.0, headroom_db=0.0, clip_count=1, lna_state="0")
        assert set(g.snapshot().keys()) == expected
        g.tick(t=HOLD_S + 1, headroom_db=0.0, clip_count=1, lna_state="0")
        assert set(g.snapshot().keys()) == expected
