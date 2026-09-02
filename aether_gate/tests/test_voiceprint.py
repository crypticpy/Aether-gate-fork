#
# Aether-gate — talker prints from synthetic overs, no hardware.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Two talkers with different rigs (TX upper edge 2.4 vs 2.9 kHz) and
different voices (3 vs 5 syllables a second) get prints that say so; short
overs and overs nobody was recalled for teach nothing.

Run:  python -m pytest aether_gate/tests/test_voiceprint.py
"""
import numpy as np

from aether_gate.core.voiceprint import VoicePrint, MIN_OVER_S

RATE = 25_000.0
BLOCK = 2000


def _over(rng, seconds, lo_hz, hi_hz, syl_hz):
    n = int(seconds * RATE)
    X = np.zeros(n, dtype=np.complex128)
    f = np.fft.fftfreq(n, 1.0 / RATE)
    sel = (f >= lo_hz) & (f < hi_hz)                     # one-sided, like an SSB passband
    X[sel] = rng.normal(size=sel.sum()) + 1j * rng.normal(size=sel.sum())
    x = np.fft.ifft(X)
    t = np.arange(n) / RATE
    env = 0.55 + 0.45 * np.sin(2 * np.pi * syl_hz * t)
    return x / np.sqrt(np.mean(np.abs(x) ** 2)) * env


def _feed(vp, x, talker, talking=True):
    for i in range(0, len(x), BLOCK):
        vp.feed(x[i:i + BLOCK], talking, talker)


def _silence(vp):
    vp.feed(np.zeros(BLOCK, dtype=np.complex128), False, None)


def test_two_talkers_get_prints_that_tell_them_apart():
    rng = np.random.default_rng(3)
    vp = VoicePrint(RATE)
    for _ in range(3):
        _feed(vp, _over(rng, 4.0, 300.0, 2400.0, 3.0), 1)
        _silence(vp)
        _feed(vp, _over(rng, 6.0, 100.0, 2900.0, 5.0), 2)
        _silence(vp)
    a, b = vp.summary(1), vp.summary(2)
    assert 2300 <= a["high_hz"] <= 2600 and 2800 <= b["high_hz"] <= 3100
    assert 200 <= a["low_hz"] <= 400 and b["low_hz"] <= 200
    assert abs(a["syllabic_hz"] - 3.0) <= 0.4 and abs(b["syllabic_hz"] - 5.0) <= 0.4
    assert abs(a["over_s"] - 4.0) <= 0.3 and abs(b["over_s"] - 6.0) <= 0.3
    assert a["overs"] == 3 and b["overs"] == 3
    assert a["centroid_hz"] < b["centroid_hz"]
    assert a["tilt_db"] is not None and a["tilt_db"] < b["tilt_db"]


def test_short_and_anonymous_overs_teach_nothing_and_forget_works():
    rng = np.random.default_rng(4)
    vp = VoicePrint(RATE)
    _feed(vp, _over(rng, MIN_OVER_S * 0.6, 300.0, 2400.0, 3.0), 1)
    _silence(vp)
    assert vp.summary(1) is None
    _feed(vp, _over(rng, 4.0, 300.0, 2400.0, 3.0), None)        # nobody recalled
    _silence(vp)
    assert vp.prints == {}
    # the id may arrive part-way through the over (a recall a few blocks in)
    x = _over(rng, 4.0, 300.0, 2400.0, 3.0)
    _feed(vp, x[: len(x) // 2], None)
    _feed(vp, x[len(x) // 2:], 7)
    _silence(vp)
    assert vp.summary(7) is not None and vp.summary(7)["overs"] == 1
    vp.forget(keep_ids={7})
    assert 7 in vp.prints
    vp.forget()
    assert vp.prints == {} and vp.summary(7) is None
