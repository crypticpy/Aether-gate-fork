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

from aether_gate.core.voiceprint import (VoicePrint, MIN_OVER_S, VOICE_CHECK_S,
                                         DIFFERENT_VOICE)

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


def test_running_over_can_be_judged_against_a_print_and_a_stranger_is_far():
    rng = np.random.default_rng(5)
    vp = VoicePrint(RATE)
    _feed(vp, _over(rng, 4.0, 300.0, 2400.0, 3.0), 1)
    _silence(vp)
    mine = vp.summary(1)
    # too early to judge, then the running over of the same voice sits close
    x = _over(rng, 2.0, 300.0, 2400.0, 3.0)
    n_early = int(0.5 * RATE)
    _feed(vp, x[:n_early], 1)
    assert vp.current() is None
    _feed(vp, x[n_early:int(VOICE_CHECK_S * RATE) + BLOCK], 1)
    same = vp.current()
    assert same is not None and vp.distance(same, mine) < DIFFERENT_VOICE
    _silence(vp)
    # another rig and voice from the same talker id reads as somebody else
    _feed(vp, _over(rng, 1.2, 100.0, 2900.0, 5.0), 1)
    other = vp.current()
    assert other is not None and vp.distance(other, mine) >= DIFFERENT_VOICE
    assert vp.distance(None, mine) is None


def test_an_over_that_is_not_this_voice_does_not_teach_the_print():
    rng = np.random.default_rng(6)
    vp = VoicePrint(RATE)
    _feed(vp, _over(rng, 4.0, 300.0, 2400.0, 3.0), 1)
    _silence(vp)
    before = vp.summary(1)
    _feed(vp, _over(rng, 3.0, 100.0, 2900.0, 5.0), 1)      # a stranger, filed under 1
    _silence(vp)
    after = vp.summary(1)
    assert after["overs"] == 1 and after["high_hz"] == before["high_hz"]
    _feed(vp, _over(rng, 3.0, 300.0, 2400.0, 3.0), 1)      # the talker again
    _silence(vp)
    assert vp.summary(1)["overs"] == 2


def _noisy(rng, x, snr_db):
    n = rng.normal(size=len(x)) + 1j * rng.normal(size=len(x))
    n *= np.sqrt(np.mean(np.abs(x) ** 2) / 10 ** (snr_db / 10) / np.mean(np.abs(n) ** 2))
    return x + n


def _noise_only(rng, seconds, level):
    n = int(seconds * RATE)
    return level * (rng.normal(size=n) + 1j * rng.normal(size=n))


def test_the_band_noise_is_taken_out_so_a_weak_over_still_reads_as_the_same_voice():
    rng = np.random.default_rng(7)
    vp = VoicePrint(RATE)
    clean = _over(rng, 4.0, 300.0, 2400.0, 3.0)
    _feed(vp, clean, 1)
    _silence(vp)
    mine = vp.summary(1)
    # the band between overs: white noise at the level a 3 dB over would sit over
    noise_level = np.sqrt(np.mean(np.abs(clean) ** 2) / 10 ** (3.0 / 10) / 2)
    _feed(vp, _noise_only(rng, 4.0, noise_level), None, talking=False)
    # the same voice at 3 dB SNR, judged a second in: still this talker
    _feed(vp, _noisy(rng, _over(rng, 1.3, 300.0, 2400.0, 3.0), 3.0), 1)
    weak = vp.current()
    assert weak is not None and vp.distance(weak, mine) < DIFFERENT_VOICE, vp.distance(weak, mine)
    _silence(vp)
    # noise alone, with the VAD stuck on, is judged nowhere and teaches nothing
    _feed(vp, _noise_only(rng, 2.0, noise_level), 1)
    assert vp.current() is None
    _silence(vp)
    assert vp.summary(1)["overs"] == 1
