#
# Aether-gate — the coherence post-filter (core/postfilter.py) on a synthetic pair.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""A talker arriving in step on two loops over independent band noise: the
post-filter takes the noise between words down to its floor and leaves the
talker's loud frames nearly as they were; the floor holds; a print with no
energy up top lets the floor go deeper there; off, the combiner is the
combiner.

Run:  python -m pytest aether_gate/tests/test_postfilter.py
"""
import numpy as np

from aether_gate.core.postfilter import PostFilter, FLOOR_DB, PRINT_EXTRA_DB
from aether_gate.core.subband import SubbandCombiner, NFFT

RATE = 25_000.0
BLOCK = 2000
S = np.array([0.8, 0.6 * np.exp(1j * 0.7)])


def _band_noise(rng, n, lo, hi, power):
    X = np.zeros(n, dtype=np.complex128)
    f = np.fft.fftfreq(n, 1.0 / RATE)
    sel = (f >= lo) & (f < hi)
    X[sel] = rng.normal(size=sel.sum()) + 1j * rng.normal(size=sel.sum())
    x = np.fft.ifft(X)
    return x * np.sqrt(power / max(1e-30, np.mean(np.abs(x) ** 2)))


def _scene(rng, seconds, snr_db, talking):
    """(pa, pb) blocks: white noise on each loop, plus a syllabic 300-2700 Hz
    talker on both through S when talking."""
    n = int(seconds * RATE)
    na = _band_noise(rng, n, 100.0, 3000.0, 1.0)
    nb = _band_noise(rng, n, 100.0, 3000.0, 1.0)
    if not talking:
        return na, nb
    t = np.arange(n) / RATE
    v = _band_noise(rng, n, 300.0, 2700.0, 10 ** (snr_db / 10)) * (0.55 + 0.45 * np.sin(2 * np.pi * 4 * t))
    return na + S[0] * v, nb + S[1] * v


def _run(post, rng, snr_db=10.0):
    sb = SubbandCombiner(RATE)
    sb.set_post(post)
    out = []
    for talking, seconds in ((False, 1.5), (True, 1.5), (False, 1.0)):
        pa, pb = _scene(rng, seconds, snr_db, talking)
        for i in range(0, len(pa) - BLOCK + 1, BLOCK):
            out.append(sb.process(pa[i:i + BLOCK], pb[i:i + BLOCK], S[1] / S[0], S, talking))
    y = np.concatenate(out)
    n = int(1.5 * RATE)
    return y[int(0.5 * RATE):n], y[n + NFFT // 2:n + NFFT // 2 + int(1.2 * RATE)], y[-int(0.7 * RATE):]


def _db(x):
    return 10 * np.log10(np.mean(np.abs(x) ** 2))


def test_the_noise_between_words_goes_to_the_floor_and_the_talker_stays():
    q0, l0, e0 = _run(False, np.random.default_rng(1))
    q1, l1, e1 = _run(True, np.random.default_rng(1))
    assert _db(q0) - _db(q1) >= 4.0                      # the quiet before the over
    assert _db(e0) - _db(e1) >= 4.0                      # and after it
    assert _db(l0) - _db(l1) <= 3.5                      # the over keeps its level...
    c = abs(np.vdot(l0, l1)) / np.sqrt(np.vdot(l0, l0).real * np.vdot(l1, l1).real)
    assert c >= 0.9                                      # ...and its shape


def test_the_floor_is_the_floor():
    pf = PostFilter(RATE, NFFT, NFFT // 2, floor_db=-3.0)
    rng = np.random.default_rng(2)
    for _ in range(20):
        Xa = rng.normal(size=NFFT) + 1j * rng.normal(size=NFFT)
        Xb = rng.normal(size=NFFT) + 1j * rng.normal(size=NFFT)
        g = pf.gain(Xa, Xb, S)
    assert np.all(g >= 10 ** (-3.0 / 20) - 1e-9) and np.all(g <= 1.0)
    assert -3.1 <= pf.status()["mean_db"] <= -2.0


def test_a_print_with_nothing_up_top_lets_the_floor_go_deeper_there():
    pf = PostFilter(RATE, NFFT, NFFT // 2)
    rng = np.random.default_rng(3)
    profile = [0.0] * 15 + [-45.0] * 17                  # nothing above 1.5 kHz
    for _ in range(20):
        Xa = rng.normal(size=NFFT) + 1j * rng.normal(size=NFFT)
        Xb = rng.normal(size=NFFT) + 1j * rng.normal(size=NFFT)
        g = pf.gain(Xa, Xb, S, profile)
    lo = 20 * np.log10(np.mean(g[(pf.f > 300) & (pf.f < 1200)]))
    hi = 20 * np.log10(np.mean(g[(pf.f > 1800) & (pf.f < 3000)]))
    assert abs(lo - FLOOR_DB) < 0.5 and abs(hi - (FLOOR_DB + PRINT_EXTRA_DB)) < 0.5


def test_a_loop_the_talker_barely_reaches_is_no_witness():
    pf = PostFilter(RATE, NFFT, NFFT // 2)
    g = pf.gain(np.ones(NFFT, dtype=complex), np.ones(NFFT, dtype=complex), np.array([1.0, 0.01]))
    assert np.all(g == 1.0)
