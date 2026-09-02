#
# Aether-gate — the per-bin passband combiner, no hardware.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""SubbandCombiner against synthetic passband pairs at 24 kHz:

  * white noise only: the output is the wideband combiner's, sample for
    sample (the STFT emits late but does not shift) (the refinement must cost nothing
    where there is nothing to refine);
  * two coherent noise sources from different directions in different
    frequency ranges: one wideband weight can null only one of them, the
    per-bin weights null both, and the talker comes through flat.

Run:  python -m pytest aether_gate/tests/test_subband.py
"""
import numpy as np
import pytest

from aether_gate.core.diversity import combine, fit_max_snr
from aether_gate.core.subband import SubbandCombiner, NFFT

RATE = 24_000.0
BLOCK = 800


def _band_noise(rng, n, lo, hi, power):
    X = np.zeros(n, dtype=np.complex128)
    f = np.fft.fftfreq(n, 1.0 / RATE)
    sel = (f >= lo) & (f < hi)
    X[sel] = rng.normal(size=sel.sum()) + 1j * rng.normal(size=sel.sum())
    x = np.fft.ifft(X)
    return x * np.sqrt(power / max(1e-30, np.mean(np.abs(x) ** 2)))


def _run(comb, pa, pb, m, s, talk):
    """Feed in BLOCK-sized pieces; talk is a per-sample bool array. The
    output is re-aligned so y[k] is the combiner's answer for input k (the
    STFT emits late, it does not shift)."""
    pend = len(comb._in_a)
    out = []
    for i in range(0, len(pa) - BLOCK + 1, BLOCK):
        out.append(comb.process(pa[i:i + BLOCK], pb[i:i + BLOCK], m, s, bool(talk[i:i + BLOCK].any())))
    return np.concatenate(out)[pend:]


def test_white_noise_only_reproduces_the_wideband_combiner_exactly():
    rng = np.random.default_rng(1)
    n = BLOCK * 60
    a = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)
    b = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2)
    m = 0.7 * np.exp(1j * 0.9)
    s = np.array([1.0, np.conj(m)])                 # the MRC steering for that weight
    comb = SubbandCombiner(RATE)
    y = _run(comb, a, b, m, s, np.zeros(n, dtype=bool))
    ref = combine(a, b, m)
    lo, hi = 5 * NFFT, len(y) - NFFT           # the settled middle
    err = np.max(np.abs(y[lo:hi] - ref[lo:hi]))
    assert err < 1e-3, err
    assert comb.refined_bins == 0 and abs(comb.extra_db) < 0.2


def test_two_directional_noises_are_nulled_per_bin_and_the_talker_stays_flat():
    rng = np.random.default_rng(2)
    n = BLOCK * 150                                   # 5 s
    s = np.array([1.0, 0.8 * np.exp(1j * 1.0)])       # the talker
    n1 = np.array([1.0, 1.0 * np.exp(-1j * 2.0)])     # a hum comb's direction, below 1.2 kHz
    n2 = np.array([1.0, 0.9 * np.exp(1j * 2.6)])      # a het's direction, above 1.6 kHz
    talk = np.zeros(n, dtype=bool)
    talk[: n // 2] = True                             # first half voiced, second half silent
    voice = _band_noise(rng, n, 300.0, 2700.0, 4.0) * talk
    hum = _band_noise(rng, n, 100.0, 1200.0, 6.0)
    het = _band_noise(rng, n, 1600.0, 2600.0, 6.0)
    white_a = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2) * 0.3
    white_b = (rng.normal(size=n) + 1j * rng.normal(size=n)) / np.sqrt(2) * 0.3
    a = s[0] * voice + n1[0] * hum + n2[0] * het + white_a
    b = s[1] * voice + n1[1] * hum + n2[1] * het + white_b
    # the wideband tracker's weight from full-band covariances
    Rn = np.cov(np.stack([a[n // 2:], b[n // 2:]]), bias=True)
    Rs = np.cov(np.stack([a[: n // 2], b[: n // 2]]), bias=True)
    m = fit_max_snr(Rs, Rn)
    comb = SubbandCombiner(RATE)
    # let it learn on the silent half first, then judge a second pass
    _run(comb, a[n // 2:], b[n // 2:], m, s, talk[n // 2:])
    y_silent = _run(comb, a[n // 2:], b[n // 2:], m, s, talk[n // 2:])
    ref_silent = combine(a[n // 2:], b[n // 2:], m)
    p_sb = np.mean(np.abs(y_silent[4 * NFFT:]) ** 2)
    p_wb = np.mean(np.abs(ref_silent[4 * NFFT:]) ** 2)
    improvement = 10 * np.log10(p_wb / p_sb)
    assert improvement >= 6.0, improvement
    assert comb.refined_bins >= 20 and comb.extra_db >= 6.0
    # the talker through the frozen per-bin weights: flat within 1.5 dB of
    # what the wideband weight gives it (weights hold: nothing is learned
    # while talking, and the 0.3 s smoothing has settled)
    vo = _band_noise(rng, n // 2, 300.0, 2700.0, 4.0)
    y_v = _run(comb, s[0] * vo, s[1] * vo, m, s, np.ones(n // 2, dtype=bool))
    r_v = combine(s[0] * vo, s[1] * vo, m)
    L = min(len(y_v), len(r_v))
    Yv = np.abs(np.fft.rfft(y_v[4 * NFFT:L].real)) ** 2
    Rv = np.abs(np.fft.rfft(r_v[4 * NFFT:L].real)) ** 2
    f = np.fft.rfftfreq(L - 4 * NFFT, 1.0 / RATE)
    edges = np.linspace(400.0, 2600.0, 12)
    ratios = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (f >= lo) & (f < hi)
        ratios.append(10 * np.log10(np.sum(Yv[sel]) / np.sum(Rv[sel])))
    ratios = np.asarray(ratios)
    assert np.max(np.abs(ratios - np.mean(ratios))) <= 1.5, ratios
    assert abs(np.mean(ratios)) <= 1.5, ratios


def test_a_block_shorter_than_a_hop_still_flows():
    comb = SubbandCombiner(RATE)
    z = np.zeros(100, dtype=np.complex128)
    s = np.array([1.0, 1.0])
    total = 0
    for _ in range(20):
        total += len(comb.process(z, z, 0j, s, False))
    # between half a frame and a whole one stays pending, whole hops come out
    assert 20 * 100 - NFFT < total <= 20 * 100 - NFFT // 2 and total % (NFFT // 2) == 0
