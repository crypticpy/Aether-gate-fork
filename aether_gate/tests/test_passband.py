#
# Aether-gate — passband phase flatness.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Run:  python -m pytest aether_gate/tests/test_passband.py"""
import numpy as np
import pytest

from aether_gate.core.passband import PassbandPhase

RATE = 125_000.0
N = 4096


def _bins(rng, delay_samples, gain=0.7, phase=0.4):
    f = np.fft.fftfreq(N, 1.0 / RATE)
    sel = (f >= 0) & (f < 3000)
    Xa = rng.normal(size=sel.sum()) + 1j * rng.normal(size=sel.sum())
    Xb = Xa * gain * np.exp(1j * phase) * np.exp(-2j * np.pi * f[sel] * delay_samples / RATE)
    return Xa, Xb, f[sel]


def test_nothing_before_a_voiced_block():
    pb = PassbandPhase(RATE)
    assert pb.status() is None
    Xa, Xb, f = _bins(np.random.default_rng(1), 0)
    pb.update(Xa, Xb, f, N, voiced=False)
    assert pb.status() is None


def test_one_weight_fits_a_flat_passband():
    rng = np.random.default_rng(2)
    pb = PassbandPhase(RATE)
    for _ in range(10):
        pb.update(*_bins(rng, 0), N, voiced=True)
    s = pb.status()
    assert s["flatness"] > 0.99
    assert abs(s["phase_slope_deg_per_khz"]) < 1.0
    assert s["coherence"] > 0.99


def test_a_delay_between_antennas_shows_as_a_slope_and_lost_flatness():
    rng = np.random.default_rng(3)
    pb = PassbandPhase(RATE)
    for _ in range(10):
        pb.update(*_bins(rng, 20), N, voiced=True)      # 20 samples = 160 us
    s = pb.status()
    # 360 deg * 160 us * 1 kHz = 57.6 deg/kHz; B lags, so S = Xa conj(Xb) leads
    assert abs(s["phase_slope_deg_per_khz"]) == pytest.approx(57.6, abs=3.0)
    # phase spread of +-86 deg over the band: sin(1.5)/1.5
    assert 0.55 < s["flatness"] < 0.78, s


def test_bins_show_where_the_phase_slopes():
    rng = np.random.default_rng(5)
    rate, n = 125_000.0, 4096
    pb = PassbandPhase(rate)
    f = np.fft.fftfreq(n, 1.0 / rate)
    sel = (f >= 0) & (f < 3000)
    for _ in range(20):
        X = rng.normal(size=sel.sum()) + 1j * rng.normal(size=sel.sum())
        # 20-sample delay on B: phase ramps across the band
        Xb = X * np.exp(-2j * np.pi * f[sel] * 20 / rate)
        pb.update(X, Xb, f[sel], n, True)
    st = pb.status()
    bins = st["bins"]
    assert len(bins) == PassbandPhase.BINS
    ph = [b["phase_deg"] for b in bins]
    assert all(p is not None for p in ph)
    assert ph[0] < 0 < ph[-1] and ph[-1] - ph[0] > 100      # monotone ramp, ~170 deg span
    assert all(abs(p - q) < 40 for p, q in zip(ph, ph[1:]))
    assert all(b["coherence"] > 0.95 for b in bins)
