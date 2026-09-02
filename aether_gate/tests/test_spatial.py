#
# Aether-gate — the per-bin noise map on synthetic two-channel spectra.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Run:  .venv/bin/python -m pytest aether_gate/tests/test_spatial.py -q"""
import numpy as np
import pytest

from aether_gate.core.spatial import (
    SOURCE_MIN_COHERENCE, SpatialMap, region_covariance,
)
from aether_gate.core.diversity import combine

RATE = 125_000.0
NB = 1024


def _noise(rng, n, p=1.0):
    return np.sqrt(p / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def _frame(rng, sources, white=1.0):
    """One block of both channels -> spectra (2, NB). sources: (lo_hz, hi_hz,
    angle, gain_b, power): band-limited coherent noise between lo and hi."""
    a = _noise(rng, NB, white); b = _noise(rng, NB, white)
    f = np.fft.fftfreq(NB, 1 / RATE)
    for lo, hi, ang, gb, p in sources:
        w = np.fft.fft(_noise(rng, NB, p))
        w[(f < lo) | (f > hi)] = 0
        wave = np.fft.ifft(w)
        a = a + wave
        b = b + gb * np.exp(1j * ang) * wave
    return np.vstack([np.fft.fft(a), np.fft.fft(b)]) / np.sqrt(NB)


def _feed(sm, rng, frames, sources, white=1.0):
    for _ in range(frames):
        sm.update(_frame(rng, sources, white), NB / RATE)


def test_coherence_is_high_only_where_a_shared_source_lives():
    rng = np.random.default_rng(1)
    sm = SpatialMap(NB, RATE)
    _feed(sm, rng, 200, [(10_000, 20_000, 1.2, 0.8, 30.0)])
    coh = sm.coherence()
    f = np.fft.fftfreq(NB, 1 / RATE)
    inside = (f > 11_000) & (f < 19_000)
    outside = (f < -5_000) | (f > 40_000)
    assert coh[inside].mean() > 0.9, coh[inside].mean()
    assert coh[outside].mean() < 0.3, coh[outside].mean()


def test_sources_are_listed_with_a_null_that_works():
    rng = np.random.default_rng(2)
    sm = SpatialMap(NB, RATE)
    srcs = [(10_000, 20_000, 1.2, 0.8, 30.0), (-40_000, -30_000, -2.0, 1.1, 10.0)]
    _feed(sm, rng, 200, srcs)
    out = sm.sources(center_hz=3_700_000.0)
    assert len(out) == 2, out
    loud = out[0]
    assert abs(loud["lo_hz"] - 3_710_000) < 1_000 and abs(loud["hi_hz"] - 3_720_000) < 1_000
    assert loud["coherence"] >= SOURCE_MIN_COHERENCE and loud["level_db"] > out[1]["level_db"]
    # apply its null to fresh data from the same source: > 15 dB down
    m = 10 ** (loud["ratio_db"] / 20) * np.exp(1j * np.radians(loud["phase_deg"]))
    X = _frame(rng, srcs[:1], white=0.001)
    f = np.fft.fftfreq(NB, 1 / RATE)
    band = (f > 11_000) & (f < 19_000)
    p_a = np.mean(np.abs(X[0, band]) ** 2)
    p_y = np.mean(np.abs(combine(X[0, band], X[1, band], m)) ** 2)
    assert 10 * np.log10(p_a / p_y) > 15.0


def test_two_adjacent_sources_from_different_bearings_are_split():
    rng = np.random.default_rng(3)
    sm = SpatialMap(NB, RATE)
    _feed(sm, rng, 200, [(10_000, 20_000, 0.3, 1.0, 30.0), (20_000, 30_000, 2.5, 1.0, 30.0)])
    out = sm.sources()
    assert len(out) == 2, out
    assert abs(out[0]["phase_deg"] - out[1]["phase_deg"]) % 360 > 60


def test_a_passing_station_does_not_enter_the_floor_map():
    rng = np.random.default_rng(4)
    sm = SpatialMap(NB, RATE)
    _feed(sm, rng, 250, [])                                  # isotropic floor only (past warm-up)
    _feed(sm, rng, 60, [(0, 3_000, 0.5, 1.0, 100.0)])         # a 20 dB station for 0.5 s
    coh = sm.coherence()
    f = np.fft.fftfreq(NB, 1 / RATE)
    assert coh[(f > 500) & (f < 2_500)].mean() < 0.3
    assert sm.sources() == []


def test_a_band_that_gets_noisier_is_followed_eventually():
    rng = np.random.default_rng(5)
    sm = SpatialMap(NB, RATE)
    _feed(sm, rng, 50, [], white=1.0)
    lvl0 = sm.map()["level_db"]
    _feed(sm, rng, int(45 * RATE / NB), [], white=10.0)      # 45 s at +10 dB
    lvl1 = sm.map()["level_db"]
    assert np.mean(lvl1) - np.mean(lvl0) > 6.0


def test_nulled_weights_fall_back_where_nothing_is_coherent():
    rng = np.random.default_rng(6)
    sm = SpatialMap(NB, RATE)
    _feed(sm, rng, 100, [(10_000, 20_000, 1.2, 0.8, 30.0)])
    m = sm.null_weights(fallback=0.5j)
    f = np.fft.fftfreq(NB, 1 / RATE)
    assert np.all(m[(f < -20_000)] == 0.5j)
    assert np.all(m[(f > 12_000) & (f < 18_000)] != 0.5j)


def test_map_is_decimated_and_ordered_low_to_high():
    rng = np.random.default_rng(7)
    sm = SpatialMap(NB, RATE)
    _feed(sm, rng, 50, [(40_000, 50_000, 1.0, 1.0, 30.0)])
    d = sm.map(center_hz=7_000_000.0, points=64)
    assert len(d["coherence"]) == 64 and len(d["level_db"]) == 64
    assert d["start_hz"] == 7_000_000.0 - RATE / 2 and abs(d["step_hz"] - RATE / 64) < 1e-6
    peak = int(np.argmax(d["coherence"]))
    assert 7_040_000 < d["start_hz"] + peak * d["step_hz"] < 7_050_000
    assert d["sources"] and abs(d["sources"][0]["lo_hz"] - 7_040_000) < 2_000


def test_region_covariance_trims_a_station_out_of_a_guard_band():
    rng = np.random.default_rng(8)
    X = _frame(rng, [(5_000, 6_000, 1.0, 1.0, 100.0)], white=1.0)
    f = np.fft.fftfreq(NB, 1 / RATE)
    idx = np.flatnonzero((f > 3_000) & (f < 9_000))
    R_all = region_covariance(X, idx)
    R_trim = region_covariance(X, idx, trim=True)
    assert np.real(R_all[0, 0]) > 5 * np.real(R_trim[0, 0])
    assert abs(R_trim[0, 1]) < 0.3 * np.real(R_trim[0, 0])
    assert region_covariance(X, []) is None


def test_trimmed_covariance_reads_as_the_noise_mean_not_the_quiet_half():
    """For noise-only bins the quieter half averages (1 - ln 2) of the mean;
    the trimmed estimate must be rescaled back, or a VAD referenced to it
    never goes quiet. With a station in the band the trim still excludes it."""
    rng = np.random.default_rng(11)
    X = (rng.normal(size=(2, 4000)) + 1j * rng.normal(size=(2, 4000))) * np.sqrt(0.5)
    idx = np.arange(4000)
    full = region_covariance(X, idx)
    trimmed = region_covariance(X, idx, trim=True)
    assert np.real(np.trace(trimmed)) == pytest.approx(np.real(np.trace(full)), rel=0.08)
    X[:, 1000:1400] *= 30.0                                  # a strong station in 10% of the bins
    hot = region_covariance(X, idx)
    trimmed = region_covariance(X, idx, trim=True)
    assert np.real(np.trace(trimmed)) == pytest.approx(np.real(np.trace(full)), rel=0.08)
    assert np.real(np.trace(hot)) > 10 * np.real(np.trace(full))
