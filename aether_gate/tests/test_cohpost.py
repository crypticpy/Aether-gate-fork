#
# Aether-gate — the coherence post-filter on the combined slice, no hardware.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""CoherencePostFilter and PauseGate against synthetic pairs, and once
against a real RSPduo capture:

  * a shaped, syllabic talker arriving in step on both loops over
    independent band noise: the filter digs it out by more than 2 dB over
    the plain combine, and the band outside the slice comes through
    untouched;
  * diffuse noise alone: there is nothing coherent to keep, and the gain
    sits on its floor;
  * two identical loops: the gain is one everywhere and the STFT is a wire;
  * the frame is a time, so 62.5 kS/s, 250 kS/s and 2.04 MS/s all get the
    same 10 ms frame, the same ~61 Hz bins and the same dig-out;
  * one long block is the same samples as many short ones;
  * the pause gate finds the gaps of a gated signal and learns the noise
    that is in them.

Run:  python -m pytest aether_gate/tests/test_cohpost.py
"""
import os

import numpy as np
import pytest

from aether_gate.core.cohpost import (
    CoherencePostFilter, PauseGate, frame_bins, FLOOR_DB,
)

RATE = 125_000.0                 # the gate's own rate on 80 m
LO, HI = 3_000.0, 5_700.0        # the slice: 2.7 kHz of phone, offset from DC
THETA = 0.9                      # the talker's phase between the loops
M = np.exp(-1j * THETA)          # ...and the weight that puts them in step
# Real RSPduo pairs off 80 m, with where the FINDER had the talker at the
# time. The files' own slice_hz is wherever the operator's slice happened to
# be parked (a CW signal on one of them), which is not where the voice is.
CAPTURES = (
    ("20260903-031814_3890250Hz_125000sps.npz", 3_906_000.0, "LSB"),   # a rag-chew
    ("20260903-030404_3891250Hz_125000sps.npz", 3_906_500.0, "LSB"),   # the same net, quiet
)


def _capture(name):
    return os.path.expanduser("~/aether-gate-captures/" + name)


def _band_noise(rng, n, rate, lo, hi, power, tilt_db=0.0):
    """Complex noise confined to [lo, hi), optionally sloping tilt_db across
    the band the way a voice does."""
    X = np.zeros(n, dtype=np.complex128)
    f = np.fft.fftfreq(n, 1.0 / rate)
    sel = (f >= lo) & (f < hi)
    k = sel.sum()
    X[sel] = rng.normal(size=k) + 1j * rng.normal(size=k)
    if tilt_db:
        shape = np.linspace(0.0, tilt_db, k)
        X[sel] *= 10.0 ** (shape[np.argsort(np.argsort(f[sel]))] / 20.0)
    x = np.fft.ifft(X)
    return x * np.sqrt(power / max(1e-30, float(np.mean(np.abs(x) ** 2))))


def _talker(rng, n, rate, snr_db, gap_s=0.4):
    """A shaped 2.7 kHz voice that keys on and off in gap_s blocks with a 4 Hz
    syllabic swing on top, scaled to snr_db against one unit of in-band noise:
    the pause gate gets gaps to learn in, and the Wiener gain gets both a
    spectrum and an envelope to be selective about."""
    v = _band_noise(rng, n, rate, LO, HI, 1.0, tilt_db=-15.0)
    t = np.arange(n) / rate
    on = (np.floor(t / gap_s).astype(int) % 2) == 0
    v = v * on * (0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t))
    return v * np.sqrt(10.0 ** (snr_db / 10.0)
                       / max(1e-30, float(np.mean(np.abs(v) ** 2))))


def _scene(rng, seconds, rate=RATE, snr_db=3.0, spur_hz=None):
    """(a, b, voice_as_combined): the talker on both loops at THETA, over
    independent band noise, plus an optional carrier outside the slice. The
    noise is spread over the whole span but scaled so that exactly one unit
    of it lands in the slice, which is what snr_db is against."""
    n = int(seconds * rate)
    v = _talker(rng, n, rate, snr_db)
    dens = rate / (HI - LO)
    a = v + _band_noise(rng, n, rate, -rate / 2, rate / 2, dens)
    b = v * np.exp(1j * THETA) + _band_noise(rng, n, rate, -rate / 2, rate / 2, dens)
    if spur_hz is not None:
        c = 3.0 * np.exp(2j * np.pi * spur_hz * np.arange(n) / rate)
        a, b = a + c, b + c
    return a, b, _combine(v, v * np.exp(1j * THETA))


def _combine(a, b):
    return (a + M * b) / np.sqrt(1.0 + abs(M) ** 2)


def _bandpass(x, rate, lo, hi):
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x), 1.0 / rate)
    return np.fft.ifft(np.where((f >= lo) & (f < hi), X, 0.0))


def _snr_db(y, ref):
    """How much of y is ref, over how much of it is not: the projection SNR,
    which does not care what gain the filter left behind."""
    g = np.vdot(ref, y) / np.vdot(ref, ref)
    err = y - g * ref
    sig = abs(g) ** 2 * float(np.vdot(ref, ref).real)
    return 10.0 * np.log10(sig / max(float(np.vdot(err, err).real), 1e-30))


def _feed(pf, y, a, b, block=4000, phase=None):
    out = [pf.process(y[i:i + block], a[i:i + block], b[i:i + block], phase)
           for i in range(0, len(y), block)]
    return np.concatenate(out)


def _dig_out(rng, seconds=2.5, rate=RATE, snr_db=3.0):
    """(improvement in dB, the filter) for one scene at one rate: what the
    post-filter's slice-band SNR is over the plain combine's, both measured
    against the talker as the combiner would have delivered them."""
    a, b, v = _scene(rng, seconds, rate, snr_db)
    y = _combine(a, b)
    pf = CoherencePostFilter(rate, LO, HI)
    z = _feed(pf, y, a, b, block=max(64, int(0.032 * rate)), phase=np.angle(M))
    d = pf.latency_samples
    skip = int(0.5 * rate)                       # the estimates settling
    end = len(z) - d
    ref = _bandpass(v, rate, LO, HI)[skip:end]
    plain = _bandpass(y, rate, LO, HI)[skip:end]
    filt = _bandpass(z[d:], rate, LO, HI)[skip:end]
    return _snr_db(filt, ref) - _snr_db(plain, ref), pf


def test_the_talker_comes_out_at_least_two_decibels_further_up():
    gain_db, pf = _dig_out(np.random.default_rng(1))
    assert gain_db >= 2.0, gain_db          # measured 3.4 dB; the oracle's is 5.6
    st = pf.status()
    assert st["coherence_mean"] > 0.3 and st["snr_out_db"] > st["snr_in_db"]
    assert st["pause_fraction"] > 0.1


def test_the_band_outside_the_slice_comes_through_untouched():
    rng = np.random.default_rng(2)
    a, b, _v = _scene(rng, 1.0, spur_hz=20_000.0)
    y = _combine(a, b)
    pf = CoherencePostFilter(RATE, LO, HI)
    z = _feed(pf, y, a, b, phase=np.angle(M))
    d = pf.latency_samples
    n = len(z) - d
    Y = np.fft.fft(y[:n]); Z = np.fft.fft(z[d:d + n])
    f = np.fft.fftfreq(n, 1.0 / RATE)
    out = (f < LO - 500.0) | (f >= HI + 500.0)
    err = np.sum(np.abs(Z[out] - Y[out]) ** 2) / np.sum(np.abs(Y[out]) ** 2)
    assert 10.0 * np.log10(max(err, 1e-30)) < -40.0, err
    k = int(np.argmin(np.abs(f - 20_000.0)))     # and the carrier itself
    assert abs(20.0 * np.log10(abs(Z[k]) / abs(Y[k]))) < 0.1


def test_diffuse_noise_alone_is_taken_down_to_the_floor():
    rng = np.random.default_rng(3)
    n = int(1.5 * RATE)
    a = _band_noise(rng, n, RATE, -RATE / 2, RATE / 2, 1.0)
    b = _band_noise(rng, n, RATE, -RATE / 2, RATE / 2, 1.0)
    pf = CoherencePostFilter(RATE, LO, HI)
    _feed(pf, _combine(a, b), a, b, phase=np.angle(M))
    st = pf.status()
    assert FLOOR_DB - 0.1 <= st["gain_mean_db"] <= FLOOR_DB + 1.5, st
    assert st["coherence_mean"] < 0.5


def test_a_gain_of_one_is_a_wire():
    rng = np.random.default_rng(4)
    n = int(0.5 * RATE)
    x = _band_noise(rng, n, RATE, -RATE / 2, RATE / 2, 1.0)
    pf = CoherencePostFilter(RATE, LO, HI)
    z = _feed(pf, x, x, x, block=3000, phase=0.0)         # two identical loops
    d = pf.latency_samples
    assert np.allclose(pf.g, 1.0)
    assert np.max(np.abs(z[d:] - x[:len(z) - d])) < 1e-4
    # ...and what the reader hands up (complex64) is what it gets back
    x32 = x.astype(np.complex64)
    pf32 = CoherencePostFilter(RATE, LO, HI)
    assert pf32.process(x32[:3000], x32[:3000], x32[:3000]).dtype == np.complex64


def test_the_same_frame_in_seconds_works_at_every_rate():
    seen = {}
    for rate in (62_500.0, 250_000.0, 2_040_000.0):
        gain_db, pf = _dig_out(np.random.default_rng(5), seconds=1.5, rate=rate)
        seen[rate] = (pf.n, round(pf.bin_hz), pf.frames, gain_db)
        assert gain_db >= 2.0, (rate, gain_db)
        assert pf.n == frame_bins(rate) and 60 <= pf.bin_hz <= 63
    frames = [v[2] for v in seen.values()]        # a frame is a time: the same
    assert max(frames) / min(frames) < 1.05, seen  # count of them at every rate


def test_one_long_block_is_the_same_samples_as_many_short_ones():
    rng = np.random.default_rng(6)
    a, b, _v = _scene(rng, 0.5)
    y = _combine(a, b)
    one = CoherencePostFilter(RATE, LO, HI)
    many = CoherencePostFilter(RATE, LO, HI)
    z1 = one.process(y, a, b, np.angle(M))
    z2 = _feed(many, y, a, b, block=997, phase=np.angle(M))
    assert len(z1) == len(z2) == len(y)
    assert np.max(np.abs(z1 - z2)) < 1e-9


def _gated(gate, seconds, rng, on_s=0.5):
    """Half a second of a talker, half a second of band: energies and the
    per-bin spectrum of each, with a chi-square scatter on the noise."""
    noise = np.full(8, 2.0)
    for i in range(int(seconds / gate.hop_s)):
        on = (int(i * gate.hop_s / on_s) % 2) == 0
        psd = noise * rng.gamma(4.0, 0.25, size=8)
        gate.update((100.0 if on else 1.0) * 2.0, psd + (200.0 if on else 0.0))


def test_the_pause_gate_finds_the_gaps_and_learns_the_noise_in_them():
    gate = PauseGate(125_000.0, 1024)
    _gated(gate, 20.0, np.random.default_rng(7))      # pause_fraction is the last 10 s
    assert 0.38 <= gate.pause_fraction <= 0.50, gate.pause_fraction
    assert gate.gaps >= 15
    assert np.all(np.abs(10.0 * np.log10(gate.noise_psd / 2.0)) < 0.5), gate.noise_psd
    st = gate.status()
    assert st["in_pause"] is True and st["hold"] is True     # ends in a gap
    assert abs(st["noise_db"] - 3.0) < 0.5


def test_the_hold_flag_leads_the_confirmed_pause():
    gate = PauseGate(125_000.0, 1024)
    _gated(gate, 4.0, np.random.default_rng(8))       # a floor to be measured against
    while gate.hold:                                  # back into the middle of an over
        gate.update(200.0)
    gate.update(2.0)                                  # ...and the moment it stops
    assert gate.hold is True and gate.in_pause is False      # the dip has begun
    for _ in range(int(gate.min_gap_s / gate.hop_s) + 1):
        gate.update(2.0)
    assert gate.in_pause is True


def test_a_steady_signal_is_not_one_long_pause():
    gate = PauseGate(125_000.0, 1024)
    for _ in range(400):
        gate.update(100.0, np.full(8, 100.0))
    assert gate.in_pause is False and gate.hold is False
    assert gate.noise_psd is None


@pytest.mark.parametrize("name,voice_hz,mode", CAPTURES)
def test_a_real_capture_does_not_come_out_worse(name, voice_hz, mode):
    if not os.path.exists(_capture(name)):
        pytest.skip("no local capture " + name)
    d = np.load(_capture(name))
    a = d["a"].astype(np.complex128)
    b = d["b"].astype(np.complex128)
    rate, center = float(d["rate_hz"]), float(d["center_hz"])
    off = voice_hz - center
    lo, hi = (off - 3000.0, off) if mode == "LSB" else (off, off + 3000.0)
    # the weight the tracker would have: the talker's phase between the loops
    n = 4096
    nf = len(a) // n
    A = np.fft.fft(a[:nf * n].reshape(nf, n) * np.hanning(n), axis=1)
    B = np.fft.fft(b[:nf * n].reshape(nf, n) * np.hanning(n), axis=1)
    f = np.fft.fftfreq(n, 1.0 / rate)
    sel = (f >= lo) & (f < hi)
    phase = float(np.angle(np.sum((A * np.conj(B))[:, sel])))
    m = np.exp(1j * phase)
    y = (a + m * b) / np.sqrt(2.0)
    pf = CoherencePostFilter(rate, lo, hi)
    z = _feed(pf, y, a, b, block=4096, phase=phase)
    st = pf.status()
    assert len(z) == len(y)
    assert st["snr_in_db"] is not None and st["snr_out_db"] is not None
    assert st["snr_out_db"] >= st["snr_in_db"] - 0.1, st
    import json
    json.dumps(st)                                   # the control port must survive it
    json.dumps(pf.gate.status())
