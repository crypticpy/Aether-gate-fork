#
# Aether-gate -- sub-band maximal-ratio combining, no hardware.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""BinWeights against synthetic scenes and against one real two-loop capture.

The claims worth holding it to:

  * where the noise is white and equal on both loops the per-bin weights ARE
    the broadband weight, and the output is the broadband combine sample for
    sample (the refinement must cost nothing where there is nothing to gain);
  * a coherent local source filling part of the slice, arriving from a
    direction the talker is not in, is nulled in its own bins while the
    talker's level does not move;
  * a stale patch of the map falls back to the broadband weight, and the
    talker's gain is identical either side of the boundary, so nothing steps;
  * the same scene at 62.5 kS/s, 250 kS/s and 2.04 MS/s scores the same;
  * a stream fed in pieces is the stream fed whole;
  * on the real capture the answer is never worse than the broadband combine,
    and the status it reports is JSON.

Run:  python -m pytest aether_gate/tests/test_binweights.py
"""
import json
import math
import os

import numpy as np
import pytest

from aether_gate.core.binweights import BinWeights, steering_for_weight
from aether_gate.core.diversity import combine, fit_max_snr
from aether_gate.core import spatial

CAPTURE = os.path.expanduser(
    "~/aether-gate-captures/20260903-030404_3891250Hz_125000sps.npz")
MAP_STEP_HZ = 62_500.0 / 2048          # 30.5 Hz, the live map's bin at 62.5 kS/s
SLICE_HALF_HZ = 1350.0                 # a 2.7 kHz slice


def _band_noise(rng, n, rate, lo, hi, power):
    """Complex noise confined to [lo, hi), with the given mean power."""
    f = np.fft.fftfreq(n, 1.0 / rate)
    sel = (f >= lo) & (f < hi)
    X = np.zeros(n, dtype=np.complex128)
    X[sel] = rng.normal(size=int(sel.sum())) + 1j * rng.normal(size=int(sel.sum()))
    x = np.fft.ifft(X)
    return x * math.sqrt(power / max(1e-30, float(np.mean(np.abs(x) ** 2))))


def _band_power(x, rate, lo, hi):
    """Mean power of x inside [lo, hi]."""
    X = np.fft.fft(x)
    f = np.fft.fftfreq(len(x), 1.0 / rate)
    sel = (f >= lo) & (f <= hi)
    return float(np.sum(np.abs(X[sel]) ** 2)) / len(x) ** 2


def _bin_covariance(a, b, nfft, hop=None):
    """Per-bin 2x2 covariance of a pair, ASCENDING frequency: what a spatial
    map holds, averaged over every frame rather than floor-tracked."""
    hop = hop or nfft
    win = np.hanning(nfft)
    R = np.zeros((nfft, 2, 2), dtype=np.complex128)
    k = 0
    for i in range(0, len(a) - nfft + 1, hop):
        X = np.fft.fft(np.stack([a[i:i + nfft], b[i:i + nfft]]) * win, axis=1)
        R += np.einsum("ik,jk->kij", X, np.conj(X))
        k += 1
    return np.fft.fftshift(R / max(k, 1), axes=0)


def _scene(rng, rate, seconds, talker_s, inter_d, inter_lo, inter_hi,
           p_talk=1.0, p_inter=100.0, p_white=0.5):
    """A talker across the whole slice, one coherent source across part of it
    from another direction, and independent noise on each loop."""
    n = int(rate * seconds)
    u = _band_noise(rng, n, rate, -SLICE_HALF_HZ, SLICE_HALF_HZ, p_talk)
    q = _band_noise(rng, n, rate, inter_lo, inter_hi, p_inter)
    wa = _band_noise(rng, n, rate, -rate / 2, rate / 2, p_white)
    wb = _band_noise(rng, n, rate, -rate / 2, rate / 2, p_white)
    talk = (talker_s[0] * u, talker_s[1] * u)
    noise = (inter_d[0] * q + wa, inter_d[1] * q + wb)
    return talk, noise


def _fresh(rate, m, nbins_map, R, stale=None):
    bw = BinWeights(rate, nbins_map, lo_hz=-SLICE_HALF_HZ, hi_hz=SLICE_HALF_HZ)
    bw.set_weight(m)
    bw.set_covariance(R, stale, -rate / 2.0, rate / nbins_map)
    return bw


def test_white_noise_bins_reproduce_the_broadband_weight_exactly():
    rate, n = 62_500.0, 8192
    m = 0.8 * np.exp(-0.6j)
    R = np.tile(np.eye(2, dtype=np.complex128), (2048, 1, 1))
    bw = _fresh(rate, m, 2048, R)
    _f, W = bw.bin_weights()
    v_wb = np.array([1.0, m]) / math.sqrt(1.0 + abs(m) ** 2)
    assert np.abs(W - v_wb).max() <= 0.01 * np.abs(v_wb).max()

    rng = np.random.default_rng(7)
    u = _band_noise(rng, n, rate, -SLICE_HALF_HZ, SLICE_HALF_HZ, 1.0)
    s = steering_for_weight(m)
    a = s[0] * u + 0.1 * _band_noise(rng, n, rate, -rate / 2, rate / 2, 1.0)
    b = s[1] * u + 0.1 * _band_noise(rng, n, rate, -rate / 2, rate / 2, 1.0)
    y = bw.apply(a, b)
    ref = combine(a, b, m)
    assert np.abs(y[bw.h:] - ref[bw.h:len(y)]).max() < 1e-3


def test_a_coherent_source_in_part_of_the_slice_is_nulled_and_the_talker_is_not():
    rate, seconds = 62_500.0, 2.0
    rng = np.random.default_rng(11)
    s = np.array([1.0, 0.8 * np.exp(0.6j)])
    d = np.array([1.0, 0.9 * np.exp(2.2j)])
    m = complex(np.conj(s[1]))                      # the talker's MRC weight
    talk, noise = _scene(rng, rate, seconds, s, d, -SLICE_HALF_HZ, -350.0)
    R = _bin_covariance(noise[0], noise[1], 2048)

    bw = _fresh(rate, m, 2048, R)
    yn = bw.apply(*noise)
    bw.reset()
    yt = bw.apply(*talk)
    lo, hi = -SLICE_HALF_HZ, SLICE_HALF_HZ
    k = slice(bw.h, len(yn))

    def gain_db(y, ref):
        return 10.0 * math.log10(_band_power(np.asarray(ref)[k], rate, lo, hi)
                                 / _band_power(y[k], rate, lo, hi))
    noise_db = gain_db(yn, combine(noise[0], noise[1], m))
    talk_db = gain_db(yt, combine(talk[0], talk[1], m))
    assert noise_db >= 6.0, f"only {noise_db:.1f} dB of noise reduction"
    assert abs(talk_db) <= 1.0, f"the talker moved by {talk_db:.2f} dB"
    assert bw.status()["gain_over_broadband_db"] >= 6.0


def test_a_stale_patch_falls_back_with_no_step_in_the_talker_level():
    rate = 62_500.0
    rng = np.random.default_rng(13)
    s = np.array([1.0, 0.8 * np.exp(0.6j)])
    d = np.array([1.0, 0.9 * np.exp(2.2j)])
    m = complex(np.conj(s[1]))
    _talk, noise = _scene(rng, rate, 1.0, s, d, -SLICE_HALF_HZ, -350.0)
    R = _bin_covariance(noise[0], noise[1], 2048)
    stale = np.zeros(2048, dtype=bool)
    stale[1024:] = True                              # everything above DC is old
    bw = _fresh(rate, m, 2048, R, stale)
    f, W = bw.bin_weights()

    v_wb = np.array([1.0, m]) / math.sqrt(1.0 + abs(m) ** 2)
    old = (f > 200.0) & (f <= SLICE_HALF_HZ)         # stale, clear of the smoother
    assert old.any()
    assert np.abs(W[old] - v_wb).max() < 1e-9

    # every bin passes the talker at exactly the broadband gain, so the level
    # cannot step at the stale boundary or at the edge of the slice
    g = W @ s
    g_wb = complex(v_wb @ s)
    assert np.abs(g - g_wb).max() < 1e-9 * abs(g_wb)
    step = np.abs(np.diff(W, axis=0)).max()
    assert step < 0.25 * np.abs(W).max(), f"weights jump by {step:.3f} bin to bin"
    st = bw.status()
    assert st["bins_stale"] > 0 and st["bins_used"] > 0


def test_the_same_scene_scores_the_same_at_62k_250k_and_2m():
    """The map is described in Hz, so the answer must not care about the rate.
    Built analytically: generating two seconds at 2.04 MS/s to learn a
    covariance whose closed form we already have would only be slower."""
    d = np.array([1.0, 0.9 * np.exp(2.2j)])
    m = complex(np.conj(0.8 * np.exp(0.6j)))
    want_nfft = {62_500.0: 2048, 250_000.0: 8192, 2_040_000.0: 65536}
    gains = []
    for rate in (62_500.0, 250_000.0, 2_040_000.0):
        nb = int(round(rate / MAP_STEP_HZ))
        f = -rate / 2.0 + np.arange(nb) * (rate / nb)
        R = np.tile(0.01 * np.eye(2, dtype=np.complex128), (nb, 1, 1))
        R[(f >= -SLICE_HALF_HZ) & (f < -350.0)] += np.outer(d, np.conj(d))
        bw = _fresh(rate, m, nb, R)
        st = bw.status()
        assert st["nfft"] == want_nfft[rate]
        assert 25.0 <= st["frame_ms"] <= 40.0
        gains.append(st["gain_over_broadband_db"])

        rng = np.random.default_rng(3)
        n = 4 * bw.n
        a = rng.normal(size=n) + 1j * rng.normal(size=n)
        b = rng.normal(size=n) + 1j * rng.normal(size=n)
        y = bw.apply(a, b)
        assert len(y) == ((n - bw.n) // bw.h + 1) * bw.h and np.isfinite(y).all()
    assert max(gains) - min(gains) <= 0.5, gains
    assert min(gains) >= 10.0, gains


def test_a_stream_fed_in_pieces_is_the_stream_fed_whole():
    rate = 62_500.0
    rng = np.random.default_rng(17)
    s = np.array([1.0, 0.8 * np.exp(0.6j)])
    d = np.array([1.0, 0.9 * np.exp(2.2j)])
    m = complex(np.conj(s[1]))
    _talk, noise = _scene(rng, rate, 0.5, s, d, -SLICE_HALF_HZ, -350.0)
    R = _bin_covariance(noise[0], noise[1], 2048)
    a, b = noise

    whole = _fresh(rate, m, 2048, R).apply(a, b)
    piece = _fresh(rate, m, 2048, R)
    out = []
    i = 0
    for step in (700, 4096, 33, 8192, 1500):
        out.append(piece.apply(a[i:i + step], b[i:i + step]))
        i += step
    out.append(piece.apply(a[i:], b[i:]))
    joined = np.concatenate(out)
    assert len(joined) == len(whole)
    assert np.abs(joined - whole).max() < 1e-9


@pytest.mark.skipif(not os.path.exists(CAPTURE), reason="no capture on this machine")
def test_the_real_capture_is_never_worse_than_the_broadband_combine(capsys):
    d = np.load(CAPTURE)
    a = d["a"].astype(np.complex128)
    b = d["b"].astype(np.complex128)
    rate = float(d["rate_hz"])
    center = float(d["center_hz"])
    slice_hz = float(d["slice_hz"])
    nb = 2048                                        # the live map's resolution

    # the map as the reader builds it: floor-tracked, frame by frame
    smap = spatial.SpatialMap(nb, rate)
    win = np.hanning(nb)
    frame_s = nb / rate
    for i in range(0, len(a) - nb + 1, nb):
        smap.update(np.fft.fft(np.stack([a[i:i + nb], b[i:i + nb]]) * win, axis=1), frame_s)
    R = np.fft.fftshift(smap.R, axes=0)
    stale = np.fft.fftshift(smap._stale) * frame_s > spatial.STALE_S

    # the weight the tracker would be on: max-SNR from the slice against its
    # own guard bands
    X = np.fft.fft(np.stack([a[:1 << 17], b[:1 << 17]]) * np.hanning(1 << 17), axis=1)
    f = np.fft.fftfreq(1 << 17, 1.0 / rate) + center
    inb = (f >= slice_hz - SLICE_HALF_HZ) & (f <= slice_hz + SLICE_HALF_HZ)
    guard = ((f >= slice_hz + 1500.0) & (f < slice_hz + 4200.0)) | \
            ((f <= slice_hz - 1500.0) & (f > slice_hz - 4200.0))
    R_in = spatial.region_covariance(X, np.flatnonzero(inb))
    R_n = spatial.region_covariance(X, np.flatnonzero(guard), trim=True)
    R_n = R_n + 1e-3 * float(np.real(np.trace(R_n))) / 2.0 * np.eye(2)
    m = fit_max_snr(R_in, R_n)

    bw = BinWeights(rate, nb, center_hz=center,
                    lo_hz=slice_hz - SLICE_HALF_HZ, hi_hz=slice_hz + SLICE_HALF_HZ)
    bw.set_weight(m)
    bw.set_covariance(R, stale, center - rate / 2.0, rate / nb)
    st = bw.status()
    with capsys.disabled():
        print(f"\n  real capture: m={m:.3f} {st}")
    assert st["gain_over_broadband_db"] >= -0.5, st
    assert st["bins_used"] > 0
    assert json.loads(json.dumps(st)) == st

    y = bw.apply(a, b)
    assert len(y) == ((len(a) - bw.n) // bw.h + 1) * bw.h
    assert np.isfinite(y).all()
