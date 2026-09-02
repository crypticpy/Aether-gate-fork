#
# Aether-gate — diversity combining on synthetic two-channel data (no hardware).
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The maths in core/diversity.py, checked on data whose answer is known.

Run:  .venv/bin/python -m pytest aether_gate/tests/test_diversity.py -q
"""
import numpy as np

import pytest

from aether_gate.core.diversity import (
    ALIGN_MIN_PEAK, REFIT_MIN_GAIN_DB, TALK_HOLD_S, WEIGHT_MAX_ABS, Aligner,
    Tracker, combine, find_lag, fit_max_snr, fit_null, weight_from_polar,
    weight_to_polar,
)

RATE = 25_000.0


def _noise(rng, n, p=1.0):
    return np.sqrt(p / 2) * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def _two_channel(rng, n, sources, white=0.01):
    """sources: list of (waveform, steering_angle_rad, gain_b). Each source
    arrives on A as-is and on B rotated by the angle and scaled by gain_b,
    which is what a coherent pair of antennas does to a plane wave."""
    a = _noise(rng, n, white)
    b = _noise(rng, n, white)
    for wave, ang, gb in sources:
        a = a + wave
        b = b + gb * np.exp(1j * ang) * wave
    return a, b


# --- alignment ------------------------------------------------------------

def test_find_lag_recovers_a_positive_and_a_negative_offset():
    rng = np.random.default_rng(1)
    n = 20_000
    common = _noise(rng, n + 400)
    for lag in (17, -23):
        a = common[200:200 + n]
        b = common[200 - lag:200 - lag + n]        # a[n] == b[n + lag]
        a = a + _noise(rng, n, 0.3)
        b = b + _noise(rng, n, 0.3)
        got, peak = find_lag(a, b, 64)
        assert got == lag, (lag, got)
        assert peak >= ALIGN_MIN_PEAK, peak


def test_find_lag_reports_no_peak_for_unrelated_channels():
    rng = np.random.default_rng(2)
    a = _noise(rng, 20_000)
    b = _noise(rng, 20_000)
    _lag, peak = find_lag(a, b, 64)
    assert peak < ALIGN_MIN_PEAK, peak


def test_aligner_delays_the_early_channel_block_by_block():
    rng = np.random.default_rng(3)
    n = 4096
    stream = _noise(rng, 6 * n + 200)
    for lag in (7, -11):
        al = Aligner()
        al.set_lag(lag)
        outs_a, outs_b = [], []
        for k in range(6):
            a = stream[100 + k * n: 100 + (k + 1) * n]
            b = stream[100 - lag + k * n: 100 - lag + (k + 1) * n]   # a[n] == b[n+lag]
            oa, ob = al.apply(a, b)
            assert len(oa) == len(ob) == n
            outs_a.append(oa); outs_b.append(ob)
        A = np.concatenate(outs_a); B = np.concatenate(outs_b)
        skip = abs(lag)                              # the hold buffer starts empty
        assert np.allclose(A[skip:], B[skip:]), lag


def test_aligner_calibrate_adopts_only_a_credible_peak():
    rng = np.random.default_rng(4)
    common = _noise(rng, 20_200)
    a = common[100:20_100]; b = common[100 - 5:20_100 - 5]
    al = Aligner()
    lag, _peak, ok = al.calibrate(a, b, 64)
    assert ok and al.aligned and al.lag == 5 and lag == 5
    al2 = Aligner()
    _lag, _peak, ok = al2.calibrate(_noise(rng, 20_000), _noise(rng, 20_000), 64)
    assert not ok and not al2.aligned and al2.lag == 0


# --- weights ----------------------------------------------------------------

def test_combine_keeps_uncorrelated_noise_at_input_level_for_any_weight():
    rng = np.random.default_rng(5)
    a = _noise(rng, 200_000); b = _noise(rng, 200_000)
    p_in = float(np.mean(np.abs(a) ** 2))
    for m in (0j, 1.0 + 0j, 0.3j, 5.0 * np.exp(1j * 2.0), WEIGHT_MAX_ABS + 0j):
        p_out = float(np.mean(np.abs(combine(a, b, m)) ** 2))
        assert abs(p_out / p_in - 1.0) < 0.03, (m, p_out / p_in)


def test_polar_round_trip_and_cap():
    m = weight_from_polar(210.0, -6.0)
    ph, ra = weight_to_polar(m)
    assert abs(ph - 210.0) < 1e-9 and abs(ra + 6.0) < 1e-9
    assert abs(weight_from_polar(0.0, 40.0)) == WEIGHT_MAX_ABS
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            weight_from_polar(bad, 0.0)
        with pytest.raises(ValueError):
            weight_from_polar(0.0, bad)


def test_fit_null_kills_a_directional_noise_source():
    rng = np.random.default_rng(6)
    n = 50_000
    qrm = _noise(rng, n, 1.0)
    a, b = _two_channel(rng, n, [(qrm, 1.1, 0.8)], white=0.01)
    X = np.vstack([a, b]); Rn = (X @ X.conj().T) / n
    m = fit_null(Rn)
    p_a = float(np.mean(np.abs(a) ** 2))
    p_y = float(np.mean(np.abs(combine(a, b, m)) ** 2))
    assert 10 * np.log10(p_a / p_y) > 15.0, (p_a, p_y)


def test_fit_max_snr_beats_either_antenna_alone():
    rng = np.random.default_rng(7)
    n = 50_000
    sig = 0.3 * _noise(rng, n)                       # a talker from one direction
    qrm = _noise(rng, n, 1.0)                        # a noise source from another
    a_n, b_n = _two_channel(rng, n, [(qrm, 2.0, 0.9)], white=0.02)
    a_s, b_s = _two_channel(rng, n, [(sig, -0.7, 1.1)], white=0.0)
    a, b = a_n + a_s, b_n + b_s
    Xn = np.vstack([a_n, b_n]); Rn = (Xn @ Xn.conj().T) / n
    X = np.vstack([a, b]); Rs = (X @ X.conj().T) / n
    m = fit_max_snr(Rs, Rn)

    def snr(ya, yn):
        return 10 * np.log10(np.mean(np.abs(ya) ** 2) / np.mean(np.abs(yn) ** 2))
    s_a = snr(a_s, a_n); s_b = snr(b_s, b_n)
    s_out = snr(combine(a_s, b_s, m), combine(a_n, b_n, m))
    assert s_out > max(s_a, s_b) + 10.0, (s_a, s_b, s_out)


# --- the tracker ------------------------------------------------------------

def _run(tracker, rng, seconds, talker, mode, block=1024, white=0.5):
    """Stream `seconds` of alternating quiet (0.6 s) and talking (1.0 s) blocks.
    talker = (angle, gain_b) of the wanted signal; a fixed noise source sits
    at angle 2.0 over an isotropic (uncorrelated) floor, so the best weight
    depends on where the talker is and not only on where the noise is."""
    t = 0.0
    while t < seconds:
        phase = t % 1.6
        talking = phase >= 0.6
        qrm = _noise(rng, block, 1.0)
        srcs = [(qrm, 2.0, 0.9)]
        if talking:
            srcs.append((1.2 * _noise(rng, block), talker[0], talker[1]))
        a, b = _two_channel(rng, block, srcs, white)
        tracker.update(a, b, mode)
        t += block / RATE


def test_tracker_learns_a_null_between_overs_and_a_beam_on_the_talker():
    rng = np.random.default_rng(8)
    tr = Tracker(RATE)
    _run(tr, rng, 4.0, (-0.7, 1.1), "track")
    assert tr.updates >= 1
    snr = tr.snr_db()
    assert snr["out"] > max(snr["a"], snr["b"]) + 6.0, snr


def test_tracker_follows_a_new_talker_within_a_second():
    rng = np.random.default_rng(9)
    tr = Tracker(RATE)
    _run(tr, rng, 4.0, (-0.7, 1.1), "track")
    m_first = tr.m
    n_first = tr.updates
    _run(tr, rng, 1.6, (0.5, 1.0), "track")          # a different direction
    assert tr.updates > n_first
    assert abs(tr.m - m_first) > 0.1
    snr = tr.snr_db()
    assert snr["out"] > max(snr["a"], snr["b"]) + 3.0, snr


def test_tracker_does_not_chatter_on_a_steady_scene():
    rng = np.random.default_rng(10)
    tr = Tracker(RATE)
    _run(tr, rng, 4.0, (-0.7, 1.1), "null")
    n = tr.updates
    assert n >= 1
    _run(tr, rng, 4.0, (-0.7, 1.1), "null")
    assert tr.updates - n <= 2, (n, tr.updates)


def test_refit_declines_without_enough_gain():
    tr = Tracker(RATE)
    tr.Rn = np.eye(2, dtype=complex)                  # white, equal: nothing to null
    assert tr.refit("null") is False
    assert tr.m == 0j and tr.updates == 0
    assert REFIT_MIN_GAIN_DB > 0


def test_null_refuses_to_select_the_quieter_antenna():
    """Uncorrelated noise has nothing to null: with channel B dead (or just
    quieter) the smallest eigenvector is 'B alone', which must not be adopted."""
    for pb in (0.5, 0.1, 1e-4):
        tr = Tracker(RATE)
        tr.Rn = np.diag([1.0, pb]).astype(complex)
        assert tr.refit("null") is False, pb
        assert tr.m == 0j


def test_null_declines_when_it_would_cost_signal():
    rng = np.random.default_rng(12)
    n = 50_000
    qrm = _noise(rng, n, 1.0)
    sig = 0.5 * _noise(rng, n)
    a_n, b_n = _two_channel(rng, n, [(qrm, 1.1, 0.8)], white=0.05)
    a_s, b_s = _two_channel(rng, n, [(sig, 1.1, 0.8)], white=0.0)   # same bearing as the QRM
    Xn = np.vstack([a_n, b_n]); X = np.vstack([a_n + a_s, b_n + b_s])
    tr = Tracker(RATE)
    tr.Rn = (Xn @ Xn.conj().T) / n
    tr.Rs = (X @ X.conj().T) / n
    assert tr.refit("null") is False       # the null would land on the talker too
    tr.Rs = None
    assert tr.refit("null") is True        # with no signal knowledge it is a valid null


def test_weak_signal_below_the_vad_is_not_learned_as_noise():
    rng = np.random.default_rng(13)
    tr = Tracker(RATE)
    block = 1024
    for _ in range(60):                                    # settle floor and Rn on noise
        a, b = _two_channel(rng, block, [], white=1.0)
        tr.update(a, b, "off")
    Rn_before = tr.Rn.copy()
    for _ in range(60):                                    # +1 dB coherent signal, one bearing
        a, b = _two_channel(rng, block, [(np.sqrt(0.26) * _noise(rng, block), 0.9, 1.0)], white=1.0)
        tr.update(a, b, "off")
        assert not tr.talking
    # Rn kept its (uncorrelated) shape: the signal's spatial signature was not booked.
    assert abs(tr.Rn[0, 1]) < 0.05 * abs(tr.Rn[0, 0]), tr.Rn
    assert np.allclose(np.diag(tr.Rn), np.diag(Rn_before), rtol=0.15)


def test_a_short_burst_does_not_train_the_signal_covariance():
    rng = np.random.default_rng(14)
    tr = Tracker(RATE)
    block = 256                                            # ~10 ms, shorter than TALK_HOLD_S
    assert block / RATE < TALK_HOLD_S
    for _ in range(200):
        a, b = _two_channel(rng, block, [], white=1.0)
        tr.update(a, b, "track")
    for _ in range(5):                                     # crash, quiet, crash, quiet ...
        a, b = _two_channel(rng, block, [(30.0 * _noise(rng, block), 2.5, 1.0)], white=1.0)
        tr.update(a, b, "track")
        assert tr.talking
        a, b = _two_channel(rng, block, [], white=1.0)
        tr.update(a, b, "track")
    assert tr.Rs is None and tr.updates == 0
