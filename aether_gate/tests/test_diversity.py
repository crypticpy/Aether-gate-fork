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
    ALIGN_MIN_PEAK, REFIT_MIN_GAIN_DB, RING_FOLD_MIN_PACKETS, TALK_HOLD_S,
    WEIGHT_MAX_ABS, Aligner, TalkerMemory, Tracker, _snr_of, blank_impulses,
    combine, combine_ramp, find_lag, fit_max_snr, fit_null, fold_ring_skew,
    ring_packet_samples, weight_from_polar, weight_to_polar,
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


def test_find_lag_prefers_the_peak_nearest_zero_over_a_ring_alias():
    """A calibration window straddling a ring wrap correlates at the true lag
    in one part and one ring length away in the other (live: 4032 = -63 +
    4095). Even when the alias's part is the larger, the near-zero peak
    wins as long as it is a real peak."""
    rng = np.random.default_rng(3)
    ring, true = 4095, -63
    alias = true + ring
    n = 24_000
    common = _noise(rng, n + 2 * ring + 200)
    a = common[ring + 100:ring + 100 + n].copy()
    b = np.empty_like(a)
    split = n * 2 // 5                                  # 40 % true, 60 % alias
    b[:split] = common[ring + 100 - true:ring + 100 - true + split]
    b[split:] = common[ring + 100 - alias + split:ring + 100 - alias + n]
    a = a + _noise(rng, n, 0.3)
    b = b + _noise(rng, n, 0.3)
    got, peak = find_lag(a, b, ring + 200)
    assert got == true, got
    assert peak >= ALIGN_MIN_PEAK


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

def _cov(a, b):
    X = np.vstack([a, b])
    return (X @ X.conj().T) / X.shape[1]


def _scene(rng, block, talker, white, qrm=(2.0, 0.9), talk_gain=1.2, t=None):
    """(R_in, R_guard) for one block. The guard band sees the same noise
    field (the QRM source at its bearing plus the isotropic floor) as the
    passband, with its own independent waveform, and never the talker.
    With t given the talker carries a 4 Hz syllabic envelope, as speech does."""
    if t is not None:
        talk_gain = talk_gain * (1.0 + 0.8 * np.cos(2 * np.pi * 4.0 * t))
    q_in = _noise(rng, block, 1.0)
    q_g = _noise(rng, block, 1.0)
    srcs_in = [(q_in, qrm[0], qrm[1])]
    if talker is not None:
        srcs_in.append((talk_gain * _noise(rng, block), talker[0], talker[1]))
    a, b = _two_channel(rng, block, srcs_in, white)
    ga, gb = _two_channel(rng, block, [(q_g, qrm[0], qrm[1])], white)
    return _cov(a, b), _cov(ga, gb)


def _run(tracker, rng, seconds, talker, mode, block=1024, white=0.5):
    """Stream `seconds` of alternating quiet (0.6 s) and talking (1.0 s) blocks.
    talker = (angle, gain_b) of the wanted signal; a fixed noise source sits
    at angle 2.0 over an isotropic (uncorrelated) floor, so the best weight
    depends on where the talker is and not only on where the noise is."""
    t = 0.0
    while t < seconds:
        phase = t % 1.6
        talking = phase >= 0.6
        R_in, R_g = _scene(rng, block, talker if talking else None, white, t=t)
        tracker.update(R_in, R_g, block, mode)
        t += block / RATE


def test_tracker_learns_a_null_between_overs_and_a_beam_on_the_talker():
    rng = np.random.default_rng(8)
    tr = Tracker(RATE)
    _run(tr, rng, 4.0, (-0.7, 1.1), "track")
    assert tr.updates >= 1
    snr = tr.snr_db()
    assert snr["out"] > max(snr["a"], snr["b"]) + 6.0, snr
    assert tr.rn_source in ("inband", "guard")


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


def test_guard_band_gives_a_null_before_anyone_pauses():
    """Someone is talking from the first block: no pause ever happens, yet the
    guard band alone lets the null land on the noise source."""
    rng = np.random.default_rng(16)
    tr = Tracker(RATE)
    block = 1024
    for _ in range(int(1.0 * RATE / block)):
        R_in, R_g = _scene(rng, block, (-0.7, 1.1), 0.05)
        tr.update(R_in, R_g, block, "null")
    assert tr.rn_source == "guard"
    assert tr.updates >= 1
    p_a = float(np.real(tr.Rn[0, 0]))
    from aether_gate.core.diversity import _out_noise
    assert 10 * np.log10(p_a / _out_noise(tr.m, tr.Rn)) > 10.0


def test_refit_declines_without_enough_gain():
    tr = Tracker(RATE)
    tr.Rn_guard = np.eye(2, dtype=complex)            # white, equal: nothing to null
    assert tr.refit("null") is False
    assert tr.m == 0j and tr.updates == 0
    assert REFIT_MIN_GAIN_DB > 0


def test_null_refuses_to_select_the_quieter_antenna():
    """Uncorrelated noise has nothing to null: with channel B dead (or just
    quieter) the smallest eigenvector is 'B alone', which must not be adopted."""
    for pb in (0.5, 0.1, 1e-4):
        tr = Tracker(RATE)
        tr.Rn_guard = np.diag([1.0, pb]).astype(complex)
        assert tr.refit("null") is False, pb
        assert tr.m == 0j


def test_null_declines_when_it_would_cost_signal():
    rng = np.random.default_rng(12)
    n = 50_000
    qrm = _noise(rng, n, 1.0)
    sig = 0.5 * _noise(rng, n)
    a_n, b_n = _two_channel(rng, n, [(qrm, 1.1, 0.8)], white=0.05)
    a_s, b_s = _two_channel(rng, n, [(sig, 1.1, 0.8)], white=0.0)   # same bearing as the QRM
    tr = Tracker(RATE)
    tr.Rn_guard = _cov(a_n, b_n)
    tr.Rs = _cov(a_n + a_s, b_n + b_s)
    assert tr.refit("null") is False       # the null would land on the talker too
    tr.Rs = None
    assert tr.refit("null") is True        # with no signal knowledge it is a valid null


def test_inband_noise_model_stays_in_force_through_a_long_over():
    """The in-band Rn is learned only between overs. A talker who holds the
    key for longer than RN_INBAND_FRESH_S must not lose the null it gives
    on a co-channel station five seconds into the over: freshness is
    measured to the start of the over, not to now."""
    from aether_gate.core.diversity import RN_INBAND_FRESH_S
    rng = np.random.default_rng(22)
    tr = Tracker(RATE)
    block = 1024
    t = 0.0
    while t < 1.0:                                       # quiet: guard noise learned
        R_in, R_g = _scene_tone(rng, block, False, None, 0.5, t)
        tr.update(R_in, R_g, block, "track"); t += block / RATE
    while t < 4.0:                                       # the tone parks in the passband
        R_in, R_g = _scene_tone(rng, block, True, None, 0.5, t)
        tr.update(R_in, R_g, block, "track"); t += block / RATE
    assert tr.steady and tr.rn_source == "inband"
    t_key = t
    while t < t_key + 2.0 * RN_INBAND_FRESH_S:           # one long over, twice the window
        R_in, R_g = _scene_tone(rng, block, True, (-0.7, 1.1), 0.5, t)
        tr.update(R_in, R_g, block, "track"); t += block / RATE
        assert tr.rn_source == "inband", (t - t_key, "the null on the tone was dropped mid-over")
    assert tr.talking and not tr.steady


def test_weak_signal_below_the_vad_is_not_learned_as_noise():
    rng = np.random.default_rng(13)
    tr = Tracker(RATE)
    block = 1024
    for _ in range(60):                                    # settle on noise
        R_in, R_g = _scene(rng, block, None, 1.0, qrm=(2.0, 0.0))
        tr.update(R_in, R_g, block, "off")
    for _ in range(160):                                   # +1.2 dB coherent signal, one bearing, > RN_INBAND_FRESH_S
        R_in, R_g = _scene(rng, block, (0.9, 1.0), 1.0, qrm=(2.0, 0.0), talk_gain=np.sqrt(0.5))
        tr.update(R_in, R_g, block, "off")
        assert not tr.talking
    # the in-band noise estimate went stale rather than learning the signal
    assert tr.rn_source == "guard"
    assert abs(tr.Rn[0, 1]) < 0.05 * abs(tr.Rn[0, 0]), tr.Rn


def test_a_short_burst_does_not_train_the_signal_covariance():
    rng = np.random.default_rng(14)
    tr = Tracker(RATE)
    block = 256                                            # ~10 ms, shorter than TALK_HOLD_S
    assert block / RATE < TALK_HOLD_S
    for _ in range(200):
        R_in, R_g = _scene(rng, block, None, 1.0)
        tr.update(R_in, R_g, block, "track")
    n0 = tr.updates                                        # idle-time noise nulls are fine
    for _ in range(5):                                     # a crash every 200 ms
        R_in, R_g = _scene(rng, block, (2.5, 1.0), 1.0, talk_gain=30.0)
        tr.update(R_in, R_g, block, "track")
        assert tr.talking
        for _q in range(20):
            R_in, R_g = _scene(rng, block, None, 1.0)
            tr.update(R_in, R_g, block, "track")
    assert tr.Rs is None and tr.updates == n0


# --- talker memory ------------------------------------------------------------

def _bearing(theta):
    """A unit steering vector at a bearing far enough from bearing 0 (and
    from every other _bearing(k) used below, k an integer or half-integer)
    that MEMORY_MATCH never merges or recalls across them."""
    v = np.array([1.0, np.exp(1j * theta)])
    return v / np.linalg.norm(v)


def test_memory_steers_a_known_talker_in_one_block():
    rng = np.random.default_rng(17)
    mem = TalkerMemory()
    tr = Tracker(RATE, memory=mem)
    _run(tr, rng, 4.0, (-0.7, 1.1), "track")            # learn talker 1
    m1 = tr.m
    _run(tr, rng, 3.2, (0.5, 1.0), "track")             # learn talker 2
    assert len(mem.entries) >= 2, mem.status(tr.t)
    # talker 1 keys up again: the weight must jump back within TALK_HOLD_S + 2 blocks
    block = 1024
    for _ in range(16):                                    # a pause > TALK_HANG_S (0.66 s)
        R_in, R_g = _scene(rng, block, None, 0.5)
        tr.update(R_in, R_g, block, "track")
    n0 = tr.updates
    for k in range(4):
        R_in, R_g = _scene(rng, block, (-0.7, 1.1), 0.5)
        tr.update(R_in, R_g, block, "track")
    assert tr.updates > n0
    assert abs(tr.m - m1) < 0.35 * max(1.0, abs(m1)), (tr.m, m1)
    st = mem.status(tr.t)
    assert all(set(e) == {"id", "name", "phase_deg", "ratio_db", "age_s",
                          "first_seen_s", "hits"} for e in st)
    # talker 1 is the live talker while its over runs, nobody after the hangover
    live = mem.talker(tr.t)
    assert live is not None and live["id"] == st[0]["id"] and live["since_s"] >= 0.0
    for _ in range(16):
        R_in, R_g = _scene(rng, block, None, 0.5)
        tr.update(R_in, R_g, block, "track")
    assert mem.talker(tr.t) is None
    mem.clear()
    assert mem.entries == [] and mem.talker(tr.t) is None


def test_memory_ids_are_stable_and_names_stick_to_them():
    mem = TalkerMemory()
    s1 = np.array([1.0, 0.6 * np.exp(1j * 0.4)]); s1 = s1 / np.linalg.norm(s1)
    s2 = np.array([1.0, 0.6 * np.exp(1j * 2.5)]); s2 = s2 / np.linalg.norm(s2)
    mem.store(s1, 0.3 + 0.2j, now=0.0)
    mem.store(s2, 0.1 + 0.0j, now=1.0)
    ids = [e["id"] for e in mem.entries]
    assert ids == [1, 2]
    assert mem.talker(now=1.5) == {"id": 2, "since_s": 0.5}
    assert mem.name(1, "  Bob K5XYZ ") and mem.entries[0]["name"] == "Bob K5XYZ"
    assert not mem.name(9, "nobody")
    mem.store(s1, 0.3 + 0.2j, now=2.0)                    # merge keeps id and name
    assert [e["id"] for e in mem.entries] == [1, 2]
    st = mem.status(now=3.0)
    assert st[0] == {"id": 1, "name": "Bob K5XYZ", "phase_deg": st[0]["phase_deg"],
                     "ratio_db": st[0]["ratio_db"], "age_s": 1.0, "first_seen_s": 3.0,
                     "hits": 0}
    assert mem.recall(s2, now=4.0) == 0.1 + 0.0j and mem.talker(4.0)["id"] == 2
    assert mem.name(1, "") and mem.entries[0]["name"] is None
    mem.release()
    assert mem.talker(5.0) is None


def test_memory_matches_on_bearing_not_exact_vector():
    mem = TalkerMemory()
    s = np.array([1.0, 0.6 * np.exp(1j * 0.4)]); s = s / np.linalg.norm(s)
    mem.store(s, 0.3 + 0.2j, now=0.0)
    near = s * np.exp(1j * 1.0)                             # same bearing, common phase
    assert mem.recall(near, now=1.0) == 0.3 + 0.2j
    far = np.array([1.0, 0.6 * np.exp(1j * 2.5)]); far = far / np.linalg.norm(far)
    assert mem.recall(far, now=1.0) is None
    for k in range(20):
        v = np.array([1.0, np.exp(1j * 0.3 * k)]) / np.sqrt(2)
        mem.store(v, complex(k), now=float(k))
    assert len(mem.entries) <= mem.max_n


# --- ramps and the blanker -----------------------------------------------------

def test_combine_ramp_ends_where_combine_is_and_has_no_step():
    rng = np.random.default_rng(18)
    n = 4096
    a = _noise(rng, n); b = _noise(rng, n)
    y = combine_ramp(a, b, 0j, 1.0 + 0j)
    assert np.allclose(y[-1], combine(a, b, 1.0 + 0j)[-1], atol=1e-3 * abs(y[-1]) + 1e-9)
    assert np.allclose(y[0], a[0], atol=1e-3 * abs(a[0]) + 1e-9)
    assert np.allclose(combine_ramp(a, b, 0.5j, 0.5j), combine(a, b, 0.5j))


def test_blanker_removes_an_impulse_from_both_channels_and_leaves_speech():
    rng = np.random.default_rng(19)
    n = 8192
    a = _noise(rng, n); b = _noise(rng, n)
    a[4000] += 60.0; b[4000] += 60.0 * np.exp(1j * 0.7)     # a 36 dB crash on both
    oa, ob, frac = blank_impulses(a, b, 12.0)
    assert oa[4000] == 0 and ob[4000] == 0 and oa[3998] == 0 and ob[4002] == 0
    assert 0 < frac < 0.002
    assert np.array_equal(oa[:3990], a[:3990])
    # an AM carrier at 100 % modulation peaks 6 dB over its mean: untouched
    t = np.arange(n) / RATE
    c = (1 + np.cos(2 * np.pi * 400 * t)) * np.exp(2j * np.pi * 1000 * t)
    ca, cb, frac = blank_impulses(c + 0.01 * a, c + 0.01 * b, 12.0)
    assert frac == 0.0


def test_fade_guard_switches_to_the_surviving_antenna_within_half_a_second():
    """B carries the talker best, so the beam leans on B; then B fades away
    in a syllable (QSB). The output must not sit under A for a time
    constant: the fade guard moves it to A alone within half a second."""
    rng = np.random.default_rng(23)
    tr = Tracker(RATE)
    block = 1024
    t = 0.0
    while t < 3.0:                                          # steady over, B 6 dB better
        R_in, R_g = _scene(rng, block, (0.4, 2.0), 0.5, qrm=(2.0, 0.0), talk_gain=3.0, t=t)
        tr.update(R_in, R_g, block, "track")
        t += block / RATE
    assert tr.updates >= 1 and abs(tr.m) > 1.0, tr.m         # leaning on B
    n0 = tr.updates
    steps = 0
    while t < 4.0:                                          # B collapses
        R_in, R_g = _scene(rng, block, (0.4, 0.05), 0.5, qrm=(2.0, 0.0), talk_gain=3.0, t=t)
        tr.update(R_in, R_g, block, "track")
        t += block / RATE
        steps += 1
        if tr.updates > n0:
            break
    assert tr.updates > n0 and steps * block / RATE <= 0.5, (steps, tr.m)
    assert tr.m == 0j, tr.m                                 # A alone


# --- steady carrier in the passband --------------------------------------------

def _scene_tone(rng, block, tone_on, talker, white, t):
    """Like _scene, with a steady tone at bearing (1.0, 1.0) in the passband
    that the guard band never sees."""
    q_in, q_g = _noise(rng, block, 1.0), _noise(rng, block, 1.0)
    srcs_in = [(q_in, 2.0, 0.9)]
    if tone_on:
        srcs_in.append((3.0 * _noise(rng, block), 1.0, 1.0))
    if talker is not None:
        g = 1.2 * (1.0 + 0.8 * np.cos(2 * np.pi * 4.0 * t))
        srcs_in.append((g * _noise(rng, block), talker[0], talker[1]))
    a, b = _two_channel(rng, block, srcs_in, white)
    ga, gb = _two_channel(rng, block, [(q_g, 2.0, 0.9)], white)
    return _cov(a, b), _cov(ga, gb)


def test_steady_carrier_becomes_noise_is_nulled_and_a_talker_over_it_is_tracked():
    rng = np.random.default_rng(21)
    tr = Tracker(RATE)
    block = 1024
    t = 0.0
    while t < 1.0:                                       # quiet: guard noise learned
        R_in, R_g = _scene_tone(rng, block, False, None, 0.5, t)
        tr.update(R_in, R_g, block, "track"); t += block / RATE
    while t < 4.0:                                       # the tone parks in the passband
        R_in, R_g = _scene_tone(rng, block, True, None, 0.5, t)
        tr.update(R_in, R_g, block, "track"); t += block / RATE
    assert tr.talking and tr.steady and tr.rn_source == "inband"
    assert tr.Rs is None and not tr.memory
    tone = _cov(*_two_channel(rng, 8192, [(3.0 * _noise(rng, 8192), 1.0, 1.0)], 0.0))
    eye = np.eye(2)                                      # _snr_of wants signal-plus-noise
    null_db = 10 * np.log10(_snr_of(0j, tone + eye, eye) / max(_snr_of(tr.m, tone + eye, eye), 1e-30))
    assert null_db > 10.0, (null_db, tr.m)               # the tone is nulled vs antenna A
    m_null = tr.m
    while t < 8.0:                                       # a talker keys up over the tone
        R_in, R_g = _scene_tone(rng, block, True, (-0.7, 1.1), 0.5, t)
        tr.update(R_in, R_g, block, "track"); t += block / RATE
    assert not tr.steady and tr.Rs is not None
    talk = _cov(*_two_channel(rng, 8192, [(1.2 * _noise(rng, 8192), -0.7, 1.1)], 0.0))
    noise = _cov(*_two_channel(rng, 8192, [(3.0 * _noise(rng, 8192), 1.0, 1.0),
                                           (_noise(rng, 8192), 2.0, 0.9)], 0.5))
    snr_out = _snr_of(tr.m, talk + noise, noise); snr_a = _snr_of(0j, talk + noise, noise)
    assert 10 * np.log10(snr_out / snr_a) > 6.0, (tr.m, m_null)


def test_memory_names_survive_a_restart_by_signature(tmp_path):
    path = str(tmp_path / "names.json")
    s1 = np.array([1.0, 0.6 * np.exp(1j * 0.4)]); s1 = s1 / np.linalg.norm(s1)
    s2 = np.array([1.0, 0.6 * np.exp(1j * 2.5)]); s2 = s2 / np.linalg.norm(s2)
    mem = TalkerMemory(names_path=path)
    mem.store(s1, 0.3j, now=0.0); mem.store(s2, 0.1, now=1.0)
    assert mem.name(1, "Bob")
    # a new run: fresh ids, and Bob's label comes back with his signature
    mem2 = TalkerMemory(names_path=path)
    mem2.store(s2, 0.1, now=0.0)
    mem2.store(s1 * np.exp(1j * 0.9), 0.3j, now=1.0)       # same bearing, other global phase
    st = mem2.status(now=2.0)
    assert [(e["id"], e["name"]) for e in st] == [(2, "Bob"), (1, None)]
    assert mem2.name(2, "")                                 # clearing forgets it on disk too
    mem3 = TalkerMemory(names_path=path)
    mem3.store(s1, 0.3j, now=0.0)
    assert mem3.entries[0]["name"] is None
    # a missing or corrupt file is not an error
    (tmp_path / "bad.json").write_text("{not json")
    assert TalkerMemory(names_path=str(tmp_path / "bad.json"))._named == []
    assert TalkerMemory(names_path=str(tmp_path / "none.json"))._named == []


def test_focus_steers_at_the_pinned_station_and_nulls_everyone_else():
    rng = np.random.default_rng(23)
    mem = TalkerMemory()
    tr = Tracker(RATE, memory=mem)
    dx, caller = (-0.7, 1.1), (0.5, 1.0)
    _run(tr, rng, 4.0, dx, "track")                      # learn the DX station
    _run(tr, rng, 3.2, caller, "track")                  # and one loud caller
    ids = {e["id"] for e in mem.entries}
    assert len(ids) >= 2
    dx_id = mem.entries[0]["id"]                         # first stored = the DX
    mem.set_focus(dx_id, tr.t)
    block = 1024

    def quiet(n):
        for _ in range(n):
            R_in, R_g = _scene(rng, block, None, 0.5)
            tr.update(R_in, R_g, block, "track")

    def over(talker, n):
        snr_out, snr_best = [], []
        for _ in range(n):
            R_in, R_g = _scene(rng, block, talker, 0.5, t=tr.t)
            tr.update(R_in, R_g, block, "track")
            if tr.Rs is not None:
                s = tr.snr_db()
                snr_out.append(s["out"]); snr_best.append(max(s["a"], s["b"]))
        return snr_out, snr_best

    # the caller keys up: within the over the output must be well below the
    # better antenna (nulled), and the tracker says so
    quiet(16)
    out, best = over(caller, 40)
    assert tr.interferer is True
    assert mem.talker(tr.t) is not None and mem.talker(tr.t)["id"] != dx_id
    assert out[-1] < best[-1] - 6.0, (out[-1], best[-1])
    f = mem.focus_status(tr.t, nulling=tr.interferer)
    assert f["nulling"] is True and f["nulled"] == 1 and f["overs"] == 0 and f["live"] is False
    # the DX keys up: their remembered beam is back and beats either antenna
    quiet(16)
    out, best = over(dx, 40)
    assert tr.interferer is False
    assert mem.talker(tr.t)["id"] == dx_id
    assert out[-1] > best[-1] + 3.0, (out[-1], best[-1])
    f = mem.focus_status(tr.t, nulling=tr.interferer)
    assert f["live"] is True and f["overs"] == 1 and f["best_db"] is not None
    # an unknown third station is nulled too, and never learned as the DX
    quiet(16)
    out, best = over((2.6, 0.8), 40)
    assert tr.interferer is True and out[-1] < best[-1] - 6.0
    assert mem.focus_status(tr.t)["nulled"] == 2
    # releasing the focus returns the tracker to following whoever talks
    mem.set_focus(None, tr.t)
    quiet(16)
    out, best = over(caller, 40)
    assert tr.interferer is False and out[-1] >= best[-1] - 1.0


def test_null_of_cancels_the_steering_vector_and_caps_the_ratio():
    from aether_gate.core.focus import null_of
    s = np.array([0.6, 0.8j])
    m = null_of(s)
    assert abs(s[0] + m * s[1]) < 1e-9
    m = null_of(np.array([1.0, 1e-4]))
    assert abs(m) == pytest.approx(WEIGHT_MAX_ABS)


def test_memory_never_evicts_the_focused_talker():
    mem = TalkerMemory(max_n=2)
    for k in range(2):
        mem.store(np.array([1.0, np.exp(1j * k)]) / np.sqrt(2), 0j, float(k))
    mem.set_focus(mem.entries[0]["id"], 2.0)
    mem.store(np.array([1.0, np.exp(1j * 2.5)]) / np.sqrt(2), 0j, 3.0)
    assert len(mem.entries) == 2
    assert mem.focus_entry() is not None


# --- eviction order: a named regular must not be bumped for a stranger --------

def test_eviction_prefers_the_oldest_cold_stranger():
    """Two one-off callers nobody has recognised again (hits == 0): a third
    fills the memory, and the LONGER-STANDING stranger is the one to go,
    not the one that just walked in."""
    mem = TalkerMemory(max_n=2)
    mem.store(_bearing(0.0), 0j, now=0.0)      # id 1: the older stranger
    mem.store(_bearing(1.5), 0j, now=1.0)      # id 2: the younger stranger
    mem.store(_bearing(3.0), 0j, now=2.0)      # id 3: a third caller, memory full
    ids = {e["id"] for e in mem.entries}
    assert ids == {2, 3}, [(e["id"], e["hits"], e["first_seen"]) for e in mem.entries]


def test_eviction_takes_a_cold_stranger_before_a_recognised_or_named_voice():
    """A sole hits-0 unnamed entry is evicted ahead of an unnamed entry that
    has been recalled before (hits > 0, even with an older last_seen) and
    ahead of a named entry -- exactly the busy-net case: the regular and the
    voice heard twice both outrank a stranger heard once."""
    mem = TalkerMemory(max_n=2)
    mem.names_path = None                      # never the operator's live file
    mem.store(_bearing(0.0), 0j, now=0.0)       # id 1
    mem.recall(_bearing(0.0), now=1.0)          # id 1: hits=1, last_seen=1.0
    mem.store(_bearing(1.5), 0j, now=2.0)       # id 2: hits=0, memory full
    # id 3 arrives already named, as if this bearing was named in an earlier
    # session (see test_memory_names_survive_a_restart_by_signature)
    mem._named.append({"s": _bearing(3.0), "name": "Cara", "voice": None})
    mem.store(_bearing(3.0), 0j, now=3.0)       # id 3, named on arrival
    assert {e["id"]: e["name"] for e in mem.entries} == {1: None, 3: "Cara"}


def test_eviction_among_unnamed_recognised_voices_takes_the_oldest_last_seen():
    """With no hits-0 stranger in the running, eviction picks the unnamed
    voice heard longest ago -- and a named regular survives even though its
    own last_seen is far older than either candidate."""
    mem = TalkerMemory(max_n=3)
    mem.names_path = None
    mem.store(_bearing(0.0), 0j, now=0.0)
    mem.name(1, "Zack")                         # id 1: named, last_seen 0.0 (oldest of all)
    mem.store(_bearing(1.5), 0j, now=1.0)
    mem.recall(_bearing(1.5), now=3.0)           # id 2: unnamed, hits=1, last_seen 3.0
    mem.store(_bearing(3.0), 0j, now=1.2)
    mem.recall(_bearing(3.0), now=2.5)           # id 3: unnamed, hits=1, last_seen 2.5 (older)
    mem._named.append({"s": _bearing(4.5), "name": "Wendy", "voice": None})
    mem.store(_bearing(4.5), 0j, now=4.0)        # id 4: named on arrival, memory full
    ids = {e["id"] for e in mem.entries}
    assert ids == {1, 2, 4}, [(e["id"], e["name"], e["hits"], e["last_seen"]) for e in mem.entries]


def test_eviction_falls_back_to_the_oldest_named_entry_only_once_all_are_named():
    """Every slot named: the fallback is the named entry heard longest ago,
    not a refusal to evict."""
    mem = TalkerMemory(max_n=2)
    mem.names_path = None
    mem.store(_bearing(0.0), 0j, now=0.0)
    mem.name(1, "Alice")                        # id 1: last_seen 0.0, oldest
    mem.store(_bearing(1.5), 0j, now=1.0)
    mem.name(2, "Bob")                          # id 2: last_seen 1.0, memory full
    mem._named.append({"s": _bearing(3.0), "name": "Charlie", "voice": None})
    mem.store(_bearing(3.0), 0j, now=2.0)        # id 3: named on arrival too
    assert {e["id"]: e["name"] for e in mem.entries} == {2: "Bob", 3: "Charlie"}


def test_idle_time_between_overs_is_spent_nulling_the_noise():
    from aether_gate.core.diversity import _out_noise
    rng = np.random.default_rng(29)
    tr = Tracker(RATE, memory=TalkerMemory())
    _run(tr, rng, 4.0, (-0.7, 1.1), "track")
    m_beam = tr.m
    block = 1024
    for _ in range(int(2.5 * RATE / block)):                 # 2.5 s of nobody
        R_in, R_g = _scene(rng, block, None, 0.5)
        tr.update(R_in, R_g, block, "track")
    # on a one-source scene the talker's beam already nulls the noise, so
    # park the weight on antenna A (as a fade switch would) to see the
    # idle null do its work
    tr.m = 0j
    for _ in range(int(1.0 * RATE / block)):
        R_in, R_g = _scene(rng, block, None, 0.5)
        tr.update(R_in, R_g, block, "track")
    assert tr.idle_null is True
    Rn = tr.Rn
    floor = min(float(Rn[0, 0].real), float(Rn[1, 1].real))
    assert _out_noise(tr.m, Rn) < 0.7 * floor, (_out_noise(tr.m, Rn), floor)
    # the talker keys up again: the beam is back and the null is history
    for _ in range(12):
        R_in, R_g = _scene(rng, block, (-0.7, 1.1), 0.5, t=tr.t)
        tr.update(R_in, R_g, block, "track")
    assert tr.idle_null is False
    assert abs(tr.m - m_beam) < 0.35 * max(1.0, abs(m_beam)), (tr.m, m_beam)


def test_a_focus_keeps_its_beam_through_idle_time():
    rng = np.random.default_rng(31)
    mem = TalkerMemory()
    tr = Tracker(RATE, memory=mem)
    _run(tr, rng, 4.0, (-0.7, 1.1), "track")
    mem.set_focus(mem.entries[0]["id"], tr.t)
    m_beam = tr.m
    block = 1024
    for _ in range(int(3.0 * RATE / block)):
        R_in, R_g = _scene(rng, block, None, 0.5)
        tr.update(R_in, R_g, block, "track")
    assert tr.idle_null is False and tr.m == m_beam


# --- fold_ring_skew: the driver's FIFO skew is packets, not delay ----------
# Every figure below is a real stream-start or post-fault lag out of the gate
# logs (three nights, four sample rates). They are all whole 504 us driver
# packets, which is what makes them the FIFO and not the antennas.

@pytest.mark.parametrize("rate,q", [(62_500.0, 31.5), (125_000.0, 63.0),
                                    (250_000.0, 126.0), (500_000.0, 252.0),
                                    (2_000_000.0, 1008.0)])
def test_a_driver_packet_is_504_microseconds_at_every_rate(rate, q):
    assert ring_packet_samples(rate) == q
    assert ring_packet_samples(rate) / rate == pytest.approx(1008 / 2e6)


@pytest.mark.parametrize("lag,rate,packets", [
    (-4158, 125_000.0, -66),         # stream start, 43 times in the logs
    (-4221, 125_000.0, -67),         # ...and 10 times
    (-4284, 125_000.0, -68),         # ...and 3
    (-2079, 62_500.0, -66),          # the same 33.264 ms at a quarter the rate
    (-8316, 250_000.0, -66),
    (-16632, 500_000.0, -66),
    (4032, 125_000.0, 64),           # after an overflow on A: B is the one ahead
])
def test_a_whole_packet_skew_folds_to_its_residual(lag, rate, packets):
    assert fold_ring_skew(lag, rate) == (0, packets)


@pytest.mark.parametrize("lag", [0, -1, 5, -63, -126, -252, -441])
def test_a_small_lag_is_left_exactly_as_measured(lag):
    """Requirement: the ordinary case must not move. Anything under
    RING_FOLD_MIN_PACKETS is the aligner's delay line to hold, including the
    one- and two-packet residuals the reader leaves behind after a fault."""
    assert abs(lag) < RING_FOLD_MIN_PACKETS * 63
    assert fold_ring_skew(lag, 125_000.0) == (lag, 0)


def test_the_fold_threshold_is_where_it_says_it_is():
    """Mutation check on RING_FOLD_MIN_PACKETS: seven packets is held, eight
    is folded, and the boundary is the constant rather than a coincidence."""
    q = int(ring_packet_samples(125_000.0))
    held = -q * (RING_FOLD_MIN_PACKETS - 1)
    folded = -q * RING_FOLD_MIN_PACKETS
    assert fold_ring_skew(held, 125_000.0) == (held, 0)
    assert fold_ring_skew(folded, 125_000.0) == (0, -RING_FOLD_MIN_PACKETS)


@pytest.mark.parametrize("resid,folds", [(-10, True), (-15, True),
                                         (-16, False), (-30, False)])
def test_only_a_lag_that_lands_on_a_packet_boundary_is_claimed(resid, folds):
    """Mutation check on RING_FOLD_TOL: a quarter of a packet (15 samples at
    125 kS/s) is as far off a whole number of packets as a skew is allowed to
    be. Further out we cannot tell skew from delay, so we report what we
    measured and drop nothing."""
    lag = -66 * 63 + resid
    assert fold_ring_skew(lag, 125_000.0) == ((resid, -66) if folds else (lag, 0))


def test_the_fold_survives_a_nonsense_rate():
    assert fold_ring_skew(-4158, 0.0) == (-4158, 0)
