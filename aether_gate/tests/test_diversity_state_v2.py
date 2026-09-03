#
# Aether-gate — the adapter's diversity v2 state, no hardware.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""What _DiversityState adds on top of alignment and a per-slice weight:
the two-channel noise blanker, the spatial map and its 'nulled' pan, source
listing and null_source, the raw two-channel capture, the guard-band
tracker feed, and the status/set contract the engine's routes rely on.

Run:  python -m pytest aether_gate/tests/test_diversity_state_v2.py
"""
import math
import os
import time

import numpy as np
import pytest

from aether_gate.adapters.diversity_state import _DiversityState

RATE = 125_000.0
BLOCK = 4096


class _FakeAdapter:
    def __init__(self, samp_rate=RATE, center_hz=3_600_000.0, mode="USB"):
        self._np = np
        self.samp_rate = float(samp_rate)
        self.center_hz = float(center_hz)
        self._mode = mode


def _aligned_state(**kw):
    st = _DiversityState(_FakeAdapter(**kw))
    st.aligner.set_lag(0, 20.0, True)
    return st


def _white(rng, n, p=1.0):
    return ((rng.normal(size=n) + 1j * rng.normal(size=n)) * np.sqrt(p / 2)).astype(np.complex64)


def _coherent_source(rng, n, f_lo, f_hi, angle, gain_b, power):
    """A band-limited noise source arriving on both antennas with a fixed
    inter-antenna phase: what a nearby switch-mode supply looks like."""
    X = np.zeros(n, dtype=np.complex128)
    f = np.fft.fftfreq(n, 1.0 / RATE)
    sel = (f >= f_lo) & (f < f_hi)
    X[sel] = rng.normal(size=sel.sum()) + 1j * rng.normal(size=sel.sum())
    x = np.fft.ifft(X)
    x *= np.sqrt(power / max(1e-30, np.mean(np.abs(x) ** 2)))    # power per sample
    return x.astype(np.complex64), (x * gain_b * np.exp(1j * angle)).astype(np.complex64)


def _feed_scene(st, rng, blocks, sources, white=1.0):
    """sources: [(f_lo, f_hi, angle, gain_b, power)] relative to the centre."""
    out = None
    for _ in range(blocks):
        a = _white(rng, BLOCK, white); b = _white(rng, BLOCK, white)
        for (lo, hi, ang, gb, p) in sources:
            sa, sb = _coherent_source(rng, BLOCK, lo, hi, ang, gb, p)
            a = a + sa; b = b + sb
        out, _pair = st.ingest(a, b)
    return out


# --- noise blanker ----------------------------------------------------------

def test_blanker_zeroes_an_impulse_on_both_channels_and_reports_it():
    rng = np.random.default_rng(1)
    st = _aligned_state()
    a = _white(rng, BLOCK); b = _white(rng, BLOCK)
    a[1000] = 300 + 0j; b[1000] = 250 + 0j                  # a lightning crack, both antennas
    st.set(nb=True, nb_db=12.0)
    _pan, (a2, b2) = st.ingest(a, b)
    assert a2[1000] == 0 and b2[1000] == 0
    assert a2[998] == 0 and b2[1002] == 0                    # widened by 2 samples
    assert np.count_nonzero(a2 == 0) <= 5
    nb = st.status()["nb"]
    assert (nb["enabled"], nb["threshold_db"]) == (True, 12.0)
    assert nb["blanked_pct"] == pytest.approx(100 * 5 / BLOCK * 0.1, rel=0.2)
    assert nb["auto"]["mode"] == "on"
    st.set(nb=False)
    _pan, (a3, b3) = st.ingest(a, b)
    assert a3[1000] == a[1000]                               # off: untouched
    assert st.status()["nb"]["blanked_pct"] == 0.0


def test_blanker_threshold_is_validated():
    st = _aligned_state()
    with pytest.raises(ValueError):
        st.set(nb_db=41.0)
    with pytest.raises(ValueError):
        st.set(nb_db=-1.0)


# --- spatial map, sources, nulled pan ----------------------------------------

def test_map_is_built_only_once_aligned_then_retuned_in_place_on_a_centre_move():
    rng = np.random.default_rng(2)
    st = _DiversityState(_FakeAdapter())
    _feed_scene(st, rng, 3, [])
    assert st.map is None and st.map_json()["available"] is False
    st.aligner.set_lag(0, 20.0, True)
    _feed_scene(st, rng, 3, [])
    first = st.map
    assert first is not None and first.frames == 3
    st.a.center_hz += 50_000.0            # same span, just moved: retune in place
    _feed_scene(st, rng, 1, [])
    assert st.map is first and st.map.frames == 4          # no reset -- one more frame accepted


def test_a_rate_change_still_rebuilds_the_map_the_live_rows_and_the_finder():
    rng = np.random.default_rng(13)
    st = _aligned_state()
    _feed_scene(st, rng, 3, [])
    first_map, first_live, first_finder = st.map, st.live, st.finder
    assert first_map is not None and first_map.frames == 3
    assert first_finder.fast_n == 3
    st.a.samp_rate = RATE * 2                                # every bin's meaning changed
    _feed_scene(st, rng, 1, [])
    assert st.map is not first_map and st.map.frames == 1
    assert st.live is not first_live
    assert st.finder is not first_finder
    # the finder counts SLOTS, not frames (core/finder.py SLOT_S): at twice
    # the span a block is half a slot long, so the frame is accepted and
    # accumulated but the ring has not gained a row from it yet
    assert st.finder.fast_n == 0 and st.finder.elapsed == pytest.approx(BLOCK / (RATE * 2))


def test_sources_list_a_coherent_noise_source_at_its_frequency():
    rng = np.random.default_rng(3)
    st = _aligned_state()
    src = (10_000.0, 14_000.0, 1.1, 0.9, 30.0)               # 10-14 kHz above centre
    _feed_scene(st, rng, 60, [src])
    srcs = st.status()["sources"]
    assert len(srcs) >= 1, srcs
    top = srcs[0]
    assert 3_600_000 + 9_000 <= top["lo_hz"] <= 3_600_000 + 11_000, top
    assert 3_600_000 + 13_000 <= top["hi_hz"] <= 3_600_000 + 15_000, top
    assert top["coherence"] >= 0.8
    mj = st.map_json()
    assert mj["available"] and len(mj["coherence"]) == 256 and mj["sources"] == srcs
    assert mj["start_hz"] == pytest.approx(3_600_000 - RATE / 2)
    assert mj["passband_hz"] is None                          # the fake has no slice
    st.a._slice_hz = 3_610_000.0
    assert st.map_json()["passband_hz"] == [3_610_000.0, 3_610_000.0 + 3_000.0]
    st.a._mode = "LSB"
    assert st.map_json()["passband_hz"] == [3_610_000.0 - 3_000.0, 3_610_000.0]


def test_nulled_pan_suppresses_the_source_and_leaves_the_rest_alone():
    rng = np.random.default_rng(4)
    st = _aligned_state()
    src = (10_000.0, 14_000.0, 1.1, 0.9, 30.0)
    _feed_scene(st, rng, 60, [src])
    st.set(pan="combined")
    comb = _feed_scene(st, rng, 1, [src])
    st.set(pan="nulled")
    nulled = _feed_scene(st, rng, 1, [src])
    f = np.fft.fftfreq(BLOCK, 1.0 / RATE)
    in_src = (f >= 10_500) & (f < 13_500)
    elsewhere = (f >= -40_000) & (f < -20_000)

    def band_power(x, sel):
        X = np.fft.fft(x)
        return float(np.mean(np.abs(X[sel]) ** 2))
    drop_db = 10 * np.log10(band_power(comb, in_src) / band_power(nulled, in_src))
    assert drop_db > 10.0, drop_db
    same_db = 10 * np.log10(band_power(comb, elsewhere) / band_power(nulled, elsewhere))
    assert abs(same_db) < 1.5, same_db
    assert nulled.dtype == comb.dtype and len(nulled) == BLOCK


def test_null_source_parks_the_slice_on_that_sources_null_in_manual_mode():
    rng = np.random.default_rng(5)
    st = _aligned_state()
    src = (10_000.0, 14_000.0, 1.1, 0.9, 30.0)
    _feed_scene(st, rng, 60, [src])
    st.set(mode="track")
    d = st.set(null_source=0)
    assert d["mode"] == "manual"
    top = d["sources"][0]
    assert d["phase_deg"] == pytest.approx(top["phase_deg"], abs=0.11)
    assert d["ratio_db"] == pytest.approx(top["ratio_db"], abs=0.11)
    # the weight really does null it: |1 + m*g*e^{j angle}| small
    m = complex(*d["weight"])
    assert abs(1 + m * 0.9 * np.exp(1j * 1.1)) < 0.15
    with pytest.raises(ValueError):
        st.set(null_source=5)


def test_hear_is_the_audio_and_pan_is_the_panadapter():
    st = _aligned_state()
    assert st.set(pan="nulled")["source"] == "combined"      # the pan does not touch the audio
    out = st.set(source="b")
    assert out["source"] == "b" and out["pan"] == "nulled"   # ...and HEAR does not touch the pan
    assert st.set(source="stereo")["source"] == "stereo"
    with pytest.raises(ValueError):
        st.set(pan="c")
    with pytest.raises(ValueError):
        st.set(source="c")


def test_memory_entries_carry_a_voice_print_learned_while_the_talker_is_live():
    rng = np.random.default_rng(13)
    st = _aligned_state(mode="USB")
    st.set(mode="track", subband=False)
    rate = 25_000.0
    pa = rng.normal(size=2000) + 1j * rng.normal(size=2000)
    pb = rng.normal(size=2000) + 1j * rng.normal(size=2000)
    st.observe(0, pa[:1024], pb[:1024])
    # nobody remembered: entries are empty, nothing is fed to a print
    assert st.status()["memory"] == []
    # a remembered, live talker: 3 s of overs through combine_passband while
    # the tracker says talking, then silence closes the over
    s_vec = np.array([1.0, 0.7 * np.exp(1j * 0.4)])
    st.memory.store(s_vec / np.linalg.norm(s_vec), 0.7 + 0j, time.monotonic())
    t = st.trackers[0]
    t.talking, t.steady = True, False
    for _ in range(40):                                   # 40 x 2000 = 3.2 s
        st.combine_passband(0, pa, pb, 0j, 0j, rate)
    t.talking = False
    st.combine_passband(0, pa, pb, 0j, 0j, rate)
    mem = st.status()["memory"]
    assert len(mem) == 1 and "voice" in mem[0]
    v = mem[0]["voice"]
    assert v is not None and v["overs"] == 1 and 3.0 <= v["over_s"] <= 3.4
    assert set(v) == {"centroid_hz", "low_hz", "high_hz", "tilt_db", "syllabic_hz", "over_s",
                      "overs", "bands_db"}
    assert len(v["bands_db"]) == 32 and max(v["bands_db"]) == 0.0
    # clearing the memory clears the prints too
    st.memory_clear()
    assert st.prints[0].prints == {}


def test_hear_a_b_stereo_hand_the_loops_through_and_the_tracker_keeps_learning():
    from aether_gate.core.diversity import combine_ramp
    rng = np.random.default_rng(12)
    st = _aligned_state(mode="USB")
    st.set(mode="track")
    pa = rng.normal(size=2000) + 1j * rng.normal(size=2000)
    pb = rng.normal(size=2000) + 1j * rng.normal(size=2000)
    m = 0.5 + 0.2j
    st.set(source="a")
    assert st.combine_passband(0, pa, pb, m, m, 25_000.0) is pa
    st.set(source="b")
    assert st.combine_passband(0, pa, pb, m, m, 25_000.0) is pb
    st.set(source="stereo")
    y = st.combine_passband(0, pa, pb[:1500], m, m, 25_000.0)
    assert y.shape == (1500, 2)
    assert np.array_equal(y[:, 0], pa[:1500]) and np.array_equal(y[:, 1], pb[:1500])
    # observe() is what learns, and it does not look at HEAR
    st.observe(0, pa[:1024], pb[:1024])
    assert 0 in st.trackers
    st.set(source="combined", subband=False)
    assert np.allclose(st.combine_passband(0, pa, pb, m, m, 25_000.0), combine_ramp(pa, pb, m, m))


# --- capture ------------------------------------------------------------------

def test_capture_writes_the_aligned_pair_with_its_metadata(tmp_path, monkeypatch):
    rng = np.random.default_rng(6)
    st = _aligned_state()
    monkeypatch.setattr(_DiversityState, "CAPTURE_DIR", str(tmp_path))
    st.aligner.set_lag(3, 20.0, True)
    path = st.capture(0.05)                                  # 6250 samples
    assert path.startswith(str(tmp_path)) and path.endswith("_3600000Hz_125000sps.npz")
    assert st.status()["capture"] == {"active": True, "path": path}
    with pytest.raises(RuntimeError):
        st.capture(1)
    for _ in range(3):
        st.ingest(_white(rng, BLOCK), _white(rng, BLOCK))
    for _ in range(50):
        if os.path.exists(path) and st.last_capture == path:
            break
        time.sleep(0.05)
    z = np.load(path)
    assert len(z["a"]) == 6250 and len(z["b"]) == 6250
    assert z["a"].dtype == np.complex64
    assert float(z["rate_hz"]) == RATE and int(z["lag_samples"]) == 3
    assert st.status()["capture"] == {"active": False, "path": path}


# --- the tracker feed -----------------------------------------------------------

def test_observe_feeds_the_tracker_and_ramps_between_weights():
    rng = np.random.default_rng(7)
    st = _aligned_state(mode="USB")
    st.set(mode="track")
    n = BLOCK
    # a talker in the USB passband (0..3 kHz of the mixed block) on both
    # antennas, a BROADBAND coherent noise source covering passband and guard
    # bands alike (a switch-mode supply), white floor: the guard band's
    # covariance is the in-band noise's, so the null is right from the start
    for k in range(120):
        a = _white(rng, n, 0.5); b = _white(rng, n, 0.5)
        na, nb_ = _coherent_source(rng, n, -6_300.0, 6_300.0, 2.0, 1.0, 4.0)
        a = a + na; b = b + nb_
        if (k // 30) % 2:                                    # 1 s over, 1 s pause
            sa, sb = _coherent_source(rng, n, 300.0, 2_700.0, -0.7, 1.1, 3.0)
            a = a + sa; b = b + sb
        m0, m1 = st.observe(0, a.astype(np.complex128), b.astype(np.complex128))
    t = st.trackers[0]
    assert t.Rn_guard is not None and t.rn_source in ("guard", "inband")
    assert t.updates >= 1
    s = st.status()
    assert s["snr_db"]["out"] > max(s["snr_db"]["a"], s["snr_db"]["b"]) + 3.0, s["snr_db"]
    assert s["rn_source"] in ("guard", "inband") and s["talk_mod"] is not None
    assert set(s) >= {"nb", "pan", "sources", "memory", "capture", "rn_source", "talk_mod",
                      "talker"}
    assert s["memory"] and all("id" in e and "name" in e for e in s["memory"])
    st.memory.names_path = None                  # never the operator's names file
    st.memory_name(s["memory"][0]["id"], "Ted")
    assert st.status()["memory"][0]["name"] == "Ted"
    with pytest.raises(ValueError):
        st.memory_name(999, "x")
    assert st.last_m[0] == m1


def test_observe_returns_previous_and_new_weight_for_the_ramp():
    rng = np.random.default_rng(8)
    st = _aligned_state()
    a = _white(rng, BLOCK); b = _white(rng, BLOCK)
    st.set(mode="manual", phase_deg=90.0, ratio_db=0.0)
    m0, m1 = st.observe(0, a, b)
    assert m0 == m1 == pytest.approx(1j)                     # first block: no step to ramp from
    st.set(mode="off")
    m0, m1 = st.observe(0, a, b)
    assert m0 == pytest.approx(1j) and m1 == 0j


def test_lsb_and_fm_pick_their_own_bands():
    st = _aligned_state()
    f = np.fft.fftfreq(BLOCK, 1.0 / RATE)
    i, g = st._bands(BLOCK, RATE, "LSB")
    assert f[i].max() < 0 and f[i].min() >= -3000
    assert set(np.sign(f[g])) == {-1.0, 1.0}
    i, g = st._bands(BLOCK, RATE, "NFM")
    assert f[i].min() >= -8000 and f[i].max() < 8000 and i.sum() > 500
    assert f[g].min() >= -8000 - 300 - 16000 and abs(f[g]).min() >= 8300


def test_focus_needs_a_known_talker_and_shows_in_status():
    st = _aligned_state()
    assert st.status()["focus"] is None
    with pytest.raises(ValueError):
        st.set(focus=3)
    st.memory.store(np.array([1.0, 0.5j]) / np.sqrt(1.25), 0.4 + 0.2j, 0.0)
    st.memory.release()                       # store() marks the entry live
    tid = st.memory.entries[0]["id"]
    d = st.set(focus=tid)
    f = d["focus"]
    assert set(f) == {"id", "name", "since_s", "live", "nulling", "overs", "nulled", "best_db"}
    assert f["id"] == tid and f["live"] is False and f["nulling"] is False
    assert st.set(focus="")["focus"] is None


def test_loop_balance_warns_only_after_a_gap_has_held():
    from aether_gate.core.balance import LoopBalance
    lb = LoopBalance()
    even = np.diag([1.0, 1.0]).astype(complex)
    sick = np.diag([1.0, 10 ** (-0.8)]).astype(complex)        # B 8 dB down
    for k in range(10):
        lb.update(even, float(k))
    assert lb.status(10.0)["warning"] is None
    assert abs(lb.status(10.0)["b_minus_a_db"]) < 0.5
    for k in range(10, 200):                                    # 190 s: not yet
        lb.update(sick, float(k))
    st = lb.status(200.0)
    assert st["warning"] is None and st["b_minus_a_db"] < -7.0
    for k in range(200, 900):
        lb.update(sick, float(k))
    w = lb.status(900.0)["warning"]
    assert w and w.startswith("B is 8 dB down for 1") and "loop" in w, w
    for k in range(900, 1100):                                  # recovers: warning clears
        lb.update(even, float(k))
    assert lb.status(1100.0)["warning"] is None


def test_status_carries_the_loop_balance():
    st = _aligned_state()
    assert st.status()["loops"] == {"b_minus_a_db": None, "warning": None}


# --- live spatial rows and the finder ---------------------------------------

def test_spatial_and_finder_follow_the_map_and_carry_the_passband():
    rng = np.random.default_rng(21)
    st = _DiversityState(_FakeAdapter())
    assert st.spatial_json() == {"available": False}
    assert st.finder_json()["available"] is False
    st.aligner.set_lag(0, 20.0, True)
    _feed_scene(st, rng, 3, [(10_000.0, 14_000.0, 1.1, 0.9, 30.0)])
    sj = st.spatial_json()
    assert sj["available"] and sj["points"] == 512 and len(sj["phase_deg"]) == 512
    assert sj["passband_hz"] is None
    st.a._slice_hz = 3_610_000.0
    assert st.spatial_json()["passband_hz"] == [3_610_000.0, 3_610_000.0 + 3_000.0]
    fj = st.finder_json()
    assert fj == {"available": False}                     # needs ~4 s of frames
    _feed_scene(st, rng, 300, [])                          # ~10 s: the source is out of the ring
    fj = st.finder_json()
    assert fj["available"] and len(fj["activity"]) == 512
    assert fj["candidates"] == [], "plain noise must never be a conversation"
    live, finder = st.live, st.finder
    assert finder.fast_n == 256                            # the ring was already full
    st.a.center_hz += 50_000.0                # same span, just moved: retuned, not rebuilt
    _feed_scene(st, rng, 1, [])
    assert st.live is live and st.finder is finder and st.finder.fast_n == 256


# --- the per-bin passband combiner ------------------------------------------
def test_combine_passband_refines_only_in_track_or_null_and_reports_it():
    from aether_gate.core.diversity import combine_ramp
    from aether_gate.core.subband import NFFT
    rng = np.random.default_rng(11)
    st = _aligned_state(mode="USB")
    rate = 25_000.0
    pa = (rng.normal(size=2000) + 1j * rng.normal(size=2000))
    pb = (rng.normal(size=2000) + 1j * rng.normal(size=2000))
    m = 0.5 + 0.2j
    # off / manual: the wideband ramp, nothing learned, nothing reported
    for mode in ("off", "manual"):
        st.set(mode=mode)
        y = st.combine_passband(0, pa, pb, m, m, rate)
        assert np.allclose(y, combine_ramp(pa, pb, m, m))
    assert st.subbands == {} and st.status()["subband"] == {"enabled": True, "bins": 0,
                                                             "extra_db": 0.0}
    # track with no tracker yet: still the ramp (nothing to refine against)
    st.set(mode="track")
    assert np.allclose(st.combine_passband(0, pa, pb, m, m, rate), combine_ramp(pa, pb, m, m))
    # once the tracker exists the STFT path takes over: hop-sized output,
    # a combiner per slice at the passband rate, status carries it
    st.observe(0, pa[:1024].astype(np.complex128), pb[:1024].astype(np.complex128))
    out = np.concatenate([st.combine_passband(0, pa, pb, m, m, rate) for _ in range(4)])
    assert len(out) % (NFFT // 2) == 0 and len(out) >= 4 * 2000 - NFFT
    assert 0 in st.subbands and st.subbands[0].rate_hz == rate
    sb = st.status()["subband"]
    assert sb["enabled"] is True and set(sb) == {"enabled", "bins", "extra_db"}
    # what the VAD calls talking is not learned; a steady carrier is
    t = st.trackers[0]
    t.talking, t.steady = True, False
    n0 = st.subbands[0]._frames
    st.combine_passband(0, pa, pb, m, m, rate)
    assert st.subbands[0]._frames == n0
    t.talking, t.steady = True, True
    st.combine_passband(0, pa, pb, m, m, rate)
    assert st.subbands[0]._frames > n0
    # a rate change (retune to another decimation) rebuilds the combiner
    st.combine_passband(0, pa, pb, m, m, 50_000.0)
    assert st.subbands[0].rate_hz == 50_000.0
    # switching it off falls back to the ramp and forgets what was learned
    st.set(subband=False)
    assert st.subbands == {} and st.status()["subband"]["enabled"] is False
    assert np.allclose(st.combine_passband(0, pa, pb, m, m, rate), combine_ramp(pa, pb, m, m))


# --- the noise profile --------------------------------------------------------
def test_noise_profile_rides_in_status_and_sees_impulses_before_the_blanker():
    rng = np.random.default_rng(12)
    st = _DiversityState(_FakeAdapter())
    st.set(nb=True)
    a = _white(rng, BLOCK); b = _white(rng, BLOCK)
    st.ingest(a, b)
    assert st.status()["noise_profile"] is None            # not aligned: nothing measured
    st.aligner.set_lag(0, 20.0, True)
    for k in range(120):                                    # ~4 s, one impulse per block
        a = _white(rng, BLOCK); b = _white(rng, BLOCK)
        i = int(rng.integers(0, BLOCK - 40))
        a[i:i + 20] += 40.0; b[i:i + 20] += 40.0
        st.ingest(a, b)
    prof = st.status()["noise_profile"]
    assert set(prof) == {"mains_hz", "hum_db", "harmonics", "impulses_per_s", "impulse_db",
                         "periodic", "seconds", "window_s", "impulse_window_s", "kinds"}
    # ~30 impulses/s at 125 kS/s in 4096-sample blocks, counted despite the blanker
    assert 20.0 <= prof["impulses_per_s"] <= 40.0, prof
    assert prof["impulse_db"] >= 20.0 and prof["mains_hz"] is None, prof
    # the finding names itself, says how long it looked, and offers the one
    # thing to do about it: the blanker is already on, so the button releases it
    kinds = {k["kind"]: k for k in prof["kinds"]}
    assert set(kinds) == {"impulse"}, prof["kinds"]
    imp = kinds["impulse"]
    assert imp["window_s"] == prof["impulse_window_s"] and imp["active"] is True
    assert imp["action"] == {"label": "UNBLANK", "route": "/diversity/set", "query": "nb=off"}
    assert "blanking" in imp["detail"]
    st.set(nb=False)
    imp = {k["kind"]: k for k in st.status()["noise_profile"]["kinds"]}["impulse"]
    assert imp["active"] is False and imp["action"]["label"] == "BLANK"
    assert imp["action"]["query"].startswith("nb=on&nb_db=")
    rec = float(imp["action"]["query"].split("=")[-1])
    assert 6.0 <= rec <= 30.0 and rec <= prof["impulse_db"], (rec, prof)


# --- the beacon watch -----------------------------------------------------------
def test_beacon_watch_rides_along_once_aligned_and_answers_its_route():
    rng = np.random.default_rng(13)
    st = _DiversityState(_FakeAdapter(center_hz=14_120_000.0))
    st.BEACONS_PATH = None                                  # no store on disk in a test
    assert st.beacons_json() == {"available": False}
    st.aligner.set_lag(0, 20.0, True)
    st.ingest(_white(rng, BLOCK), _white(rng, BLOCK))
    out = st.beacons_json()
    assert out["available"] and out["band_hz"] == 14_100_000.0
    assert out["now"]["call"] and 0.0 <= out["now"]["seconds_left"] <= 10.0
    assert out["results"] == []
    # off 20 m: the watch idles (no band in span)
    st.a.center_hz = 7_200_000.0
    st.ingest(_white(rng, BLOCK), _white(rng, BLOCK))
    assert st.beacons_json()["band_hz"] is None


# --- the noise profile's kinds: what it is and what to do about it -------------
class _FixedProfile:
    def __init__(self, **over):
        self.rate_hz = RATE
        self.d = {"mains_hz": None, "hum_db": 0.0, "harmonics": 0, "impulses_per_s": 0.0,
                  "impulse_db": None, "periodic": [], "seconds": 2.0, "window_s": 2.0,
                  "impulse_window_s": 4.0}
        self.d.update(over)

    def update(self, a, b):
        pass

    def status(self):
        return dict(self.d)


class _FixedFilter:
    def __init__(self, found, notches=()):
        self.found = found
        self.notches = list(notches)

    def status(self):
        return {"notches": [{"hz": h, "width_hz": 140, "depth_db": -30.0} for h in self.notches],
                "anf": {"enabled": True, "found_hz": [f for f, _ in self.found],
                        "depth_db": [d for _, d in self.found]}}


def _kinds(st):
    return {k["kind"]: k for k in st.status()["noise_profile"]["kinds"]}


def test_mains_hum_offers_a_null_only_when_the_noise_has_a_direction():
    st = _DiversityState(_FakeAdapter())
    st.profile = _FixedProfile(mains_hz=60.0, hum_db=18.0, harmonics=5)
    k = _kinds(st)
    assert set(k) == {"mains"}
    m = k["mains"]
    assert m["label"] == "Mains hum · 60 Hz grid" and m["detail"] == "120 Hz comb, 5 harmonics"
    assert m["db"] == 18.0 and m["window_s"] == 2.0 and m["active"] is False
    assert m["action"] is None and m["why"] == "no noise estimate yet"
    # a tracker whose noise covariance is isotropic: nothing to null
    st.set(mode="track")
    from aether_gate.core.diversity import Tracker
    t = Tracker(RATE)
    t.Rn_guard = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)    # Rn reads the guard band
    st.trackers[0] = t
    m = _kinds(st)["mains"]
    assert m["action"] is None and m["why"].startswith("not directional enough")
    # a directional noise: the button nulls it
    t.Rn_guard = np.array([[1.0, 0.9], [0.9, 1.0]], dtype=complex)
    m = _kinds(st)["mains"]
    assert m["action"] == {"label": "NULL", "route": "/diversity/set", "query": "mode=null"}
    st.set(mode="null")
    m = _kinds(st)["mains"]
    assert m["active"] is True and m["action"]["label"] == "NULLED"
    assert m["action"]["query"] == "mode=track"


def test_periodic_lines_tones_and_the_floor_each_say_what_they_are():
    st = _DiversityState(_FakeAdapter())
    st.profile = _FixedProfile(periodic=[{"hz": 182.0, "db": 14.0}])
    st.a._filt = _FixedFilter(found=[(1240.0, -31.0), (2010.0, -28.0)], notches=[2000.0])
    k = st.status()["noise_profile"]["kinds"]
    assert [x["kind"] for x in k] == ["periodic", "tone", "tone"]
    assert k[0]["label"] == "Periodic · 182 Hz" and k[0]["action"] is None and k[0]["why"]
    assert k[1]["label"] == "Tone · 1240 Hz" and k[1]["db"] == -31.0
    assert k[1]["action"] == {"label": "NOTCH", "route": "/filter/notch",
                              "query": "add=1240&width=160"}
    assert k[2]["active"] is True and k[2]["action"] is None      # 2010 sits in the 2000 notch
    st.profile = _FixedProfile()
    st.a._filt = None
    k = st.status()["noise_profile"]["kinds"]
    assert [x["kind"] for x in k] == ["floor"] and k[0]["db"] is None
    assert k[0]["detail"].startswith("nothing mains-locked")


def test_the_station_grid_builds_the_beacon_watch_and_rides_in_its_status(tmp_path):
    st = _DiversityState(_FakeAdapter())
    st.BEACONS_PATH = str(tmp_path / "beacons.json")
    assert st.beacons is None and st.beacons_json()["available"] is False
    st.set(grid="EM10")
    assert st.beacons is not None and st.beacons_json()["station_grid"] == "EM10"
    with pytest.raises(ValueError):
        st.set(grid="ZZ99")
    assert st.beacons_json()["station_grid"] == "EM10"
    st.set(grid="")
    assert st.beacons_json()["station_grid"] is None


def test_the_profile_arms_the_blanker_unless_the_operator_says_on_or_off():
    st = _aligned_state()
    assert st.status()["nb"]["auto"]["mode"] == "auto"
    st.set(nb="on")
    assert st.nb_on is True
    assert st.status()["nb"]["auto"]["mode"] == "on"
    st.set(nb=False)                    # the old boolean callers still work
    assert st.nb_on is False
    assert st.status()["nb"]["auto"]["mode"] == "off"
    st.set(nb="auto")
    assert st.status()["nb"]["auto"]["mode"] == "auto"
    with pytest.raises(ValueError):
        st.set(nb="sometimes")


def test_a_polled_noise_verdict_is_kept_in_the_site_log_and_the_compass_says_why_not_yet():
    st = _aligned_state()
    rng = np.random.default_rng(3)
    for _ in range(3):
        st.ingest(_white(rng, BLOCK), _white(rng, BLOCK))
    st.status()                                     # the poll is what writes
    kinds = [e["kind"] for e in st.sitelog.read()]
    assert kinds.count("noise") == 1
    cp = st.compass_json()
    assert cp["available"] is False and "3" in cp["reason"]


# --- the optional stages (adapters.diversity_enhance) ---------------------------
def test_post_v2_stands_in_for_the_subband_stage_and_says_so():
    from aether_gate.core.diversity import combine_ramp
    rng = np.random.default_rng(21)
    st = _aligned_state(mode="USB")
    st.set(mode="track")
    rate = 25_000.0
    pa = rng.normal(size=4096) + 1j * rng.normal(size=4096)
    pb = rng.normal(size=4096) + 1j * rng.normal(size=4096)
    m = 0.5 + 0.2j
    assert st.status()["post"]["version"] == 1
    st.set(post="v2")
    assert st.post_on and st.enh.post_v2
    out = np.concatenate([st.combine_passband(0, pa, pb, m, m, rate) for _ in range(3)])
    # one frame of delay, then sample for sample; the sub-band combiner is
    # never built for this slice
    assert len(out) == 3 * 4096 and 0 not in st.subbands
    ps = st.status()["post"]
    assert ps["enabled"] and ps["version"] == 2 and ps["nfft"] == 256
    assert "gate" in ps and set(ps["gate"]) >= {"gaps", "noise_db"}
    assert "hold" in ps and "in_pause" in ps
    # back to v1: the ramp again until the tracker exists
    st.set(post=True)
    assert not st.enh.post_v2 and st.status()["post"]["version"] == 1
    assert np.allclose(st.combine_passband(0, pa, pb, m, m, rate), combine_ramp(pa, pb, m, m))
    st.set(post=False)
    assert not st.post_on


def test_mrc_refines_the_pan_from_the_map_and_reports_it():
    rng = np.random.default_rng(22)
    st = _aligned_state()
    st.set(mode="track")
    st.a._slice_hz = 3_601_000.0
    assert st.status()["mrc"] == {"enabled": False}
    st.set(mrc=True)
    # no map yet: the broadband weight (mrc_pan says None)
    out = _feed_scene(st, rng, 1, [])
    assert len(out) == BLOCK
    # a few frames on: the weights exist, the status carries their worth
    _feed_scene(st, rng, 6, [(-20_000, -10_000, 1.0, 1.0, 40.0)])
    ms = st.status()["mrc"]
    assert ms["enabled"] and ms["nfft"] == 4096 and ms["band_hz"] == [3_601_000.0, 3_604_000.0]
    assert ms["bins_used"] > 0
    st.set(mrc=False)
    assert st.status()["mrc"] == {"enabled": False} and st.enh._bw is None


def test_time_signals_ride_along_and_score_into_the_site_log():
    from aether_gate.core import timesignals as ts
    rng = np.random.default_rng(23)
    st = _DiversityState(_FakeAdapter(center_hz=3_330_000.0))      # CHU 3.330
    st.BEACONS_PATH = None
    assert st.timesignals_json() == {"available": False}
    st.aligner.set_lag(0, 20.0, True)
    st.set(grid="EM10")
    st.ingest(_white(rng, BLOCK), _white(rng, BLOCK))
    out = st.timesignals_json()
    assert out["available"] and out["freq_hz"] == 3_330_000.0
    assert out["station_grid"] == "EM10"
    # a whole window, then the gap after it: one result, kept in the log
    w = st.enh.timesignals
    t0 = (int(time.time() // ts.PERIOD_S) + 1) * ts.PERIOD_S
    n = int(BLOCK)
    for k in range(int(ts.WINDOW_S * RATE / n) + 2):
        w.update(_white(rng, n), _white(rng, n), 3_330_000.0, t0 + k * n / RATE)
    w.update(_white(rng, n), _white(rng, n), 3_330_000.0, t0 + ts.WINDOW_S + 1.0)
    assert w.last is not None
    st.ingest(_white(rng, BLOCK), _white(rng, BLOCK))            # the state notices
    assert list(st.sitelog.read(kind="beacon"))[-1]["callsign"] == "CHU"
    # a shared carrier can be named; a station that is not there cannot
    st.set(assume_hz=10_000_000.0, assume_call="WWVH")
    assert st.timesignals_json()["assumed"] == {"10000000": "WWVH"}
    with pytest.raises(ValueError):
        st.set(assume_hz=10_000_000.0, assume_call="CHU")


def test_the_compass_is_asked_at_the_slice_and_cached(monkeypatch):
    from aether_gate.adapters import diversity_enhance as de
    st = _aligned_state()
    st.a._slice_hz = 3_850_000.0
    calls = []

    def fake(log, bands_hz=None, since=None, phase_deg=None, f_hz=None):
        calls.append((phase_deg, f_hz))
        return {"available": False, "reason": "none yet"}
    monkeypatch.setattr(de._cp(), "compass_json", fake)
    assert st.compass_json()["reason"] == "none yet"
    st.compass_json()
    assert calls == [(-0.0, 3_850_000.0)] or calls == [(0.0, 3_850_000.0)]
    st.set(mode="manual", phase_deg=40.0)
    st.compass_json()
    assert len(calls) == 2 and calls[-1][0] == -40.0


def test_the_noise_bearing_rides_on_the_compass_payload_and_into_the_site_log(monkeypatch):
    """core/noisebearing wired up: the map's coherent floor becomes one
    bearing under the compass payload's "noise" key, and the same number is
    stamped on the site log's noise line. The compass fit is made up here --
    the fit itself is core/compass' business, tested there."""
    from aether_gate.core import compass
    st = _aligned_state()
    st.memory.names_path = None
    # nothing fed yet: the key is there and says why it is empty
    assert st.compass_json()["noise"] == {
        "available": False, "kind": None, "phase_deg": None, "coherence": None,
        "bearing_deg": None, "mirror_deg": None, "bins": 0, "since": None,
        "reason": "no spatial map yet"}
    fit = compass.GlobalFit(True, dtau_s=12e-9, d_m=4.0, baseline_deg=70.0,
                            quality=0.9, n_beacons=8, n_bands=3)
    monkeypatch.setattr(st.enh, "global_fit", lambda log, now: fit)
    st.enh._noise = None                                 # the 5 s cache, from above
    rng = np.random.default_rng(31)
    # 35 kHz of coherent hash below the centre, all of it from one direction
    _feed_scene(st, rng, 12, [(-45_000, -10_000, 1.1, 1.0, 1.0)])
    nb = st.compass_json()["noise"]
    assert nb["available"] and nb["kind"] == "floor" and nb["bins"] >= 100
    assert nb["coherence"] >= 0.4 and nb["since"] is not None
    # the phase went in as b = a * exp(1.1i), which is what the log means by
    # a ratio, and the made-up compass turns it into a bearing and its mirror
    assert abs((nb["phase_deg"] - math.degrees(1.1) + 180.0) % 360.0 - 180.0) <= 5.0
    seen = fit.bearing_from_phase(nb["phase_deg"], st.a.center_hz)["bearings_deg"]
    assert min(abs(nb["bearing_deg"] - b) for b in seen) <= 1.0
    assert nb["mirror_deg"] is not None
    st.status()                                          # the poll writes the line
    line = list(st.sitelog.read(kind="noise"))[-1]
    assert line["noise_bearing_deg"] == nb["bearing_deg"]
    assert line["noise_coherence"] is None or line["noise_coherence"] >= 0.0
