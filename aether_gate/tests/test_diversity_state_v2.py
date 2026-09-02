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
    assert st.status()["nb"] == {"enabled": True, "threshold_db": 12.0,
                                 "blanked_pct": pytest.approx(100 * 5 / BLOCK * 0.1, rel=0.2)}
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

def test_map_is_built_only_once_aligned_and_rebuilt_on_retune():
    rng = np.random.default_rng(2)
    st = _DiversityState(_FakeAdapter())
    _feed_scene(st, rng, 3, [])
    assert st.map is None and st.map_json()["available"] is False
    st.aligner.set_lag(0, 20.0, True)
    _feed_scene(st, rng, 3, [])
    first = st.map
    assert first is not None and first.frames == 3
    st.a.center_hz += 50_000.0                               # the bins are absolute frequencies
    _feed_scene(st, rng, 1, [])
    assert st.map is not first and st.map.frames == 1


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


def test_pan_and_source_are_one_setting():
    st = _aligned_state()
    assert st.set(pan="nulled")["source"] == "combined"      # v1 vocabulary has no 'nulled'
    assert st.set(source="b")["pan"] == "b"
    with pytest.raises(ValueError):
        st.set(pan="c")


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
