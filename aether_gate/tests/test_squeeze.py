#
# Aether-gate — SQUEEZE on synthetic two-channel spectra (no hardware).
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""core/squeeze.py, core/comb.py, the tracker's squeeze hook, and the notch
fallback in core/filter.py -- checked against STFT frames built directly in
the frequency domain (Squeeze.refresh()'s own input), so a target's ratio,
phase and coherence are set exactly rather than fitted from time-domain
noise.

Run:  .venv/bin/python -m pytest aether_gate/tests/test_squeeze.py -q
"""
import math

import numpy as np
import pytest

from aether_gate.core import comb
from aether_gate.core.diversity import Tracker, steering_of
from aether_gate.core.filter import SliceFilter, response_at
from aether_gate.core.squeeze import Squeeze
from aether_gate.adapters import diversity_squeeze as dsq

RATE = 48_000.0
N = 8192                          # bin_hz ~5.86 Hz -- fine enough for 1 kHz-spaced teeth
LO, HI = -3500.0, 3500.0          # a wide SSB-ish passband, baseband (signed) hertz


def _f():
    return np.fft.fftfreq(N, 1.0 / RATE)


def _floor(rng, power=1.0):
    """Independent white spectra on both channels -- the "nothing here"
    baseline every scene starts from."""
    f = _f()
    Xa = (rng.normal(size=N) + 1j * rng.normal(size=N)) * math.sqrt(power / 2.0)
    Xb = (rng.normal(size=N) + 1j * rng.normal(size=N)) * math.sqrt(power / 2.0)
    return Xa, Xb, f


def _add(rng, Xa, Xb, f, hz, ratio_db, phase_deg, power, coherent_frac=1.0, span_hz=40.0):
    """Add one source centred on hz: coherent_frac=1 makes B an EXACT scaled,
    rotated copy of A in those bins (coherence -> 1 exactly); 0 makes B an
    independent draw of the same power (coherence -> 0); in between mixes
    the two, so the true coherence is ~sqrt(coherent_frac)."""
    idx = np.flatnonzero(np.abs(f - hz) <= span_hz)
    if len(idx) == 0:
        idx = np.array([int(np.argmin(np.abs(f - hz)))])
    k = 10.0 ** (ratio_db / 20.0) * np.exp(1j * math.radians(phase_deg))
    src = (rng.normal(size=len(idx)) + 1j * rng.normal(size=len(idx))) / math.sqrt(2.0)
    src *= math.sqrt(power / max(1e-30, np.mean(np.abs(src) ** 2)))
    coh_part = k * src
    if coherent_frac >= 0.999:
        b_part = coh_part
    else:
        indep = (rng.normal(size=len(idx)) + 1j * rng.normal(size=len(idx))) / math.sqrt(2.0)
        indep *= abs(k) * math.sqrt(power)
        cf = max(0.0, coherent_frac)
        b_part = math.sqrt(cf) * coh_part + math.sqrt(1.0 - cf) * indep
    Xa[idx] += src
    Xb[idx] += b_part
    return idx


def _out_power(m, R):
    v = np.array([1.0, m], dtype=np.complex128)
    return float(np.real(v @ R @ np.conj(v))) / (1.0 + abs(m) ** 2)


def _region_depth_db(Xa, Xb, f, hz, width, m_before, m_after):
    """The same before/after measurement Squeeze.depth_db makes, recomputed
    independently in the test for one region -- used to check per-tooth
    depth without relying on core.squeeze's own aggregate number."""
    sel = np.abs(f - hz) <= width / 2.0
    A, B = Xa[sel], Xb[sel]
    R = np.array([[np.sum(np.abs(A) ** 2), np.sum(A * np.conj(B))],
                 [np.sum(np.conj(A) * B), np.sum(np.abs(B) ** 2)]], dtype=np.complex128)
    return 10.0 * math.log10(max(_out_power(m_before, R), 1e-30)
                             / max(_out_power(m_after, R), 1e-30))


# --- signal target: the null tool -------------------------------------------

def test_signal_target_coherent_holds_the_null_tool():
    rng = np.random.default_rng(1)
    sq = Squeeze(refresh_s=0.25)
    sq.set(1500.0, None, now=0.0)         # DEFAULT_WIDTH_HZ=300 -> +-150
    for _ in range(6):
        Xa, Xb, f = _floor(rng, power=1.0)
        # span covers the whole measurement window: no floor-only bins to
        # dilute the coherence estimate
        _add(rng, Xa, Xb, f, 1500.0, ratio_db=-4.0, phase_deg=157.0, power=40.0, span_hz=150.0)
        sq.refresh(np.stack([Xa, Xb]), f, LO, HI, dt=0.3, m_current=0j, now=0.0)
    assert sq.held and sq.reason is None
    assert sq.tool == "null"
    assert sq.coherence > 0.9
    assert sq.phase_deg == pytest.approx(157.0, abs=3.0)
    assert sq.ratio_db == pytest.approx(-4.0, abs=1.0)
    assert sq.depth_db >= 15.0
    assert "nulled" in sq.why


def test_too_weak_is_refused_and_the_weight_in_use_is_left_alone():
    rng = np.random.default_rng(2)
    sq = Squeeze(refresh_s=0.25)
    sq.set(1500.0, None, now=0.0)
    Xa, Xb, f = _floor(rng, power=1.0)
    _add(rng, Xa, Xb, f, 1500.0, ratio_db=0.0, phase_deg=10.0, power=1.02)   # barely over the floor
    sq.refresh(np.stack([Xa, Xb]), f, LO, HI, dt=0.3, m_current=0.4j, now=0.0)
    assert sq.held is False and sq.reason == "too weak"
    assert sq.tool is None and sq.depth_db is None


def test_outside_the_passband_is_refused():
    sq = Squeeze(refresh_s=0.25)
    sq.set(5000.0, None, now=0.0)                        # past HI=3500
    f = _f()
    Xa = np.zeros(N, dtype=np.complex128)
    Xb = np.zeros(N, dtype=np.complex128)
    sq.refresh(np.stack([Xa, Xb]), f, LO, HI, dt=0.3, m_current=0j, now=0.0)
    assert sq.held is False and sq.reason == "outside the passband"


# --- coherence chooses the tool, with hysteresis ----------------------------

def test_incoherent_signal_target_falls_to_the_notch_tool():
    rng = np.random.default_rng(3)
    sq = Squeeze(refresh_s=0.25)
    # a wide window (~270 bins): a single-frame coherence estimate under true
    # independence needs many bins to sit reliably clear of NOTCH_ENTER_COHERENCE
    sq.set(1500.0, 1600.0, now=0.0)
    for _ in range(6):
        Xa, Xb, f = _floor(rng, power=1.0)
        _add(rng, Xa, Xb, f, 1500.0, ratio_db=0.0, phase_deg=0.0, power=40.0,
            coherent_frac=0.0, span_hz=800.0)
        sq.refresh(np.stack([Xa, Xb]), f, LO, HI, dt=0.3, m_current=0j, now=0.0)
    assert sq.held and sq.reason is None
    assert sq.tool == "notch"
    assert sq.coherence < 0.35
    assert "not one direction" in sq.why and "notched" in sq.why


def test_coherence_hysteresis_does_not_flap_and_switches_at_the_right_edges():
    rng = np.random.default_rng(4)
    sq = Squeeze(refresh_s=0.25)
    sq.set(1500.0, 1600.0, now=0.0)       # a wide window: a stable estimate at every cf

    def _step(coherent_frac):
        Xa, Xb, f = _floor(rng, power=1.0)
        _add(rng, Xa, Xb, f, 1500.0, ratio_db=0.0, phase_deg=30.0, power=40.0,
            coherent_frac=coherent_frac, span_hz=800.0)
        sq.refresh(np.stack([Xa, Xb]), f, LO, HI, dt=0.3, m_current=0j, now=0.0)
        prev = sq.tool
        return prev

    # fully coherent: the null holds
    for _ in range(4):
        _step(1.0)
    assert sq.tool == "null" and sq.coherence >= 0.5

    # fully incoherent: the null gives way to the notch
    for _ in range(4):
        prev = sq.tool
        _step(0.0)
    assert sq.tool == "notch" and sq.coherence < 0.35

    # every step from here obeys the hysteresis rule for whatever it just measured
    for cf in (0.7, 0.3, 0.9, 0.1, 0.6):
        prev = sq.tool
        _step(cf)
        want = ("notch" if sq.coherence < 0.35 else "null") if prev == "null" else \
               ("null" if sq.coherence >= 0.5 else "notch")
        assert sq.tool == want, (prev, sq.coherence, sq.tool, want)


# --- comb target -------------------------------------------------------------

SPACING, OFFSET = 1000.0, 350.0
TEETH = [k * SPACING + OFFSET for k in range(-3, 4)]   # 7 teeth, -2650..3350 Hz


def _comb_scene(rng, ratio_db=-3.0, phase_deg=120.0, teeth=TEETH, tooth_power=300.0,
                talker=True):
    Xa, Xb, f = _floor(rng, power=1.0)
    for hz in teeth:
        # span matches the analysis window (comb.TOOTH_WIDTH_HZ/2) so every
        # bin the null/depth measurement pools is actually the tooth, not floor
        _add(rng, Xa, Xb, f, hz, ratio_db, phase_deg, tooth_power, span_hz=comb.TOOTH_WIDTH_HZ / 2.0)
    if talker:
        # a broadband "voice" from a different direction, spread thinly so no
        # single bin looks like a tooth to comb._peaks's own local-median test;
        # carved away from each tooth's own analysis window so a real, much
        # louder switch-mode comb (the reason it is worth a SQUEEZE at all)
        # is not diluted bin-for-bin by a talker sharing the exact tooth bins
        sel = (f >= -2800.0) & (f < 2800.0) & ~comb.teeth_mask(f, teeth, comb.TOOTH_WIDTH_HZ)
        idx = np.flatnonzero(sel)
        # 110 deg keeps the talker's own steering well clear of the comb's
        # (-3 dB / 120 deg): nulling the comb costs it well under 3 dB
        k = 1.3 * np.exp(1j * math.radians(110.0))
        src = (rng.normal(size=len(idx)) + 1j * rng.normal(size=len(idx))) / math.sqrt(2.0)
        src *= math.sqrt(3.0 / max(1e-30, np.mean(np.abs(src) ** 2)))
        Xa[idx] += src
        Xb[idx] += k * src
    return Xa, Xb, f


def test_comb_auto_detect_recovers_spacing_and_offset_under_a_talker():
    rng = np.random.default_rng(5)
    sq = Squeeze(refresh_s=0.25)
    sq.set_comb_auto(now=0.0)
    # ~2 s of independent blocks so the talker's own periodogram noise
    # averages down and only the steady teeth accumulate coherently
    for _ in range(24):
        Xa, Xb, f = _comb_scene(rng)
        sq.refresh(np.stack([Xa, Xb]), f, LO, HI, dt=0.1, m_current=0j, now=0.0)
    assert sq.comb_spacing_hz is not None, sq.reason
    assert sq.comb_spacing_hz == pytest.approx(SPACING, rel=0.01)
    assert (sq.comb_offset_hz % SPACING) == pytest.approx(OFFSET, abs=20.0)
    assert sq.target == "comb"


def test_comb_explicit_holds_null_deep_on_every_tooth_and_spares_the_talker():
    rng = np.random.default_rng(6)
    sq = Squeeze(refresh_s=0.25)
    sq.set_comb(SPACING, OFFSET, now=0.0)
    Xa = Xb = f = None
    for _ in range(6):
        Xa, Xb, f = _comb_scene(rng, teeth=TEETH)
        sq.refresh(np.stack([Xa, Xb]), f, LO, HI, dt=0.3, m_current=0j, now=0.0, ref_hz=0.0)
    assert sq.held and sq.tool == "null", sq.reason
    assert sorted(round(x) for x in sq.teeth_in_band) == sorted(round(t) for t in TEETH)
    assert sq.teeth_seen == len(TEETH)
    from aether_gate.core.focus import null_of
    m_null = null_of(sq.s)
    for hz in TEETH:
        d = _region_depth_db(Xa, Xb, f, hz, comb.TOOTH_WIDTH_HZ, 0j, m_null)
        assert d >= 15.0, (hz, d)
    # the talker: s_talker = [1, 1.3*e^{j110deg}] built into _comb_scene
    s_t = np.array([1.0, 1.3 * np.exp(1j * math.radians(110.0))])
    g_before = abs(s_t[0])
    g_after = abs(s_t[0] + m_null * s_t[1])
    cost_db = 20.0 * math.log10(max(g_before, 1e-12) / max(g_after, 1e-12))
    assert abs(cost_db) <= 3.0, cost_db


def test_comb_retune_keeps_the_same_absolute_teeth_held():
    # a wider physical comb than TEETH alone: retuning slides the window, so
    # teeth at its edges change even though the comb itself has not -- inject
    # enough of the (infinite) comb that both windows are fully populated
    wide = [k * SPACING + OFFSET for k in range(-8, 12)]
    rng = np.random.default_rng(7)
    sq = Squeeze(refresh_s=0.25)
    sq.set_comb(SPACING, OFFSET, now=0.0)
    Xa, Xb, f = _comb_scene(rng, teeth=wide)                    # ref_hz=0: baseband == absolute
    sq.refresh(np.stack([Xa, Xb]), f, LO, HI, dt=0.3, m_current=0j, now=0.0, ref_hz=0.0)
    want0 = comb.teeth_in_band(SPACING, OFFSET, LO, HI, 0.0)
    assert sq.held and sorted(round(x) for x in sq.teeth_in_band) == sorted(round(x) for x in want0)
    # retune 2.5 kHz: the SAME absolute comb, now 2.5 kHz lower in baseband
    Xa2, Xb2, f2 = _comb_scene(rng, teeth=[t - 2500.0 for t in wide])
    sq.refresh(np.stack([Xa2, Xb2]), f2, LO, HI, dt=0.3, m_current=0j, now=1.0, ref_hz=2500.0)
    assert sq.held
    assert sq.comb_spacing_hz == pytest.approx(SPACING, rel=0.001)
    assert (sq.comb_offset_hz % SPACING) == pytest.approx(OFFSET, abs=1.0)
    want1 = comb.teeth_in_band(SPACING, OFFSET, LO, HI, 2500.0)
    assert sorted(round(x) for x in sq.teeth_in_band) == sorted(round(x) for x in want1)
    assert want1 != want0        # the retune genuinely moved which teeth are in band


def test_comb_with_only_two_teeth_is_refused():
    rng = np.random.default_rng(8)
    sq = Squeeze(refresh_s=0.25)
    sq.set_comb_auto(now=0.0)
    for _ in range(24):
        Xa, Xb, f = _comb_scene(rng, teeth=[350.0, 1350.0], talker=False)
        sq.refresh(np.stack([Xa, Xb]), f, LO, HI, dt=0.1, m_current=0j, now=0.0)
    assert sq.held is False
    assert sq.reason == "no comb found"
    assert sq.target == "comb"


# --- status shape, release, the tracker's hook ------------------------------

STATUS_KEYS = {"hz", "width_hz", "held", "reason", "tool", "why", "phase_deg", "ratio_db",
              "coherence", "depth_db", "scope", "target", "comb", "since"}


def test_status_has_every_key_held_and_off():
    sq = Squeeze()
    assert STATUS_KEYS <= set(sq.status().keys())
    sq.set(1000.0, None, now=0.0)
    assert STATUS_KEYS <= set(sq.status().keys())
    sq.off()
    st = sq.status()
    assert STATUS_KEYS <= set(st.keys())
    assert st["hz"] is None and st["held"] is False and st["target"] == "signal"


def test_tracker_squeeze_hook_wins_and_release_lets_the_tracker_refit():
    t = Tracker(25_000.0)
    n = int(25_000.0 * 0.3)
    Rg = np.array([[1.0, 0.3 + 0.1j], [0.3 - 0.1j, 1.0]], dtype=np.complex128)
    Ri = np.array([[1.0, 0.3 + 0.1j], [0.3 - 0.1j, 1.0]], dtype=np.complex128)

    class _FakeSqueeze:
        held = True
        scope = "passband"
        null_m = 0.6 * np.exp(1j * 1.2)

    t.update(Ri, Rg, n, "track", squeeze=_FakeSqueeze())
    assert t.m == _FakeSqueeze.null_m
    for _ in range(40):
        t.update(Ri, Rg, n, "track", squeeze=None)
    assert t.m != _FakeSqueeze.null_m


# --- the notch bank in core/notchbank.py, wired via core/filter.py ---------

def test_filter_squeeze_notches_are_deep_regardless_of_shape():
    """The whole point of core/notchbank.py: the OLD behaviour (squeeze
    notches folded into the FIR design) only reached real depth with
    "sharp"'s long Kaiser window -- "soft"'s wide main lobe barely dented a
    notch this narrow (see git history of this test). The dedicated bank
    does not care what `shape` the operator picked."""
    for shape in ("soft", "sharp"):
        filt = SliceFilter(24_000.0)
        filt.set(low=100.0, high=2900.0, shape=shape)
        filt.set_squeeze_notches([(t, 140.0) for t in (650.0, 1650.0)])
        filt.apply(np.zeros(64, dtype=np.complex128))   # forces the FIR redesign
        for hz in (650.0, 1650.0):
            assert filt.combined_response_db(hz) <= -25.0, (shape, hz)
        # a talker's own frequency, well clear of either tooth, is untouched
        assert abs(filt.combined_response_db(1200.0)) <= 1.0, shape
        assert filt.spec.notches == []                   # never mixed into the operator's own table
        # the FIR's OWN taps no longer carry the squeeze notch at all -- it
        # is entirely the bank's doing now
        assert response_at(filt.taps, 24_000.0, 650.0) > -3.0, shape


SLICE_RATE = 24_000.0
SLICE_BLOCK = 819                # a realistic post-decimation block (test_filter.py's own figure)


def _voice_like_slice(seconds, amp, low=200.0, high=2800.0, seed=11):
    """Band-limited noise, a stand-in for a talker (same recipe as
    test_filter.py's own _voice_like, duplicated here to keep this file
    self-contained)."""
    from aether_gate.core.filter import design_taps as _dt
    rng = np.random.default_rng(seed)
    total = int(seconds * SLICE_RATE)
    w = amp * (rng.standard_normal(total + 1100) + 1j * rng.standard_normal(total + 1100))
    return np.convolve(w, _dt(SLICE_RATE, low, high, "sharp"), mode="valid")[:total]


def _feed_slice(sf, sig):
    out = []
    for i in range(0, len(sig) - SLICE_BLOCK + 1, SLICE_BLOCK):
        out.append(sf.apply(sig[i:i + SLICE_BLOCK], 0))
    return np.concatenate(out)


def _tone_db_slice(y, hz):
    """A matched-filter correlation against one exact tone over the whole
    record: at ~2 s this is a ~0.5 Hz-wide bin, comfortably clear of teeth
    60 Hz apart or a talker's own broadband floor."""
    t = np.arange(len(y)) / SLICE_RATE
    return 20.0 * math.log10(abs(np.mean(y * np.exp(-2j * math.pi * hz * t))) + 1e-30)


def _band_db_slice(y, lo, hi, avoid_hz, guard=45.0):
    """The talker band's own periodogram level, away from any tooth -- the
    "the talker is untouched" half of the comb test. y is complex baseband
    IQ (fft, not rfft: there is no real-signal symmetry to exploit here)."""
    Y = np.fft.fft(y)
    f = np.fft.fftfreq(len(y), 1.0 / SLICE_RATE)
    sel = (f >= lo) & (f <= hi)
    for hz in avoid_hz:
        sel &= np.abs(f - hz) > guard
    return 10.0 * math.log10(float(np.mean(np.abs(Y[sel]) ** 2)) + 1e-30)


def _comb_scene_slice(teeth, tone_db_below_talker=-20.0, talker_amp=0.05):
    talker = _voice_like_slice(2.0, talker_amp, low=350.0, high=2400.0)
    t = np.arange(len(talker)) / SLICE_RATE
    tone_amp = talker_amp * 10.0 ** (tone_db_below_talker / 20.0)
    sig = talker + sum(tone_amp * np.exp(2j * math.pi * hz * t) for hz in teeth)
    return sig.astype(np.complex128)


def _comb_case(shape):
    from aether_gate.core.comb import TOOTH_WIDTH_HZ
    teeth = [700.0 + 60.0 * k for k in range(5)]        # a 60 Hz comb, 5 teeth in-band
    sig = _comb_scene_slice(teeth)

    before_sf = SliceFilter(SLICE_RATE)
    before_sf.set(low=350.0, high=2400.0, shape=shape)
    before = _feed_slice(before_sf, sig.copy())
    before_band = _band_db_slice(before, 350.0, 2400.0, teeth)

    after_sf = SliceFilter(SLICE_RATE)
    after_sf.set(low=350.0, high=2400.0, shape=shape)
    after_sf.set_squeeze_notches([(hz, TOOTH_WIDTH_HZ) for hz in teeth])
    after = _feed_slice(after_sf, sig.copy())
    after_band = _band_db_slice(after, 350.0, 2400.0, teeth)

    for hz in teeth:
        depth = _tone_db_slice(before, hz) - _tone_db_slice(after, hz)
        assert depth >= 25.0, (shape, hz, depth)
    assert abs(after_band - before_band) <= 1.0, (shape, before_band, after_band)


def test_comb_notch_bank_soft_shape_deep_teeth_talker_untouched():
    _comb_case("soft")


def test_comb_notch_bank_sharp_shape_deep_teeth_talker_untouched():
    _comb_case("sharp")


def test_signal_squeeze_notch_bank_deep_at_centre_and_shows_in_response_db():
    hz, width = 1200.0, 300.0
    talker = _voice_like_slice(2.0, 0.05)
    t = np.arange(len(talker)) / SLICE_RATE
    tone = 0.05 * 10.0 ** (-20.0 / 20.0) * np.exp(2j * math.pi * hz * t)
    sig = (talker + tone).astype(np.complex128)

    before_sf = SliceFilter(SLICE_RATE)
    before_sf.set(low=100.0, high=2900.0, shape="soft")
    before = _feed_slice(before_sf, sig.copy())

    after_sf = SliceFilter(SLICE_RATE)
    after_sf.set(low=100.0, high=2900.0, shape="soft")
    after_sf.set_squeeze_notches([(hz, width)])
    after = _feed_slice(after_sf, sig.copy())

    assert _tone_db_slice(before, hz) - _tone_db_slice(after, hz) >= 25.0

    rdb = after_sf.response_db(points=512)
    f = np.array(rdb["hz"], dtype=float)
    idx = int(np.argmin(np.abs(f - hz)))
    assert rdb["db"][idx] <= -20.0, rdb["db"][idx]        # the curve the VISUAL tab draws shows it
    assert rdb["db"][int(np.argmin(np.abs(f - 300.0)))] > -3.0   # elsewhere in the passband, untouched


def test_squeeze_notch_bank_coefficient_cache_skips_recompute_when_unchanged():
    from aether_gate.core.notchbank import NotchBank
    bank = NotchBank(SLICE_RATE)
    bank.set_targets([(1000.0, 100.0), (1400.0, 100.0)])
    assert bank.recomputes == 1
    bank.set_targets([(1000.0, 100.0), (1400.0, 100.0)])         # unchanged -> no rebuild
    assert bank.recomputes == 1
    bank.set_targets([(1000.04, 100.0), (1400.0, 100.0)])        # rounds to the same table
    assert bank.recomputes == 1
    bank.set_targets([(1050.0, 100.0), (1400.0, 100.0)])         # a genuinely new target
    assert bank.recomputes == 2
    bank.set_targets([])
    assert bank.recomputes == 3
    bank.set_targets([])
    assert bank.recomputes == 3


def test_slice_filter_set_squeeze_notches_is_a_no_op_when_unchanged():
    sf = SliceFilter(SLICE_RATE)
    sf.set_squeeze_notches([(1000.0, 100.0)])
    assert sf._squeeze_bank.recomputes == 1
    sf.set_squeeze_notches([(1000.0, 100.0)])
    assert sf._squeeze_bank.recomputes == 1
    assert sf.squeeze_notches == [(1000.0, 100.0)]
    sf.set_squeeze_notches([])
    assert sf._squeeze_bank.recomputes == 2
    assert sf.squeeze_notches == []


def test_diversity_squeeze_sync_notches_wires_the_adapter_to_the_filter():
    class _A:
        pass
    class _State:
        pass
    a = _A()
    a._filt = SliceFilter(RATE)
    a._filt.set(low=100.0, high=2900.0)
    st = _State()
    st.a = a
    sq = Squeeze()
    sq.set_comb(SPACING, OFFSET, now=0.0)
    sq.teeth_in_band = [350.0, 1350.0]
    sq.tool, sq.held = "notch", True
    dsq.sync_notches(st, sq)
    assert sorted(round(hz, 1) for hz, _w in a._filt.squeeze_notches) == [350.0, 1350.0]
    sq.tool = "null"
    dsq.sync_notches(st, sq)
    assert a._filt.squeeze_notches == []


# --- the adapter's set()/status() path --------------------------------------

def test_diversity_state_set_squeeze_then_off():
    from aether_gate.adapters.diversity_state import _DiversityState

    class _FakeAdapter:
        def __init__(self):
            self._np = np
            self.samp_rate = 125_000.0
            self.center_hz = 3_600_000.0
            self._mode = "USB"

    st = _DiversityState(_FakeAdapter())
    st.memory.names_path = None
    st.set(squeeze=1500, squeeze_width=250)
    out = st.status()["squeeze"]
    assert STATUS_KEYS <= set(out.keys())
    assert out["hz"] == 1500 and out["width_hz"] == 250 and out["target"] == "signal"
    assert out["held"] is False and out["reason"] == "not measured yet"
    st.set(squeeze="off")
    out = st.status()["squeeze"]
    assert out["hz"] is None and out["held"] is False
