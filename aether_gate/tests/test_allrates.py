#
# Aether-gate — the diversity DSP driven at every RSPduo rate and every HF
# centre the operator actually asked for, no hardware.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""So far only 125 k / 250 k S/s on 80 m and 20 m have been exercised live.
The standing rule is every band 160 m .. 10 m and every span the RSPduo
offers, 62.5 k .. 2.04 MS/s. This file drives finder, spatial, the noise
profile, the sub-band combiner, kinds, the AGC, the auto contour, the voice
print and the talker memory at all six rates (and, where a centre is an
input, all six band centres), against synthetic input built the way
adapters/diversity_state.py builds it for the real reader thread:

  * finder / spatial / kinds     one (2, MAP_BINS) natural-order spectrum
    per CHUNK-sample block (_map_update: MAP_BINS = 2048, a Hann window,
    frame_s = CHUNK / rate, CHUNK = 4096 -- adapters/soapy.py's raw block
    read length);
  * noise profile                the raw aligned block pair, before the
    windowed FFT (ingest() feeds NoiseProfile.update() straight off the
    reader, at CHUNK samples a call);
  * noise bearing                not a streaming stage at all: it reads the
    spatial map the reader has already built, once per poll, so its row
    below is one call per span against a map fed the same frames as the
    spatial test above (adapters/diversity_enhance.py caches it for 5 s);
  * sub-band / voice print / contour   the demodulated PASSBAND pair, at
    _pd_rate = samp_rate // AUDIO_RATE's floor (adapters/soapy.py
    _init_demod), which is why their own "rate" argument is a few values
    from 24.0 to 31.25 kHz rather than the RF span itself;
  * AGC                          fixed at AUDIO_RATE (24 kHz): soapy.py's
    audio path always resamples pd_rate -> AUDIO_RATE before agc.process
    runs (adapters/soapy.py:1941), so the RF span never reaches it. Its
    row in the table below is the same call at every span, for the record;
  * talker memory                takes no rate at all -- entries are
    steering vectors and weights, unitless. Its row is likewise one call
    repeated per span, for the record.

One real defect was found and fixed here (see test_finder_window_width_
tracks_voice_width_at_every_rate and its neighbour): Finder's decimated
grid held SPATIAL_POINTS = 512 fixed regardless of the span, so step_hz
grew with the rate and the ~2.7 kHz voice window it is built around
ballooned to 5.9 kHz at 1.02 MS/s and 11.95 kHz at 2.04 MS/s -- 2-4x too
coarse to localise a station "within one bin" as the operator's brief
requires. core/finder.py now scales its default `points` with the span,
capped at nbins (MAP_BINS), which reproduces the exact old behaviour at
every rate already exercised live (<= 500 kS/s, where step_hz's growth had
not yet outrun VOICE_WIDTH_HZ) and holds the window near 2.7-3.0 kHz at
1.02 and 2.04 MS/s too. See POINTS_REFERENCE_HZ / _default_points there.

Run:  nice -n 15 ./.venv/bin/python -m pytest aether_gate/tests/test_allrates.py -q
"""
import json
import math
import time

import numpy as np
import pytest

from aether_gate.core import kinds
from aether_gate.core.agc import Agc
from aether_gate.core.contour import fit_contour, PROFILE_BANDS
from aether_gate.core.diversity import ALIGN_MIN_PEAK, combine
from aether_gate.core.finder import (DIAL_GRID_HZ, EDGE_MARGIN_HZ, FAST_FRAMES,
                                     Finder, LiveSpatial, VOICE_SCORE)
from aether_gate.core import compass, noisebearing
from aether_gate.core.noiseprofile import NoiseProfile
from aether_gate.core.spatial import SOURCE_MIN_COHERENCE, SpatialMap
from aether_gate.core.subband import NFFT, SubbandCombiner
from aether_gate.core.talkermemory import TalkerMemory
from aether_gate.core.voiceprint import VoicePrint
from aether_gate.core import alignsearch
from aether_gate.tests.test_alignsearch import _pair as _lag_pair

# --- the matrix --------------------------------------------------------
RATES_HZ = (62_500.0, 125_000.0, 250_000.0, 500_000.0, 1_000_000.0, 2_040_000.0)
CENTERS_HZ = (1_900_000.0, 3_800_000.0, 7_100_000.0, 14_200_000.0, 21_200_000.0,
             28_400_000.0)      # 160 m .. 10 m
MAP_BINS = 2048           # adapters/diversity_state.py _DiversityState.MAP_BINS
CHUNK = 4096               # adapters/soapy.py's raw block read length (see there, ~928)
AUDIO_RATE = 24_000.0      # adapters/soapy.py AUDIO_RATE

REALTIME_S = 0.5           # the operator's bound at 2.04 MS/s
REALTIME_GENERAL_S = 1.0   # "nothing takes longer than real time" at any span


def _pd_rate(samp_rate, audio_rate=AUDIO_RATE):
    """adapters/soapy.py _init_demod's post-decimation rate: what subband,
    the voice print and the passband tracker actually see, whatever the RF
    span is (self._decim = max(1, int(samp_rate // AUDIO_RATE)))."""
    decim = max(1, int(samp_rate // audio_rate))
    return samp_rate / decim


PD_RATES_HZ = tuple(_pd_rate(r) for r in RATES_HZ)

# --- the report table: filled in as the tests run, printed at the end --
TIMING_ROWS = []       # (module, rate_label, input_s, wall_s, realtime_factor)


def _record(module, rate, input_s, wall_s):
    factor = wall_s / input_s if input_s else float("nan")
    TIMING_ROWS.append((module, rate, input_s, wall_s, factor))
    return factor


def _rid(r):
    return f"{r / 1e3:.1f}k" if r < 1e6 else f"{r / 1e6:.3f}M"


def _cid(c):
    return f"{c / 1e6:.1f}M"


def _assert_finite_json(obj):
    """obj must round-trip through json.dumps with only finite numbers --
    the contract every status()/candidates()/map() dict on the wire has to
    meet (a NaN or a bare numpy scalar breaks the HTTP layer)."""
    text = json.dumps(obj)  # raises on a numpy scalar that slipped through round()/float()
    assert "NaN" not in text and "Infinity" not in text, text


@pytest.fixture(scope="session", autouse=True)
def _print_timing_table():
    yield
    if not TIMING_ROWS:
        return
    print("\n\n=== test_allrates timing table (module, rate, input_s, wall_s, realtime_factor) ===")
    for module, rate, input_s, wall_s, factor in TIMING_ROWS:
        print(f"{module:<24} {rate:<10} input={input_s:6.3f}s  wall={wall_s:7.4f}s  "
              f"factor={factor:6.3f}x")


# =========================================================================
# finder / live spatial / spatial map / kinds
# =========================================================================

def _white_frame(rng, n):
    Xa = (rng.normal(size=n) + 1j * rng.normal(size=n)) / math.sqrt(2)
    Xb = (rng.normal(size=n) + 1j * rng.normal(size=n)) / math.sqrt(2)
    return Xa, Xb


def _voice_scene_frames(rng, n_frames, rate, offset_hz, phase_deg=50.0, ratio_db=-2.0,
                        snr=24.0, syllable_hz=4.0, nbins=MAP_BINS):
    """Frames shaped exactly as _map_update hands them to map/live/finder:
    a natural-FFT-order (2, nbins) spectrum per CHUNK-sample block, with a
    coherent 300-2700 Hz voice patch offset_hz from the centre (fixed
    inter-loop phase/ratio, syllable-gated) on a white floor on both loops.
    Yields (X, sel): sel is the boolean index of the voice bins, natural
    order, for the caller to check the spatial map against directly."""
    f = np.fft.fftfreq(nbins, 1.0 / rate)
    lo, hi = offset_hz + 300.0, offset_hz + 2700.0
    sel = (f >= lo) & (f < hi)
    # at 2.04 MS/s a MAP_BINS raw bin is ~1 kHz (rate/2048), so a 2.4 kHz
    # voice patch is only 2-3 raw bins there -- expected, not a bug.
    assert sel.sum() >= 2, "the voice patch fell outside the span for this rate"
    ratio = 10 ** (ratio_db / 20.0)
    phase = math.radians(phase_deg)
    frame_s = CHUNK / rate
    t = 0.0
    for _ in range(n_frames):
        Xa, Xb = _white_frame(rng, nbins)
        env = 1.0 if (t * syllable_hz) % 1.0 < 0.5 else 0.05
        s = ((rng.normal(size=sel.sum()) + 1j * rng.normal(size=sel.sum()))
             * math.sqrt(snr * env / 2))
        Xa[sel] += s
        Xb[sel] += s * ratio * np.exp(1j * phase)
        yield np.stack([Xa, Xb]), sel
        t += frame_s


FRAME_TARGET_S = 5.0   # real seconds of scene fed per (rate, centre) below.
                       # The finder scores nothing until its ring holds
                       # FAST_FRAMES // 2 SLOTS of SLOT_S (core/finder.py), which
                       # is ~4.2 s of band at every span -- frames alone are not
                       # enough above 125 kS/s, where they arrive far faster than
                       # syllables happen.


def _finder_spatial_case(rate, center, seed):
    rng = np.random.default_rng(seed)
    offset = 0.30 * (rate / 2.0)     # inside the span, clear of DC and the edges
    frame_s = CHUNK / rate
    n_frames = max(FAST_FRAMES, int(math.ceil(FRAME_TARGET_S / frame_s)))
    fd = Finder(MAP_BINS, rate)
    live = LiveSpatial(MAP_BINS, rate)
    sm = SpatialMap(MAP_BINS, rate)
    sel = None
    t0 = time.perf_counter()
    for X, sel in _voice_scene_frames(rng, n_frames, rate, offset):
        fd.update(X, frame_s)
        live.update(X, frame_s)
        sm.update(X, frame_s)
    wall = time.perf_counter() - t0
    return fd, live, sm, sel, offset, n_frames, frame_s, wall


@pytest.mark.parametrize("center", CENTERS_HZ, ids=_cid)
@pytest.mark.parametrize("rate", RATES_HZ, ids=_rid)
def test_finder_locates_the_voice_patch_and_labels_it_voice(rate, center):
    fd, live, sm, sel, offset, n_frames, frame_s, wall = _finder_spatial_case(
        rate, center, seed=(int(rate) ^ int(center)) & 0xFFFFFFFF)
    input_s = n_frames * frame_s
    _record("finder+live+spatial", _rid(rate), input_s, wall)

    out = fd.candidates(center, live)
    assert out["available"], f"no candidates at all at {rate} / {center}"
    _assert_finite_json(out)
    cands = out["candidates"]
    assert cands, f"the voice patch was not found at rate={rate} centre={center}"

    bin_hz = rate / MAP_BINS
    tol_hz = max(bin_hz, 500.0)
    true_lo, true_hi = center + offset + 300.0, center + offset + 2700.0
    true_mid = 0.5 * (true_lo + true_hi)

    # the dial is DELIBERATELY not the energy's centre (core/finder.py
    # _dial_hz: it sits EDGE_MARGIN_HZ outside the energy, then snaps to the
    # DIAL_GRID_HZ grid) plus the window's own half-width, on top of the
    # brief's bin/500 Hz localisation tolerance.
    slop = tol_hz + EDGE_MARGIN_HZ + DIAL_GRID_HZ

    def _near(c):
        return abs(c["hz"] - true_mid) <= slop + c["width_hz"] / 2

    hit = next((c for c in cands if _near(c)), None)
    assert hit is not None, (
        f"no candidate near {true_mid:.0f} Hz (tol {tol_hz:.0f}) at rate={rate} "
        f"centre={center}: {[c['hz'] for c in cands]}")
    assert hit["kind"] == "voice", hit
    assert hit["score"] >= VOICE_SCORE, hit


@pytest.mark.parametrize("rate", RATES_HZ, ids=_rid)
def test_spatial_map_phase_is_right_at_the_source_and_low_elsewhere(rate):
    # centre does not change the spatial-map maths (only the labels it
    # reports frequencies under), so this is rate-only: SOURCE_MIN_COHERENCE
    # and the map's own bins are what is under test.
    fd, live, sm, sel, offset, n_frames, frame_s, wall = _finder_spatial_case(
        rate, center=10_000_000.0, seed=int(rate))
    coh, steer, _m, _lvl = sm._analyse()
    # spatial.py's steer is angle(R[:,0,1]) = angle(Xa * conj(Xb)): for a
    # source injected on B as Xb = ratio*exp(i*phase_deg)*Xa (which is how
    # _voice_scene_frames builds it), that comes out to -phase_deg, not
    # +phase_deg -- verified against SpatialMap directly, not assumed.
    src_phase = -math.radians(50.0)
    inside = np.degrees(np.abs(((steer[sel] - src_phase + math.pi) % (2 * math.pi)) - math.pi))
    assert np.all(coh[sel] >= SOURCE_MIN_COHERENCE), (
        f"coherence at the source dropped below {SOURCE_MIN_COHERENCE} at rate={rate}: "
        f"{coh[sel].min():.2f}")
    assert np.mean(inside) <= 15.0, f"mean phase error {np.mean(inside):.1f} deg at rate={rate}"
    outside = ~sel
    # far from the source: a handful of bins next to the patch can still be
    # smoothed into it (SMOOTH_BINS=9), so the median over the whole rest of
    # the span is the honest measure of "elsewhere is not coherent".
    assert np.median(coh[outside]) < 0.3, (
        f"coherence away from the source is not low at rate={rate}: "
        f"{np.median(coh[outside]):.2f}")


@pytest.mark.parametrize("rate", RATES_HZ, ids=_rid)
def test_finder_window_width_tracks_voice_width_at_every_rate(rate):
    """Regression guard for the fix in core/finder.py (_default_points):
    the window Finder scores voice against must stay close to
    VOICE_WIDTH_HZ (2700 Hz) at every span, not balloon once step_hz
    outgrows it. See the module docstring above for the defect this
    replaces."""
    from aether_gate.core.finder import VOICE_WIDTH_HZ
    fd = Finder(MAP_BINS, rate)
    width_hz = fd.win * fd.step_hz
    assert width_hz <= 1.5 * VOICE_WIDTH_HZ, (
        f"finder window at rate={rate} is {width_hz:.0f} Hz, "
        f"{width_hz / VOICE_WIDTH_HZ:.2f}x VOICE_WIDTH_HZ")


@pytest.mark.parametrize("rate", RATES_HZ, ids=_rid)
def test_finder_and_spatial_process_one_second_faster_than_real_time(rate):
    frame_s = CHUNK / rate
    n_frames = max(1, round(1.0 / frame_s))
    rng = np.random.default_rng(1)
    frames = [np.stack(_white_frame(rng, MAP_BINS)) for _ in range(n_frames)]
    fd = Finder(MAP_BINS, rate)
    live = LiveSpatial(MAP_BINS, rate)
    sm = SpatialMap(MAP_BINS, rate)
    t0 = time.perf_counter()
    for X in frames:
        fd.update(X, frame_s)
        live.update(X, frame_s)
        sm.update(X, frame_s)
    wall = time.perf_counter() - t0
    _record("finder+live+spatial (rt)", _rid(rate), 1.0, wall)
    assert wall < REALTIME_GENERAL_S, f"{rate}: {wall:.3f}s for 1s of input"
    if rate == max(RATES_HZ):
        assert wall < REALTIME_S, f"{rate}: {wall:.3f}s for 1s of input (must be < {REALTIME_S}s)"


# =========================================================================
# noise bearing
# =========================================================================

NOISE_BEARING_DEG = 200.0    # where the made-up pair below is told the hash is
NOISE_BASELINE_DEG = 70.0
NOISE_SNR = 4.0              # 7 dB over the floor: hash, not a transmission


def _noise_scene(rate, seed, bearing_deg=NOISE_BEARING_DEG, station=None):
    """A spatial map fed a coherent hash band covering 5-90 % of the span's
    upper half at a phase a made-up compass says is `bearing_deg`, and that
    compass. Everything is a FRACTION of the span, so the same scene is the
    same scene at 62.5 k and at 2.04 MS/s. `station` adds a 3 kHz signal
    25 dB over the hash, pointing somewhere else, present from the first
    frame so the map's floor tracker learns it rather than refusing it."""
    fit = compass.GlobalFit(True, dtau_s=12e-9, d_m=4.0,
                            baseline_deg=NOISE_BASELINE_DEG, quality=0.95,
                            n_beacons=8, n_bands=3)
    center = 7_100_000.0
    f = np.fft.fftfreq(MAP_BINS, 1.0 / rate)
    sel = (f >= 0.05 * rate / 2) & (f < 0.90 * rate / 2)
    at_hz = center + float(np.mean(f[sel]))
    phase = math.radians(fit.phase_at(bearing_deg, at_hz))
    rng = np.random.default_rng(seed)
    sm = SpatialMap(MAP_BINS, rate)
    frame_s = CHUNK / rate
    for _ in range(40):
        Xa, Xb = _white_frame(rng, MAP_BINS)
        s = ((rng.normal(size=int(sel.sum())) + 1j * rng.normal(size=int(sel.sum())))
             * math.sqrt(NOISE_SNR / 2))
        Xa[sel] += s
        Xb[sel] += s * np.exp(1j * phase)          # B relative to A, the log's sign
        if station is not None:
            k = int(station.sum())
            q = (rng.normal(size=k) + 1j * rng.normal(size=k)) * math.sqrt(300.0 / 2)
            Xa[station] += q
            Xb[station] += q * np.exp(1j * math.radians(140.0))
        sm.update(np.stack([Xa, Xb]), frame_s)
    return sm, fit, center, at_hz


@pytest.mark.parametrize("rate", RATES_HZ, ids=_rid)
def test_noise_bearing_finds_the_same_hash_at_every_rate(rate):
    """One local noise source, described as a fraction of the span, has to
    come back as the same bearing whether the span is 62.5 kHz or 2.04 MHz:
    every threshold in core/noisebearing.py that is a frequency (the 50 kHz
    neighbourhood a station is judged against, the 20 kHz a transmission
    cannot be wider than, the guard around DC) is derived from the rate, not
    from a bin count."""
    sm, fit, center, at_hz = _noise_scene(rate, seed=int(rate))
    t0 = time.perf_counter()
    out = noisebearing.noise_bearing(sm, {"impulses_per_s": 8.0}, fit, center, rate,
                                     now=1000.0,
                                     history=noisebearing.BearingHistory())
    _record("noisebearing", _rid(rate), 1.0, time.perf_counter() - t0)
    _assert_finite_json(out)
    assert out["available"], out["reason"]
    assert out["kind"] == "impulse"
    assert out["bins"] >= noisebearing.MIN_BINS, out
    assert out["coherence"] >= noisebearing.MIN_COHERENCE, out
    err = abs((out["bearing_deg"] - NOISE_BEARING_DEG + 180.0) % 360.0 - 180.0)
    assert err <= 5.0, f"bearing {out['bearing_deg']} at rate={rate} ({out['reason']})"
    # the mirror the two elements cannot tell apart, and nothing else
    assert abs((out["mirror_deg"] - (2 * NOISE_BASELINE_DEG - out["bearing_deg"])
                + 180.0) % 360.0 - 180.0) <= 0.2
    assert out["since"] == 1000.0


@pytest.mark.parametrize("rate", RATES_HZ, ids=_rid)
def test_noise_bearing_leaves_a_station_out_at_every_rate(rate):
    """A 3 kHz signal 25 dB over the hash is a transmission at every span --
    at 62.5 kS/s it is 100 bins wide and at 2.04 MS/s it is three, and
    either way it must not get a vote in where the noise is."""
    f = np.fft.fftfreq(MAP_BINS, 1.0 / rate)
    station = (f >= 0.30 * rate / 2) & (f < 0.30 * rate / 2 + 3_000.0)
    sm, fit, center, _at = _noise_scene(rate, seed=int(rate) ^ 0x5EED,
                                        station=station)
    _coh, _steer, _m, level = sm._analyse()
    assert noisebearing.station_mask(level, rate)[station].all()
    out = noisebearing.noise_bearing(sm, None, fit, center, rate)
    err = abs((out["bearing_deg"] - NOISE_BEARING_DEG + 180.0) % 360.0 - 180.0)
    assert err <= 5.0, f"the station moved the bearing to {out['bearing_deg']}"


# =========================================================================
# noise profile
# =========================================================================

IMPULSE_FEED_S = 4.0


def _plain_noise_profile_status(rate, seed):
    rng = np.random.default_rng(seed)
    n = int(IMPULSE_FEED_S * rate)
    a = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    b = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    prof = NoiseProfile(rate)
    for i in range(0, n - CHUNK + 1, CHUNK):
        prof.update(a[i:i + CHUNK], b[i:i + CHUNK])
    return prof.status()


def test_noiseprofile_impulse_rate_on_plain_noise_is_a_few_per_second_at_every_rate():
    """core/noiseprofile.py's impulse test runs on FULL-RATE samples, and a
    white sequence's per-sample threshold-crossing probability is, in
    theory, roughly rate-independent -- so the crossing COUNT per second
    could scale with the sample rate rather than describing the physical
    impulse content. Measured across all six rates below: the 12 dB bar
    (IMPULSE_DB) over a chi-square(4) median is such a deep excursion that
    the crossing rate stays at (or indistinguishable from) zero at every
    rate over IMPULSE_FEED_S seconds -- the scaling risk is real in theory
    but does not show up in a run this short at any of these rates. Reported
    per the brief; not fixed, because there is nothing here yet to fix."""
    rows = []
    for i, rate in enumerate(RATES_HZ):
        st = _plain_noise_profile_status(rate, seed=1000 + i)
        rows.append((rate, st))
        _assert_finite_json(st)
        assert st["impulses_per_s"] < 5.0, f"rate={rate}: {st['impulses_per_s']}"
        print(f"noiseprofile @ {rate:.0f}: impulses_per_s={st['impulses_per_s']} "
              f"impulse_db={st['impulse_db']}")
    lo_rate, lo_st = rows[0]
    hi_rate, hi_st = rows[-1]
    if hi_st["impulses_per_s"] > 1.0 and lo_st["impulses_per_s"] < 0.2:
        pytest.xfail(
            f"impulses_per_s scales with the sample rate on plain noise: "
            f"{lo_st['impulses_per_s']}/s at {lo_rate:.0f} Hz vs "
            f"{hi_st['impulses_per_s']}/s at {hi_rate:.0f} Hz. The threshold test in "
            f"NoiseProfile.update() runs on the raw samples, not the decimated "
            f"envelope; report before fixing (core/noiseprofile.py update()).")


@pytest.mark.parametrize("rate", RATES_HZ, ids=_rid)
def test_noiseprofile_processes_one_second_faster_than_real_time(rate):
    rng = np.random.default_rng(2)
    n = int(rate)
    a = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    b = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    prof = NoiseProfile(rate)
    t0 = time.perf_counter()
    for i in range(0, n - CHUNK + 1, CHUNK):
        prof.update(a[i:i + CHUNK], b[i:i + CHUNK])
    wall = time.perf_counter() - t0
    _record("noiseprofile (rt)", _rid(rate), 1.0, wall)
    assert wall < REALTIME_GENERAL_S, f"{rate}: {wall:.3f}s for 1s of input"
    if rate == max(RATES_HZ):
        assert wall < REALTIME_S, f"{rate}: {wall:.3f}s for 1s of input (must be < {REALTIME_S}s)"


# =========================================================================
# sub-band combiner (runs at _pd_rate, not the RF span)
# =========================================================================

SUB_BLOCK = 800


def _subband_case(pd_rate, seconds=1.6):
    rng = np.random.default_rng(int(pd_rate))
    n = int(SUB_BLOCK * max(4, round(seconds * pd_rate / SUB_BLOCK)))
    a = (rng.normal(size=n) + 1j * rng.normal(size=n)) / math.sqrt(2)
    b = (rng.normal(size=n) + 1j * rng.normal(size=n)) / math.sqrt(2)
    m = 0.7 * np.exp(1j * 0.9)
    s = np.array([1.0, np.conj(m)])
    comb = SubbandCombiner(pd_rate)
    pend = len(comb._in_a)
    out = []
    t0 = time.perf_counter()
    for i in range(0, n - SUB_BLOCK + 1, SUB_BLOCK):
        out.append(comb.process(a[i:i + SUB_BLOCK], b[i:i + SUB_BLOCK], m, s, False))
    wall = time.perf_counter() - t0
    y = np.concatenate(out)[pend:]
    ref = combine(a, b, m)
    return comb, y, ref, wall, n / pd_rate


@pytest.mark.parametrize("rate", RATES_HZ, ids=_rid)
def test_subband_reproduces_the_wideband_combiner_on_white_noise_at_every_pd_rate(rate):
    pd_rate = _pd_rate(rate)
    comb, y, ref, wall, input_s = _subband_case(pd_rate)
    _record("subband", f"{_rid(rate)}->pd{pd_rate:.0f}", input_s, wall)
    lo, hi = 5 * NFFT, len(y) - NFFT
    assert hi > lo, f"pd_rate={pd_rate} too short a run to have a settled middle"
    err = np.max(np.abs(y[lo:hi] - ref[lo:hi]))
    assert err < 1e-3, f"pd_rate={pd_rate}: {err}"
    assert comb.refined_bins == 0 and abs(comb.extra_db) < 0.2
    assert np.all(np.isfinite(y))
    _assert_finite_json(comb.status())


@pytest.mark.parametrize("rate", RATES_HZ, ids=_rid)
def test_subband_processes_one_second_faster_than_real_time(rate):
    pd_rate = _pd_rate(rate)
    comb, y, ref, wall, input_s = _subband_case(pd_rate, seconds=1.0)
    _record("subband (rt)", f"{_rid(rate)}->pd{pd_rate:.0f}", input_s, wall)
    assert wall < REALTIME_GENERAL_S * max(1.0, input_s)
    if rate == max(RATES_HZ):
        assert wall < REALTIME_S, f"pd_rate={pd_rate}: {wall:.3f}s for {input_s:.2f}s of input"


# =========================================================================
# AGC (always at AUDIO_RATE -- see the module docstring)
# =========================================================================

AGC_BLOCK = 480    # 20 ms at 24 kHz, a plausible audio callback size


@pytest.mark.parametrize("rate", RATES_HZ, ids=_rid)
def test_agc_is_finite_and_realtime_regardless_of_the_rf_span(rate):
    """Agc (core/agc.py) is always constructed with its default rate_hz =
    24000 (adapters filter.py: self.agc = Agc()) and always fed the audio
    AFTER the fractional resampler lands it on AUDIO_RATE (adapters/soapy.py
    ~1941: self._filt.agc.process(audio)), so the RF span never reaches it.
    `rate` here is a label only, carried through so this module gets a row
    per span in the table like the others; the call itself is identical
    every time."""
    rng = np.random.default_rng(int(rate) & 0xFFFF)
    agc = Agc(rate_hz=AUDIO_RATE)
    n_blocks = int(round(AUDIO_RATE / AGC_BLOCK))
    t0 = time.perf_counter()
    for i in range(n_blocks):
        loud = i % 5 == 0
        audio = rng.normal(size=AGC_BLOCK) * (0.9 if loud else 0.02)
        out = agc.process(audio)
        assert np.all(np.isfinite(out)) and np.max(np.abs(out)) <= 1.0 + 1e-9
    wall = time.perf_counter() - t0
    _record("agc", _rid(rate), 1.0, wall)
    assert wall < REALTIME_S      # 24 kHz audio, a fixed rate: comfortably real time everywhere
    _assert_finite_json(agc.status())


# =========================================================================
# auto contour, fed from a voice print at every pd_rate
# =========================================================================

CONTOUR_BLOCK = 2000


def _bumped_over(rng, seconds, pd_rate, lo_hz, hi_hz, bump_lo, bump_hi, bump_db):
    """A band-limited voice-shaped over (as test_voiceprint._over builds
    it) with a deliberate microphone-style bump laid on top, so fit_contour
    has something to find."""
    n = int(seconds * pd_rate)
    X = np.zeros(n, dtype=np.complex128)
    f = np.fft.fftfreq(n, 1.0 / pd_rate)
    sel = (f >= lo_hz) & (f < hi_hz)
    X[sel] = rng.normal(size=sel.sum()) + 1j * rng.normal(size=sel.sum())
    bump = (f >= bump_lo) & (f < bump_hi)
    X[bump] *= 10 ** (bump_db / 20.0)
    x = np.fft.ifft(X)
    t = np.arange(n) / pd_rate
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 4.0 * t)
    return x / np.sqrt(np.mean(np.abs(x) ** 2)) * env


@pytest.mark.parametrize("rate", RATES_HZ, ids=_rid)
def test_contour_fits_a_finite_bell_from_a_voiceprint_at_every_pd_rate(rate):
    pd_rate = _pd_rate(rate)
    rng = np.random.default_rng(int(pd_rate))
    vp = VoicePrint(pd_rate)
    x = _bumped_over(rng, 3.0, pd_rate, 300.0, 2700.0, 1000.0, 1400.0, 10.0)
    t0 = time.perf_counter()
    for i in range(0, len(x), CONTOUR_BLOCK):
        vp.feed(x[i:i + CONTOUR_BLOCK], True, 1)
    vp.feed(np.zeros(CONTOUR_BLOCK), False, None)     # ends the over
    wall_vp = time.perf_counter() - t0

    s = vp.summary(1)
    assert s is not None, f"pd_rate={pd_rate}: the over taught nothing"
    bands_db = s["bands_db"]
    assert len(bands_db) == PROFILE_BANDS
    assert all(math.isfinite(x) for x in bands_db)

    t1 = time.perf_counter()
    bell = fit_contour(bands_db)
    wall_fit = time.perf_counter() - t1
    _record("voiceprint+contour", _rid(rate), 3.0, wall_vp + wall_fit)

    if bell is not None:
        hz, db, width = bell
        assert math.isfinite(hz) and math.isfinite(db) and math.isfinite(width)
        assert 300.0 <= hz <= 2500.0
        assert abs(db) <= 6.0
        assert 200.0 <= width <= 1200.0
    _assert_finite_json({"summary": s, "bell": bell})


# =========================================================================
# voice print round trip
# =========================================================================

def _voice_over(rng, seconds, lo_hz, hi_hz, syl_hz, rate):
    """As test_voiceprint.py's _over, parametrised by rate rather than
    that file's fixed module-level RATE (25 kHz): a band-limited,
    syllable-gated one-sided (SSB-shaped) over."""
    n = int(seconds * rate)
    X = np.zeros(n, dtype=np.complex128)
    f = np.fft.fftfreq(n, 1.0 / rate)
    sel = (f >= lo_hz) & (f < hi_hz)
    X[sel] = rng.normal(size=sel.sum()) + 1j * rng.normal(size=sel.sum())
    x = np.fft.ifft(X)
    t = np.arange(n) / rate
    env = 0.55 + 0.45 * np.sin(2 * np.pi * syl_hz * t)
    return x / np.sqrt(np.mean(np.abs(x) ** 2)) * env


@pytest.mark.parametrize("rate", RATES_HZ, ids=_rid)
def test_voiceprint_round_trips_two_talkers_at_every_pd_rate(rate):
    pd_rate = _pd_rate(rate)
    rng = np.random.default_rng(int(pd_rate) + 1)
    vp = VoicePrint(pd_rate)
    t0 = time.perf_counter()
    for _ in range(2):
        _feed(vp, _voice_over(rng, 3.0, 300.0, 2400.0, 3.0, pd_rate), 1, pd_rate)
        _silence(vp, pd_rate)
        _feed(vp, _voice_over(rng, 3.0, 100.0, 2900.0, 5.0, pd_rate), 2, pd_rate)
        _silence(vp, pd_rate)
    wall = time.perf_counter() - t0
    _record("voiceprint", _rid(rate), 12.0, wall)

    a, b = vp.summary(1), vp.summary(2)
    assert a is not None and b is not None, f"pd_rate={pd_rate}: a talker's print never formed"
    assert a["overs"] == 2 and b["overs"] == 2
    assert abs(a["syllabic_hz"] - 3.0) <= 0.6
    assert abs(b["syllabic_hz"] - 5.0) <= 0.6
    assert a["centroid_hz"] < b["centroid_hz"]
    _assert_finite_json({"a": a, "b": b})


def _feed(vp, x, talker, pd_rate, block=None):
    block = block or CONTOUR_BLOCK
    for i in range(0, len(x), block):
        vp.feed(x[i:i + block], True, talker)


def _silence(vp, pd_rate, seconds=0.3, block=None):
    block = block or CONTOUR_BLOCK
    vp.feed(np.zeros(int(seconds * pd_rate)), False, None)


# =========================================================================
# talker memory: no rate parameter at all, tested once per span for the
# record -- see the module docstring above.
# =========================================================================

def _steering(phase, ratio=0.75):
    s = np.array([1.0, ratio * np.exp(1j * phase)])
    return s / np.linalg.norm(s)


@pytest.mark.parametrize("rate", RATES_HZ, ids=_rid)
def test_talkermemory_store_and_recall_round_trips_regardless_of_span(rate):
    mem = TalkerMemory(names_path=None)    # never the operator's real names file
    s1, s2 = _steering(0.3), _steering(2.4)
    m1, m2 = 0.8 * np.exp(1j * 0.3), 0.5 * np.exp(1j * 2.4)
    t0 = time.perf_counter()
    mem.store(s1, m1, now=0.0)
    got1 = mem.recall(s1, now=1.0)
    got_none = mem.recall(s2, now=1.0)
    mem.store(s2, m2, now=2.0)
    got2 = mem.recall(s2, now=3.0)
    wall = time.perf_counter() - t0
    _record("talkermemory", _rid(rate), 0.0, max(wall, 1e-6))

    assert got1 is not None and abs(got1 - m1) < 1e-9
    assert got_none is None
    assert got2 is not None and abs(got2 - m2) < 1e-9
    assert len(mem.entries) == 2
    _assert_finite_json(mem.status(4.0))
    assert wall < REALTIME_S


# =========================================================================
# alignsearch: fill in the two rates the existing suite does not cover
# (62.5 k, 250 k, 1 M and 2.04 M are pinned in test_alignsearch.py already)
# =========================================================================

@pytest.mark.parametrize("rate,lag", [(125_000.0, -4_158), (500_000.0, 16_632)])
def test_alignsearch_finds_the_offset_at_the_two_remaining_rates(rate, lag):
    rng = np.random.default_rng(int(abs(lag)) % 101)
    n = min(int(0.5 * rate), 1_000_000)
    a, b = _lag_pair(n, lag, rng)
    t0 = time.perf_counter()
    found, peak = alignsearch.measure_lag(a, b, rate)
    wall = time.perf_counter() - t0
    _record("alignsearch", _rid(rate), n / rate, wall)
    assert found == lag
    assert peak >= ALIGN_MIN_PEAK


@pytest.mark.parametrize("rate", RATES_HZ, ids=_rid)
def test_alignsearch_measure_lag_is_faster_than_the_window_it_measures(rate):
    """The reader thread cannot stall on this: measuring the CAL_SECONDS
    (0.5 s) window must take a good deal less than 0.5 s itself."""
    rng = np.random.default_rng(7)
    n = min(int(0.5 * rate), 1_000_000)
    a, b = _lag_pair(n, 123, rng)
    t0 = time.perf_counter()
    alignsearch.measure_lag(a, b, rate)
    wall = time.perf_counter() - t0
    input_s = n / rate
    _record("alignsearch (rt)", _rid(rate), input_s, wall)
    assert wall < REALTIME_GENERAL_S * max(1.0, input_s)
    if rate == max(RATES_HZ):
        assert wall < REALTIME_S, f"{rate}: {wall:.3f}s for a {input_s:.2f}s window"
