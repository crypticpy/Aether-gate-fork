#
# Aether-gate — the noise's bearing, from the map's floor and the compass.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""core/noisebearing.py: the coherent floor of the spatial map, averaged into
one phase and handed to a fitted compass, is the direction to walk in.

The scenes here are built the way adapters/diversity_state.py builds the map
— a natural-FFT-order (2, 2048) spectrum per 4096-sample block — with a
band-limited coherent source standing on a white floor at a known inter-loop
phase, and a compass.GlobalFit made up rather than fitted, so the bearing
that comes back can be checked against the one that went in.

The phase convention is the site log's, loop B relative to loop A: a source
built as Xb = exp(i phi) Xa comes back as +phi (the map's own steering angle
is the other sign; see test_allrates' spatial-map test, where the same thing
is checked against SpatialMap directly).

Run:  python -m pytest aether_gate/tests/test_noisebearing.py
"""
import json
import math

import numpy as np
import pytest

from aether_gate.core import compass, noisebearing, spatial

NB = 2048              # adapters/diversity_state.py MAP_BINS
CHUNK = 4096           # adapters/soapy.py's raw block read length
RATE = 125_000.0
CENTER = 3_800_000.0
BASELINE_DEG = 70.0
# a mains comb, as core/noiseprofile.py's status() reports one
HUM = {"mains_hz": 60.0, "hum_db": 14.0, "harmonics": 4, "impulses_per_s": 0.2,
       "impulse_db": None, "periodic": []}
# the noise band used by most of the scenes: 5-40 kHz off the centre, wide
# enough that no station-width rule touches it, clear of DC and the edges
NOISE_BAND = (5_000.0, 40_000.0)


def _fit(d_m=4.0, dtau_ns=12.0, baseline_deg=BASELINE_DEG):
    """A pair whose geometry is known because it was made up: 4 m apart on
    070, 12 ns of cable difference."""
    return compass.GlobalFit(True, dtau_s=dtau_ns * 1e-9, d_m=d_m,
                             baseline_deg=baseline_deg, quality=0.95,
                             n_beacons=8, n_bands=3)


def _sel(lo_hz, hi_hz, rate=RATE, nbins=NB):
    f = np.fft.fftfreq(nbins, 1.0 / rate)
    return (f >= lo_hz) & (f < hi_hz)


def _mean_hz(lo_hz, hi_hz, rate=RATE, center=CENTER, nbins=NB):
    """Where noise_bearing will say the phase was measured: the middle of
    the bins it averages."""
    f = np.fft.fftfreq(nbins, 1.0 / rate)
    return center + float(np.mean(f[_sel(lo_hz, hi_hz, rate, nbins)]))


def _map(rng, sources, frames=40, rate=RATE, nbins=NB):
    """A SpatialMap fed `frames` of white floor carrying coherent sources:
    (lo_hz, hi_hz, phase_deg, snr) each, offsets from the centre, phase as
    the site log means it."""
    sm = spatial.SpatialMap(nbins, rate)
    rows = [(_sel(lo, hi, rate, nbins), math.radians(ph), snr)
            for lo, hi, ph, snr in sources]
    for _ in range(frames):
        Xa = (rng.normal(size=nbins) + 1j * rng.normal(size=nbins)) / math.sqrt(2)
        Xb = (rng.normal(size=nbins) + 1j * rng.normal(size=nbins)) / math.sqrt(2)
        for sel, ph, snr in rows:
            k = int(sel.sum())
            s = (rng.normal(size=k) + 1j * rng.normal(size=k)) * math.sqrt(snr / 2)
            Xa[sel] += s
            Xb[sel] += s * np.exp(1j * ph)
        sm.update(np.stack([Xa, Xb]), CHUNK / rate)
    return sm


def _wrap(d):
    return (d + 180.0) % 360.0 - 180.0


# --- the bearing ------------------------------------------------------------

def test_a_hum_comb_from_a_known_bearing_comes_back_as_that_bearing():
    fit = _fit()
    at = _mean_hz(*NOISE_BAND)
    phase = fit.phase_at(200.0, at)                 # what 200 deg true looks like
    sm = _map(np.random.default_rng(4), [(*NOISE_BAND, phase, 4.0)])
    out = noisebearing.noise_bearing(sm, HUM, fit, CENTER, RATE, now=1000.0,
                                     history=noisebearing.BearingHistory())
    assert out["available"] and out["kind"] == "hum"
    assert abs(_wrap(out["phase_deg"] - phase)) <= 5.0
    assert abs(_wrap(out["bearing_deg"] - 200.0)) <= 5.0
    # the mirror is that bearing reflected about the baseline, and it is one
    # of the answers the compass itself gives for the same phase
    assert abs(_wrap(out["mirror_deg"] - (2 * BASELINE_DEG - out["bearing_deg"]))) <= 0.2
    seen = fit.bearing_from_phase(out["phase_deg"], at)["bearings_deg"]
    assert min(abs(_wrap(out["mirror_deg"] - b)) for b in seen) <= 0.5
    assert out["bins"] >= noisebearing.MIN_BINS and out["coherence"] >= 0.6
    assert out["since"] == 1000.0 and "3.8" in out["reason"]
    json.dumps(out)                                  # the wire takes it


def test_the_same_floor_read_at_the_frequency_it_was_measured_at():
    """The phase a pair measures grows with frequency, so a bearing asked at
    the dial and one asked where the noise actually sits are different
    answers. This is the second: a source high in the span is read at ITS
    frequency, not the centre's."""
    fit = _fit(d_m=30.0)                          # a wide pair: 2 pi f d / c bites
    band = (300_000.0, 480_000.0)
    rate = 1_000_000.0
    at = _mean_hz(*band, rate=rate)
    phase = fit.phase_at(115.0, at)
    sm = _map(np.random.default_rng(5), [(*band, phase, 4.0)], rate=rate)
    out = noisebearing.noise_bearing(sm, None, fit, CENTER, rate)
    assert out["kind"] == "floor"                 # no profile: the band itself
    near = min(abs(_wrap(out["bearing_deg"] - b))
               for b in fit.bearing_from_phase(phase, at)["bearings_deg"])
    assert near <= 5.0
    # asked at the centre instead, the same phase is a different bearing --
    # which is exactly the error not making this correction would be
    at_centre = fit.bearing_from_phase(phase, CENTER)["bearings_deg"]
    assert min(abs(_wrap(out["bearing_deg"] - b)) for b in at_centre) > 5.0


def test_a_station_in_the_noise_is_left_out_of_the_bearing():
    """A loud, narrow signal sitting inside the hash points somewhere else.
    It is a transmission, not the noise, and it does not get a vote."""
    fit = _fit()
    at = _mean_hz(*NOISE_BAND)
    noise = fit.phase_at(200.0, at)
    sm = _map(np.random.default_rng(6),
              [(*NOISE_BAND, noise, 4.0),
               (18_000.0, 21_000.0, noise + 90.0, 300.0)])       # +25 dB, 3 kHz
    out = noisebearing.noise_bearing(sm, HUM, fit, CENTER, RATE)
    assert abs(_wrap(out["bearing_deg"] - 200.0)) <= 5.0
    # ... and the station's bins really were dropped, not merely outvoted
    coh, _st, _m, level = sm._analyse()
    hot = noisebearing.station_mask(level, RATE)
    assert hot[_sel(18_500.0, 20_500.0)].all()
    assert not hot[_sel(*NOISE_BAND) & ~_sel(17_000.0, 22_000.0)].any()


def test_bins_that_disagree_have_a_phase_but_no_bearing():
    """Two halves of the band pointing 180 apart is not one source. There is
    still a phase and a bin count -- the operator can see the mess -- but no
    bearing, and the reason says why."""
    fit = _fit()
    sm = _map(np.random.default_rng(7),
              [(5_000.0, 20_000.0, 90.0, 4.0), (25_000.0, 40_000.0, -90.0, 4.0)])
    out = noisebearing.noise_bearing(sm, HUM, fit, CENTER, RATE)
    assert out["available"] and out["bins"] >= noisebearing.MIN_BINS
    assert out["coherence"] < noisebearing.MIN_COHERENCE
    assert out["bearing_deg"] is None and out["mirror_deg"] is None
    assert "no one direction" in out["reason"]
    assert out["since"] is None


def test_without_a_compass_fit_there_is_a_phase_and_the_reason_it_is_only_that():
    sm = _map(np.random.default_rng(8), [(*NOISE_BAND, 30.0, 4.0)])
    out = noisebearing.noise_bearing(sm, HUM, None, CENTER, RATE)
    assert out["available"] and abs(_wrap(out["phase_deg"] - 30.0)) <= 5.0
    assert out["bearing_deg"] is None and "no beacon fit yet" in out["reason"]
    unfitted = compass.fit_global([])                     # nothing heard at all
    out = noisebearing.noise_bearing(sm, HUM, unfitted, CENTER, RATE)
    assert out["bearing_deg"] is None and unfitted.reason in out["reason"]


def test_a_band_with_nothing_coherent_in_it_says_so():
    sm = _map(np.random.default_rng(9), [])               # white floor, both loops
    out = noisebearing.noise_bearing(sm, HUM, _fit(), CENTER, RATE)
    assert out["available"] is False and out["bearing_deg"] is None
    assert out["kind"] is None and out["bins"] < noisebearing.MIN_BINS
    assert "needed" in out["reason"]
    json.dumps(out)


def test_no_map_and_no_rate_are_answers_too():
    blank = noisebearing.noise_bearing(None, HUM, _fit(), CENTER, RATE)
    assert blank["available"] is False and blank["reason"] == "no spatial map yet"
    assert set(blank) == {"available", "kind", "phase_deg", "coherence",
                          "bearing_deg", "mirror_deg", "bins", "since", "reason"}
    sm = spatial.SpatialMap(NB, RATE)                     # built, never fed
    assert noisebearing.noise_bearing(sm, HUM, _fit(), CENTER).get("reason") \
        == "no spatial map yet"


# --- what kind of noise it is ------------------------------------------------

@pytest.mark.parametrize("status,kind", [
    (HUM, "hum"),
    ({**HUM, "harmonics": 0}, "floor"),
    ({**HUM, "hum_db": 3.0}, "floor"),                     # under LINE_MIN_DB
    ({**HUM, "mains_hz": None, "periodic": [{"hz": 1000.0, "db": 11.0}]}, "lines"),
    ({"impulses_per_s": 8.0}, "impulse"),
    ({"impulses_per_s": 0.2}, "floor"),
    (None, "floor"),
])
def test_the_kind_is_the_profiles_verdict_in_the_order_an_operator_says_it(status, kind):
    assert noisebearing.kind_of(status) == kind


# --- since -------------------------------------------------------------------

def test_since_holds_while_the_bearing_holds_and_restarts_when_it_moves():
    h = noisebearing.BearingHistory()
    assert h.hold(100.0, 10.0) == 10.0
    assert h.hold(100.0 + noisebearing.HOLD_DEG - 1.0, 20.0) == 10.0    # same source
    assert h.hold(200.0, 30.0) == 30.0                                 # a different one
    # nobody asked for longer than the gap: whatever is there now is new
    assert h.hold(200.0, 30.0 + noisebearing.HOLD_GAP_S + 1.0) == 30.0 + \
        noisebearing.HOLD_GAP_S + 1.0
    assert h.hold(None, 200.0) is None and h.hold(200.0, 210.0) == 210.0


def test_since_survives_the_next_poll_of_the_same_noise():
    fit = _fit()
    at = _mean_hz(*NOISE_BAND)
    phase = fit.phase_at(200.0, at)
    hist = noisebearing.BearingHistory()
    rng = np.random.default_rng(11)
    first = noisebearing.noise_bearing(_map(rng, [(*NOISE_BAND, phase, 4.0)]), HUM,
                                       fit, CENTER, RATE, now=500.0, history=hist)
    again = noisebearing.noise_bearing(_map(rng, [(*NOISE_BAND, phase, 4.0)]), HUM,
                                       fit, CENTER, RATE, now=530.0, history=hist)
    assert first["since"] == 500.0 and again["since"] == 500.0
    assert abs(_wrap(again["bearing_deg"] - first["bearing_deg"])) <= 5.0
