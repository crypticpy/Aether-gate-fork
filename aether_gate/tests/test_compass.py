#
# Aether-gate -- fitting the pair's geometry from beacons, wraps and all.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Synthetic beacons from a known array: pick a phi0, an A and a B, work out
what each of five or six directions would measure, wrap it the way a real
phase measurement is wrapped, add a degree or two of scatter, and ask the
fit for the array back. It has to come back to within a degree even when the
phases cross +-180 between one beacon and the next -- which is the whole
reason the fit is done on complex ratios and not on numbers.

The rest is the refusals: three beacons is the minimum because there are
three unknowns, and three beacons in two directions is still two directions
however far apart they are. And the mirror: the answer to "what bearing is
this phase" is always two bearings, and it always will be until there is a
third loop off the line.

Run:  python -m pytest aether_gate/tests/test_compass.py
"""
import math

import numpy as np
import pytest

from aether_gate.core import compass
from aether_gate.core.compass import (BandFit, compass_json, fit, fit_from_log,
                                      pattern_from_log)
from aether_gate.core.sitelog import SiteLog

BAND = 14_100_000.0
PHI0, A, B = 120.0, 1.9, -1.4        # k = 2.36 rad, baseline 323.6 deg true


def _phase(bearing_deg, phi0=PHI0, a=A, b=B):
    """What the model says the pair measures towards a bearing, wrapped to
    +-180 exactly as a real phase measurement is."""
    t = math.radians(bearing_deg)
    return compass._wrap180(phi0 + math.degrees(a * math.cos(t) + b * math.sin(t)))


def _ratios(bearings, noise_deg=0.0, seed=3, phi0=PHI0, a=A, b=B, mag=1.0):
    rng = np.random.default_rng(seed)
    out = []
    for t in bearings:
        p = _phase(t, phi0, a, b) + (rng.normal(0.0, noise_deg) if noise_deg else 0.0)
        out.append(mag * np.exp(1j * math.radians(p)))
    return out


def test_a_known_array_comes_back_from_phases_that_cross_the_wrap():
    bearings = [10.0, 75.0, 140.0, 215.0, 300.0, 350.0]
    phases = [_phase(t) for t in bearings]
    # the set really does wrap: neighbours 2 degrees apart in phase would be
    # 358 apart to anything that fits unwrapped numbers
    assert max(phases) > 70.0 and min(phases) < -140.0
    f = fit(bearings, _ratios(bearings, noise_deg=1.0), band_hz=BAND)
    assert f.available and f.n_beacons == 6
    assert f.phi0_deg == pytest.approx(PHI0, abs=1.0)
    assert f.k == pytest.approx(math.hypot(A, B), abs=0.02)
    assert f.baseline_deg == pytest.approx(math.degrees(math.atan2(B, A)) % 360.0, abs=1.0)
    assert f.quality > 0.99
    assert max(abs(r["residual_deg"]) for r in f.residuals) < 3.0
    d = f.as_dict()
    assert d["band_hz"] == BAND and d["n_beacons"] == 6
    assert d["spacing_wavelengths"] == pytest.approx(f.k / (2 * math.pi), abs=1e-4)
    assert [r["bearing_deg"] for r in d["beacons"]] == sorted(bearings)


def test_five_well_spread_beacons_are_enough_and_four_are_not():
    five = [10.0, 75.0, 140.0, 215.0, 300.0]
    f = fit(five, _ratios(five, noise_deg=1.0), band_hz=BAND)
    assert f.available and f.unique and f.k == pytest.approx(math.hypot(A, B), abs=0.05)
    # Four is three unknowns and four wrapped measurements: a WIDER pair whose
    # extra turn lands on the same four phases fits them exactly too. The
    # quality cannot tell -- both are 1.0 -- so the fit says so instead.
    four = [20.0, 130.0, 200.0, 310.0]
    g = fit(four, _ratios(four), band_hz=BAND)
    assert g.available and g.quality > 0.999
    assert not g.unique and len(g.alias_k) > 1
    assert any(abs(k - math.hypot(A, B)) < 0.05 for k in g.alias_k)
    assert g.as_dict()["alias_k"] == [round(k, 4) for k in g.alias_k]


def test_two_beacons_are_not_a_fit_and_the_reason_says_so():
    bearings = [30.0, 200.0]
    f = fit(bearings, _ratios(bearings))
    assert not f.available and "2 beacon" in f.reason and "3 needed" in f.reason
    assert f.as_dict() == {"available": False, "reason": f.reason, "band_hz": None,
                           "n_beacons": 0}
    assert f.bearing_from_phase(40.0) == {"available": False, "reason": f.reason,
                                          "bearings_deg": []}


def test_beacons_from_only_two_directions_are_rejected_however_far_apart():
    bearings = [0.0, 1.0, 180.0, 181.0, 180.5]       # two clusters, 180 apart
    f = fit(bearings, _ratios(bearings))
    assert not f.available and "distinct bearing" in f.reason
    assert compass._spread_deg(bearings) == pytest.approx(181.0, abs=0.1)
    assert compass._distinct(bearings) == 2


def test_beacons_bunched_in_one_quadrant_do_not_span_enough_compass():
    bearings = [10.0, 25.0, 40.0, 55.0]
    f = fit(bearings, _ratios(bearings))
    assert not f.available and "span 45 deg" in f.reason


def test_a_ratio_of_zero_or_no_bearing_is_dropped_not_fitted():
    bearings = [10.0, None, 140.0, 215.0, 300.0, 350.0]
    ratios = _ratios([10.0, 0.0, 140.0, 215.0, 300.0, 350.0])
    ratios[3] = 0j                                     # a beacon that was never heard
    f = fit(bearings, ratios, weights=[1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
    assert f.available and f.n_beacons == 3             # no bearing, no ratio, no weight
    assert [r["bearing_deg"] for r in f.residuals] == [10.0, 140.0, 350.0]


def test_bearing_from_phase_gives_the_mirror_pair_and_round_trips():
    bearings = [10.0, 75.0, 140.0, 215.0, 300.0, 350.0]
    f = fit(bearings, _ratios(bearings), band_hz=BAND)
    for truth in (75.0, 200.0, 340.0):
        ans = f.bearing_from_phase(_phase(truth))
        assert ans["available"] and not ans["outside_model"]
        assert len(ans["bearings_deg"]) == 2 and "reflection" in ans["mirror"]
        assert min(abs(b - truth) for b in ans["bearings_deg"]) < 0.5
        # the pair really is a mirror about the baseline, and both fit
        lo, hi = ans["bearings_deg"]
        mid = (lo + hi) / 2.0
        assert min(abs(mid - f.baseline_deg), abs(mid + 180.0 - f.baseline_deg),
                   abs(mid - 180.0 - f.baseline_deg)) < 0.5
        for b in ans["bearings_deg"]:
            assert f.phase_at(b) == pytest.approx(_phase(truth), abs=0.5)


def test_a_phase_the_model_cannot_reach_is_clamped_and_flagged():
    bearings = [10.0, 75.0, 140.0, 215.0, 300.0, 350.0]
    f = fit(bearings, _ratios(bearings))
    ans = f.bearing_from_phase(f.phi0_deg + 170.0)     # k is 2.36 rad, 135 deg
    assert ans["outside_model"] and len(ans["bearings_deg"]) == 1
    assert ans["bearings_deg"][0] == pytest.approx(f.baseline_deg, abs=0.5)
    assert f.bearing_from_phase(f.phi0_deg)["outside_model"] is False


def _write(log, bearings, phases=None, band=BAND, calls=None, coherence=0.9,
           snr=(20.0, 18.0)):
    for i, t in enumerate(bearings):
        p = _phase(t) if phases is None else phases[i]
        z = np.exp(1j * math.radians(p))
        log.beacon(band_hz=band, callsign=(calls[i] if calls else f"B{i}"),
                   bearing_deg=t, distance_km=1000.0 + 10.0 * i,
                   snr_a_db=snr[0], snr_b_db=snr[1], floor_a_db=-24.0, floor_b_db=-23.5,
                   ratio=complex(z), coherence=coherence, steps_heard=3,
                   lowest_w=1.0, mrc_gain_db=2.0)


def test_the_fit_is_built_from_the_log_and_takes_the_latest_line_per_callsign(tmp_path):
    log = SiteLog(str(tmp_path / "site-log.jsonl"))
    calls = ["4U1UN", "W6WX", "ZL6B", "OH2B", "CS3B", "YV5B"]
    bearings = [10.0, 75.0, 140.0, 215.0, 300.0, 350.0]
    # yesterday W6WX was heard through a wall of noise and its phase was junk
    _write(log, [75.0], phases=[_phase(75.0) + 140.0], calls=["W6WX"], coherence=0.2)
    _write(log, bearings, calls=calls)                 # today, properly
    _write(log, bearings, band=21_150_000.0, calls=calls)
    f = fit_from_log(log, BAND)
    assert f.available and f.n_beacons == 6            # not 7: one line per callsign
    assert sorted(r["call"] for r in f.residuals) == sorted(calls)
    assert f.phi0_deg == pytest.approx(PHI0, abs=1.0)
    assert f.k == pytest.approx(math.hypot(A, B), abs=0.02)
    assert f.band_hz == BAND
    assert fit_from_log(log, 28_200_000.0).available is False
    # weights: coherence times a soft SNR term, so a marginal beacon still counts
    w = {r["call"]: r["weight"] for r in f.residuals}
    assert all(x == pytest.approx(0.847, abs=0.005) for x in w.values())   # 0.9 * SNR term
    faint = compass._weight({"coherence": 0.9, "snr_a_db": 3.0, "snr_b_db": -2.0})
    assert 0.0 < faint < 0.2                            # heard, but not worth a degree
    assert compass._weight({"coherence": None}) == 0.0


def test_the_pattern_table_is_one_row_per_beacon_sorted_by_bearing(tmp_path):
    log = SiteLog(str(tmp_path / "site-log.jsonl"))
    _write(log, [300.0, 10.0, 140.0], calls=["CS3B", "4U1UN", "ZL6B"], snr=(20.0, 12.5))
    rows = pattern_from_log(log, BAND)
    assert [r["bearing_deg"] for r in rows] == [10.0, 140.0, 300.0]
    assert [r["call"] for r in rows] == ["4U1UN", "ZL6B", "CS3B"]
    assert all(r["a_minus_b_db"] == 7.5 for r in rows)      # loop A hears them better
    assert rows[0]["floor_a_db"] == -24.0 and rows[0]["distance_km"] is not None
    assert pattern_from_log(log, 28_200_000.0) == []


def test_compass_json_covers_every_band_the_log_has_and_names_the_mirror(tmp_path):
    log = SiteLog(str(tmp_path / "site-log.jsonl"))
    bearings = [10.0, 75.0, 140.0, 215.0, 300.0]
    _write(log, bearings, calls=[f"C{i}" for i in range(5)])
    _write(log, [10.0, 75.0], band=21_150_000.0, calls=["C0", "C1"])
    out = compass_json(log, phase_deg=_phase(75.0))
    assert out["available"] and out["fitted"] == 1 and len(out["bands"]) == 2
    assert "reflection" in out["mirror"] and "theta_b" in out["model"]
    got = {b["band_hz"]: b for b in out["bands"]}
    assert got[BAND]["available"] and got[BAND]["phi0_deg"] == pytest.approx(PHI0, abs=1.0)
    assert len(got[BAND]["pattern"]) == 5
    assert 75.0 in got[BAND]["bearing_from_phase"]["bearings_deg"]
    assert got[21_150_000.0]["available"] is False
    assert "2 beacon" in got[21_150_000.0]["reason"]
    assert compass_json(log, bands_hz=[]) == {"available": False, "bands": [],
                                              "reason": "no beacon band heard yet, 3 needed",
                                              "fitted": 0,
                                              "model": out["model"],
                                              "mirror": BandFit.MIRROR}
    import json
    assert json.loads(json.dumps(out))["fitted"] == 1
