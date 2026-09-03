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
from aether_gate.core.compass import (C_M_S, BandFit, compass_json, fit,
                                      fit_from_log, fit_global,
                                      fit_global_from_log, pattern_from_log)
from aether_gate.core.sitelog import SiteLog

BAND = 14_100_000.0
PHI0, A, B = 120.0, 1.9, -1.4        # k = 2.36 rad, baseline 323.6 deg true
# the global model's truth: 12.5 ns of cable difference, 3.2 m apart on 065
DTAU, D_M, THETA_B = 12.5e-9, 3.2, 65.0
# 14.100, 18.110 and 24.930 are NOT all multiples of one frequency, so the
# whole-turn alias in dtau that 14.1/21.15/28.2 share is not in these
HIGH_BANDS = (14_100_000.0, 18_110_000.0, 24_930_000.0)


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
    empty = compass_json(log, bands_hz=[])
    assert set(empty) == {"available", "reason", "bands", "fitted", "model",
                          "global", "mirror"}
    assert (empty["available"], empty["bands"], empty["fitted"]) == (False, [], 0)
    assert empty["reason"] == "no beacon band heard yet, 3 needed"
    assert empty["model"] == out["model"] and empty["mirror"] == BandFit.MIRROR
    import json
    assert json.loads(json.dumps(out))["fitted"] == 1


# --- the same pair, on every band ---------------------------------------------

def _gphase(bearing_deg, f_hz, dtau=DTAU, d_m=D_M, theta_b=THETA_B):
    """What a real pair of loops measures: a cable delay plus the geometry,
    both of which grow with frequency. Wrapped, as a phase always is."""
    th = math.radians(bearing_deg - theta_b)
    return compass._wrap180(math.degrees(
        2.0 * math.pi * f_hz * (dtau + (d_m / C_M_S) * math.cos(th))))


def _gevent(bearing_deg, f_hz, call="C0", noise_deg=0.0, rng=None,
            dtau=DTAU, d_m=D_M, theta_b=THETA_B, coherence=0.9):
    """One site-log beacon line towards a known bearing on one frequency."""
    p = _gphase(bearing_deg, f_hz, dtau, d_m, theta_b)
    if noise_deg and rng is not None:
        p += rng.normal(0.0, noise_deg)
    z = np.exp(1j * math.radians(p))
    return {"kind": "beacon", "band_hz": f_hz, "callsign": call,
            "bearing_deg": bearing_deg, "ratio": [z.real, z.imag],
            "coherence": coherence, "snr_a_db": 20.0, "snr_b_db": 18.0}


def _gevents(bearings, bands, noise_deg=0.0, seed=5, **kw):
    rng = np.random.default_rng(seed)
    return [_gevent(t, f, "C%d" % i, noise_deg, rng, **kw)
            for f in bands for i, t in enumerate(bearings)]


def test_beacons_on_three_bands_give_the_pair_itself_delay_spacing_and_bearing():
    bearings = [10.0, 75.0, 140.0, 215.0, 300.0]
    g = fit_global(_gevents(bearings, HIGH_BANDS, noise_deg=2.0))
    assert g.available and g.n_beacons == 15 and g.n_bands == 3
    assert g.dtau_ns == pytest.approx(DTAU * 1e9, abs=2.0)
    assert g.d_m == pytest.approx(D_M, abs=0.3)
    assert g.baseline_deg == pytest.approx(THETA_B, abs=2.0)
    assert g.quality > 0.99 and g.unique and g.alternatives == []
    assert [b["band_hz"] for b in g.bands] == sorted(HIGH_BANDS)
    assert all(b["n_beacons"] == 5 and b["max_residual_deg"] < 8.0 for b in g.bands)
    d = g.as_dict()
    assert d["dtau_ns"] == round(g.dtau_ns, 2) and d["n_bands"] == 3
    assert len(d["beacons"]) == 15 and "reflection" in d["mirror"]
    # every band's own k is the one 2 pi f d / c predicts
    for band in HIGH_BANDS:
        assert g.k_at(band) == pytest.approx(2 * math.pi * band * D_M / C_M_S, abs=0.05)


def test_a_second_band_breaks_the_alias_four_beacons_on_one_band_cannot():
    four = [20.0, 130.0, 200.0, 310.0]
    one = _gevents(four, [BAND])
    per_band = fit(four, [complex(*e["ratio"]) for e in one], band_hz=BAND)
    assert per_band.available and not per_band.unique      # four wrapped phases
    assert len(per_band.alias_k) > 1
    # the SAME four directions, heard again on 18.110: one spacing explains
    # both bands and the wider alias explains neither
    g = fit_global(one + _gevents(four, [18_110_000.0]))
    assert g.available and g.unique and g.alternatives == []
    assert g.d_m == pytest.approx(D_M, abs=0.1)
    assert g.baseline_deg == pytest.approx(THETA_B, abs=1.0)
    assert g.dtau_ns == pytest.approx(DTAU * 1e9, abs=1.0)


def test_a_compass_earned_on_the_high_bands_answers_on_eighty_metres():
    g = fit_global(_gevents([10.0, 75.0, 140.0, 215.0, 300.0], HIGH_BANDS,
                            noise_deg=1.0))
    assert g.available
    # not 245 or 65: a bearing ON the baseline is the edge of the pattern,
    # where a tenth of a degree of phase error is outside the model
    for truth in (30.0, 110.0, 200.0):
        ans = g.bearing_from_phase(_gphase(truth, 3_800_000.0), 3_800_000.0)
        assert ans["available"] and not ans["outside_model"]
        assert ans["f_hz"] == 3_800_000.0 and not ans["grating_lobes"]
        assert len(ans["bearings_deg"]) == 2
        assert min(abs(b - truth) for b in ans["bearings_deg"]) < 5.0
        # and the model's own phase at 3.8 MHz inverts exactly
        for b in ans["bearings_deg"]:
            assert g.phase_at(b, 3_800_000.0) == pytest.approx(
                compass._wrap180(_gphase(truth, 3_800_000.0)), abs=1.0)


def test_a_pair_over_half_a_wavelength_apart_has_grating_lobes():
    g = fit_global(_gevents([10.0, 75.0, 140.0, 215.0, 300.0], HIGH_BANDS,
                            d_m=12.0))
    assert g.available and g.d_m == pytest.approx(12.0, abs=0.1)
    ans = g.bearing_from_phase(_gphase(110.0, 28_200_000.0, d_m=12.0), 28_200_000.0)
    assert ans["grating_lobes"] and ans["d_over_lambda"] > 0.5
    assert len(ans["bearings_deg"]) > 2                    # the extra answers
    assert min(abs(b - 110.0) for b in ans["bearings_deg"]) < 2.0
    # ... and on 80 m the same pair is a fraction of a wavelength again
    low = g.bearing_from_phase(_gphase(110.0, 3_800_000.0, d_m=12.0), 3_800_000.0)
    assert not low["grating_lobes"] and len(low["bearings_deg"]) == 2


def test_one_band_fits_the_geometry_but_cannot_pin_the_cable_delay():
    five = [10.0, 75.0, 140.0, 215.0, 300.0]
    g = fit_global(_gevents(five, [BAND], noise_deg=0.5))
    assert g.available and g.n_bands == 1
    assert g.d_m == pytest.approx(D_M, abs=0.3)
    assert g.baseline_deg == pytest.approx(THETA_B, abs=2.0)
    # a whole turn at 14.100 is 70.9 ns of delay and lands on the same
    # phases: the fit takes the shortest cable and NAMES the others
    assert not g.unique and g.alternatives
    step = 1e9 / BAND
    for alt in g.alternatives:
        assert alt["d_m"] == pytest.approx(g.d_m, abs=0.05)
        turns = (alt["dtau_ns"] - g.dtau_ns) / step
        assert abs(turns - round(turns)) < 0.05 and round(turns) != 0


def test_the_global_fit_says_why_when_the_beacons_are_too_few_or_bunched():
    five = [10.0, 75.0, 140.0, 215.0, 300.0]
    g = fit_global(_gevents(five[:3], [BAND]))
    assert not g.available and "3 beacon" in g.reason and "4 needed" in g.reason
    assert g.as_dict() == {"available": False, "reason": g.reason, "n_beacons": 3,
                           "n_bands": 1, "model": compass.GLOBAL_MODEL}
    g = fit_global(_gevents(five[:4], [BAND]))
    assert not g.available and "on one band" in g.reason and "5 needed" in g.reason
    assert fit_global(_gevents(five[:2], HIGH_BANDS)).available          # 6 over 3
    bunched = fit_global(_gevents([10.0, 25.0, 40.0, 55.0, 70.0], HIGH_BANDS))
    assert not bunched.available and "span 60 deg" in bunched.reason
    assert bunched.bearing_from_phase(20.0, 3_800_000.0) == {
        "available": False, "reason": bunched.reason, "bearings_deg": []}
    assert bunched.phase_at(20.0, 3_800_000.0) is None
    # a line with no ratio, no bearing or no coherence is not a measurement
    thin = _gevents(five, HIGH_BANDS)
    for e in thin[:11]:
        e["ratio"] = None
    assert fit_global(thin).n_beacons == 4


def test_the_log_feeds_the_global_fit_and_the_compass_answers_at_the_slice(tmp_path):
    log = SiteLog(str(tmp_path / "site-log.jsonl"))
    bearings = [10.0, 75.0, 140.0, 215.0, 300.0]
    calls = ["4U1UN", "W6WX", "ZL6B", "OH2B", "CS3B"]
    for f in HIGH_BANDS:
        for call, t in zip(calls, bearings):
            log.beacon(band_hz=f, callsign=call, bearing_deg=t, distance_km=5000.0,
                       snr_a_db=20.0, snr_b_db=18.0, floor_a_db=-24.0,
                       floor_b_db=-23.5, coherence=0.9,
                       ratio=np.exp(1j * math.radians(_gphase(t, f))))
    g = fit_global_from_log(log)
    assert g.available and g.n_beacons == 15 and g.unique
    assert g.d_m == pytest.approx(D_M, abs=0.1)
    truth = 250.0
    out = compass_json(log, phase_deg=_gphase(truth, 3_800_000.0), f_hz=3_800_000.0)
    assert out["global"]["available"] and out["global"]["n_bands"] == 3
    assert min(abs(b - truth) for b in out["bearing"]["bearings_deg"]) < 5.0
    # the per-band fits are the check on the global one, and they agree
    for row in out["bands"]:
        assert row["available"] and abs(row["vs_global"]["disagreement_deg"]) < 3.0
        assert row["vs_global"]["k_global"] == pytest.approx(row["k"], abs=0.05)
    # no frequency, no global bearing -- the per-band answers still stand
    quiet = compass_json(log, phase_deg=_gphase(truth, BAND))
    assert "bearing" not in quiet and quiet["global"]["available"]
    assert quiet["bands"][0]["bearing_from_phase"]["available"]
