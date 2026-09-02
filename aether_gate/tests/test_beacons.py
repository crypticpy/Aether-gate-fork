#
# Aether-gate — the NCDXF beacon watch, no hardware.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The schedule (who is on which band when, from UTC alone) and a synthetic
slot: a keyed carrier 40 Hz off nominal at four descending power steps on
both loops with a fixed phase, then noise. The watch must name the beacon
from the clock, hear the steps the SNR allows, and read the pair's phase.

Run:  python -m pytest aether_gate/tests/test_beacons.py
"""
import numpy as np
import pytest

from aether_gate.core.beacons import (BeaconWatch, on_air, slot_at, BANDS_HZ, BEACONS,
                                      GRIDS, SLOT_S, SLOTS, bearing_distance, grid_to_latlon)

RATE = 125_000.0
BLOCK = 4096
T0 = 1_800_000_000.0 - (1_800_000_000.0 % (SLOT_S * SLOTS))     # a cycle boundary


def test_schedule_follows_utc_and_shifts_one_slot_per_band():
    assert slot_at(T0) == 0 and slot_at(T0 + 10.0) == 1 and slot_at(T0 + 179.9) == 17
    assert on_air(T0, BANDS_HZ[0])[0] == "4U1UN"
    assert on_air(T0, BANDS_HZ[1])[0] == "YV5B"          # one slot behind on the next band
    assert on_air(T0 + 10.0, BANDS_HZ[1])[0] == "4U1UN"
    assert on_air(T0 + 40.0, BANDS_HZ[4])[0] == "4U1UN"
    assert on_air(T0 + 25.0, BANDS_HZ[0])[0] == BEACONS[2][0]
    assert on_air(T0 + 3.0, BANDS_HZ[0])[2] == pytest.approx(7.0)


def _slot(rng, snr_db, phase=0.9, ratio=0.8, offset_hz=40.0, noise=1.0, rel_hz=0.0):
    """One 10 s slot at the sample rate: 2 s of keyed callsign, four 1 s
    dashes at 0/-10/-20/-30 dB, silence. The carrier sits rel_hz from the
    span's centre (the beacon frequency's place in the span) plus offset_hz
    (the beacon's error from nominal). Returns (a, b)."""
    n = int(SLOT_S * RATE)
    t = np.arange(n) / RATE
    offset_hz = rel_hz + offset_hz
    # snr_db is in a 500 Hz bandwidth: the carrier against the noise that
    # falls in 500 Hz of the RATE-wide floor (noise is the per-loop power)
    amp = np.sqrt(noise * (500.0 / RATE) * 10 ** (snr_db / 10.0))
    env = np.zeros(n)
    key = ((t * 8.0) % 1.0) < 0.5                               # ~dits at 8 Hz
    env[(t >= 0.5) & (t < 2.5)] = 1.0
    env[(t >= 0.5) & (t < 2.5)] *= key[(t >= 0.5) & (t < 2.5)]
    for s in range(4):
        env[(t >= 2.5 + s) & (t < 3.5 + s)] = 10 ** (-s * 10.0 / 20.0)
    carrier = amp * env * np.exp(2j * np.pi * offset_hz * t)
    a = (rng.normal(size=n) + 1j * rng.normal(size=n)) * np.sqrt(noise / 2) + carrier
    b = (rng.normal(size=n) + 1j * rng.normal(size=n)) * np.sqrt(noise / 2) \
        + carrier * ratio * np.exp(1j * phase)
    return a, b


def _feed(watch, a, b, center_hz, t_start):
    n = len(a)
    for i in range(0, n - BLOCK + 1, BLOCK):
        watch.update(a[i:i + BLOCK], b[i:i + BLOCK], center_hz, t_start + i / RATE)


def test_nothing_happens_when_no_beacon_frequency_is_in_span():
    w = BeaconWatch(RATE)
    rng = np.random.default_rng(1)
    a, b = _slot(rng, 30.0)
    _feed(w, a[:BLOCK * 8], b[:BLOCK * 8], 7_200_000.0, T0)
    assert w.band_hz is None and w.results == {}
    assert w.status(T0)["now"] is None


def test_a_strong_beacon_is_named_heard_on_all_four_steps_with_its_phase():
    w = BeaconWatch(RATE)
    rng = np.random.default_rng(2)
    center = 14_120_000.0                                        # 14.100 sits 20 kHz below
    a, b = _slot(rng, 40.0, phase=0.9, ratio=0.8, rel_hz=BANDS_HZ[0] - center)
    _feed(w, a, b, center, T0 + 20.0)                            # slot 2: W6WX on 14.100
    q, qb = _slot(rng, -40.0)                                     # the next slot, empty
    _feed(w, q[:BLOCK * 4], qb[:BLOCK * 4], center, T0 + 30.0)   # crossing the boundary scores it
    r = w.results[(BANDS_HZ[0], "W6WX")]
    assert r["heard"] and r["at"] == pytest.approx(T0 + 20.0)
    assert r["location"].startswith("Mt Umunhum")
    assert r["offset_hz"] == pytest.approx(40.0, abs=25.0)
    assert 33.0 <= r["snr_db"] <= 47.0, r
    assert r["steps_heard"] == 4 and r["lowest_w"] == 0.1, r
    assert r["steps_db"][0] - r["steps_db"][3] == pytest.approx(30.0, abs=4.0), r
    assert r["phase_deg"] == pytest.approx(np.degrees(-0.9), abs=6.0), r
    assert r["coherence"] >= 0.9
    assert r["snr_a"] - r["snr_b"] == pytest.approx(-20 * np.log10(0.8), abs=1.5), r
    assert 1.5 <= r["gain_db"] <= 2.5, r
    st = w.status(T0 + 31.0)
    assert st["now"]["call"] == "KH6RS" and st["results"][0]["call"] == "W6WX"


def test_a_weak_beacon_hears_only_the_top_step_and_a_missing_one_is_not_heard():
    w = BeaconWatch(RATE)
    rng = np.random.default_rng(3)
    center = 21_140_000.0                                        # 21.150 is 10 kHz up
    a, b = _slot(rng, 12.0, rel_hz=BANDS_HZ[2] - center)
    _feed(w, a, b, center, T0)                                   # slot 0 on 21.150: OA4B
    e, eb = _slot(rng, -60.0)
    _feed(w, e, eb, center, T0 + 10.0)                           # slot 1: empty
    z, zb = _slot(rng, -60.0)
    _feed(w, z[:BLOCK * 3], zb[:BLOCK * 3], center, T0 + 20.0)
    weak = w.results[(BANDS_HZ[2], "OA4B")]
    assert weak["heard"] and 1 <= weak["steps_heard"] <= 2 and weak["lowest_w"] in (100.0, 10.0)
    silent = w.results[(BANDS_HZ[2], "YV5B")]
    assert not silent["heard"] and silent["steps_heard"] == 0 and silent["lowest_w"] is None
    assert silent["phase_deg"] is None and silent["gain_db"] is None


# --- where each beacon is, and what the samples add up to ----------------------

def test_every_beacon_has_a_locator_and_the_geometry_is_right():
    assert set(GRIDS) == {c for c, _ in BEACONS}
    lat, lon = grid_to_latlon("EM10")                       # central Texas
    assert (lat, lon) == pytest.approx((30.5, -97.0))
    lat6, lon6 = grid_to_latlon("EM10cf")
    assert abs(lat6 - lat) < 0.5 and abs(lon6 - lon) < 1.0
    brg, km = bearing_distance(lat, lon, *grid_to_latlon(GRIDS["W6WX"]))
    assert 285 <= brg <= 305 and 2200 <= km <= 2600           # California: WNW, ~2400 km
    brg, km = bearing_distance(lat, lon, *grid_to_latlon(GRIDS["ZL6B"]))
    assert 225 <= brg <= 240 and 11500 <= km <= 12500         # New Zealand: SW, long path
    for bad in ("", "E1", "ZZ99", "EM1x", "EM10zz", "12ab"):
        with pytest.raises(ValueError):
            grid_to_latlon(bad)


def test_results_gain_bearings_when_the_station_grid_is_known_and_survive_a_restart(tmp_path):
    store = tmp_path / "beacons.json"
    w = BeaconWatch(RATE, store_path=str(store))
    rng = np.random.default_rng(4)
    center = 14_120_000.0
    a, b = _slot(rng, 40.0, phase=0.9, ratio=0.8, rel_hz=BANDS_HZ[0] - center)
    _feed(w, a, b, center, T0 + 20.0)                            # W6WX
    q, qb = _slot(rng, -40.0)
    _feed(w, q[:BLOCK * 4], qb[:BLOCK * 4], center, T0 + 30.0)
    r = w.results[(BANDS_HZ[0], "W6WX")]
    assert r["bearing_deg"] is None and r["distance_km"] is None and r["grid"] == "CM97bd"
    assert r["samples"] == 1 and r["heard_n"] == 1 and r["snr_mean_db"] == r["snr_db"]
    st = w.status(T0 + 31.0)
    assert st["station_grid"] is None and st["pattern"] == []
    assert st["propagation"] == [{"band_hz": BANDS_HZ[0], "sampled": 1, "heard": 1, "of": SLOTS,
                                  "best_w": 0.1, "median_snr_db": r["snr_db"],
                                  "updated": r["at"]}]
    w.set_station("EM10")
    r = w.results[(BANDS_HZ[0], "W6WX")]
    assert 285 <= r["bearing_deg"] <= 305 and 2200 <= r["distance_km"] <= 2600
    st = w.status(T0 + 31.0)
    assert st["station_grid"] == "EM10"
    assert st["pattern"] == [{"call": "W6WX", "band_hz": BANDS_HZ[0],
                              "bearing_deg": r["bearing_deg"], "distance_km": r["distance_km"],
                              "b_minus_a_db": pytest.approx(r["snr_b"] - r["snr_a"], abs=0.11),
                              "phase_deg": r["phase_deg"], "snr_db": r["snr_db"]}]
    with pytest.raises(ValueError):
        w.set_station("ZZ99")
    assert w.station_grid == "EM10"
    # a second sample of the same beacon merges into the running record
    a2, b2 = _slot(rng, 30.0, phase=0.9, ratio=0.8, rel_hz=BANDS_HZ[0] - center)
    _feed(w, a2, b2, center, T0 + 180.0 + 20.0)
    _feed(w, q[:BLOCK * 4], qb[:BLOCK * 4], center, T0 + 180.0 + 30.0)
    r2 = w.results[(BANDS_HZ[0], "W6WX")]
    assert r2["samples"] == 2 and r2["heard_n"] == 2 and r2["at"] == pytest.approx(T0 + 200.0)
    assert r2["snr_mean_db"] == pytest.approx((r["snr_db"] + r2["snr_db"]) / 2, abs=0.11)
    # a fresh watch on the same store remembers the grid and the samples
    w2 = BeaconWatch(RATE, store_path=str(store))
    assert w2.station_grid == "EM10"
    got = w2.results[(BANDS_HZ[0], "W6WX")]
    assert got["samples"] == 2 and got["bearing_deg"] == r2["bearing_deg"]
    assert w2.status(T0)["propagation"][0]["heard"] == 1
    w2.set_station("")
    assert w2.station_grid is None and w2.results[(BANDS_HZ[0], "W6WX")]["bearing_deg"] is None
    assert BeaconWatch(RATE, store_path=str(store)).station_grid is None
