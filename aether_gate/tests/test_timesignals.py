#
# Aether-gate -- the standard-frequency stations as low-band known directions.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The table first: who is on which frequency, which frequencies two of them
share, and where each of them is from a station in central Texas. Then a
synthetic carrier -- ten seconds of it, unkeyed, the way a time signal
actually sounds to a narrowband detector -- fed to the watch on the minute,
which has to come back with the SAME dict a beacon slot produces, so the
site log and the compass never learn there is a difference.

And the awkward one: 10 MHz is WWV and WWVH and BPM at once. It is scored,
because the SNRs are propagation, but it is NOT a direction, and the
compass has to drop it until the operator says which of the three he is
listening to.

Run:  python -m pytest aether_gate/tests/test_timesignals.py
"""
import math

import numpy as np
import pytest

from aether_gate.core import compass
from aether_gate.core.beacons import grid_to_latlon
from aether_gate.core.sitelog import SiteLog
from aether_gate.core.timesignals import (CARRIERS, PERIOD_S, STATIONS,
                                          UNAMBIGUOUS_HZ, TimeSignalWatch,
                                          shared_with, station_table)

RATE = 125_000.0
BLOCK = 4096
T0 = 1_800_000_000.0                      # a minute boundary in UTC
SHARED = (2_500_000.0, 5_000_000.0, 10_000_000.0, 15_000_000.0)


def test_the_table_names_every_carrier_and_flags_the_shared_ones():
    assert set(STATIONS) == {"WWV", "WWVH", "CHU", "RWM", "BPM"}
    for f in SHARED:
        assert CARRIERS[f] == ("BPM", "WWV", "WWVH")
    assert CARRIERS[20_000_000.0] == CARRIERS[25_000_000.0] == ("WWV",)
    assert CARRIERS[7_850_000.0] == ("CHU",) and CARRIERS[9_996_000.0] == ("RWM",)
    # RWM sits 4 kHz below WWV, well outside the +-100 Hz search bin
    assert 5_000_000.0 - 4_996_000.0 == 4_000.0
    assert shared_with(10_000_000.0, "WWV") == ["BPM", "WWVH"]
    assert shared_with(10_000_000.0) == ["BPM", "WWV", "WWVH"]
    assert shared_with(3_330_000.0, "CHU") == []
    # what is left to fit a compass with, and three of them are below 10 MHz
    assert set(UNAMBIGUOUS_HZ) == {3_330_000.0, 4_996_000.0, 7_850_000.0,
                                   9_996_000.0, 14_670_000.0, 14_996_000.0,
                                   20_000_000.0, 25_000_000.0}
    assert sum(1 for f in UNAMBIGUOUS_HZ if f < 10e6) == 4


def test_every_station_has_a_locator_and_a_bearing_from_the_operators_grid():
    rows = {r["call"]: r for r in station_table("EM10")}       # central Texas
    for r in rows.values():
        grid_to_latlon(r["grid"])                               # parses, or raises
    assert 320 <= rows["WWV"]["bearing_deg"] <= 340            # Colorado, NNW
    assert 1_100 <= rows["WWV"]["distance_km"] <= 1_500
    assert 30 <= rows["CHU"]["bearing_deg"] <= 55              # Ottawa, NE
    assert 2_200 <= rows["CHU"]["distance_km"] <= 2_800
    assert 265 <= rows["WWVH"]["bearing_deg"] <= 290           # Hawaii, W
    assert 5_900 <= rows["WWVH"]["distance_km"] <= 6_500
    assert 10 <= rows["RWM"]["bearing_deg"] <= 40              # Moscow, over the pole
    assert 9_000 <= rows["RWM"]["distance_km"] <= 10_000
    assert 315 <= rows["BPM"]["bearing_deg"] <= 345            # Shaanxi, the other pole
    assert rows["WWVH"]["unambiguous_hz"] == []                # every WWVH carrier is shared
    assert rows["CHU"]["unambiguous_hz"] == [3_330_000.0, 7_850_000.0, 14_670_000.0]
    blind = {r["call"]: r for r in station_table()}
    assert all(r["bearing_deg"] is None and r["distance_km"] is None
               for r in blind.values())


def _carrier(rng, snr_db, phase=0.7, ratio=0.8, offset_hz=12.0, rel_hz=0.0,
             seconds=10.0, noise=1.0):
    """A continuous carrier -- what a time signal is between its ticks -- at
    rel_hz from the span's centre plus offset_hz of the transmitter's own
    error. snr_db is in the 500 Hz CW bandwidth, as everywhere else."""
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    amp = math.sqrt(noise * (500.0 / RATE) * 10 ** (snr_db / 10.0))
    carrier = amp * np.exp(2j * np.pi * (rel_hz + offset_hz) * t)
    a = (rng.normal(size=n) + 1j * rng.normal(size=n)) * math.sqrt(noise / 2) + carrier
    b = (rng.normal(size=n) + 1j * rng.normal(size=n)) * math.sqrt(noise / 2) \
        + carrier * ratio * np.exp(1j * phase)
    return a, b


def _feed(watch, a, b, center_hz, t_start):
    for i in range(0, len(a) - BLOCK + 1, BLOCK):
        watch.update(a[i:i + BLOCK], b[i:i + BLOCK], center_hz, t_start + i / RATE)


def _window(watch, center_hz, f_hz, t0, snr_db=30.0, rng=None, **kw):
    """One scored window: ten seconds on the minute, then a block past the
    end of it, which is what tells the watch the window is over."""
    rng = rng or np.random.default_rng(9)
    a, b = _carrier(rng, snr_db, rel_hz=f_hz - center_hz, **kw)
    _feed(watch, a, b, center_hz, t0)
    q, qb = _carrier(rng, -60.0, rel_hz=f_hz - center_hz, seconds=0.5)
    _feed(watch, q, qb, center_hz, t0 + 10.0)
    return watch.last


def test_a_time_signal_window_scores_into_the_shape_a_beacon_slot_has(tmp_path):
    w = TimeSignalWatch(RATE)
    w.set_station("EM10")
    center = 7_860_000.0                             # CHU on 7.850, 10 kHz down
    r = _window(w, center, 7_850_000.0, T0, snr_db=30.0)
    assert r is not None and r["call"] == "CHU" and r["source"] == "time_signal"
    assert r["band_hz"] == 7_850_000.0 and r["at"] == T0
    assert r["heard"] and 24.0 <= r["snr_db"] <= 36.0, r
    assert r["offset_hz"] == pytest.approx(12.0, abs=25.0)
    assert r["phase_deg"] == pytest.approx(math.degrees(-0.7), abs=6.0), r
    assert r["coherence"] >= 0.9 and len(r["ratio"]) == 2
    assert r["snr_a"] - r["snr_b"] == pytest.approx(-20 * math.log10(0.8), abs=1.5)
    assert r["floor_a_db"] is not None and r["floor_b_db"] is not None
    assert 30 <= r["bearing_deg"] <= 55 and 2_200 <= r["distance_km"] <= 2_800
    # the two fields only a four-step NCDXF beacon has
    assert r["steps_heard"] is None and r["lowest_w"] is None
    assert r["shared_with"] == [] and r["ambiguous"] is False
    assert r["samples"] == 1 and r["heard_n"] == 1 and r["snr_mean_db"] == r["snr_db"]
    # ... and the site log writes it by exactly the beacon path
    log = SiteLog(str(tmp_path / "site-log.jsonl"))
    line = log.beacon_result(r)
    assert line["kind"] == "beacon" and line["callsign"] == "CHU"
    assert line["band_hz"] == 7_850_000.0 and line["ratio"] == r["ratio"]
    assert line["steps_heard"] is None and line["lowest_w"] is None
    assert line["snr_a_db"] == r["snr_a"] and line["coherence"] == r["coherence"]
    assert compass._weight(line) > 0.5                 # the compass will use it
    st = w.status(T0 + 11.0)
    assert st["freq_hz"] == 7_850_000.0 and st["in_window"] is False
    assert st["results"] == [r] and st["last"] is r
    assert st["station_grid"] == "EM10" and len(st["stations"]) == 5


def test_a_shared_carrier_is_measured_but_is_not_a_direction():
    w = TimeSignalWatch(RATE)
    w.set_station("EM10")
    center = 10_010_000.0                            # 10.000 is 10 kHz down,
    r = _window(w, center, 10_000_000.0, T0, snr_db=30.0)     # RWM's 9.996 further
    assert r["call"] == "BPM/WWV/WWVH" and r["ambiguous"] is True
    assert r["shared_with"] == ["BPM", "WWV", "WWVH"] and r["assumed"] is False
    assert r["heard"] and r["ratio"] is not None       # the propagation is real
    assert r["bearing_deg"] is None and r["distance_km"] is None
    # no bearing, no vote: the global fit cannot use a line like this
    assert compass._global_rows([{"band_hz": r["band_hz"], "ratio": r["ratio"],
                                  "bearing_deg": r["bearing_deg"],
                                  "coherence": r["coherence"]}]) == []
    # the operator decides Kauai is the only one propagating, and says so
    with pytest.raises(ValueError):
        w.set_assumed(10_000_000.0, "CHU")
    w.set_assumed(10_000_000.0, "WWVH")
    r2 = _window(w, center, 10_000_000.0, T0 + PERIOD_S, snr_db=30.0)
    assert r2["call"] == "WWVH" and r2["ambiguous"] is False and r2["assumed"] is True
    assert r2["shared_with"] == ["BPM", "WWV"]
    assert 265 <= r2["bearing_deg"] <= 290
    assert w.status(T0 + PERIOD_S + 11.0)["assumed"] == {"10000000": "WWVH"}
    w.set_assumed(10_000_000.0, None)
    assert w.assume == {}


def test_one_window_a_minute_and_none_at_all_off_a_carrier():
    w = TimeSignalWatch(RATE)
    rng = np.random.default_rng(4)
    a, b = _carrier(rng, 30.0, seconds=1.0)
    _feed(w, a, b, 14_200_000.0, T0)                 # no carrier within the span
    assert w.freq_hz is None and w.results == {} and w.last is None
    assert w.status(T0)["in_window"] is False
    # a window that begins after the first ten seconds of the minute is a
    # partial one, and a partial window is not a measurement
    center = 4_986_000.0
    _window(w, center, 4_996_000.0, T0 + 40.0, snr_db=30.0, rng=rng)
    assert w.freq_hz == 4_996_000.0 and w.last is None
    # ... and the same minute does not get a second try
    r = _window(w, center, 4_996_000.0, T0, snr_db=30.0, rng=rng)
    assert r is not None and r["call"] == "RWM" and r["at"] == T0
    assert w.status(T0 + 5.0)["next_window_s"] == pytest.approx(55.0, abs=0.1)
