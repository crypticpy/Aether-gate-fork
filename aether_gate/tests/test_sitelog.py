#
# Aether-gate -- the site log: one line per event, and it stays readable.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""What the pair measures is worth nothing if it is thrown away when the
window repaints. These are the promises: both kinds of event round-trip
(the beacon one carrying the COMPLEX ratio, not just its angle), the noise
verdict does not fill a disk with the same sentence, the directory appears
by itself, and a path that cannot be written is said once and never raised
into the DSP thread that called it.

Run:  python -m pytest aether_gate/tests/test_sitelog.py
"""
import json
import math

import numpy as np
import pytest

from aether_gate.core.beacons import BANDS_HZ, BeaconWatch, SLOT_S, SLOTS
from aether_gate.core.sitelog import SiteLog

RATE = 125_000.0
BLOCK = 4096
T0 = 1_800_000_000.0 - (1_800_000_000.0 % (SLOT_S * SLOTS))     # a cycle boundary


class _Clock:
    """A clock the test winds by hand."""

    def __init__(self, t=1_800_000_000.0):
        self.t = float(t)

    def __call__(self):
        return self.t


def _log(tmp_path, clock=None, name="site-log.jsonl"):
    return SiteLog(str(tmp_path / name), clock=clock or _Clock())


def _noise_fields(**over):
    f = dict(samp_rate=125_000.0, center_hz=14_120_000.0, mains_hz=60.0, hum_db=14.2,
             harmonics=4, impulses_per_s=2.1, impulse_db=17.5,
             lines=[{"hz": 480.0, "db": 11.0}], noise_coherence=0.62)
    f.update(over)
    return f


def test_a_beacon_event_round_trips_with_its_complex_ratio(tmp_path):
    log = _log(tmp_path)
    log.beacon(band_hz=14_100_000.0, callsign="W6WX", bearing_deg=295.0,
               distance_km=2410.0, snr_a_db=21.4, snr_b_db=18.9,
               floor_a_db=-121.5, floor_b_db=-120.2, ratio=complex(0.62, -0.48),
               coherence=0.91, steps_heard=3, lowest_w=1.0, mrc_gain_db=2.3)
    (rec,) = list(log.read())
    assert rec["kind"] == "beacon" and rec["callsign"] == "W6WX"
    assert rec["t"].endswith("+00:00") and rec["t"].startswith("2027-")
    assert rec["ratio"] == [0.62, -0.48]                    # complex, not an angle
    assert complex(*rec["ratio"]) == pytest.approx(complex(0.62, -0.48))
    assert rec["bearing_deg"] == 295.0 and rec["distance_km"] == 2410.0
    assert (rec["snr_a_db"], rec["snr_b_db"]) == (21.4, 18.9)
    assert (rec["floor_a_db"], rec["floor_b_db"]) == (-121.5, -120.2)
    assert rec["steps_heard"] == 3 and rec["lowest_w"] == 1.0
    assert rec["mrc_gain_db"] == 2.3 and rec["coherence"] == 0.91
    # a ratio given as a pair, or missing entirely, is just as welcome
    log.beacon(band_hz=14_100_000.0, callsign="OH2B", ratio=[0.1, 0.2])
    log.beacon(band_hz=14_100_000.0, callsign="ZL6B")
    oh2b, zl6b = list(log.read())[1:]
    assert oh2b["ratio"] == [0.1, 0.2] and zl6b["ratio"] is None
    assert log.written == 3 and log.error is None


def test_a_noise_event_passes_the_profile_lines_through(tmp_path):
    log = _log(tmp_path)
    log.noise(**_noise_fields())
    (rec,) = list(log.read(kind="noise"))
    assert rec["kind"] == "noise" and rec["mains_hz"] == 60.0 and rec["harmonics"] == 4
    assert rec["hum_db"] == 14.2 and rec["impulses_per_s"] == 2.1
    assert rec["impulse_db"] == 17.5 and rec["noise_coherence"] == 0.62
    assert rec["lines"] == [{"hz": 480.0, "db": 11.0}]      # straight through
    assert rec["samp_rate"] == 125_000.0 and rec["center_hz"] == 14_120_000.0


def test_the_profile_status_can_be_handed_over_whole(tmp_path):
    log = _log(tmp_path)
    status = {"mains_hz": 50.0, "hum_db": 9.9, "harmonics": 2, "impulses_per_s": 0.4,
              "impulse_db": None, "periodic": [{"hz": 1000.0, "db": 9.0}], "seconds": 2.0}
    log.noise_status(status, 125_000.0, 7_074_000.0, noise_coherence=0.3)
    (rec,) = list(log.read(kind="noise"))
    assert rec["mains_hz"] == 50.0 and rec["harmonics"] == 2
    assert rec["lines"] == [{"hz": 1000.0, "db": 9.0}] and rec["impulse_db"] is None
    assert rec["noise_coherence"] == 0.3


def test_the_same_noise_verdict_writes_one_line_a_minute(tmp_path):
    clock = _Clock()
    log = _log(tmp_path, clock)
    assert log.noise(**_noise_fields()) is not None         # the first is always kept
    for dt in (1.0, 5.0, 30.0, 59.0):                       # the profile speaks each second
        clock.t += dt - (clock.t - 1_800_000_000.0)
        assert log.noise(**_noise_fields(hum_db=14.4)) is None
    clock.t = 1_800_000_000.0 + 61.0
    assert log.noise(**_noise_fields(hum_db=14.4)) is not None
    assert log.written == 2 and log.skipped == 4
    # a day of an unchanging verdict is 1440 lines, not 86400
    assert len(list(log.read(kind="noise"))) == 2


def test_a_changed_verdict_writes_a_line_before_the_minute_is_out(tmp_path):
    clock = _Clock()
    log = _log(tmp_path, clock)
    log.noise(**_noise_fields())
    clock.t += 10.0
    assert log.noise(**_noise_fields(mains_hz=50.0)) is not None      # another grid
    clock.t += 10.0
    assert log.noise(**_noise_fields(mains_hz=50.0, impulses_per_s=90.0)) is not None
    clock.t += 10.0
    assert log.noise(**_noise_fields(mains_hz=50.0, impulses_per_s=90.0,
                                     lines=[{"hz": 1200.0, "db": 14.0}])) is not None
    clock.t += 10.0                                          # ... but breathing is not a change
    assert log.noise(**_noise_fields(mains_hz=50.0, impulses_per_s=95.0,
                                     lines=[{"hz": 1202.0, "db": 15.0}])) is None
    clock.t += 1.0                                           # the mains comb goes away
    assert log.noise(**_noise_fields(mains_hz=None, impulses_per_s=0.0)) is not None
    clock.t += 2.0                                           # ... and a verdict that flickers
    assert log.noise(**_noise_fields(mains_hz=50.0)) is None  # on a bucket edge still waits
    assert log.written == 5


def test_read_filters_by_kind_and_by_time(tmp_path):
    clock = _Clock()
    log = _log(tmp_path, clock)
    log.beacon(band_hz=14_100_000.0, callsign="4U1UN")
    log.noise(**_noise_fields())
    clock.t += 3600.0
    log.beacon(band_hz=21_150_000.0, callsign="OH2B")
    assert [r["kind"] for r in log.read()] == ["beacon", "noise", "beacon"]
    assert [r["callsign"] for r in log.read(kind="beacon")] == ["4U1UN", "OH2B"]
    assert len(list(log.read(kind="noise"))) == 1
    recent = list(log.read(since=1_800_000_000.0 + 60.0))
    assert [r["callsign"] for r in recent] == ["OH2B"]
    assert list(log.read(since=1_800_000_000.0 + 7200.0)) == []
    # an ISO string is a time too
    assert len(list(log.read(since="2027-01-15T00:00:00+00:00"))) == 3


def test_the_directory_is_made_on_the_first_line(tmp_path):
    path = tmp_path / "not" / "there" / "yet" / "site-log.jsonl"
    log = SiteLog(str(path), clock=_Clock())
    assert list(log.read()) == []                            # no file, no exception
    log.beacon(band_hz=14_100_000.0, callsign="CS3B")
    assert path.exists() and log.error is None
    assert json.loads(path.read_text().splitlines()[0])["callsign"] == "CS3B"


def test_an_unwritable_path_is_said_once_and_never_raised(tmp_path, capsys):
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where a directory should be\n")
    log = SiteLog(str(blocked / "site-log.jsonl"), clock=_Clock())
    for _ in range(5):
        assert log.noise(force=True, **_noise_fields()) is None
    assert log.beacon(band_hz=14_100_000.0, callsign="OA4B") is None
    assert log.written == 0 and log.error is not None
    assert capsys.readouterr().out.count("[sitelog]") == 1    # said once, not per block
    assert list(log.read()) == []
    assert log.status()["error"] == log.error


def test_a_line_a_crash_truncated_is_skipped_not_raised(tmp_path):
    path = tmp_path / "site-log.jsonl"
    log = SiteLog(str(path), clock=_Clock())
    log.beacon(band_hz=14_100_000.0, callsign="YV5B")
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"t": "2027-01-15T00:00:00+00:00", "kind": "bea')
    rows = list(log.read())
    assert [r["callsign"] for r in rows] == ["YV5B"]


def _slot(rng, snr_db, phase=0.9, ratio=0.8, offset_hz=40.0, noise=1.0, rel_hz=0.0):
    """One 10 s beacon slot: a keyed callsign then four descending dashes,
    loop B a fixed ratio and phase from loop A (see test_beacons)."""
    n = int(SLOT_S * RATE)
    t = np.arange(n) / RATE
    off = rel_hz + offset_hz
    amp = math.sqrt(noise * (500.0 / RATE) * 10 ** (snr_db / 10.0))
    env = np.zeros(n)
    key = ((t * 8.0) % 1.0) < 0.5
    env[(t >= 0.5) & (t < 2.5)] = 1.0
    env[(t >= 0.5) & (t < 2.5)] *= key[(t >= 0.5) & (t < 2.5)]
    for s in range(4):
        env[(t >= 2.5 + s) & (t < 3.5 + s)] = 10 ** (-s * 10.0 / 20.0)
    carrier = amp * env * np.exp(2j * np.pi * off * t)
    a = (rng.normal(size=n) + 1j * rng.normal(size=n)) * math.sqrt(noise / 2) + carrier
    b = (rng.normal(size=n) + 1j * rng.normal(size=n)) * math.sqrt(noise / 2) \
        + carrier * ratio * np.exp(1j * phase)
    return a, b


def test_a_scored_slot_carries_its_complex_ratio_and_floors_into_the_log(tmp_path):
    watch = BeaconWatch(RATE)
    watch.set_station("EM10")
    rng = np.random.default_rng(11)
    center = 14_120_000.0
    a, b = _slot(rng, 40.0, phase=0.9, ratio=0.8, rel_hz=BANDS_HZ[0] - center)
    for i in range(0, len(a) - BLOCK + 1, BLOCK):
        watch.update(a[i:i + BLOCK], b[i:i + BLOCK], center, T0 + 20.0 + i / RATE)
    watch.update(a[:BLOCK], b[:BLOCK], center, T0 + 31.0)          # ends the slot
    res = watch.results[(BANDS_HZ[0], "W6WX")]
    r = complex(*res["ratio"])
    assert abs(r) == pytest.approx(0.8, abs=0.02)                  # B relative to A
    assert math.degrees(math.atan2(r.imag, r.real)) == pytest.approx(math.degrees(0.9), abs=2.0)
    assert res["phase_deg"] == pytest.approx(-math.degrees(0.9), abs=2.0)   # the other way
    # both loops saw the same unit-power noise: -24 dB in a 500 Hz bandwidth
    assert res["floor_a_db"] == pytest.approx(-24.0, abs=1.0)
    assert res["floor_b_db"] == pytest.approx(-24.0, abs=1.0)
    assert res["snr_a"] + res["floor_a_db"] == pytest.approx(40.0 - 24.0, abs=2.0)

    log = _log(tmp_path)
    rec = log.beacon_result(res)
    assert rec["callsign"] == "W6WX" and rec["band_hz"] == BANDS_HZ[0]
    assert rec["ratio"] == res["ratio"] and rec["coherence"] == res["coherence"]
    assert rec["floor_a_db"] == res["floor_a_db"] and rec["snr_b_db"] == res["snr_b"]
    assert rec["mrc_gain_db"] == res["gain_db"] and rec["steps_heard"] == 4
    assert 285 <= rec["bearing_deg"] <= 305 and 2200 <= rec["distance_km"] <= 2600
    assert log.beacon_result(None) is None


def test_the_antenna_note_is_stamped_on_every_line_and_is_its_own_verdict(tmp_path):
    t = [T0]
    log = SiteLog(tmp_path / "site.jsonl", clock=lambda: t[0])
    assert log.noise(**_noise_fields())["antenna"] is None
    log.antenna = "K-480WLA HF, gain HIGH"
    t[0] += 6.0                                 # same verdict, but a new note
    assert log.noise(**_noise_fields())["antenna"] == "K-480WLA HF, gain HIGH"
    assert log.beacon(band_hz=14.1e6, callsign="W6WX")["antenna"] == "K-480WLA HF, gain HIGH"
    assert log.status()["antenna"] == "K-480WLA HF, gain HIGH"
    log.antenna = ""
    assert log.status()["antenna"] is None
    assert [r.get("antenna") for r in log.read()] == [None, "K-480WLA HF, gain HIGH",
                                                     "K-480WLA HF, gain HIGH"]
