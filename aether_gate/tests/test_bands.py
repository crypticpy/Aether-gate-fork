#
# Aether-gate — G1: the band table, and the gate noticing a retune.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""core/bands.py is pure, so the table and BandWatch are driven straight; the
publication half is driven through a _DiversityState on a fake adapter, the
same one test_soapy_diversity_state.py uses.

Each test names the mutation it catches in its own body.

Run:  .venv/bin/python -m pytest aether_gate/tests/test_bands.py -q
"""
import numpy as np

from aether_gate.adapters.diversity_state import _DiversityState
from aether_gate.core import bands

M20 = 14_175_000            # (14.000 + 14.350) / 2
M40 = 7_150_000
M80 = 3_750_000


class _FakeAdapter:
    """Enough of SoapyAdapter for status() -- no hardware, no stream. The
    centre is a quarter-span above the slice, which is where soapy's
    offset tuning actually puts it (_dc_offset_hz)."""

    _mode = "USB"
    _filt = None

    def __init__(self, slice_hz=14_212_000.0, samp_rate=2_040_000.0):
        self._np = np
        self.samp_rate = float(samp_rate)
        self._slice_hz = float(slice_hz)
        self.center_hz = self._slice_hz + 0.25 * self.samp_rate

    def tune(self, slice_hz):
        self._slice_hz = float(slice_hz)
        self.center_hz = self._slice_hz + 0.25 * self.samp_rate


# --- the table --------------------------------------------------------------

def test_every_band_edge_is_inside_its_own_band_and_the_centre_is_the_key():
    """Mutation: an edge off by a kilohertz, or a Region 1 edge (40 m ending
    at 7.200, 6 m at 52.000) -- 7.290 and 53.000 would then be nowhere."""
    for name, lo, hi in bands.BANDS:
        centre = (lo + hi) // 2
        assert bands.band_of(lo) == centre, name
        assert bands.band_of(hi) == centre, name
        assert bands.band_of((lo + hi) / 2.0) == centre, name
        assert bands.band_name(lo) == name
    assert bands.band_of(7_290_000) == M40          # Region 2 runs to 7.300
    assert bands.band_of(53_000_000) == bands.band_of(50_000_000)


def test_off_the_bands_is_none_not_the_nearest_band():
    """Mutation: a table that clamps instead of returning None would call
    the 5 MHz broadcast band 60 m and remember talkers against it."""
    for hz in (0, 1_799_999, 5_000_000, 13_999_999, 14_350_001, 30_000_000):
        assert bands.band_of(hz) is None, hz
    assert bands.band_name(5_000_000) is None
    assert bands.band_of(None) is None and bands.band_of("nonsense") is None
    assert bands.band_of(float("nan")) is None


def test_band_changed_compares_bands_not_frequencies():
    """Mutation: `a != b` on the raw hertz -- every dial move would then be a
    band change, and the governor would put the squeeze back on every nudge."""
    assert bands.band_changed(14_020_000, 14_340_000) is False
    assert bands.band_changed(14_200_000, 7_150_000) is True
    # both off the bands is not a change: out-of-band tuning must not churn
    assert bands.band_changed(4_500_000, 4_900_000) is False
    assert bands.band_changed(14_200_000, 14_400_000) is True


# --- BandWatch --------------------------------------------------------------

def test_the_first_tune_seeds_the_band_and_is_not_a_change():
    """Mutation: stamping band_changed_at on the first reading would have the
    app announce a band change every time the gate started."""
    seen = []
    w = bands.BandWatch(lambda b, hz: seen.append((b, hz)))
    assert w.tune(14_200_000, 1000.0) is False
    assert w.band_hz == M20 and w.band_changed_at is None
    assert seen == [(M20, 14_200_000.0)]


def test_a_move_inside_the_band_is_not_a_change_and_a_move_across_it_is():
    w = bands.BandWatch()
    w.tune(14_200_000, 1000.0)
    assert w.tune(14_340_000, 1001.0) is False and w.band_changed_at is None
    assert w.tune(7_150_000, 1002.0) is True
    assert w.band_hz == M40 and w.band_changed_at == 1002.0


def test_retuned_stamps_the_clock_and_says_whether_the_centre_left_the_band():
    w = bands.BandWatch()
    assert w.retuned(14_200_000, 14_300_000, 500.0) is False
    assert w.retuned_at == 500.0 and w.band_hz is None      # the band is tune()'s
    assert w.retuned(14_200_000, 7_100_000, 501.0) is True
    assert w.retuned_at == 501.0


def test_status_is_the_three_keys_and_nothing_else():
    assert set(bands.BandWatch().status()) == {"retuned_at", "band_hz",
                                               "band_changed_at"}


# --- what /diversity publishes ---------------------------------------------

def test_the_band_comes_from_the_slice_not_the_hardware_centre():
    """Mutation: band_of(center_hz). The centre is parked a quarter span off
    the slice (0.51 MHz at 2.04 MS/s), so 14.212 would report band None and
    every band-keyed memory in the gate would be filed under nothing."""
    a = _FakeAdapter(14_212_000.0)
    st = _DiversityState(a)
    assert bands.band_of(a.center_hz) is None       # the trap this test guards
    s = st.status()
    assert s["band_hz"] == M20
    assert s["retuned_at"] is None and s["band_changed_at"] is None


def test_a_retune_stamps_retuned_at_and_the_new_band_reaches_the_memory():
    """Mutation: an on_retune that only stamps, or one that never tells the
    talker memory -- the fingerprints would go on matching across the band."""
    a = _FakeAdapter(14_212_000.0)
    st = _DiversityState(a)
    assert st.memory.band_hz == M20
    a.tune(7_120_000.0)
    st.on_retune(14_722_000.0, 7_630_000.0)
    s = st.status()
    assert s["retuned_at"] is not None
    assert s["band_hz"] == M40 and s["band_changed_at"] is not None
    assert st.memory.band_hz == M40 and st.memory.center_hz == 7_120_000.0


def test_a_slice_move_inside_the_band_stamps_nothing():
    """Mutation: stamping band_changed_at on every status read. The dial moves
    inside the window without any hardware retune at all, so this is the path
    that would fire on a 2 kHz nudge."""
    a = _FakeAdapter(14_212_000.0)
    st = _DiversityState(a)
    st.status()
    a.tune(14_290_000.0)
    s = st.status()
    assert s["band_hz"] == M20 and s["band_changed_at"] is None
    a.tune(3_790_000.0)
    assert st.status()["band_hz"] == M80


def test_the_new_keys_are_additive():
    """Mutation: a rename. The app reads every one of these by name."""
    st = _DiversityState(_FakeAdapter())
    s = st.status()
    for k in ("available", "mode", "source", "memory", "talker", "nb", "squeeze",
              "retuned_at", "band_hz", "band_changed_at"):
        assert k in s, k
