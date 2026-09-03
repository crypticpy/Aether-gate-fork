#
# Aether-gate — the filter per talker (core/filter.py, TALKER_SETTINGS).
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""A known talker's filter comes back the block they key up; a new talker
restarts the automatics from their own print; the filter stays with the
last voice through the silence; "smooth" glides, "fast" snaps; the memory
table carries each talker's filter; the whole thing can be turned off.

Run:  python -m pytest aether_gate/tests/test_talker_filter.py
"""
import numpy as np
import pytest

from aether_gate.core.filter import (SliceFilter, design_taps, AUTO_MIN_HIGH_HZ,
                                     SPEC_WARMUP_BLOCKS)
from aether_gate.core.engine import _filter_kwargs

RATE = 25000.0
N = 819


class _Mic:
    """Who is talking (the talker memory's id) and what their print says."""
    def __init__(self):
        self.talker = None
        self.prints = {}

    def talker_source(self):
        return self.talker

    def print_source(self):
        return self.prints.get(self.talker)


def _filter(mic, **kw):
    sf = SliceFilter(RATE, print_source=mic.print_source)
    sf.talker_source = mic.talker_source
    if kw:
        sf.set(**kw)
    return sf


def _voice(seconds, lo, hi, seed=3, amp=0.05):
    rng = np.random.default_rng(seed)
    total = (int(seconds * RATE) // N) * N
    w = amp * (rng.standard_normal(total + 1100) + 1j * rng.standard_normal(total + 1100))
    v = np.convolve(w, design_taps(RATE, lo, hi, "sharp"), mode="valid")[:total]
    return v + 0.0005 * (rng.standard_normal(total) + 1j * rng.standard_normal(total))


def _run(sf, sig):
    for i in range(0, len(sig) - N + 1, N):
        sf.apply(sig[i:i + N], 0)


def _blocks(sf, n=1):
    _run(sf, _voice(n * N / RATE + 0.01, 300, 2700))


def test_a_known_talkers_filter_comes_back_the_block_they_key_up():
    mic = _Mic()
    sf = _filter(mic)
    mic.talker = 1
    _blocks(sf)
    sf.set(low=300, high=2700, shape="sharp", threshold_db=12, contour=True, contour_db=4)
    mic.talker = 2
    _blocks(sf)                                   # #1's filter is stored as #2 keys up
    sf.set(low=200, high=3100, shape="soft", threshold_db=20, contour=False)
    st = sf.status()
    assert st["talker"] == {"enabled": True, "snap": "fast", "id": 2, "remembered": [1]}
    mic.talker = 1
    _blocks(sf)                                   # ONE block: #1 is back
    st = sf.status()
    assert (st["set_low_hz"], st["set_high_hz"], st["shape"]) == (300, 2700, "sharp")
    assert st["agc"]["threshold_db"] == 12.0 and sf.agc.threshold_db == 12.0
    assert st["contour"]["enabled"] is True and st["contour"]["db"] == 4.0
    assert st["talker"]["id"] == 1 and st["talker"]["remembered"] == [1, 2]
    mic.talker = 2
    _blocks(sf)
    st = sf.status()
    assert (st["set_low_hz"], st["set_high_hz"], st["shape"]) == (200, 3100, "soft")
    assert st["agc"]["threshold_db"] == 20.0 and st["contour"]["enabled"] is False


def test_notches_and_the_blanker_are_about_the_frequency_not_the_talker():
    mic = _Mic()
    sf = _filter(mic)
    mic.talker = 1
    _blocks(sf)
    sf.notch_add(1000, 120)
    sf.set(nb=True, nb_db=9, agc="slow")
    mic.talker = 2
    _blocks(sf)
    st = sf.status()
    assert [n["hz"] for n in st["notches"]] == [1000.0]
    assert st["nb"]["enabled"] is True and st["nb"]["threshold_db"] == 9.0
    assert st["agc"]["mode"] == "slow"
    sf.notch_clear()
    mic.talker = 1
    _blocks(sf)
    assert sf.status()["notches"] == []          # not brought back with #1


def test_the_filter_stays_with_the_last_voice_through_the_silence():
    mic = _Mic()
    sf = _filter(mic)
    mic.talker = 1
    _blocks(sf)
    sf.set(low=300, high=2700)
    mic.talker = None                             # the over ends
    _blocks(sf)
    sf.set(high=2500)                             # an edit in the silence is for #1
    mic.talker = 1
    _blocks(sf)
    assert sf.status()["set_high_hz"] == 2500 and sf.status()["talker"]["id"] == 1
    mic.talker = 2
    _blocks(sf)
    assert sf.talker_filter_summary(1)["high_hz"] == 2500


def test_a_new_talker_restarts_auto_from_their_own_print_at_once():
    mic = _Mic()
    sf = _filter(mic, low=100, high=3200, auto=True)
    mic.talker = 1
    _run(sf, _voice(3.0, 350, 2100))              # the spectrum fit for #1
    st = sf.status()
    assert st["auto"]["source"] == "spectrum" and 2400 <= st["auto"]["high_hz"] <= 2450
    mic.prints[2] = {"low_hz": 300, "high_hz": 2800, "tilt_db": -4.0}
    mic.talker = 2
    _blocks(sf)                                   # one block, no glide
    st = sf.status()
    assert st["auto"]["source"] == "print"
    assert 190 <= st["auto"]["low_hz"] <= 300 and 2850 <= st["auto"]["high_hz"] <= 2950, st["auto"]
    assert st["low_hz"] == st["auto"]["low_hz"]   # in force, not just reported
    # a talker with neither a print nor a filter: the spectrum starts over,
    # the edges follow this voice, not the last one
    mic.talker = 3
    _run(sf, _voice(2.0, 350, 2100, seed=7))
    st = sf.status()
    assert st["auto"]["source"] == "spectrum" and 2400 <= st["auto"]["high_hz"] <= 2450


def test_smooth_glides_where_fast_snaps():
    for snap, at_once in (("fast", True), ("smooth", False)):
        mic = _Mic()
        sf = _filter(mic, low=100, high=3400, auto=True, talker_snap=snap)
        mic.talker = 1
        _run(sf, _voice(3.0, 350, 2100))          # #1: 350..2400
        mic.talker = 2
        _run(sf, _voice(3.0, 350, 3100, seed=5))  # #2: wide, then back to #1
        assert sf.status()["auto"]["high_hz"] >= 3100
        mic.talker = 1
        _blocks(sf)
        hi = sf.status()["auto"]["high_hz"]
        assert (hi <= 2450) is at_once, (snap, hi)
        # fast: the restored edges survive the first noisy blocks of the
        # new spectrum; smooth: they arrive within ~2 s
        _run(sf, _voice(0.6, 350, 2100, seed=9))
        assert (sf.status()["auto"]["high_hz"] <= 2450) is at_once, (snap, sf.status()["auto"])
        _run(sf, _voice(1.6, 350, 2100, seed=11))
        assert sf.status()["auto"]["high_hz"] <= 2450, (snap, sf.status()["auto"])


def test_turned_off_nothing_follows_and_forget_forgets():
    mic = _Mic()
    sf = _filter(mic, talker=False)
    mic.talker = 1
    _blocks(sf)
    sf.set(low=300, high=2700)
    mic.talker = 2
    _blocks(sf)
    sf.set(low=200, high=3100)
    mic.talker = 1
    _blocks(sf)
    st = sf.status()
    assert st["set_low_hz"] == 200 and st["talker"] == {"enabled": False, "snap": "fast",
                                                        "id": None, "remembered": []}
    sf.set(talker=True)
    _blocks(sf)                                   # #1 takes the live filter from here
    mic.talker = 2
    _blocks(sf)
    assert sf.status()["talker"]["remembered"] == [1]
    sf.talker_forget(keep_ids={2})
    assert sf.status()["talker"]["remembered"] == []
    sf.talker_forget()
    assert sf.status()["talker"]["id"] is None


def test_the_summary_for_the_memory_table():
    mic = _Mic()
    sf = _filter(mic)
    assert sf.talker_filter_summary(1) is None
    mic.talker = 1
    _blocks(sf)
    sf.set(low=300, high=2700, shape="sharp", threshold_db=8)
    live = sf.talker_filter_summary(1)             # the live one, without a store
    assert live == {"low_hz": 300, "high_hz": 2700, "shape": "sharp", "auto": False,
                    "auto_eq": False, "contour": False, "threshold_db": 8.0, "live": True}
    mic.talker = 2
    _blocks(sf)
    assert sf.talker_filter_summary(1)["live"] is False
    with pytest.raises(ValueError):
        sf.set(talker_snap="jumpy")


def test_the_query_keys():
    assert _filter_kwargs({"talker": ["on"], "talker_snap": ["Smooth"]}) == \
        {"talker": True, "talker_snap": "smooth"}
