#
# Aether-gate — the auto contour (core/contour.py, SliceFilter.auto_contour).
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""A speech-shaped print earns no bell; a proximity boom earns a cut where
the boom is, a scooped presence a lift; the bell rides the print into the
filter's response, is the talker's own, and a hand on the CONTOUR knobs
takes over from it.

Run:  python -m pytest aether_gate/tests/test_auto_contour.py
"""
import numpy as np

from aether_gate.core.contour import (fit_contour, SPEECH_DB, PROFILE_HZ,
                                      AUTO_CONTOUR_MAX_DB, AUTO_CONTOUR_STRENGTH)
from aether_gate.core.filter import SliceFilter, response_at, SPEC_WARMUP_BLOCKS
from aether_gate.tests.test_talker_filter import _Mic, _filter, _blocks, RATE


def _profile(tilt_db_per_oct=0.0, bump=None):
    """A speech-shaped profile with a tilt and, optionally, a (lo, hi, dB)
    microphone bump laid on top; dB re its own peak like the print's."""
    p = SPEECH_DB + tilt_db_per_oct * np.log2(PROFILE_HZ / 500.0)
    if bump is not None:
        lo, hi, db = bump
        p = p + db * ((PROFILE_HZ >= lo) & (PROFILE_HZ < hi))
    return p - p.max()


def test_speech_shaped_prints_with_any_tilt_earn_no_bell():
    assert fit_contour(_profile()) is None
    assert fit_contour(_profile(-4.0)) is None            # a bassy station: auto EQ's job
    assert fit_contour(_profile(+3.0)) is None
    assert fit_contour(np.zeros(31)) is None               # not a profile


def test_a_boom_is_cut_where_it_is_and_a_scoop_is_lifted_but_never_all_of_it():
    hz, db, width = fit_contour(_profile(-2.0, bump=(400.0, 700.0, 6.0)))
    assert 400 <= hz <= 700 and 200 <= width <= 500
    assert -AUTO_CONTOUR_MAX_DB <= db < 0 and abs(db) <= AUTO_CONTOUR_STRENGTH * 6.0 + 0.5
    hz, db, width = fit_contour(_profile(0.0, bump=(1700.0, 2300.0, -7.0)))
    assert 1700 <= hz <= 2300 and 400 <= width <= 900 and 0 < db <= AUTO_CONTOUR_MAX_DB
    # a wall of a boom is capped, not matched
    hz, db, _w = fit_contour(_profile(-2.0, bump=(400.0, 700.0, 12.0)))
    assert 400 <= hz <= 700 and db == -AUTO_CONTOUR_MAX_DB


def _print(bump):
    return {"low_hz": 300, "high_hz": 2600, "tilt_db": -6.0,
            "bands_db": [round(float(x), 1) for x in _profile(0.0, bump)]}


def test_the_bell_rides_the_print_into_the_response_and_is_the_talkers_own():
    mic = _Mic()
    mic.prints[1] = _print((400.0, 700.0, 6.0))            # a boomy microphone
    mic.prints[2] = None                                    # no print yet
    sf = _filter(mic)
    assert sf.status()["contour"] == {"enabled": False, "hz": None, "db": 0.0,
                                      "width_hz": None, "auto": True, "source": None}
    flat = response_at(sf.taps, RATE, 550.0)
    mic.talker = 1
    _blocks(sf, SPEC_WARMUP_BLOCKS + 8)
    c = sf.status()["contour"]
    assert c["auto"] and c["enabled"] and c["source"] == "print" and c["db"] < 0
    assert 400 <= c["hz"] <= 700
    assert response_at(sf.taps, RATE, 550.0) < flat - 1.5   # the boom is taken down
    # a talker with no print keys up: no bell, the block they arrive
    mic.talker = 2
    _blocks(sf, 1)
    assert not sf.status()["contour"]["enabled"]
    assert response_at(sf.taps, RATE, 550.0) > flat - 0.5
    # and the boomy one gets theirs back at once, before any spectrum settles
    mic.talker = 1
    _blocks(sf, 1)
    assert sf.status()["contour"]["enabled"] and response_at(sf.taps, RATE, 550.0) < flat - 1.5


def test_a_hand_on_the_contour_knobs_takes_over_and_off_clears_the_bell():
    mic = _Mic()
    mic.prints[1] = _print((400.0, 700.0, 6.0))
    mic.talker = 1
    sf = _filter(mic)
    _blocks(sf, SPEC_WARMUP_BLOCKS + 8)
    assert sf.status()["contour"]["auto"]
    sf.set(contour_db=3.0)
    c = sf.status()["contour"]
    assert c == {"enabled": False, "hz": 1200.0, "db": 3.0, "width_hz": 600.0,
                 "auto": False, "source": "manual"}
    sf.set(contour=True)
    assert sf.status()["contour"]["enabled"]
    sf.set(auto_contour=True)
    _blocks(sf, 4)
    assert sf.status()["contour"]["auto"] and sf.status()["contour"]["source"] == "print"
    sf.set(auto_contour=False)
    assert sf.status()["contour"]["auto"] is False and sf.status()["contour"]["db"] == 3.0
