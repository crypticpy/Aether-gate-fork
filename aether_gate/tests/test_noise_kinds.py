#
# Aether-gate — noise_profile.kinds[].since (adapters/noise_kinds.py).
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Each kinds[] row's `since` is the epoch the finding was first seen, read
off the NoiseProfile the rest of the row was built from
(NoiseProfile.kind_since, core/noiseprofile.py) -- keyed so mains, impulse,
each periodic line and each ANF tone hold their own clock, none of them
moved just because this poll asked for it again. A caller with no profile
to persist against (None) gets `since: None` on every row instead.

Run:  python -m pytest aether_gate/tests/test_noise_kinds.py
"""
from aether_gate.adapters import noise_kinds as nk
from aether_gate.core.noiseprofile import NoiseProfile

RATE = 125_000.0


def _st(**over):
    d = {"mains_hz": None, "hum_db": 0.0, "harmonics": 0, "impulses_per_s": 0.0,
         "impulse_db": None, "periodic": [], "window_s": 2.0, "impulse_window_s": 4.0}
    d.update(over)
    return d


def _kinds(st, profile=None, filt_status=None):
    rows = nk.noise_kinds(st, None, "off", False, 0.0, 0.6, filt_status, profile)
    return {r["kind"]: r for r in rows}


def test_since_is_none_with_no_profile_to_persist_against():
    k = _kinds(_st(mains_hz=60.0, hum_db=18.0, harmonics=5))
    assert k["mains"]["since"] is None


def test_since_is_a_fresh_float_epoch_the_first_time_a_finding_is_seen():
    prof = NoiseProfile(RATE)
    k = _kinds(_st(mains_hz=60.0, hum_db=18.0, harmonics=5), profile=prof)
    since = k["mains"]["since"]
    assert isinstance(since, float) and since > 0.0
    assert prof.kind_since("mains", since + 999.0) == since   # the profile agrees


def test_since_holds_across_re_detections_on_the_same_profile():
    prof = NoiseProfile(RATE)
    st = _st(mains_hz=60.0, hum_db=18.0, harmonics=5)
    first = _kinds(st, profile=prof)["mains"]["since"]
    second = _kinds(st, profile=prof)["mains"]["since"]      # a later poll, same finding
    assert second == first


def test_each_kind_and_each_periodic_line_uses_its_own_persisted_key():
    prof = NoiseProfile(RATE)
    prof.kind_since("mains", 111.0)
    prof.kind_since("impulse", 222.0)
    prof.kind_since("periodic:182", 333.0)
    prof.kind_since("periodic:900", 444.0)
    st = _st(mains_hz=60.0, hum_db=18.0, harmonics=5,
             impulse_db=20.0, impulses_per_s=5.0,
             periodic=[{"hz": 182.0, "db": 14.0}, {"hz": 900.0, "db": 9.0}])
    rows = nk.noise_kinds(st, None, "off", False, 0.0, 0.6, None, prof)
    periodic = [r for r in rows if r["kind"] == "periodic"]
    by_id = {r["kind"]: r for r in rows if r["kind"] != "periodic"}
    assert by_id["mains"]["since"] == 111.0
    assert by_id["impulse"]["since"] == 222.0
    assert {r["since"] for r in periodic} == {333.0, 444.0}


def test_a_tone_and_the_floor_fallback_each_key_their_own_since():
    prof = NoiseProfile(RATE)
    prof.kind_since("tone:1240", 555.0)
    filt_status = {"notches": [],
                   "anf": {"enabled": True, "found_hz": [1240.0], "depth_db": [-31.0]}}
    k = _kinds(_st(), profile=prof, filt_status=filt_status)
    assert k["tone"]["since"] == 555.0
    prof2 = NoiseProfile(RATE)
    prof2.kind_since("floor", 777.0)
    assert _kinds(_st(), profile=prof2)["floor"]["since"] == 777.0
