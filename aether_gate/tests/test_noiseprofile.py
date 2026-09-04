#
# Aether-gate — the noise profile, no hardware.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Synthetic pairs into NoiseProfile: a 60 Hz grid's rectifier hum on the
noise (120 Hz comb), a 50 Hz one (100 Hz), an electric fence's impulses,
and plain noise, which must report nothing.

Run:  python -m pytest aether_gate/tests/test_noiseprofile.py
"""
import numpy as np

from aether_gate.core.noiseprofile import NoiseProfile

RATE = 125_000.0
BLOCK = 4096


def _feed(prof, rng, seconds, mains_hz=None, depth=0.5, impulses_per_s=0.0, tone_hz=None):
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    env = np.ones(n)
    if mains_hz is not None:
        f2 = 2.0 * mains_hz
        # a rectifier's envelope: |sin| has even harmonics of the mains
        env += depth * (np.cos(2 * np.pi * f2 * t) + 0.5 * np.cos(2 * np.pi * 2 * f2 * t)
                        + 0.25 * np.cos(2 * np.pi * 3 * f2 * t))
    if tone_hz is not None:
        env += 0.3 * np.cos(2 * np.pi * tone_hz * t)
    env = np.maximum(env, 0.05)
    a = (rng.normal(size=n) + 1j * rng.normal(size=n)) * np.sqrt(env / 2)
    b = (rng.normal(size=n) + 1j * rng.normal(size=n)) * np.sqrt(env / 2)
    if impulses_per_s:
        # a fence or an arc is not a clock: random spacing, so the envelope's
        # spectrum has no comb of its own
        k = int(impulses_per_s * seconds)
        idx = np.sort(rng.integers(0, n - 40, size=k))
        for i in idx:
            a[i:i + 30] += 30.0 * np.exp(1j * rng.uniform(0, 2 * np.pi))
            b[i:i + 30] += 30.0 * np.exp(1j * rng.uniform(0, 2 * np.pi))
    for i in range(0, n - BLOCK + 1, BLOCK):
        prof.update(a[i:i + BLOCK], b[i:i + BLOCK])
    return prof.status()


def test_plain_noise_reports_nothing():
    rng = np.random.default_rng(1)
    s = _feed(NoiseProfile(RATE), rng, 4.0)
    assert s["mains_hz"] is None and s["harmonics"] == 0 and s["hum_db"] == 0.0
    assert s["impulses_per_s"] < 1.0 and s["periodic"] == []
    assert s["seconds"] >= 1.9


def test_a_60_hz_rectifier_hum_is_a_120_hz_comb():
    rng = np.random.default_rng(2)
    s = _feed(NoiseProfile(RATE), rng, 4.0, mains_hz=60.0)
    assert s["mains_hz"] == 60.0, s
    assert s["harmonics"] >= 3 and s["hum_db"] >= 15.0, s
    assert not any(abs(p["hz"] - 100.0) < 3 for p in s["periodic"]), s


def test_a_50_hz_grid_is_told_from_a_60_hz_one():
    rng = np.random.default_rng(3)
    s = _feed(NoiseProfile(RATE), rng, 4.0, mains_hz=50.0)
    assert s["mains_hz"] == 50.0 and s["harmonics"] >= 3, s


def test_impulses_are_counted_and_sized():
    rng = np.random.default_rng(4)
    s = _feed(NoiseProfile(RATE), rng, 4.0, impulses_per_s=20.0)
    assert 12.0 <= s["impulses_per_s"] <= 28.0, s
    assert s["impulse_db"] is not None and s["impulse_db"] >= 20.0, s
    assert s["mains_hz"] is None, s


def test_a_non_mains_line_is_listed_as_periodic():
    rng = np.random.default_rng(5)
    s = _feed(NoiseProfile(RATE), rng, 4.0, tone_hz=333.0)
    assert s["mains_hz"] is None, s
    assert s["periodic"] and abs(s["periodic"][0]["hz"] - 333.0) <= 1.5, s


def test_kind_since_holds_the_first_epoch_across_re_detections():
    prof = NoiseProfile(RATE)
    t0 = prof.kind_since("mains", 1000.0)
    assert isinstance(t0, float) and t0 == 1000.0
    # the same key, later "now": the first-seen epoch does not move
    assert prof.kind_since("mains", 1005.0) == 1000.0
    # a different key gets its own clock, independent of "mains"
    assert prof.kind_since("impulse", 1002.0) == 1002.0
    assert prof.kind_since("mains", 1010.0) == 1000.0


def test_kind_since_resets_only_when_the_profile_itself_is_replaced():
    a = NoiseProfile(RATE)
    a.kind_since("mains", 500.0)
    b = NoiseProfile(RATE)                              # a fresh profile: a retune
    assert b.kind_since("mains", 900.0) == 900.0         # not 500.0 -- a's clock, not shared
    assert a.kind_since("mains", 999.0) == 500.0         # a's own clock is untouched
