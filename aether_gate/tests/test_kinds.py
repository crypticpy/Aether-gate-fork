#
# Aether-gate — telling a conversation from a keyed tone, no hardware.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Five synthetic signals into the same Finder the band feeds, one per kind.

An SSB-shaped patch modulated at syllable rate, a tone keyed on and off at
about twelve words a minute, three constant-envelope tones eight hundred
hertz apart standing in for RTTY or MFSK, a bare carrier, and the two faces
of noise: an empty stretch of band, and a spark that fills a slice of it one
frame in twenty-four. The finder must name each of them, and say how sure
it is.

Run:  python -m pytest aether_gate/tests/test_kinds.py
"""
import numpy as np

from aether_gate.core import kinds
from aether_gate.core.finder import Finder, WINDOW_STEP_POINTS

NBINS = 2048
RATE = 125_000.0
FRAME_S = 4096 / RATE                 # ~30 frames a second, as the reader runs
CENTER = 3_800_000.0
FRAMES = 300                          # ~10 s: the fast ring holds 8.5 s of it

TONE_AMP = 30.0                       # ~13 dB of window SNR in one map point
DATA_AMP = 17.0                       # the same power shared over three tones
DATA_SPACING_HZ = 400.0
CW_HZ = 5.0                           # 0.1 s on, 0.1 s off: about 12 wpm
SYLLABLE_HZ = 4.0
IMPULSE_EVERY = 24                    # one loud frame in twenty-four
IMPULSE_POWER = 60.0
IMPULSE_WIDTH_HZ = 20_000.0


def _bin(f, hz):
    return int(np.argmin(np.abs(f - hz)))


def _frames(rng, n, voice=None, cw=None, data=None, carrier=None, impulse=None,
            snr=20.0):
    """n frames of (2, NBINS) spectra, natural FFT order, offsets in Hz."""
    f = np.fft.fftfreq(NBINS, 1.0 / RATE)
    t = 0.0
    for i in range(n):
        X = [(rng.normal(size=NBINS) + 1j * rng.normal(size=NBINS)) / np.sqrt(2)
             for _ in range(2)]
        if voice is not None:
            sel = (f >= voice[0]) & (f < voice[1])
            env = 1.0 if (t * SYLLABLE_HZ) % 1.0 < 0.5 else 0.05
            s = ((rng.normal(size=int(sel.sum())) + 1j * rng.normal(size=int(sel.sum())))
                 * np.sqrt(snr * env / 2))
            for x in X:
                x[sel] += s
        if cw is not None and (t * CW_HZ) % 1.0 < 0.5:
            for x in X:
                x[_bin(f, cw)] += TONE_AMP * np.exp(1j * 2 * np.pi * 0.37 * t)
        if data is not None:
            for j in range(3):
                k = _bin(f, data + j * DATA_SPACING_HZ)
                for x in X:
                    x[k] += DATA_AMP * np.exp(1j * 2 * np.pi * (0.2 + 0.1 * j) * t)
        if carrier is not None:
            for x in X:
                x[_bin(f, carrier)] += TONE_AMP * np.exp(1j * 2 * np.pi * 0.37 * t)
        if impulse is not None and i % IMPULSE_EVERY == 0:
            sel = np.abs(f - impulse) < IMPULSE_WIDTH_HZ / 2
            for x in X:
                x[sel] += ((rng.normal(size=int(sel.sum()))
                            + 1j * rng.normal(size=int(sel.sum())))
                           * np.sqrt(IMPULSE_POWER / 2))
        yield np.stack(X)
        t += FRAME_S


def _run(rng, n=FRAMES, **kw):
    fd = Finder(NBINS, RATE)
    for X in _frames(rng, n, **kw):
        fd.update(X, FRAME_S)
    return fd


def _verdict(fd, hz_rel, radius=3):
    """What the finder calls the window sitting on `hz_rel`, and how sure.

    The window is chosen the way the finder's own ranking would choose it --
    the strongest one within a few steps of the frequency -- so an off-by-one
    in this test's arithmetic cannot silently grade a neighbouring window.
    """
    point = (hz_rel + RATE / 2) / fd.step_hz - 0.5
    mid = int(round((point - fd.win / 2.0) / WINDOW_STEP_POINTS))
    lo = max(0, mid - radius)
    hi = min(fd.nwin, mid + radius + 1)
    snr = fd._last["snr_db"][lo:hi]
    w = lo + int(np.argmax(snr))
    code, conf = fd.window_kinds()
    return kinds.name(code[w]), float(conf[w])


def test_a_syllabic_patch_is_voice():
    fd = _run(np.random.default_rng(21), voice=(20_000.0, 22_600.0))
    kind, conf = _verdict(fd, 21_300.0)
    assert kind == "voice", (kind, conf)
    assert conf >= 0.5


def test_a_keyed_tone_is_cw():
    fd = _run(np.random.default_rng(22), cw=-15_000.0)
    kind, conf = _verdict(fd, -15_000.0)
    assert kind == "cw", (kind, conf)
    assert conf >= 0.5


def test_constant_envelope_tones_are_data():
    fd = _run(np.random.default_rng(23), data=8_000.0)
    kind, conf = _verdict(fd, 8_400.0)
    assert kind == "data", (kind, conf)
    assert conf >= 0.4


def test_a_bare_carrier_is_a_carrier():
    fd = _run(np.random.default_rng(24), carrier=-30_000.0)
    kind, conf = _verdict(fd, -30_000.0)
    assert kind == "carrier", (kind, conf)
    assert conf >= 0.5


def test_empty_band_is_noise():
    fd = _run(np.random.default_rng(25))
    kind, conf = _verdict(fd, 12_000.0)
    assert kind == "noise", (kind, conf)
    assert conf >= 0.8


def test_a_spark_one_frame_in_twenty_four_is_noise_not_voice():
    fd = _run(np.random.default_rng(26), impulse=-40_000.0)
    kind, conf = _verdict(fd, -40_000.0)
    assert kind == "noise", (kind, conf)
    assert conf >= 0.4


def test_every_window_gets_a_verdict_inside_the_contract():
    """Whatever is on the band, the arrays the payload is built from are the
    right length, name one of the five kinds, and never claim more than
    certainty or less than none."""
    fd = _run(np.random.default_rng(27), voice=(-6_000.0, -3_400.0),
              carrier=25_000.0, cw=-15_000.0)
    code, conf = fd.window_kinds()
    assert code.shape == conf.shape == (fd.nwin,)
    assert set(kinds.name(c) for c in code) <= set(kinds.KINDS)
    assert np.all(conf >= 0.0) and np.all(conf <= 1.0)


def test_the_verdict_survives_a_map_shorter_than_the_windows_expect():
    """A short spectrum must hold the last window rather than raise: the
    finder's window count and the map's length are computed apart, and one
    day a resolution change will make them disagree by a point."""
    n, nwin, win = 40, 6, 11
    rng = np.random.default_rng(28)
    W = rng.gamma(4.0, 0.25, size=(n, nwin)) * win
    floor = rng.gamma(4.0, 0.25, size=n)
    mean_points = np.ones(win + 2 * WINDOW_STEP_POINTS)      # two windows short
    code, conf = kinds.classify(W, floor, mean_points, np.zeros(nwin), np.full(nwin, 0.1),
                                np.full(nwin, 0.2), np.full(nwin, 0.5),
                                win, WINDOW_STEP_POINTS, 244.0)
    assert code.shape == conf.shape == (nwin,)
    assert all(kinds.name(c) == "noise" for c in code)       # nothing over the floor
