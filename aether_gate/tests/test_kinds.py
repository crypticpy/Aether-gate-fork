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
import math
import os

import numpy as np
import pytest

from aether_gate.core import kinds
from aether_gate.core.finder import (FAST_FRAMES, Finder, KIND_HOLD_ROWS, SLOT_S,
                                      WINDOW_STEP_POINTS)

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


# =========================================================================
# The grid the 2026-09-03 defect was found and fixed against
# =========================================================================
# Honest CW and honest SSB voice at every span and every workable SNR.
#
# The five cases above build their voice out of a FLAT patch of band noise,
# which is the one shape the old occupied-width measure got right: it counted
# the points within 6 dB of the window's strongest one, and a flat patch is all
# of them. A real SSB signal is not flat -- its long-term spectrum falls 15-20
# dB from the strongest formant region to the top of the passband -- so that
# measure read real phone as 500-1000 Hz wide, narrower than a keyed tone is
# supposed to be, and on 2026-09-03 every conversation on 80 m and 40 m came
# back from the live gate as "cw" at a high score. See kinds.py BW_ENERGY_FRAC.
#
# So the signals below are built the way the band builds them: a keyed tone
# sending random Morse at 25 wpm with raised-cosine edges, and an SSB patch with
# a -6 dB/octave tilt on it and a jittered syllabic gate. They are fed at all six
# RSPduo spans, because a defect that only shows at one resolution is a defect
# nobody finds.
GRID_RATES_HZ = (62_500.0, 125_000.0, 250_000.0, 500_000.0, 1_020_000.0, 2_040_000.0)
GRID_SNR_DB = (3.0, 6.0, 10.0, 20.0)
CHUNK = 4096                     # adapters/soapy.py's raw block read length
GRID_S = 0.001                   # envelope grid, finer than any keying edge
VOICE_LO_HZ, VOICE_HI_HZ = 300.0, 2700.0
TILT_DB_PER_OCT = -6.0           # an SSB signal's long-term spectrum above the knee
TILT_KNEE_HZ = 500.0
GATE_FLOOR = 0.01                # 20 dB down between syllables, between elements
SYLLABLE_HZ = 4.0
SYLLABLE_DUTY = 0.40             # a talker holds the frequency 40% of the time
WPM = 25.0
RISE_S = 0.005                   # keying edge: ~200 Hz of sidebands, no clicks
MORSE = {"a": ".-", "b": "-...", "c": "-.-.", "d": "-..", "e": ".", "f": "..-.",
         "g": "--.", "h": "....", "i": "..", "j": ".---", "k": "-.-", "l": ".-..",
         "m": "--", "n": "-.", "o": "---", "p": ".--.", "q": "--.-", "r": ".-.",
         "s": "...", "t": "-", "u": "..-", "v": "...-", "w": ".--", "x": "-..-",
         "y": "-.--", "z": "--..", "0": "-----", "1": ".----", "2": "..---",
         "3": "...--", "4": "....-", "5": ".....", "6": "-....", "7": "--...",
         "8": "---..", "9": "----."}
ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def _smooth(g, seconds):
    k = max(3, int(round(seconds / GRID_S)) | 1)
    w = np.hanning(k)
    return np.convolve(g, w / w.sum(), mode="same")


def _syllabic_envelope(rng, total_s):
    """Speech's own envelope on a 1 ms grid: on SYLLABLE_DUTY of the time at
    SYLLABLE_HZ, the syllable length jittered so it is a talker and not a
    metronome, 20 dB down in between."""
    g = np.full(int(math.ceil(total_s / GRID_S)) + 1, GATE_FLOOR)
    t = 0.0
    while t < total_s:
        period = float(rng.uniform(0.7, 1.3)) / SYLLABLE_HZ
        g[int(t / GRID_S):int(min(total_s, t + period * SYLLABLE_DUTY) / GRID_S)] = 1.0
        t += period
    return _smooth(g, 0.030)               # a syllable does not start in a microsecond


def _keyed_envelope(rng, total_s):
    """A keyed envelope on the same grid: random Morse at WPM (PARIS timing),
    proper element/character/word spacing, RISE_S edges."""
    g = np.full(int(math.ceil(total_s / GRID_S)) + 1, GATE_FLOOR)
    dot = 1.2 / WPM
    t = 0.0
    while t < total_s:
        for el in MORSE[ALPHABET[int(rng.integers(len(ALPHABET)))]]:
            on = (3 if el == "-" else 1) * dot
            g[int(t / GRID_S):int(min(total_s, t + on) / GRID_S)] = 1.0
            t += on + dot                                  # inter-element space
        t += 2 * dot                                       # inter-character space
        if rng.random() < 0.2:
            t += 4 * dot                                   # word space
    return _smooth(g, RISE_S)


def _tilt(f_audio):
    """Relative POWER of an SSB signal's long-term spectrum at f_audio."""
    octaves = np.log2(np.maximum(np.maximum(f_audio, 1.0) / TILT_KNEE_HZ, 1.0))
    return 10.0 ** (TILT_DB_PER_OCT * octaves / 10.0)


def _grid_scene(rng, rate, n_frames, kind, offset_hz, tilt=True):
    """Yield (Xn, Xs) per frame: the (2, NBINS) noise spectrum pair and the
    unit-power signal spectrum pair, Hann-windowed and transformed exactly as
    adapters/diversity_state._map_update does it. The caller adds amp * Xs to
    Xn to set the SNR, which the FFT's linearity makes identical to having
    summed the two in the time domain -- and lets one pass over the scene feed
    a Finder per SNR."""
    total_s = n_frames * CHUNK / rate + NBINS / rate
    grid_t = np.arange(int(math.ceil(total_s / GRID_S)) + 1) * GRID_S
    env = (_keyed_envelope(rng, total_s) if kind == "cw"
           else _syllabic_envelope(rng, total_s))
    norm = math.sqrt(float(np.mean(env)))       # unit mean power over the run
    w = np.hanning(NBINS)
    f = np.fft.fftfreq(NBINS, 1.0 / rate)
    aud = offset_hz - f                         # LSB: the audio sits below the carrier
    sel = (aud >= VOICE_LO_HZ) & (aud <= VOICE_HI_HZ)
    mask = np.zeros(NBINS)
    mask[sel] = _tilt(aud[sel]) if tilt else 1.0
    mask *= NBINS / max(float(mask.sum()), 1e-30)
    root = np.sqrt(mask)
    for i in range(n_frames):
        t = (i * CHUNK + np.arange(NBINS)) / rate
        e = np.interp(t, grid_t, env)
        Xn = np.fft.fft(np.stack([
            (rng.normal(size=NBINS) + 1j * rng.normal(size=NBINS)) / math.sqrt(2) * w
            for _ in range(2)]), axis=1)
        if kind == "cw":
            s = np.sqrt(e) * np.exp(2j * np.pi * offset_hz * t) / norm
        else:
            u = (rng.normal(size=NBINS) + 1j * rng.normal(size=NBINS)) / math.sqrt(2)
            s = np.fft.ifft(np.fft.fft(u) * root) * np.sqrt(e) / norm
        yield Xn, np.fft.fft(np.stack([s, s]) * w, axis=1)


_GRID_CACHE = {}


def _grid(rate, kind, tilt=True, seed=11):
    """(kind name, confidence, measured snr_db) per GRID_SNR_DB, from ONE pass
    over the scene feeding one Finder per SNR."""
    key = (rate, kind, tilt, seed)
    if key in _GRID_CACHE:
        return _GRID_CACHE[key]
    frame_s = CHUNK / rate
    # enough frames to fill the ring past the finder's own gate at every span
    per_slot = max(1, int(math.ceil(SLOT_S / frame_s)))
    n_frames = (FAST_FRAMES // 2 + 8) * per_slot
    offset = 0.20 * rate / 2                    # inside the span, clear of DC and the edge
    fds = [Finder(NBINS, rate) for _ in GRID_SNR_DB]
    win_hz = fds[0].win * fds[0].step_hz
    amps = [math.sqrt(10.0 ** (s / 10.0) * win_hz / rate) for s in GRID_SNR_DB]
    rng = np.random.default_rng(seed)
    for Xn, Xs in _grid_scene(rng, rate, n_frames, kind, offset, tilt=tilt):
        for fd, a in zip(fds, amps):
            fd.update(Xn + a * Xs, frame_s)
    out = []
    for fd in fds:
        assert fd._last is not None, f"the finder never scored at rate={rate}"
        point = (offset + rate / 2) / fd.step_hz - 0.5
        mid = int(round((point - fd.win / 2.0) / WINDOW_STEP_POINTS))
        lo, hi = max(0, mid - 4), min(fd.nwin, mid + 5)
        w = lo + int(np.argmax(fd._last["snr_db"][lo:hi]))
        code, conf = fd.window_kinds()
        out.append((kinds.name(code[w]), float(conf[w]), float(fd._last["snr_db"][w])))
    _GRID_CACHE[key] = out
    return out


@pytest.mark.parametrize("snr_db", GRID_SNR_DB, ids=lambda s: f"{s:.0f}dB")
@pytest.mark.parametrize("rate", GRID_RATES_HZ, ids=lambda r: f"{r / 1e3:.0f}k")
def test_a_real_ssb_signal_is_voice_at_every_span_and_snr(rate, snr_db):
    """The defect this grid exists for: at 5-8 dB on 80 m every talker came
    back "cw". A sloped SSB spectrum must be voice at every span."""
    i = GRID_SNR_DB.index(snr_db)
    kind, conf, meas = _grid(rate, "voice")[i]
    assert kind == "voice", (rate, snr_db, kind, conf, meas)
    assert meas == pytest.approx(snr_db, abs=2.5), (rate, snr_db, meas)


@pytest.mark.parametrize("snr_db", GRID_SNR_DB, ids=lambda s: f"{s:.0f}dB")
@pytest.mark.parametrize("rate", GRID_RATES_HZ, ids=lambda r: f"{r / 1e3:.0f}k")
def test_a_keyed_tone_is_cw_and_never_voice_at_every_span_and_snr(rate, snr_db):
    """And the other way round, which the fix must not buy with the first:
    25 wpm Morse is never somebody talking."""
    i = GRID_SNR_DB.index(snr_db)
    kind, conf, meas = _grid(rate, "cw")[i]
    assert kind == "cw", (rate, snr_db, kind, conf, meas)
    assert meas == pytest.approx(snr_db, abs=2.5), (rate, snr_db, meas)


@pytest.mark.parametrize("rate", GRID_RATES_HZ, ids=lambda r: f"{r / 1e3:.0f}k")
def test_a_flat_ssb_patch_is_still_voice_at_every_span(rate):
    """The untilted patch the older cases in this file use, at every span:
    the width measure had to change to see a real signal, and it must not
    have stopped seeing the ideal one."""
    for snr_db, (kind, conf, _meas) in zip(GRID_SNR_DB, _grid(rate, "voice", tilt=False)):
        assert kind == "voice", (rate, snr_db, kind, conf)


# =========================================================================
# ...and the same question put to the band itself
# =========================================================================
CAPTURE_DIR = os.path.expanduser("~/aether-gate-captures")
# 6 s of 80 m phone (3.828-3.954 MHz) recorded 2026-09-02, four talkers in it
# and no CW: the recording the "everything is cw" report was reproduced on.
PHONE_CAPTURE = "20260902-231538_3891250Hz_125000sps.npz"


def _replay(path):
    """A capture from adapters/diversity_state's /diversity/capture through a
    Finder, framed exactly as _map_update frames the live reader."""
    d = np.load(path)
    a, b, lag = d["a"], d["b"], int(d["lag_samples"])
    if lag > 0:
        a, b = a[lag:], b[:len(b) - lag]
    elif lag < 0:
        a, b = a[:len(a) + lag], b[-lag:]
    rate = float(d["rate_hz"])
    fd = Finder(NBINS, rate)
    w = np.hanning(NBINS)
    for i in range(0, min(len(a), len(b)) - CHUNK + 1, CHUNK):
        fd.update(np.fft.fft(np.stack([a[i:i + NBINS], b[i:i + NBINS]]) * w, axis=1),
                  CHUNK / rate)
    return fd, float(d["center_hz"])


@pytest.mark.skipif(not os.path.exists(os.path.join(CAPTURE_DIR, PHONE_CAPTURE)),
                    reason=f"no {PHONE_CAPTURE} under {CAPTURE_DIR}")
def test_a_recorded_stretch_of_80m_phone_is_called_voice():
    """Off-air, not synthesised. Every candidate the finder raises on this
    recording is a talker in the 80 m phone band; before the width measure
    changed it called all four of them "cw" at 0.6-1.0 confidence."""
    fd, center = _replay(os.path.join(CAPTURE_DIR, PHONE_CAPTURE))
    out = fd.candidates(center)
    assert out["available"], out
    cands = out["candidates"]
    # the finder lists what is THERE now, not only what it can name (see
    # finder_report), so the recording's four talkers come back among other
    # detections rather than as the whole list
    assert len(cands) >= 5, cands
    assert all(3_700_000 <= c["hz"] <= 4_000_000 for c in cands), cands
    assert len([c for c in cands if c["kind"] == "voice"]) >= 4, cands
    # ...and the defect itself: a phone-wide window is never a keyed tone
    assert not [c for c in cands
                if c["kind"] == "cw" and c["occupied_hz"] > kinds.NARROW_HZ[1]], cands


# =========================================================================
# ...and again the next night, when the answer would not sit still
# =========================================================================
# 25 s of the same band recorded 2026-09-03 at 04:34 local, spanning the two
# reads of /diversity/finder thirty seconds apart that the operator reported:
# the first came back "cw" on four of eight candidates -- two of them at
# confidence 1.0 -- and the second came back "voice" on the same four. Nobody
# sends Morse on 80 m phone at half past four in the morning.
#
# Replayed through the finder as adapters/diversity_state._map_update frames it,
# that recording reproduces the flap exactly: before the fix its 21 scored rows
# put "cw" on 22 of 119 candidate rows, up to 1.00 confidence, and the same
# window read voice-cw-voice-cw from one second to the next.
FLAP_CAPTURE = "20260903-043417_3937250Hz_125000sps.npz"
# The one thing on that recording that is genuinely not a conversation: two
# carriers 977 Hz apart at 3.8815 and 3.8824 MHz, each about 490 Hz wide with
# clear floor between them, switching on and off 4 dB over the floor. A narrow
# keyed pair is what it is, and the gate is entitled to say so.
NOT_A_TALKER_HZ = 3_882_000.0
NOT_A_TALKER_MARGIN_HZ = 1_500.0
FLAP_PATH = os.path.join(CAPTURE_DIR, FLAP_CAPTURE)
needs_flap_capture = pytest.mark.skipif(
    not os.path.exists(FLAP_PATH), reason=f"no {FLAP_CAPTURE} under {CAPTURE_DIR}")


def _replay_rows(path, finder=Finder):
    """A capture through a Finder, with the candidate list as it stood after
    every scored row: what a read of /diversity/finder a second apart would
    have said each time, which is the thing that was not sitting still."""
    d = np.load(path)
    a, b, lag = d["a"], d["b"], int(d["lag_samples"])
    if lag > 0:
        a, b = a[lag:], b[:len(b) - lag]
    elif lag < 0:
        a, b = a[:len(a) + lag], b[-lag:]
    rate, center = float(d["rate_hz"]), float(d["center_hz"])
    fd = finder(NBINS, rate)
    w = np.hanning(NBINS)
    rows = []
    for i in range(0, min(len(a), len(b)) - CHUNK + 1, CHUNK):
        before = fd.slow_n
        fd.update(np.fft.fft(np.stack([a[i:i + NBINS], b[i:i + NBINS]]) * w, axis=1),
                  CHUNK / rate)
        if fd.slow_n != before:
            rows.append((fd.elapsed, fd.candidates(center)["candidates"]))
    return fd, center, rows


@needs_flap_capture
def test_no_talker_on_a_recorded_80m_evening_is_ever_called_cw():
    """Every candidate on the recording, at every one of its scored rows --
    not just at the end, because the operator reads the gate whenever they
    read it."""
    _fd, _center, rows = _replay_rows(FLAP_PATH)
    assert len(rows) >= 15, len(rows)
    for t, cands in rows:
        assert cands, t
        for c in cands:
            if abs(c["hz"] - NOT_A_TALKER_HZ) <= NOT_A_TALKER_MARGIN_HZ:
                continue
            if c["occupied_hz"] <= kinds.NARROW_HZ[1]:
                continue          # a narrow column may be a keyed tone, and is
            assert c["kind"] != "cw", (round(t, 1), c)


@needs_flap_capture
def test_the_verdict_on_a_recorded_80m_evening_stops_flapping():
    """No candidate may change its mind more than twice in twenty-five
    seconds, and none may be confidently anything it was not a moment ago:
    "cw 1.0" for thirty seconds on a conversation is the complaint."""
    _fd, _center, rows = _replay_rows(FLAP_PATH)
    seen, changes = {}, {}
    for _t, cands in rows:
        for c in cands:
            was = seen.get(c["hz"])
            if was is not None and was != c["kind"]:
                changes[c["hz"]] = changes.get(c["hz"], 0) + 1
            seen[c["hz"]] = c["kind"]
    assert changes == {} or max(changes.values()) <= 2, changes
    talkers = [c for _t, cands in rows for c in cands
               if abs(c["hz"] - NOT_A_TALKER_HZ) > NOT_A_TALKER_MARGIN_HZ
               and c["occupied_hz"] > kinds.NARROW_HZ[1]]
    assert not [c for c in talkers if c["kind"] == "cw"], talkers


class _Unheld(Finder):
    """The same finder showing each row's verdict raw, as it did before."""

    def _hold(self, kind, kconf):
        return kind, kconf


@needs_flap_capture
def test_holding_the_verdict_halves_the_flapping_across_the_whole_span():
    """Not only on the candidates: every window of the map, counted over the
    stored rows. The hold is worth about a factor of two, which is the
    difference between a display that reads and one that strobes."""
    held, _c, _r = _replay_rows(FLAP_PATH)
    raw, _c, _r = _replay_rows(FLAP_PATH, finder=_Unheld)
    hk = held._slow_rows()[2]
    rk = raw._slow_rows()[2]
    assert hk.shape == rk.shape and hk.shape[0] >= 15
    held_changes = int(np.count_nonzero(np.diff(hk, axis=0)))
    raw_changes = int(np.count_nonzero(np.diff(rk, axis=0)))
    assert raw_changes > 100, raw_changes           # the flap is in the recording
    assert held_changes * 2 <= raw_changes, (held_changes, raw_changes)


def test_a_confident_verdict_gives_way_only_to_a_run_of_contradictions():
    """The hold, on its own terms: one confident row establishes a verdict,
    a single contradiction never takes it, and KIND_HOLD_ROWS of them do."""
    fd = Finder(NBINS, RATE)
    voice = np.full(fd.nwin, kinds.KINDS.index("voice"), dtype=np.int8)
    cw = np.full(fd.nwin, kinds.KINDS.index("cw"), dtype=np.int8)
    sure = np.ones(fd.nwin, dtype=np.float32)
    kind, conf = fd._hold(voice, sure)               # nothing yet to protect
    assert kinds.name(kind[0]) == "voice" and conf[0] == pytest.approx(1.0)
    kind, _ = fd._hold(cw, sure)                     # one contradiction: no
    assert kinds.name(kind[0]) == "voice"
    kind, _ = fd._hold(voice, sure)                  # ...and the run is broken
    assert kinds.name(kind[0]) == "voice"
    for _ in range(KIND_HOLD_ROWS):
        kind, conf = fd._hold(cw, sure)
    assert kinds.name(kind[0]) == "cw" and conf[0] == pytest.approx(1.0)


def test_a_confident_verdict_decays_before_it_changes():
    """The other half of the same rule. A contradicted verdict loses its
    confidence first -- so the display goes "voice 1.0", "voice 0.4", "cw"
    rather than "voice 1.0", "cw 1.0" -- and a verdict held at 1.0 outlasts a
    hesitant challenger that a verdict held at 0.2 gives way to."""
    fd = Finder(NBINS, RATE)
    voice = np.full(fd.nwin, kinds.KINDS.index("voice"), dtype=np.int8)
    cw = np.full(fd.nwin, kinds.KINDS.index("cw"), dtype=np.int8)
    hesitant = np.full(fd.nwin, 0.2, dtype=np.float32)
    fd._hold(voice, np.ones(fd.nwin, dtype=np.float32))
    kind, conf = fd._hold(cw, np.full(fd.nwin, 0.6, dtype=np.float32))
    assert kinds.name(kind[0]) == "voice" and conf[0] == pytest.approx(0.4)

    # a verdict with nothing left to spend goes on the run length alone
    weak = Finder(NBINS, RATE)
    weak._hold(voice, hesitant)
    for _ in range(KIND_HOLD_ROWS):
        kind, _ = weak._hold(cw, hesitant)
    assert kinds.name(kind[0]) == "cw"
    # ...and the same three rows of the same hesitant answer do not move one
    # that was sure of itself
    strong = Finder(NBINS, RATE)
    strong._hold(voice, np.ones(fd.nwin, dtype=np.float32))
    for _ in range(KIND_HOLD_ROWS):
        kind, conf = strong._hold(cw, hesitant)
    assert kinds.name(kind[0]) == "voice" and conf[0] == pytest.approx(0.4)
