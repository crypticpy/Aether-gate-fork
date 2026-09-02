#
# Aether-gate — the live spatial rows and the conversation finder, no hardware.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Synthetic frames straight into LiveSpatial and Finder: a voice-shaped
patch (syllabic modulation, both loops, fixed inter-loop phase), a steady
carrier, and band noise. The finder must rank the voice first, refuse the
carrier and the noise, put the dial just below the energy for USB, and
carry the pair's phase there; the spatial rows must show that phase only
where the source is.

Run:  python -m pytest aether_gate/tests/test_finder.py
"""
import numpy as np
import pytest

from aether_gate.core.finder import (Finder, LiveSpatial, VOICE_SCORE, FAST_FRAMES,
                                     SLOW_PERIOD_S)

NBINS = 2048
RATE = 125_000.0
FRAME_S = 4096 / RATE
CENTER = 14_100_000.0


def _frames(rng, n, voice=None, carrier=None, phase=1.2, ratio=0.8, snr=20.0):
    """Yield n frames of (2, NBINS) spectra in natural FFT order.

    voice: (lo_hz, hi_hz) relative to centre, energy modulated at 4 Hz with a
    hard on/off syllable pattern; carrier: hz relative to centre, steady.
    """
    f = np.fft.fftfreq(NBINS, 1.0 / RATE)
    t = 0.0
    for _ in range(n):
        Xa = (rng.normal(size=NBINS) + 1j * rng.normal(size=NBINS)) / np.sqrt(2)
        Xb = (rng.normal(size=NBINS) + 1j * rng.normal(size=NBINS)) / np.sqrt(2)
        if voice is not None:
            sel = (f >= voice[0]) & (f < voice[1])
            env = 1.0 if (t * 4.0) % 1.0 < 0.5 else 0.05          # syllables at 4 Hz
            s = (rng.normal(size=sel.sum()) + 1j * rng.normal(size=sel.sum())) * np.sqrt(snr * env / 2)
            Xa[sel] += s
            Xb[sel] += s * ratio * np.exp(1j * phase)
        if carrier is not None:
            k = int(np.argmin(np.abs(f - carrier)))
            s = np.sqrt(snr * 40) * np.exp(1j * 2 * np.pi * 0.37 * t)
            Xa[k] += s
            Xb[k] += s * ratio * np.exp(1j * phase)
        yield np.stack([Xa, Xb])
        t += FRAME_S


def _run(rng, n, **kw):
    live = LiveSpatial(NBINS, RATE)
    fd = Finder(NBINS, RATE)
    for X in _frames(rng, n, **kw):
        live.update(X, FRAME_S)
        fd.update(X, FRAME_S)
    return live, fd


def test_nothing_is_reported_before_there_is_data():
    fd = Finder(NBINS, RATE)
    live = LiveSpatial(NBINS, RATE)
    assert fd.candidates(CENTER, live) == {"available": False}
    assert live.rows(CENTER) is None


def test_voice_is_found_first_and_the_carrier_and_noise_are_not():
    rng = np.random.default_rng(7)
    live, fd = _run(rng, 400, voice=(20_000.0, 22_600.0), carrier=-30_000.0)
    out = fd.candidates(CENTER, live)
    assert out["available"] and len(out["activity"]) == out["points"] == 512
    cands = out["candidates"]
    assert cands, "the voice patch was not found"
    top = cands[0]
    assert top["score"] >= VOICE_SCORE and top["mode"] == "USB"
    # USB: the dial sits just below the energy, which starts 20 kHz up
    assert CENTER + 19_000 <= top["hz"] <= CENTER + 20_100, top
    assert top["syllabic"] >= 0.6 and top["depth"] >= 0.4 and top["snr_db"] >= 5.0
    assert top["active_s"] >= 3 * SLOW_PERIOD_S and top["last_s"] is not None
    # the carrier (steady) and the noise never make the list
    for c in cands[1:]:
        assert not (CENTER - 32_000 <= c["hz"] <= CENTER - 28_000), c
    assert all(c["score"] >= VOICE_SCORE for c in cands)
    # the pair's phase and level ratio are read at the candidate
    assert top["phase_deg"] == pytest.approx(np.degrees(-1.2), abs=8.0)
    assert top["ratio_db"] == pytest.approx(20 * np.log10(0.8), abs=1.5)
    assert top["coherence"] >= 0.6            # syllable gaps are noise-only time
    # equal-ish loops: the pair can earn close to the 0.8^2 MRC gain
    assert 1.5 <= top["gain_db"] <= 3.1, top


def test_activity_strip_lights_only_where_the_voice_is():
    rng = np.random.default_rng(8)
    live, fd = _run(rng, 400, voice=(20_000.0, 22_600.0))
    out = fd.candidates(CENTER, live)
    act = np.asarray(out["activity"])
    step = RATE / 512
    lo = int((RATE / 2 + 20_000) / step)
    hi = int((RATE / 2 + 22_600) / step)
    assert np.max(act[lo:hi]) >= 0.5
    assert np.max(np.concatenate([act[:lo - 20], act[hi + 20:]])) < 0.2


def test_spatial_rows_carry_the_phase_only_where_the_source_is():
    rng = np.random.default_rng(9)
    live, _ = _run(rng, FAST_FRAMES // 4, voice=(20_000.0, 22_600.0), phase=2.0, snr=40.0)
    rows = live.rows(CENTER)
    assert rows["points"] == 512 and len(rows["phase_deg"]) == 512
    assert rows["start_hz"] == pytest.approx(CENTER - RATE / 2)
    step = rows["step_hz"]
    coh = np.asarray(rows["coherence"])
    ph = np.asarray(rows["phase_deg"])
    lo = int((RATE / 2 + 20_000) / step) + 1
    hi = int((RATE / 2 + 22_600) / step) - 1
    assert np.min(coh[lo:hi]) >= 0.7
    assert np.all(np.abs(((ph[lo:hi] - np.degrees(-2.0)) + 180) % 360 - 180) <= 10)
    # plain noise is incoherent: the EMA has ~8 frames x 4 bins of averaging
    assert np.median(coh[:lo - 10]) < 0.15


def test_a_conversation_that_stopped_is_still_listed_with_its_age():
    rng = np.random.default_rng(10)
    live = LiveSpatial(NBINS, RATE)
    fd = Finder(NBINS, RATE)
    for X in _frames(rng, 300, voice=(-10_000.0, -7_400.0)):
        live.update(X, FRAME_S)
        fd.update(X, FRAME_S)
    for X in _frames(rng, 450):                     # ~15 s of silence
        live.update(X, FRAME_S)
        fd.update(X, FRAME_S)
    out = fd.candidates(CENTER, live)
    assert out["candidates"], "a 30 s recent window must keep it listed"
    top = out["candidates"][0]
    # ~15 s of silence, minus the ~4 s the 8.5 s scoring window keeps hearing it
    assert 8.0 <= top["last_s"] <= 14.0, top
    assert top["active_s"] >= 3 * SLOW_PERIOD_S
