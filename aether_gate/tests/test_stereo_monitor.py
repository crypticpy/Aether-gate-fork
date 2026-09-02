#
# Aether-gate — the diversity stereo monitor through the adapter's audio path.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""HEAR = stereo puts loop A in the left ear and loop B in the right, through
the real NCO / FIR / resampler / AGC path of the soapy adapter, and the
engine's packetiser interleaves the pairs. No hardware: the tuple blocks a
dual-tuner reader would queue are fed directly, with a stand-in for the
diversity state that answers combine_passband the way HEAR does.

Run:  python -m pytest aether_gate/tests/test_stereo_monitor.py
"""
import numpy as np

from aether_gate.adapters.soapy import SoapyAdapter, AUDIO_RATE
from aether_gate.core.engine import audio_frames

SAMP = 125_000.0
CHUNK = 480


class _Hear:
    """What _DiversityState.combine_passband does for HEAR, nothing else."""

    def __init__(self, hear):
        self.hear = hear
        self.active_slice = 0

    def observe(self, sid, xa, xb):
        return 0j, 0j

    def combine_passband(self, sid, pa, pb, m0, m1, rate_hz):
        if self.hear == "a":
            return pa
        if self.hear == "b":
            return pb
        if self.hear == "stereo":
            n = min(len(pa), len(pb))
            return np.stack([pa[:n], pb[:n]], axis=1)
        return (pa + pb) / np.sqrt(2.0)


def _adapter(hear):
    a = SoapyAdapter(driver="sdrplay", samp_rate=SAMP, center_hz=7_100_000.0)
    a._np = np
    a._init_demod()
    a._mode = "USB"
    a._div = _Hear(hear)
    return a


def _feed(a, seconds, hz_a, hz_b, block=4096):
    t0 = 0
    for _ in range(int(seconds * SAMP) // block):
        t = (t0 + np.arange(block)) / SAMP
        a._audio_q.append((0.1 * np.exp(2j * np.pi * hz_a * t).astype(np.complex64),
                           0.1 * np.exp(2j * np.pi * hz_b * t).astype(np.complex64)))
        t0 += block


def _peak_hz(sig):
    sig = np.asarray(sig, dtype=float)
    spec = np.abs(np.fft.rfft(sig * np.hanning(len(sig))))
    return np.argmax(spec) * AUDIO_RATE / len(sig)


def _pull(a, n_chunks):
    out = []
    for _ in range(n_chunks):
        c = a.get_audio(CHUNK, slice_id=0)
        if c is not None:
            out.extend(c)
    return out


def test_stereo_puts_loop_a_left_and_loop_b_right():
    a = _adapter("stereo")
    _feed(a, 2.0, 1_000.0, 2_000.0)
    out = _pull(a, 2 * AUDIO_RATE // CHUNK)
    assert out and isinstance(out[0], list) and len(out[0]) == 2
    pairs = np.asarray(out)[AUDIO_RATE // 2:]                  # past the AGC's settling
    assert abs(_peak_hz(pairs[:, 0]) - 1_000.0) < 10
    assert abs(_peak_hz(pairs[:, 1]) - 2_000.0) < 10
    # one AGC for both ears: an equal pair of tones comes out at equal level
    assert abs(20 * np.log10(np.std(pairs[:, 0]) / np.std(pairs[:, 1]))) < 1.0


def test_switching_hear_swaps_the_buffer_shape_without_an_error():
    a = _adapter("stereo")
    _feed(a, 1.0, 1_000.0, 2_000.0)
    assert isinstance(_pull(a, 10)[0], list)
    a._div.hear = "b"
    _feed(a, 1.0, 1_000.0, 2_000.0)
    mono = _pull(a, AUDIO_RATE // CHUNK)
    assert isinstance(mono[-1], float)
    assert abs(_peak_hz(mono[-AUDIO_RATE // 2:]) - 2_000.0) < 10


def test_audio_frames_interleaves_pairs_and_doubles_mono():
    assert audio_frames([0.5, -0.5]) == [0.5, 0.5, -0.5, -0.5]
    assert audio_frames([[0.1, 0.2], [0.3, 0.4]]) == [0.1, 0.2, 0.3, 0.4]
    assert audio_frames([0.5, -0.5], reduced_bw=True) == [0.5, -0.5]
    assert audio_frames([[0.1, 0.2], [0.3, 0.4]], reduced_bw=True) == [0.1, 0.3]
    assert audio_frames([]) == []
