#
# Aether-gate — the audio AGC: attack, decay, hang and a threshold, in real
# time constants. Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The AGC that follows the receive filter (see filter.py). Chunk-rate,
with a floor tracker so the gain never winds the band noise between words
up to the loudness of the voice."""
import math

import numpy as np

AGC_MODES = {                        # attack_ms, decay_ms, hang_ms
    "fast": (2.0, 100.0, 0.0),
    "med": (5.0, 250.0, 250.0),
    "slow": (5.0, 500.0, 500.0),
    "long": (5.0, 2000.0, 1000.0),
    "off": None,
}
AGC_MAX_GAIN = 1000.0                # 60 dB
AGC_THRESHOLD_DB = 20.0
AGC_FLOOR_RISE_MS = 8000.0           # the floor tracker follows speech up this slowly


class Agc:
    """Chunk-rate AGC with attack, decay and hang in milliseconds. The gain
    is ramped across each chunk so a level step never clicks."""

    def __init__(self, target=0.25, rate_hz=24000.0):
        self.target = float(target)
        self.rate_hz = float(rate_hz)
        self.level = 0.05
        self.gain = None
        self.hang_left_ms = 0.0
        # THRESHOLD (a radio's AGC-T). Without it this is a leveller: between
        # words the decay winds the gain up until the band noise sits at the
        # same loudness as the voice did, and speech comes out soft and
        # mumbling with the noise pumping up around every gap. The floor
        # tracker follows the quietest recent chunks; the gain may never lift
        # that floor above target - threshold_db. 0 is the old leveller.
        self.threshold_db = AGC_THRESHOLD_DB
        self.floor = None
        self.set("med")

    def set(self, mode=None, attack_ms=None, decay_ms=None, hang_ms=None, threshold_db=None):
        if threshold_db is not None:
            v = float(threshold_db)
            if not (0.0 <= v <= 60.0):
                raise ValueError("threshold_db must be 0..60")
            self.threshold_db = v
        if mode is not None:
            if mode not in AGC_MODES:
                raise ValueError(f"agc mode must be one of {sorted(AGC_MODES)}")
            self.mode = mode
            if AGC_MODES[mode] is not None:
                self.attack_ms, self.decay_ms, self.hang_ms = AGC_MODES[mode]
        for name, v in (("attack_ms", attack_ms), ("decay_ms", decay_ms), ("hang_ms", hang_ms)):
            if v is not None:
                v = float(v)
                if not (0.0 <= v <= 10000.0):
                    raise ValueError(f"{name} must be 0..10000")
                setattr(self, name, v)

    def process(self, audio):
        np_ = np
        n = len(audio)
        if n == 0:
            return audio
        if self.mode == "off":
            g = self.target / max(self.level, 1e-4)
            return np_.clip(audio * g, -1.0, 1.0)
        chunk_ms = 1000.0 * n / self.rate_hz
        rms = float(np_.sqrt(np_.mean(audio * audio)) + 1e-9)
        if self.floor is None or rms < self.floor:
            self.floor = rms if self.floor is None else self.floor + 0.5 * (rms - self.floor)
        else:
            self.floor += (1.0 - math.exp(-chunk_ms / AGC_FLOOR_RISE_MS)) * (rms - self.floor)
        if rms > self.level:
            a = 1.0 - math.exp(-chunk_ms / max(self.attack_ms, 1e-3))
            self.level += a * (rms - self.level)
            self.hang_left_ms = self.hang_ms
        elif self.hang_left_ms > 0:
            self.hang_left_ms -= chunk_ms
        else:
            a = 1.0 - math.exp(-chunk_ms / max(self.decay_ms, 1e-3))
            self.level += a * (rms - self.level)
        g_new = min(self.target / max(self.level, 1e-4), AGC_MAX_GAIN)
        floor_target = self.target * 10 ** (-self.threshold_db / 20.0)
        g_new = min(g_new, floor_target / max(self.floor, 1e-5))
        g_old = self.gain if self.gain is not None else g_new
        ramp = np_.linspace(g_old, g_new, n)
        out = audio * (ramp[:, None] if audio.ndim == 2 else ramp)
        self.gain = g_new
        return np_.clip(out, -1.0, 1.0)

    def status(self):
        return {"mode": self.mode, "attack_ms": self.attack_ms, "decay_ms": self.decay_ms,
                "hang_ms": self.hang_ms, "threshold_db": self.threshold_db,
                "gain_db": round(20 * math.log10(self.gain), 1) if self.gain else None}
