#
# Aether-gate — how well one complex weight fits the whole passband.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The diversity combiner applies one complex weight to the whole demod
passband. That is right when the inter-antenna phase of the wanted signal
is the same at every audio frequency; multipath with a delay spread of a
millisecond or more makes it slope across even 3 kHz, and then a weight
per sub-band would do better. This measures which case we are in.

The cross-spectrum between the antennas is accumulated over voiced blocks
of the passband. `flatness` is |sum S| / sum |S|: 1.0 when one weight fits
every bin, lower the more the phase varies. `phase_slope_deg_per_khz` is a
weighted straight-line fit of the phase across the band, relative to its
mean, and is meaningful while the spread stays inside +-180 degrees (a
steeper slope shows up as low flatness regardless).
"""
import math

import numpy as np


class PassbandPhase:
    def __init__(self, rate_hz, tc_s=2.0):
        self.rate_hz = float(rate_hz)
        self.tc_s = float(tc_s)
        self.S = None                      # cross-spectrum Xa * conj(Xb), per bin
        self.Pa = None
        self.Pb = None
        self.f = None                      # bin frequencies (Hz, relative to the slice)

    def update(self, Xa, Xb, f, n, voiced):
        """One block's in-band FFT bins for both antennas; only voiced
        blocks count (the noise's phase says nothing about the talker)."""
        if not voiced or len(Xa) == 0:
            return
        S = Xa * np.conj(Xb)
        Pa = np.abs(Xa) ** 2
        Pb = np.abs(Xb) ** 2
        if self.S is None or self.f is None or len(self.f) != len(f) or not np.array_equal(self.f, f):
            self.S, self.Pa, self.Pb, self.f = S, Pa, Pb, np.array(f, dtype=float)
            return
        al = 1.0 - math.exp(-n / self.rate_hz / self.tc_s)
        self.S = (1 - al) * self.S + al * S
        self.Pa = (1 - al) * self.Pa + al * Pa
        self.Pb = (1 - al) * self.Pb + al * Pb

    def status(self):
        if self.S is None:
            return None
        w = np.abs(self.S)
        tot = float(w.sum())
        if tot <= 0:
            return None
        total = self.S.sum()
        flatness = abs(total) / tot
        ref = total / max(abs(total), 1e-30)
        ph = np.angle(self.S * np.conj(ref))            # relative to the mean phase
        fk = self.f / 1000.0
        fc = float((w * fk).sum() / tot)
        x = fk - fc
        den = float((w * x * x).sum())
        slope = float((w * x * ph).sum() / den) if den > 0 else 0.0
        coh = tot / max(float(np.sqrt(self.Pa * self.Pb).sum()), 1e-30)
        return {"flatness": round(float(flatness), 3),
                "phase_slope_deg_per_khz": round(math.degrees(slope), 1),
                "coherence": round(float(coh), 2)}
