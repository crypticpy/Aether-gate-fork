#
# Aether-gate — what kind of noise this is: mains-locked, impulsive, or just noise.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The spatial map says where the noise comes from and how coherent it is;
this says what it IS, from the envelope of the two loops' power over time.

  * A switch-mode supply, an LED driver, a dimmer, a charger: its RF noise is
    modulated at twice the mains frequency (both half cycles of the rectifier)
    and harmonics of that. The envelope's spectrum shows a comb at
    100/120 Hz, 200/240, ... The comb's depth is how much of the band's
    noise power rides on it, and 50 vs 60 Hz tells you which grid.
  * An electric fence, ignition, a thermostat arcing, PLT bursts: impulses.
    The full-rate envelope crosses a threshold well above the floor a few
    times a second (or hundreds). Their rate and their excess over the floor
    are what a blanker or a subtractor needs to know.
  * Anything else periodic in the envelope (a beacon keying, a 1 kHz tone
    modulator) shows as the strongest non-mains line.

Envelope: |a|^2 + |b|^2 per sample, block-averaged x64 (~1.95 kHz at
125 kS/s) into a 2 s ring, analysed once a second as a Welch average of
half-second Hann segments: 2 Hz resolution, which separates 100 from
120 Hz cleanly and finds the comb's harmonics up to ~900 Hz, and enough
averaging that a line 8 dB over the median floor is a line and not the
spectrum's own chi-square tail.
"""
import math

import numpy as np

DECIM = 64
RING_S = 2.0
PERIOD_S = 1.0
MAINS_HZ = (50.0, 60.0)
HARMONICS = 8
SEGMENT_S = 0.5
LINE_MIN_DB = 8.0           # a harmonic counts when it stands this far over the floor
IMPULSE_DB = 12.0           # over the block's median power: Gaussian noise itself
                            # (chi-square, 4 dof) crosses 9 dB a few times a second
IMPULSE_TC_S = 4.0


class NoiseProfile:
    def __init__(self, rate_hz, decim=DECIM):
        self.rate_hz = float(rate_hz)
        self.decim = int(decim)
        self.env_rate = self.rate_hz / self.decim
        self.ring = np.zeros(int(RING_S * self.env_rate), dtype=np.float64)
        self.ring_i = 0
        self.ring_n = 0
        self._since = 0.0
        self._imp_count = 0
        self._imp_excess = []
        self._imp_s = 0.0
        self._analyses = 0
        self.impulses_per_s = 0.0
        self.impulse_db = 0.0
        self.impulse_seen = 0
        self.mains_hz = None
        self.hum_db = 0.0
        self.harmonics = 0
        self.periodic = []               # [(hz, db)] the strongest non-mains lines

    def update(self, a, b):
        """One aligned raw block pair (before the blanker)."""
        n = min(len(a), len(b))
        if n < self.decim:
            return
        p = np.abs(a[:n]) ** 2 + np.abs(b[:n]) ** 2
        # impulses at full rate: rising edges over the block's median
        med = float(np.median(p))
        hot = p > med * 10.0 ** (IMPULSE_DB / 10.0)
        if hot.any():
            edges = np.flatnonzero(hot[1:] & ~hot[:-1])
            k = int(len(edges) + (1 if hot[0] else 0))
            self._imp_count += k
            self._imp_excess.append(10.0 * math.log10(float(np.max(p[hot])) / max(med, 1e-30)))
        self._imp_s += n / self.rate_hz
        # the decimated envelope into the ring
        m = n // self.decim
        env = p[:m * self.decim].reshape(m, self.decim).mean(axis=1)
        for x in env:                                   # m is ~64: a loop is fine
            self.ring[self.ring_i] = x
            self.ring_i = (self.ring_i + 1) % len(self.ring)
        self.ring_n = min(self.ring_n + m, len(self.ring))
        self._since += n / self.rate_hz
        if self._since >= PERIOD_S and self.ring_n >= len(self.ring) // 2:
            self._since = 0.0
            self._analyse()

    def _analyse(self):
        # impulses: rate over the last second, smoothed
        rate = self._imp_count / max(self._imp_s, 1e-6)
        self._analyses += 1
        # a running mean until the time constant has been seen, then an EMA
        al = max(1.0 - math.exp(-self._imp_s / IMPULSE_TC_S), 1.0 / self._analyses)
        self.impulses_per_s += al * (rate - self.impulses_per_s)
        if self._imp_excess:
            self.impulse_db += al * (float(np.median(self._imp_excess)) - self.impulse_db)
            self.impulse_seen += 1
        self._imp_count, self._imp_excess, self._imp_s = 0, [], 0.0
        # the envelope's spectrum
        if self.ring_n < len(self.ring):
            e = self.ring[:self.ring_n]
        else:
            e = np.concatenate([self.ring[self.ring_i:], self.ring[:self.ring_i]])
        mean = float(np.mean(e))
        if mean <= 0:
            return
        x = e / mean - 1.0
        seg = min(len(x), int(SEGMENT_S * self.env_rate))
        hop = max(1, seg // 2)
        win = np.hanning(seg)
        starts = range(0, len(x) - seg + 1, hop)
        P = np.zeros(seg // 2 + 1)
        for i in starts:
            P += np.abs(np.fft.rfft(x[i:i + seg] * win)) ** 2
        P /= max(1, len(starts)) * seg
        f = np.fft.rfftfreq(seg, 1.0 / self.env_rate)
        lo = int(np.searchsorted(f, 20.0))
        floor = float(np.median(P[lo:])) if len(P) > lo + 8 else 0.0
        if floor <= 0:
            return
        db = 10.0 * np.log10(np.maximum(P, 1e-30) / floor)
        res = f[1] - f[0]
        tol = max(1, int(round(1.5 / res)))          # about +-2 Hz around each harmonic

        def line_at(hz):
            k = int(round(hz / res))
            if k + tol >= len(db):
                return 0.0
            return float(np.max(db[max(1, k - tol):k + tol + 1]))

        best = None
        for m_hz in MAINS_HZ:
            f2 = 2.0 * m_hz                            # the rectifier's fundamental
            lines = [line_at(f2 * (h + 1)) for h in range(HARMONICS)]
            count = sum(1 for d in lines if d >= LINE_MIN_DB)
            strength = max(lines)
            if count >= 1 and (best is None or (count, strength) > (best[1], best[2])):
                best = (m_hz, count, strength, lines)
        if best is None:
            self.mains_hz, self.hum_db, self.harmonics = None, 0.0, 0
            mask = np.zeros(len(db), dtype=bool)
        else:
            self.mains_hz, self.harmonics, self.hum_db = best[0], best[1], round(best[2], 1)
            mask = np.zeros(len(db), dtype=bool)
            for h in range(HARMONICS):
                k = int(round(2.0 * best[0] * (h + 1) / res))
                mask[max(1, k - tol):k + tol + 1] = True
        # the strongest lines that are not the mains comb
        cand = np.where(mask, -np.inf, db)
        cand[:lo] = -np.inf
        out = []
        for _ in range(3):
            k = int(np.argmax(cand))
            if cand[k] < LINE_MIN_DB:
                break
            out.append((round(float(f[k]), 1), round(float(cand[k]), 1)))
            cand[max(0, k - tol):k + tol + 1] = -np.inf
        self.periodic = out

    def status(self):
        return {
            "mains_hz": self.mains_hz,
            "hum_db": round(float(self.hum_db), 1),
            "harmonics": int(self.harmonics),
            "impulses_per_s": round(float(self.impulses_per_s), 1),
            "impulse_db": round(float(self.impulse_db), 1) if self.impulse_seen else None,
            "periodic": [{"hz": hz, "db": d} for hz, d in self.periodic],
            "seconds": round(float(self.ring_n / self.env_rate), 1),
            # what each number looks back over: the lines come from the
            # envelope ring, the impulse rate from a smoothed count
            "window_s": round(float(RING_S), 1),
            "impulse_window_s": round(float(IMPULSE_TC_S), 1),
        }
