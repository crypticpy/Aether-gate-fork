#
# Aether-gate — the coherence post-filter: what the two loops agree on stays.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Two loops give one degree of freedom to the combiner, and it spends it
on a beam or a null. But they carry a second thing the combiner cannot
spend: in every bin, the wanted talker arrives on both loops in step (that
is what the steering vector says), and the band noise does not. The
cross-spectrum between the two loops, each steered onto the talker, is an
estimate of the talker's power alone; the mean of their auto-spectra is
talker plus noise. Their ratio is the Wiener gain for that bin, made from
spatial evidence instead of a single-channel guess about what noise sounds
like -- so it is not the app's noise reduction wearing a different hat,
and the two stack.

  S_k = Re(phi_ab)                       the talker, on a short clock
  N_k = 0.5 (phi_aa + phi_bb) - S_k       the noise, on a long one
  G_k = S_k / (S_k + N_k),   clipped to [floor, 1]

The noise is what the loops do NOT share, and that part of the band
changes slowly, so it is smoothed over a second: a Wiener gain made from
an 80 ms estimate of both alone flickers between syllables (musical
noise); made from a steady noise and a quick signal it does not.

Coherent noise (a local source) shows up in Re(phi_ab) too and is left
alone here: that is the per-bin null's job. The floor is what keeps this
from sounding like a bad NR: a bin is never taken more than FLOOR_DB down,
the gain is smoothed across three bins and over ~80 ms, and where the
talker's own print says they have no energy at all the floor is let a
little deeper -- the microphone was never there, only the band.
"""
import math

import numpy as np

FLOOR_DB = -6.0
TC_S = 0.08                 # the talker's cross-spectrum memory
NOISE_TC_S = 1.0            # the incoherent part's memory
SMOOTH_BINS = 3
MIN_STEER = 0.1             # a loop the talker barely reaches is not a witness
PRINT_EDGE_DB = -20.0       # the print's edge (voiceprint.EDGE_DB)
PRINT_EXTRA_DB = -6.0       # ...beyond which the floor may go this much deeper
PRINT_BAND_HZ = 100.0


def _smooth(v, k=SMOOTH_BINS):
    if k <= 1 or len(v) < k:
        return v
    pad = k // 2
    vp = np.concatenate([np.repeat(v[:1], pad), v, np.repeat(v[-1:], pad)])
    c = np.concatenate([[0.0], np.cumsum(vp)])
    return (c[k:] - c[:-k]) / k


class PostFilter:
    def __init__(self, rate_hz, nfft, hop, floor_db=FLOOR_DB):
        self.rate_hz = float(rate_hz)
        self.n = int(nfft)
        self.floor_db = float(floor_db)
        self.al = 1.0 - math.exp(-hop / self.rate_hz / TC_S)
        self.al_n = 1.0 - math.exp(-hop / self.rate_hz / NOISE_TC_S)
        self.f = np.abs(np.fft.fftfreq(self.n, 1.0 / self.rate_hz))
        self.phi_aa = self.phi_bb = self.phi_ab = None
        self.noise = None                # (n,) the incoherent power, smoothed slowly
        self._frames = 0
        self.g = np.ones(self.n)
        self.mean_db = 0.0               # what the last frame was taken down by, on average

    def _floor(self, profile_db):
        floor = np.full(self.n, self.floor_db)
        if profile_db is None:
            return 10 ** (floor / 20.0)
        p = np.asarray(profile_db, dtype=float)
        band = np.minimum((self.f / PRINT_BAND_HZ).astype(int), len(p) - 1)
        below = np.clip((p[band] - PRINT_EDGE_DB) / PRINT_EDGE_DB, 0.0, 1.0)   # 0 at the edge, 1 at 2x
        return 10 ** ((floor + PRINT_EXTRA_DB * below) / 20.0)

    def gain(self, Xa, Xb, s, profile_db=None):
        """Per-bin amplitude gain for this frame from the two loops' spectra
        and the talker's steering vector s (2,)."""
        s = np.asarray(s, dtype=np.complex128)
        if abs(s[0]) < MIN_STEER or abs(s[1]) < MIN_STEER:
            return np.ones(self.n)
        a = Xa / s[0]
        b = Xb / s[1]
        aa = np.abs(a) ** 2
        bb = np.abs(b) ** 2
        ab = a * np.conj(b)
        self._frames += 1
        if self.phi_aa is None:
            self.phi_aa, self.phi_bb, self.phi_ab = aa, bb, ab
        else:
            self.phi_aa += self.al * (aa - self.phi_aa)
            self.phi_bb += self.al * (bb - self.phi_bb)
            self.phi_ab += self.al * (ab - self.phi_ab)
        sig = np.maximum(np.real(self.phi_ab), 0.0)
        inc = np.maximum(0.5 * (self.phi_aa + self.phi_bb) - sig, 0.0)
        if self.noise is None:
            self.noise = inc
        else:
            self.noise += max(self.al_n, 1.0 / self._frames) * (inc - self.noise)
        g = sig / np.maximum(sig + self.noise, 1e-30)
        g = _smooth(np.clip(g, 0.0, 1.0))
        g = np.maximum(g, self._floor(profile_db))
        self.g = g
        self.mean_db = float(20.0 * np.log10(max(float(np.mean(g)), 1e-6)))
        return g

    def status(self):
        return {"floor_db": self.floor_db, "mean_db": round(self.mean_db, 1)}
