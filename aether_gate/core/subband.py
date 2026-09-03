#
# Aether-gate — a weight per passband bin: the wideband weight, refined where
# the noise has a direction.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The tracker fits ONE complex weight to the whole demod passband. Two loops
give one degree of freedom, and one weight spends it once: on the talker's
beam, or on one interferer's null. But the passband has hundreds of bins,
and each bin is its own two-element problem. A mains-locked comb below
1 kHz from the neighbour's supply and a heterodyne at 2.4 kHz from the
other side come from different directions, and a weight per bin can null
each in its own bins while every other bin keeps the beam.

This is a per-bin MVDR refinement, in the STFT domain, of the weight the
tracker already chose:

  * a noise covariance per bin, learned only while nobody is talking (the
    tracker's VAD), with a two-second memory;
  * in bins where that noise is coherent between the loops, the weight is
    Rn_k^-1 s normalised so the talker's steering vector s passes with unit
    gain (distortionless), i.e. a null on whatever is there that is NOT
    the talker; elsewhere the wideband weight, normalised the same way;
  * the whole thing scaled so a bin of white noise sounds exactly as it did
    through the wideband combiner. With no coherent noise anywhere the
    output IS the wideband combiner's, sample for sample (after the STFT's
    half-frame delay).

Weights are smoothed across three bins and over ~0.3 s so a fast-changing
per-bin null never turns into musical noise. sqrt-Hann analysis and
synthesis at 50 % overlap reconstruct exactly for constant weights.
"""
import math

import numpy as np

from .diversity import WEIGHT_MAX_ABS
from .postfilter import PostFilter

NFFT = 512
MIN_COHERENCE = 0.4        # a bin's noise needs this much to earn its own weight
NOISE_TC_S = 2.0
WEIGHT_TC_S = 0.3
SMOOTH_BINS = 3
LOAD = 1e-2                # diagonal loading, fraction of the bin's trace
WARMUP_FRAMES = 64         # ~0.7 s of noise before a bin may earn its own weight:
                           # one frame's covariance is rank one, coherent everywhere


def _smooth_bins(v, k=SMOOTH_BINS):
    if k <= 1 or len(v) < k:
        return v
    pad = k // 2
    vp = np.concatenate([np.repeat(v[:1], pad), v, np.repeat(v[-1:], pad)])
    c = np.concatenate([[0j], np.cumsum(vp)])
    return (c[k:] - c[:-k]) / k


class SubbandCombiner:
    def __init__(self, rate_hz, nfft=NFFT):
        self.rate_hz = float(rate_hz)
        self.n = int(nfft)
        self.h = self.n // 2
        self.win = np.sqrt(np.hanning(self.n + 1)[:-1])        # periodic sqrt-Hann
        self.Rn = None                     # (n, 2, 2) per-bin noise covariance
        self.v = None                      # (n, 2) per-bin weight in use
        self._in_a = np.zeros(0, dtype=np.complex128)
        self._in_b = np.zeros(0, dtype=np.complex128)
        self._ola = np.zeros(self.n, dtype=np.complex128)
        self._frames = 0                   # noise frames learned so far
        self.refined_bins = 0
        self.extra_db = 0.0               # noise the per-bin weights remove beyond the wideband one
        self.post = None                  # PostFilter, when the coherence post-filter is on
        self._squeeze = None              # ([(hz, width_hz), ...], null_m), while a SQUEEZE is forcing bins

    def set_post(self, on, floor_db=None):
        if not on:
            self.post = None
        elif self.post is None or (floor_db is not None and self.post.floor_db != floor_db):
            self.post = PostFilter(self.rate_hz, self.n, self.h,
                                   **({} if floor_db is None else {"floor_db": floor_db}))

    def set_squeeze(self, held, targets, null_m):
        """A held core.squeeze target forces its own bins to its null (see
        _force_null); anything else here leaves every bin to the MVDR fit
        below, same as before SQUEEZE existed. targets: [(hz, width_hz),
        ...] -- one region for a signal target, one per tooth for a comb,
        all forced to the SAME null_m: one source, one steering vector."""
        self._squeeze = (list(targets), complex(null_m)) if held and targets else None

    def _force_null(self, v, s):
        """Overwrite v's bins in every target's span -- each dilated by two
        bins either side, the same padding the tracker's own null gets
        nulling an interferer's over -- with the squeeze's null,
        distortionless on s like every other bin (see _weights): the talker
        still passes at unit gain everywhere the squeeze itself is not."""
        targets, m_null = self._squeeze
        bin_hz = self.rate_hz / self.n
        freqs = np.fft.fftfreq(self.n, 1.0 / self.rate_hz)
        mask = np.zeros(self.n, dtype=bool)
        for hz, width in targets:
            mask |= np.abs(freqs - hz) <= width / 2.0 + 2.0 * bin_hz
        if not mask.any():
            return v
        vt = np.array([1.0, m_null], dtype=np.complex128)
        g = vt @ s
        v = v.copy()
        v[mask] = vt / (g if abs(g) > 1e-9 else 1.0)
        return v

    # --- the noise model -------------------------------------------------------
    def _learn(self, Xa, Xb):
        inst = np.stack([np.stack([Xa * np.conj(Xa), Xa * np.conj(Xb)], axis=-1),
                         np.stack([Xb * np.conj(Xa), Xb * np.conj(Xb)], axis=-1)], axis=-2)
        self._frames += 1
        if self.Rn is None:
            self.Rn = inst
            return
        # a running mean until the window is full, then the EMA: otherwise the
        # rank-one seed frame dominates for hundreds of frames
        al = max(1.0 - math.exp(-self.h / self.rate_hz / NOISE_TC_S), 1.0 / self._frames)
        self.Rn += al * (inst - self.Rn)

    def _weights(self, m, s):
        """Per-bin v (n, 2) such that y_k = v_k^T x_k; distortionless on s."""
        s = np.asarray(s, dtype=np.complex128)
        v_wb = np.array([1.0, m], dtype=np.complex128)
        g_wb = v_wb @ s                    # the wideband weight's gain on the talker
        if abs(g_wb) < 1e-9:
            g_wb = 1.0
        base = np.tile(v_wb / g_wb, (self.n, 1))
        if self.Rn is None or self._frames < WARMUP_FRAMES:
            return base, 0, 0.0
        R = self.Rn
        tr = np.real(R[:, 0, 0] + R[:, 1, 1])
        coh = np.abs(R[:, 0, 1]) ** 2 / np.maximum(np.real(R[:, 0, 0]) * np.real(R[:, 1, 1]), 1e-30)
        use = (coh >= MIN_COHERENCE) & (tr > 0)
        v = base.copy()
        if use.any():
            Rl = R[use] + (LOAD * 0.5 * tr[use])[:, None, None] * np.eye(2)
            u = np.linalg.solve(Rl, np.broadcast_to(s, (int(use.sum()), 2))[..., None])[..., 0]
            u = u / (np.conj(s) @ u.T)[:, None]          # u^H s = 1  (u = Rn^-1 s / s^H Rn^-1 s)
            vk = np.conj(u)
            # keep the ratio inside the wideband cap, then re-impose the constraint
            ratio = np.abs(vk[:, 1]) / np.maximum(np.abs(vk[:, 0]), 1e-30)
            big = ratio > WEIGHT_MAX_ABS
            vk[big, 1] *= WEIGHT_MAX_ABS / ratio[big]
            vk = vk / (vk @ s)[:, None]
            v[use] = vk
        # what the refinement buys, in dB, on the learned noise
        def noise_of(vv):
            return np.real(np.einsum("ki,kij,kj->k", vv, R, np.conj(vv)))
        n_wb = float(np.sum(noise_of(base)))
        n_sb = float(np.sum(noise_of(v)))
        extra = 10.0 * math.log10(max(n_wb, 1e-30) / max(n_sb, 1e-30)) if n_wb > 0 else 0.0
        return v, int(use.sum()), extra

    # --- the audio path --------------------------------------------------------
    def process(self, pa, pb, m, s, talking, profile_db=None):
        """One block of the passband pair -> the combined block, delayed by
        half a frame. m: the tracker's wideband weight for this block; s: the
        talker's steering vector (2,); talking: the tracker's VAD; profile_db:
        the talker's print (100 Hz bands, dB re peak) for the post-filter's
        floor, if there is one."""
        pa = np.asarray(pa, dtype=np.complex128)
        pb = np.asarray(pb, dtype=np.complex128)
        n_in = min(len(pa), len(pb))
        self._in_a = np.concatenate([self._in_a, pa[:n_in]])
        self._in_b = np.concatenate([self._in_b, pb[:n_in]])
        s = np.asarray(s, dtype=np.complex128)
        # the wideband combiner's gain on the talker, so levels match it
        scale = (s[0] + m * s[1]) / math.sqrt(1.0 + abs(m) ** 2)
        out = []
        while len(self._in_a) >= self.n:
            fa = self._in_a[:self.n] * self.win
            fb = self._in_b[:self.n] * self.win
            self._in_a = self._in_a[self.h:]
            self._in_b = self._in_b[self.h:]
            Xa = np.fft.fft(fa)
            Xb = np.fft.fft(fb)
            if not talking:
                self._learn(Xa, Xb)
            v, self.refined_bins, self.extra_db = self._weights(m, s)
            if self._squeeze is not None:
                v = self._force_null(v, s)
            v = np.stack([_smooth_bins(v[:, 0]), _smooth_bins(v[:, 1])], axis=1)
            if self.v is None:
                self.v = v
            else:
                al = 1.0 - math.exp(-self.h / self.rate_hz / WEIGHT_TC_S)
                self.v += al * (v - self.v)
            Y = scale * (self.v[:, 0] * Xa + self.v[:, 1] * Xb)
            if self.post is not None:
                Y = Y * self.post.gain(Xa, Xb, s, profile_db)
            y = np.fft.ifft(Y) * self.win
            self._ola += y
            out.append(self._ola[:self.h].copy())
            self._ola = np.concatenate([self._ola[self.h:], np.zeros(self.h, dtype=np.complex128)])
        if not out:
            return np.zeros(0, dtype=np.complex128)
        return np.concatenate(out)

    def status(self):
        return {"bins": int(self.refined_bins), "extra_db": round(float(self.extra_db), 1)}
