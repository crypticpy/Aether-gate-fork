#
# Aether-gate -- one weight per FFT bin of a slice, taken from the band's own
# noise map.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The slice's weight, spent again in every bin it covers.

A two-element array has one degree of freedom, and `diversity.combine` spends
it once for the whole slice: y = (a + m b) / sqrt(1 + |m|^2). That is the
right answer only when the noise underneath the talker looks the same at
every frequency. It rarely does. A switching supply lands as a comb, a
neighbour's plasma television as a slab, a carrier as one line -- each from
its own direction, each occupying its own handful of bins. One weight has to
pick a favourite.

`SpatialMap` has already measured the thing that would settle the argument:
a 2x2 floor covariance R_k for every bin of the whole span, learned only
while that bin sat at its floor. This module turns that map into a weight
per bin,

    w_k = R_k^-1 s / (s^H R_k^-1 s)          (in the g^H x convention)

the MVDR/MRC weight: the least noise this pair can make in bin k while the
talker's steering vector s still passes with gain exactly one. Written in
the multiplier convention the rest of the code uses (y = w^T x, w = [1, m]
up to a scale) and rescaled by the gain the BROADBAND weight already had on
s, so that

  * a bin whose noise is white and equal on both loops gets back exactly
    [1, m] / sqrt(1 + |m|^2) -- the refinement costs nothing where there is
    nothing to refine;
  * every bin, refined or not, passes the talker at the same gain, so there
    is no step at the edge of the slice, at the edge of a stale patch, or
    at the moment the per-bin path switches in. Smoothing across bins is a
    convex combination of weights that all satisfy w^T s = g, so the
    smoothed weight satisfies it too: the constraint survives the filter.

Where the map has nothing to say -- outside the slice, off the end of the
map, or in a bin whose floor is stale -- the bin keeps the broadband weight.

Applied by 50 %-overlap STFT with a sqrt-Hann window at both ends, which
reconstructs exactly for a weight that does not vary. The frame is ~20 ms
whatever the sample rate, so the cost per second of signal is ~61 frames of
three length-NFFT transforms at every rate; NFFT grows with the rate, so the
cost grows as rate * log(rate), not faster. Measured on one core of an M4:
2.8 ms of CPU per second of signal at 62.5 kS/s (0.3 % of a core), 5.8 ms at
125 kS/s, and 138 ms per second at 2.04 MS/s (14 % of a core).
"""
import math

import numpy as np

from .diversity import WEIGHT_MAX_ABS

FRAME_S = 0.020
NFFT_MIN = 256
NFFT_MAX = 65_536
# Diagonal loading, as a fraction of the bin's mean power. A bin whose floor
# covariance is nearly rank one (one dominant source and nothing else) has an
# inverse that would hand back an enormous weight fitted to one sample's worth
# of noise; this bounds it at ~20 dB of null depth, which is deeper than two
# real antennas hold anyway. It is a multiple of I, so it cancels exactly in
# the distortionless normalisation when the bin IS white: loading never moves
# the white-noise answer.
LOAD = 1e-2
# Weights are averaged over this many neighbouring bins. One bin of the map is
# one noisy sample of a 2x2 covariance; and a weight that jumps bin to bin is a
# filter whose impulse response is longer than the frame, which wraps around.
SMOOTH_BINS = 5


def _next_pow2(x):
    return 1 << max(0, int(math.ceil(math.log2(max(x, 1.0)))))


def nfft_for(rate_hz, frame_s=FRAME_S):
    """The transform length that makes a frame about frame_s long at this
    rate, a power of two, clamped so 62.5 kS/s and 2.04 MS/s both land on
    something sane."""
    return int(min(NFFT_MAX, max(NFFT_MIN, _next_pow2(float(rate_hz) * float(frame_s)))))


def _smooth(W, k):
    """Moving average of a (n, 2) complex array along the bin axis, edge-padded."""
    if k <= 1 or W.shape[0] < k:
        return W
    pad = k // 2
    Wp = np.concatenate([np.repeat(W[:1], pad, axis=0), W, np.repeat(W[-1:], pad, axis=0)])
    c = np.cumsum(Wp, axis=0)
    c = np.concatenate([np.zeros_like(c[:1]), c])
    return (c[k:] - c[:-k]) / k


def steering_for_weight(m):
    """The steering vector for which `m` IS the maximal-ratio weight.

    combine()'s weight is w = [1, m] up to a scale, and y = w^T x, so the
    matched filter in white noise is w = conj(s): s = [1, conj(m)].
    """
    return np.array([1.0, np.conj(complex(m))], dtype=np.complex128)


class BinWeights:
    """Per-bin MVDR weights over a slice, applied by overlap-add.

    rate_hz     the sample rate of the blocks handed to apply().
    nbins_map   the resolution of the spatial map that will feed
                set_covariance(); only used to default step_hz.
    center_hz   the absolute frequency the blocks are centred on. Every other
                frequency this object is told is absolute too.
    """

    def __init__(self, rate_hz, nbins_map=0, *, center_hz=0.0, lo_hz=None, hi_hz=None,
                 frame_s=FRAME_S, load=LOAD, smooth_bins=SMOOTH_BINS):
        self.rate_hz = float(rate_hz)
        self.nbins_map = int(nbins_map)
        self.center_hz = float(center_hz)
        self.lo_hz = None if lo_hz is None else float(lo_hz)
        self.hi_hz = None if hi_hz is None else float(hi_hz)
        self.load = float(load)
        self.smooth_bins = int(smooth_bins)
        self.n = nfft_for(self.rate_hz, frame_s)
        self.h = self.n // 2
        self.frame_s = self.n / self.rate_hz
        self.win = np.sqrt(np.hanning(self.n + 1)[:-1])        # periodic sqrt-Hann
        self.m = 0j
        self.s = steering_for_weight(0j)
        self.R = None                  # (nbins_map, 2, 2), ASCENDING frequency
        self.stale = None              # (nbins_map,) bool, ascending
        self.start_hz = None           # centre frequency of R[0]
        self.step_hz = None
        self._W = None                 # (n, 2) weights, natural FFT order
        self._stats = None
        self._in_a = np.zeros(0, dtype=np.complex128)
        self._in_b = np.zeros(0, dtype=np.complex128)
        self._ola = np.zeros(self.n, dtype=np.complex128)

    # --- what the slice and the map tell it ---------------------------------
    def set_weight(self, m, s=None):
        """The slice's broadband weight, and optionally the talker's steering
        vector. Without s the steering implied by m is used (see
        steering_for_weight), which is what makes a white bin reproduce the
        broadband combine exactly."""
        self.m = complex(m)
        self.s = (steering_for_weight(self.m) if s is None
                  else np.asarray(s, dtype=np.complex128).reshape(2))
        self._W = None

    def set_covariance(self, R, stale_mask=None, start_hz=None, step_hz=None):
        """The map's per-bin noise covariance, in ASCENDING frequency order
        (SpatialMap keeps R in natural FFT order: pass np.fft.fftshift(R,
        axes=0)). R[i] is the covariance at start_hz + i * step_hz.
        stale_mask marks bins whose floor is too old to trust; R=None clears
        the map and every bin falls back to the broadband weight."""
        if R is None:
            self.R = self.stale = None
            self._W = None
            return
        R = np.asarray(R, dtype=np.complex128)
        if R.ndim != 3 or R.shape[1:] != (2, 2):
            raise ValueError("R must be (nbins, 2, 2)")
        self.R = R
        self.stale = (np.zeros(R.shape[0], dtype=bool) if stale_mask is None
                      else np.asarray(stale_mask, dtype=bool).reshape(R.shape[0]))
        self.step_hz = float(step_hz) if step_hz is not None else (
            self.step_hz if self.step_hz is not None else self.rate_hz / R.shape[0])
        self.start_hz = float(start_hz) if start_hz is not None else (
            self.start_hz if self.start_hz is not None
            else self.center_hz - self.rate_hz / 2.0)
        self._W = None

    def set_band(self, lo_hz, hi_hz):
        """The slice's edges in absolute Hz. Only these bins are refined."""
        self.lo_hz, self.hi_hz = float(lo_hz), float(hi_hz)
        self._W = None

    def set_center(self, center_hz):
        """The blocks moved: the FFT bins mean different frequencies now."""
        if float(center_hz) != self.center_hz:
            self.center_hz = float(center_hz)
            self._W = None

    def reset(self):
        """Forget the overlap-add tail (a discontinuity is coming)."""
        self._in_a = np.zeros(0, dtype=np.complex128)
        self._in_b = np.zeros(0, dtype=np.complex128)
        self._ola = np.zeros(self.n, dtype=np.complex128)

    # --- the weights ---------------------------------------------------------
    def freqs(self):
        """The FFT bin centres in absolute Hz, ascending."""
        return np.fft.fftshift(np.fft.fftfreq(self.n, 1.0 / self.rate_hz)) + self.center_hz

    def _interp(self, f):
        """The map's covariance at the frequencies f (any order), linearly
        interpolated between map bins, plus a mask of which are usable.
        A convex combination of two Hermitian PSD matrices is Hermitian PSD,
        so the interpolated R is still a covariance."""
        nmap = self.R.shape[0]
        pos = (f - self.start_hz) / self.step_hz
        ok = (pos >= 0.0) & (pos <= nmap - 1)
        i0 = np.clip(np.floor(pos), 0, max(nmap - 2, 0)).astype(np.int64)
        i1 = np.minimum(i0 + 1, nmap - 1)
        fr = np.clip(pos - i0, 0.0, 1.0)[:, None, None]
        R = (1.0 - fr) * self.R[i0] + fr * self.R[i1]
        ok &= ~self.stale[i0] & ~self.stale[i1]
        ok &= np.real(R[:, 0, 0] + R[:, 1, 1]) > 0.0
        return R, ok

    def _mvdr(self, R):
        """Distortionless per-bin weights in the multiplier convention:
        w^T s = 1, minimising w^T R conj(w)."""
        s = self.s
        tr = np.real(R[:, 0, 0] + R[:, 1, 1])
        Rl = R + (self.load * 0.5 * tr)[:, None, None] * np.eye(2)
        u = np.linalg.solve(Rl, np.broadcast_to(s, (R.shape[0], 2))[..., None])[..., 0]
        den = np.conj(s) @ u.T                      # s^H Rl^-1 s, real and positive
        w = np.conj(u / np.where(np.abs(den) > 1e-300, den, 1.0)[:, None])
        # keep the pair's ratio inside the cap the rest of the code honours,
        # then re-impose the constraint the cap just broke
        ratio = np.abs(w[:, 1]) / np.maximum(np.abs(w[:, 0]), 1e-300)
        big = ratio > WEIGHT_MAX_ABS
        if big.any():
            w[big, 1] *= WEIGHT_MAX_ABS / ratio[big]
        g = w @ s
        w = w / np.where(np.abs(g) > 1e-300, g, 1.0)[:, None]
        return w, np.isfinite(w).all(axis=1)

    def _weights(self):
        """(n, 2) weights in natural FFT order; cached until something moves."""
        if self._W is not None:
            return self._W
        n = self.n
        f = self.freqs()                                   # ascending
        v_wb = np.array([1.0, self.m], dtype=np.complex128) \
            / math.sqrt(1.0 + abs(self.m) ** 2)
        g_wb = complex(v_wb @ self.s)                      # broadband gain on the talker
        W = np.tile(v_wb, (n, 1))
        in_band = np.zeros(n, dtype=bool)
        if self.lo_hz is not None and self.hi_hz is not None:
            in_band = (f >= self.lo_hz) & (f <= self.hi_hz)
        used = np.zeros(n, dtype=bool)
        R_band = None
        have = self.R is not None and in_band.any() and abs(g_wb) > 1e-9
        if have:
            R_band, ok = self._interp(f[in_band])
            w, fine = self._mvdr(R_band)
            ok &= fine
            idx = np.flatnonzero(in_band)
            W[idx[ok]] = g_wb * w[ok]
            used[idx[ok]] = True
        W = _smooth(W, self.smooth_bins)
        self._stats = self._score(W, v_wb, in_band, used, R_band)
        self._W = np.fft.ifftshift(W, axes=0)
        return self._W

    def _score(self, W, v_wb, in_band, used, R_band):
        """What the refinement bought, on the map's own noise model."""
        st = {"bins_used": int(used.sum()),
              "bins_stale": int(in_band.sum() - used.sum()),
              "loading_db": round(10.0 * math.log10(max(self.load, 1e-30)), 1),
              "nfft": int(self.n), "frame_ms": round(1000.0 * self.frame_s, 1),
              "gain_over_broadband_db": 0.0}
        if R_band is None or not in_band.any():
            return st
        Wb = W[in_band]

        def power(V):
            return float(np.sum(np.real(np.einsum("ki,kij,kj->k", V, R_band, np.conj(V)))))
        n_wb = power(np.broadcast_to(v_wb, Wb.shape))
        n_sb = power(Wb)
        if n_wb > 0.0 and n_sb > 0.0:
            g = 10.0 * math.log10(n_wb / n_sb)
            st["gain_over_broadband_db"] = round(g, 2) if abs(g) > 5e-3 else 0.0
        return st

    def bin_weights(self):
        """(freq_hz, W) both ascending: the weight actually applied per bin."""
        return self.freqs(), np.fft.fftshift(self._weights(), axes=0)

    # --- the signal path ------------------------------------------------------
    def apply(self, a_block, b_block):
        """One aligned block pair in, the combined block out.

        The STFT buffers, so the answer for the first half-frame of a run is a
        window ramp and the output is shorter than the input until the buffer
        fills; after that it keeps up sample for sample.
        """
        a = np.asarray(a_block, dtype=np.complex128).ravel()
        b = np.asarray(b_block, dtype=np.complex128).ravel()
        k = min(len(a), len(b))
        self._in_a = np.concatenate([self._in_a, a[:k]])
        self._in_b = np.concatenate([self._in_b, b[:k]])
        W = self._weights()
        n, h = self.n, self.h
        out = []
        while len(self._in_a) >= n:
            Xa = np.fft.fft(self._in_a[:n] * self.win)
            Xb = np.fft.fft(self._in_b[:n] * self.win)
            self._in_a = self._in_a[h:]
            self._in_b = self._in_b[h:]
            self._ola += np.fft.ifft(W[:, 0] * Xa + W[:, 1] * Xb) * self.win
            out.append(self._ola[:h].copy())
            self._ola = np.concatenate([self._ola[h:], np.zeros(h, dtype=np.complex128)])
        if not out:
            return np.zeros(0, dtype=np.complex128)
        return np.concatenate(out)

    def status(self):
        """JSON-ready: what the per-bin path is doing and what it is worth."""
        self._weights()
        st = dict(self._stats)
        st["band_hz"] = None if self.lo_hz is None else [self.lo_hz, self.hi_hz]
        return st
