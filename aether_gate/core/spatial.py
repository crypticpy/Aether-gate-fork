#
# Aether-gate — per-bin spatial statistics of a coherent antenna pair.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""A noise map: what every frequency bin says about where its energy comes from.

The panadapter FFT of both channels is a free by-product of every block the
reader hands up. Keeping a small channel-by-channel covariance PER BIN turns
it into a map of the band's spatial structure:

  coherence     how much of a bin's power the two antennas share. Coherent
                power comes from one direction (a local noise source, a
                station); incoherent power is the sky and the receivers.
                A nullable noise source shows up as a run of coherent bins
                with no station in them.

  null weight   per bin, the two-element weight that minimises that bin's
                floor power. Applied per bin to the panadapter it shows the
                band as the array could hear it ("nulled" pan mode); applied
                as a slice's manual weight it nulls a source with one click.

  sources       contiguous runs of coherent bins with a consistent steering
                vector, listed with their frequency span and null weight.

The covariance is FLOOR-TRACKED: a frame updates a bin only while the power
around that bin sits within FLOOR_MAX_RATIO of what has already been learned
there, so passing stations do not become "noise" and the map describes what
is persistently there. (A single bin of a single frame has a chi-square
scatter as large as its mean, so the test is made on a neighbourhood of
bins and the ratio is generous; the first WARMUP_S accept everything.) A
bin that has not been accepted for STALE_S is updated anyway, so a band
that genuinely gets noisier is followed, just slowly.

Shapes carry a channel axis (nbins, N, N): an N-element array is the same
code with N = 2 today. Only the multiplier form of the null weight (one
complex m for y = a + m b) is two-element specific.
"""
import functools
import math

import numpy as np

from .diversity import WEIGHT_MAX_ABS, weight_to_polar

# A source needs this much coherence over at least this many bins to be listed.


def _reg_lower_gamma(a, x):
    """P(a, x), the regularised lower incomplete gamma, by its series."""
    term = 1.0 / math.gamma(a + 1)
    total = term
    for k in range(1, 500):
        term *= x / (a + k)
        total += term
        if term < 1e-15 * total:
            break
    return total * math.exp(-x) * x ** a


@functools.lru_cache(maxsize=None)
def trim_half_mean(n_channels):
    """E[p | p < median(p)] / E[p] when p is the SUM of n_channels bin powers
    of noise, i.e. Gamma(n, 1): 1 - ln 2 = 0.307 for one channel, 0.474 for
    two. The quieter-half trim divides by this so it still reads as the
    noise's mean (found live: uncorrected, a VAD referenced to the trimmed
    guard band never went quiet)."""
    n = int(n_channels)
    lo, hi = 0.0, 4.0 * n + 8.0
    for _ in range(80):                                     # median by bisection
        mid = 0.5 * (lo + hi)
        if _reg_lower_gamma(n, mid) < 0.5:
            lo = mid
        else:
            hi = mid
    m = 0.5 * (lo + hi)
    return n * _reg_lower_gamma(n + 1, m) / _reg_lower_gamma(n, m) / n

SOURCE_MIN_COHERENCE = 0.5
SOURCE_MIN_BINS = 3
# A run of coherent bins is split where the steering phase jumps by more than
# this between neighbours: two sources adjacent in frequency, not one.
SOURCE_PHASE_SPLIT_DEG = 45.0
SOURCE_MAX = 8
# Covariance entries are averaged across this many neighbouring bins before
# any weight or coherence is read from them: a single bin is one noisy sample.
SMOOTH_BINS = 9
# A neighbourhood (SMOOTH_BINS wide) whose frame power is within this ratio
# of its learned floor is accepted as floor. Nine bins of one frame have an
# 18-degree-of-freedom chi-square scatter: 2x the mean is a 0.1 % event for
# noise, and a station 6 dB up never passes.
FLOOR_MAX_RATIO = 2.0
WARMUP_S = 1.0
# A bin refused for this long is updated regardless (the band got noisier).
STALE_S = 30.0
# A bin that has just entered the span on a retune (see SpatialMap.retune):
# large enough that _stale * any real frame_s clears STALE_S, so it is
# accepted unconditionally on the very next frame rather than waiting 30 s.
RETUNE_FRESH_STALE = 1_000_000
# What a retune-fresh bin's covariance starts at: small enough to be noise
# next to any real signal, isotropic (a multiple of I) so _analyse reads it
# as incoherent rather than the divide-by-near-zero of an exact 0 matrix.
RETUNE_FRESH_EPS = 1e-6


def _shift_bins(arr, shift, fill, axis=0):
    """arr's `axis`, in ASCENDING (low-to-high frequency) order, translated
    by `shift` positions: new[i] = old[i + shift]. Positions that would come
    from off the end (the span moved past what was ever measured there) are
    set to `fill` instead. shift == 0 is a no-op; |shift| >= the axis length
    is nothing but fill."""
    arr = np.asarray(arr)
    n = arr.shape[axis]
    if shift == 0:
        return arr
    if abs(shift) >= n:
        out = np.empty_like(arr)
        out[...] = fill
        return out
    rolled = np.roll(arr, -shift, axis=axis)
    idx = [slice(None)] * arr.ndim
    idx[axis] = slice(n - shift, n) if shift > 0 else slice(0, -shift)
    rolled[tuple(idx)] = fill
    return rolled


def _shift_natural(arr, shift, fill, axis=0):
    """As _shift_bins, but `axis` is in NATURAL FFT order (bin 0 = DC, the
    upper half negative frequency) rather than ascending: the covariance
    R and the staleness counter are both kept that way, matching the FFT
    the reader hands in."""
    if shift == 0:
        return arr
    s = np.fft.fftshift(arr, axes=axis)
    return np.fft.ifftshift(_shift_bins(s, shift, fill, axis=axis), axes=axis)


def region_covariance(X, idx, trim=False):
    """Mean per-bin covariance of the bins idx of the spectra X (N, nbins).

    trim=True averages only the quieter half of the bins (by their summed
    power), so a station that happens to sit inside a guard band does not
    masquerade as noise — and rescales the result by trim_half_mean() so it
    still reads as the noise's mean rather than 3-5 dB under it.
    Returns an (N, N) Hermitian matrix, or None when idx is empty.
    """
    X = np.asarray(X)
    if len(idx) == 0:
        return None
    S = X[:, idx]
    if trim and S.shape[1] >= 4:
        p = np.sum(np.abs(S) ** 2, axis=0)
        keep = p <= np.median(p)
        S = S[:, keep]
        return (S @ S.conj().T) / (S.shape[1] * trim_half_mean(S.shape[0]))
    return (S @ S.conj().T) / S.shape[1]


def _smooth1(x, k):
    """Moving average of a 1-D array (edge-padded)."""
    if k <= 1 or len(x) < k:
        return x
    pad = k // 2
    xp = np.concatenate([np.repeat(x[:1], pad), x, np.repeat(x[-1:], pad)])
    c = np.concatenate([[0.0], np.cumsum(xp)])
    return (c[k:] - c[:-k]) / k


def _smooth(R, k):
    """Moving average of (nbins, N, N) along the bin axis (edge-padded)."""
    if k <= 1 or R.shape[0] < k:
        return R
    pad = k // 2
    Rp = np.concatenate([np.repeat(R[:1], pad, axis=0), R, np.repeat(R[-1:], pad, axis=0)])
    c = np.cumsum(Rp, axis=0)
    c = np.concatenate([np.zeros_like(c[:1]), c])
    return (c[k:] - c[:-k]) / k


class SpatialMap:
    def __init__(self, nbins, rate_hz, channels=2, cov_tc_s=10.0):
        self.nbins = int(nbins)
        self.rate_hz = float(rate_hz)
        self.n = int(channels)
        self.cov_tc_s = float(cov_tc_s)
        self.R = None                       # (nbins, N, N) floor covariance, natural FFT order
        self._stale = None                  # (nbins,) frames since the bin last accepted
        self.frames = 0
        self._cache = None

    # --- reader thread -----------------------------------------------------
    def retune(self, delta_hz):
        """The hardware centre moved by delta_hz at the same sample rate:
        slide the floor covariance along frequency so each bin's learned
        history stays with its own absolute frequency instead of being
        thrown away. Bins that enter from off-span start at a tiny ISOTROPIC
        covariance (RETUNE_FRESH_EPS * I) rather than an exact zero matrix:
        an exact zero makes _analyse's eigenvalue ratio degenerate (0/0) and
        reads as perfectly coherent, which would paint a phantom source right
        on the fresh boundary. A staleness flag forces acceptance on the very
        next frame (see RETUNE_FRESH_STALE) rather than sitting out STALE_S;
        a shift past the whole span is nothing left to slide, so it resets."""
        if self.R is None:
            return
        bin_hz = self.rate_hz / self.nbins
        shift = int(round(float(delta_hz) / bin_hz))
        if shift == 0:
            return
        if abs(shift) >= self.nbins:
            self.R = None
            self._stale = None
            self.frames = 0
            self._cache = None
            return
        fresh = RETUNE_FRESH_EPS * np.eye(self.n, dtype=self.R.dtype)
        self.R = _shift_natural(self.R, shift, fresh, axis=0)
        self._stale = _shift_natural(self._stale, shift, RETUNE_FRESH_STALE, axis=0)
        self._cache = None

    def update(self, X, frame_s):
        """One frame of spectra X (N, nbins), which took frame_s of signal."""
        X = np.asarray(X, dtype=np.complex128)
        inst = X.T[:, :, None] * np.conj(X.T[:, None, :])         # (nbins, N, N)
        self._cache = None
        self.frames += 1
        if self.R is None:
            self.R = inst
            self._stale = np.zeros(self.nbins, dtype=np.int64)
            return
        p = _smooth1(np.real(np.einsum("kii->k", inst)), SMOOTH_BINS)
        ref = _smooth1(np.real(np.einsum("kii->k", self.R)), SMOOTH_BINS)
        self._stale += 1
        accept = (p <= FLOOR_MAX_RATIO * ref) | (self._stale * frame_s > STALE_S) \
            | (self.frames * frame_s <= WARMUP_S)
        # A running mean until the time constant's worth of frames has been
        # seen: an EMA seeded from one frame would let that single (rank-one)
        # frame dominate for most of a time constant, which reads as coherence.
        al = max(1.0 - math.exp(-frame_s / self.cov_tc_s), 1.0 / self.frames)
        self.R[accept] += al * (inst[accept] - self.R[accept])
        self._stale[accept] = 0

    # --- on demand (control port / pan) --------------------------------------
    def _analyse(self):
        """Smoothed floor covariance -> (coherence, steering phase, null m, level)."""
        if self._cache is not None:
            return self._cache
        if self.R is None:
            z = np.zeros(self.nbins)
            self._cache = (z, z, np.zeros(self.nbins, dtype=np.complex128), np.full(self.nbins, -200.0))
            return self._cache
        R = _smooth(self.R, SMOOTH_BINS)
        tr = np.real(np.einsum("kii->k", R))
        load = 1e-3 * tr / self.n + 1e-30
        Rl = R + load[:, None, None] * np.eye(self.n)
        vals, vecs = np.linalg.eigh(Rl)                            # ascending per bin
        lam_max = vals[:, -1]
        coh = (lam_max / np.maximum(tr, 1e-30) - 1.0 / self.n) / (1.0 - 1.0 / self.n)
        coh = np.clip(coh, 0.0, 1.0)
        steer = np.angle(R[:, 0, 1]) if self.n >= 2 else np.zeros(self.nbins)
        w = vecs[:, :, 0]                                          # smallest eigenvector
        if self.n == 2:
            w0, w1 = w[:, 0], w[:, 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                m = np.conj(w1 / w0)
            m = np.where(np.isfinite(m), m, WEIGHT_MAX_ABS * np.exp(1j * np.angle(np.conj(w1) * w0)))
            big = np.abs(m) > WEIGHT_MAX_ABS
            m = np.where(big, m / np.maximum(np.abs(m), 1e-30) * WEIGHT_MAX_ABS, m)
        else:
            m = np.zeros(self.nbins, dtype=np.complex128)
        level = 10.0 * np.log10(np.maximum(tr / self.n, 1e-30))
        self._cache = (coh, steer, m, level)
        return self._cache

    def coherence(self):
        return self._analyse()[0]

    def null_weights(self, fallback=0j, min_coherence=SOURCE_MIN_COHERENCE):
        """Per-bin m for the 'nulled' pan: the bin's null where it is coherent
        enough to mean something, `fallback` (the slice's weight) elsewhere."""
        coh, _steer, m, _lvl = self._analyse()
        return np.where(coh >= min_coherence, m, fallback)

    def _freq(self, k, center_hz):
        """Centre frequency of natural-order bin k."""
        k = np.asarray(k)
        kk = np.where(k < self.nbins / 2, k, k - self.nbins)
        return center_hz + kk * self.rate_hz / self.nbins

    def sources(self, center_hz=0.0):
        """Runs of coherent bins with a consistent steering phase, loudest first."""
        coh, steer, m, level = self._analyse()
        order = np.fft.fftshift(np.arange(self.nbins))             # low freq -> high
        c, s, mm, lv = coh[order], steer[order], m[order], level[order]
        f = self._freq(order, center_hz)
        step = self.rate_hz / self.nbins
        out = []
        k = 0
        while k < self.nbins:
            if c[k] < SOURCE_MIN_COHERENCE:
                k += 1
                continue
            j = k + 1
            while j < self.nbins and c[j] >= SOURCE_MIN_COHERENCE:
                d = abs((s[j] - s[j - 1] + math.pi) % (2 * math.pi) - math.pi)
                if math.degrees(d) > SOURCE_PHASE_SPLIT_DEG:
                    break
                j += 1
            if j - k >= SOURCE_MIN_BINS:
                seg = slice(k, j)
                w = np.mean(mm[seg])
                ph, ra = weight_to_polar(complex(w))
                out.append({
                    "lo_hz": float(f[k] - step / 2), "hi_hz": float(f[j - 1] + step / 2),
                    "phase_deg": round(ph, 1), "ratio_db": round(ra, 1),
                    "coherence": round(float(np.mean(c[seg])), 2),
                    "level_db": round(float(np.mean(lv[seg])), 1),
                    "bins": int(j - k),
                })
            k = j
        out.sort(key=lambda d: -d["level_db"])
        return out[:SOURCE_MAX]

    def map(self, center_hz=0.0, points=256):
        """Decimated coherence/level for a strip display, low frequency first."""
        coh, _s, _m, level = self._analyse()
        order = np.fft.fftshift(np.arange(self.nbins))
        c, lv = coh[order], level[order]
        points = max(1, min(int(points), self.nbins))
        edges = np.linspace(0, self.nbins, points + 1).astype(int)
        cc = [float(np.max(c[a:b])) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
        ll = [float(np.mean(lv[a:b])) if b > a else -200.0 for a, b in zip(edges[:-1], edges[1:])]
        return {
            "start_hz": float(center_hz - self.rate_hz / 2),
            "step_hz": float(self.rate_hz / points),
            "coherence": [round(x, 3) for x in cc],
            "level_db": [round(x, 1) for x in ll],
            "sources": self.sources(center_hz),
        }
