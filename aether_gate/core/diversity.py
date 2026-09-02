#
# Aether-gate — two-element diversity combining for a coherent dual-tuner SDR.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Two antennas, one complex weight, a steerable null.

An RSPduo in dual-tuner mode samples both tuners from one clock and one
local oscillator, so the two IQ streams differ only by what the two antennas
saw: the same sky-wave signal at two amplitudes and phases, and the same
local noise at two other amplitudes and phases. The combined output

    y = (a + m * b) / sqrt(1 + |m|^2)

with a single complex multiplier m therefore behaves like a third antenna
whose pattern can be steered by choosing m, and in particular m can be
chosen to put a null on the loudest noise source, or to maximise the
signal-to-noise ratio of whatever is talking right now. That is the whole
algorithm; everything in this module is about choosing m honestly.

Three pieces, each hardware-free and unit-tested on synthetic data:

  find_lag / Aligner  The driver hands the two channels up without their
                      sample counters, so they may be offset by a whole
                      number of samples. The offset is measured once by
                      cross-correlating the two streams on whatever they
                      share (band noise is enough) and then held.

  combine             The weighted sum above, normalised so that equal,
                      uncorrelated noise on both inputs comes out at the
                      same level whatever m is, which keeps the S-meter
                      and the audio AGC from jumping as the weight moves.

  Tracker             Keeps a 2x2 noise covariance (updated between overs)
                      and a 2x2 signal covariance (updated while someone is
                      talking) of the demodulator's passband, and refits m
                      every refresh interval: the smallest eigenvector of
                      the noise covariance for "null", the generalised
                      eigenvector that maximises SNR for "track". Both are
                      closed form on a 2x2, so a refit costs microseconds
                      and can follow a new talker within a syllable.
"""
import math

import numpy as np

# Alignment is accepted only when the correlation peak stands this far above
# the window's median. The correlation is phase-transformed (GCC-PHAT): every
# frequency bin votes with unit weight, so the peak height is the coherence
# between the channels and the floor is 1/sqrt(FFT length). Band noise from
# two loops a few metres apart is only partly coherent, but 0.5 s of it at
# 125 kS/s is an FFT of 2^17 bins, so even 20 % coherence stands ~60x above
# the floor. The largest of ~16 000 pure-noise bins in the search window is
# itself ~6x the median, so the bar sits well above that: a ratio this low
# only happens when one channel is dead, the two are not coherent, or the
# true offset is outside the search window.
ALIGN_MIN_PEAK = 10.0

# |m| is capped so a dead or disconnected channel B cannot be amplified into
# the output by a fit that "sees" less noise there (20 dB either way).
WEIGHT_MAX_ABS = 10.0

# A refit replaces the current weight only if it predicts at least this
# much improvement, so the weight does not chatter on estimation noise.
REFIT_MIN_GAIN_DB = 0.3

# Block power this far above the tracked floor counts as "someone talking".
# Deliberately low: a weak DX signal only lifts the passband a couple of dB,
# and the block mean of ~1000 noise samples wanders by ~3%, so 1.5 is still
# far outside the noise's own scatter.
VAD_RATIO = 1.5                     # ~ +1.8 dB


def find_lag(a, b, max_lag):
    """Integer offset between two coherent channels.

    Returns (lag, peak_ratio) where a[n] lines up with b[n + lag]: a positive
    lag means channel B runs LATE by that many samples. peak_ratio is the
    correlation peak over the median of the search window, the confidence
    that there is a peak at all (see ALIGN_MIN_PEAK).

    The cross-spectrum is whitened before the inverse transform (the phase
    transform, GCC-PHAT), so a DC/LO spur or one strong broadcast carrier
    that both antennas hear cannot smear the correlation into a broad hump
    that drifts by hundreds of samples between measurements; only the phase
    slope across the band, which is the delay, survives.
    """
    n = min(len(a), len(b))
    max_lag = int(min(max_lag, n - 1))
    if n < 8 or max_lag < 0:
        return 0, 0.0
    a = np.asarray(a[:n], dtype=np.complex128)
    b = np.asarray(b[:n], dtype=np.complex128)
    a = a - a.mean()
    b = b - b.mean()
    m = 1 << int(2 * n - 1).bit_length()          # linear, not circular, correlation
    cross = np.fft.fft(a, m) * np.conj(np.fft.fft(b, m))
    mag = np.abs(cross)
    cross /= mag + 1e-12 * (float(mag.max()) or 1.0)
    xc = np.fft.ifft(cross)
    # xc[k] = sum_n a[n] conj(b[n-k]) peaks where a[n] ~ b[n-k], i.e. lag = -k.
    win = np.concatenate([xc[m - max_lag:], xc[:max_lag + 1]]) if max_lag else xc[:1]
    mag = np.abs(win)
    i = int(np.argmax(mag))
    k = i - max_lag
    med = float(np.median(mag))
    ratio = float(mag[i] / med) if med > 0 else 0.0
    return -k, ratio


class Aligner:
    """Delays whichever channel runs early so both come out sample-aligned."""

    def __init__(self):
        self.lag = 0
        self.aligned = False
        self.peak = 0.0
        self._hold = np.zeros(0, dtype=np.complex64)

    def set_lag(self, lag, peak=0.0, aligned=True):
        self.lag = int(lag)
        self.peak = float(peak)
        self.aligned = bool(aligned)
        self._hold = np.zeros(abs(self.lag), dtype=np.complex64)

    def calibrate(self, a, b, max_lag):
        """Measure the lag on raw (unaligned) history and adopt it if credible."""
        lag, peak = find_lag(a, b, max_lag)
        ok = peak >= ALIGN_MIN_PEAK
        self.set_lag(lag if ok else 0, peak, ok)
        return lag, peak, ok

    def apply(self, a, b):
        """One block pair in, one aligned block pair out (same lengths)."""
        if self.lag == 0:
            return a, b
        # a[n] ~ b[n + lag]: with lag > 0 the event reaches B's index later,
        # so A is early and must wait |lag| samples; with lag < 0, B waits.
        early = a if self.lag > 0 else b
        x = np.concatenate([self._hold, early])
        self._hold = x[len(early):]
        delayed = x[:len(early)]
        return (delayed, b) if self.lag > 0 else (a, delayed)


def weight_from_polar(phase_deg, ratio_db):
    """The manual weight: |m| from a dB ratio, angle from degrees."""
    mag = min(WEIGHT_MAX_ABS, 10.0 ** (float(ratio_db) / 20.0))
    return mag * np.exp(1j * math.radians(float(phase_deg)))


def weight_to_polar(m):
    """(phase_deg in [0, 360), ratio_db) of a weight; ratio floors at -100 dB."""
    mag = abs(m)
    ratio = 20.0 * math.log10(mag) if mag > 1e-5 else -100.0
    return math.degrees(math.atan2(m.imag, m.real)) % 360.0, ratio


def combine(a, b, m):
    """y = (a + m b) / sqrt(1 + |m|^2); m = 0 is channel A alone."""
    if m == 0:
        return a
    return (a + m * b) * (1.0 / math.sqrt(1.0 + abs(m) ** 2))


def _snr_of(m, Rs, Rn):
    """Linear SNR of y = a + m b given signal-plus-noise and noise covariances.

    |y|^2 = v^T R conj(v) with v = [1, m], so the quadratic form uses conj(v)
    on the right; both forms are real for Hermitian R.
    """
    v = np.array([1.0, m], dtype=np.complex128)
    cv = np.conj(v)
    noise = float(np.real(v @ Rn @ cv))
    total = float(np.real(v @ Rs @ cv))
    if noise <= 0.0:
        return 0.0
    return max(0.0, (total - noise) / noise)


def fit_null(Rn):
    """The weight that minimises noise power: the smallest eigenvector of Rn.

    Solves min over unit w of w^H Rn w, then converts w (which multiplies
    the conjugate-free vector as y = w^H x) to the multiplier form
    y ∝ a + m b, m = conj(w1 / w0).
    """
    vals, vecs = np.linalg.eigh(Rn)
    w = vecs[:, int(np.argmin(vals))]
    return _to_multiplier(w)


def fit_max_snr(Rs, Rn):
    """The weight that maximises SNR: principal generalised eigenvector of
    (Rs - Rn, Rn). For a single talker Rs - Rn is rank one, so this is the
    matched filter Rn^-1 s in disguise."""
    S = Rs - Rn
    vals, vecs = np.linalg.eig(np.linalg.solve(Rn, S))
    w = vecs[:, int(np.argmax(np.real(vals)))]
    return _to_multiplier(w)


def _to_multiplier(w):
    w0, w1 = complex(w[0]), complex(w[1])
    if abs(w0) < abs(w1) / WEIGHT_MAX_ABS:
        # The fit wants (almost) channel B alone: cap at the allowed ratio,
        # keeping the phase it asked for.
        ang = np.angle(np.conj(w1) * w0) if abs(w0) > 0 else 0.0
        return WEIGHT_MAX_ABS * np.exp(1j * ang)
    m = complex(np.conj(w1 / w0))
    if abs(m) > WEIGHT_MAX_ABS:
        m = m / abs(m) * WEIGHT_MAX_ABS
    return m


class Tracker:
    """Estimates the passband covariances and refits the weight on a clock.

    Feed it the two channels of the demodulator's passband, block by block,
    at rate_hz. It decides for itself whether a block is "talking" (power
    above the tracked floor) and books it into the signal covariance, or
    quiet and books it into the noise covariance. Every refresh_s it refits
    the weight for the requested mode, and adopts the new weight only if it
    predicts at least REFIT_MIN_GAIN_DB of improvement over the current one.
    """

    def __init__(self, rate_hz, refresh_s=0.25, noise_tc_s=2.0, signal_tc_s=0.3,
                 floor_rise_s=8.0):
        self.rate_hz = float(rate_hz)
        self.refresh_s = float(refresh_s)
        self.noise_tc_s = float(noise_tc_s)
        self.signal_tc_s = float(signal_tc_s)
        self.floor_rise_s = float(floor_rise_s)
        self.Rn = None
        self.Rs = None
        self.floor = None
        self.m = 0j
        self.updates = 0
        self.talking = False
        self._since_fit = 0.0

    def _alpha(self, n, tc_s):
        """EMA coefficient for a block of n samples against a time constant."""
        return 1.0 - math.exp(-n / (self.rate_hz * tc_s)) if tc_s > 0 else 1.0

    def update(self, a, b, mode):
        n = min(len(a), len(b))
        if n == 0:
            return
        X = np.vstack([np.asarray(a[:n], dtype=np.complex128),
                       np.asarray(b[:n], dtype=np.complex128)])
        R = (X @ X.conj().T) / n
        p = float(np.real(np.trace(R))) / 2.0
        if self.floor is None:
            self.floor = p
        elif p < self.floor:
            self.floor += 0.3 * (p - self.floor)          # fast down
        else:
            self.floor += self._alpha(n, self.floor_rise_s) * (p - self.floor)
        self.talking = p > VAD_RATIO * self.floor
        if self.talking:
            al = self._alpha(n, self.signal_tc_s)
            self.Rs = R if self.Rs is None else (1 - al) * self.Rs + al * R
        else:
            al = self._alpha(n, self.noise_tc_s)
            self.Rn = R if self.Rn is None else (1 - al) * self.Rn + al * R
        self._since_fit += n / self.rate_hz
        if mode in ("null", "track") and self._since_fit >= self.refresh_s:
            self._since_fit = 0.0
            self.refit(mode)

    def _loaded_noise(self):
        Rn = self.Rn
        load = 1e-3 * float(np.real(np.trace(Rn))) / 2.0 + 1e-30
        return Rn + load * np.eye(2)

    def refit(self, mode):
        """Fit m for the mode; adopt it only if it is a real improvement."""
        if self.Rn is None:
            return False
        Rn = self._loaded_noise()
        if mode == "null":
            cand = fit_null(Rn)
            gain = _out_noise(self.m, Rn) / max(_out_noise(cand, Rn), 1e-30)
            gain_db = 10.0 * math.log10(max(gain, 1e-30))
        elif mode == "track":
            if self.Rs is None:
                return False
            cand = fit_max_snr(self.Rs, Rn)
            s_new = _snr_of(cand, self.Rs, Rn)
            s_cur = _snr_of(self.m, self.Rs, Rn)
            gain_db = 10.0 * math.log10(max(s_new, 1e-30) / max(s_cur, 1e-30))
        else:
            return False
        if gain_db >= REFIT_MIN_GAIN_DB:
            self.m = cand
            self.updates += 1
            return True
        return False

    def snr_db(self, m=None):
        """{"a", "b", "out"} SNRs in dB from the current covariances, or Nones."""
        if self.Rn is None or self.Rs is None:
            return {"a": None, "b": None, "out": None}
        Rn = self._loaded_noise()
        m = self.m if m is None else m

        def db(x):
            return round(10.0 * math.log10(x), 1) if x > 0 else -100.0
        sa = (float(np.real(self.Rs[0, 0] - Rn[0, 0])) / float(np.real(Rn[0, 0])))
        sb = (float(np.real(self.Rs[1, 1] - Rn[1, 1])) / float(np.real(Rn[1, 1])))
        return {"a": db(max(sa, 0.0)), "b": db(max(sb, 0.0)),
                "out": db(_snr_of(m, self.Rs, Rn))}


def _out_noise(m, Rn):
    """Noise power at combine()'s output for weight m, normalisation included."""
    v = np.array([1.0, m], dtype=np.complex128)
    return float(np.real(v @ Rn @ np.conj(v))) / (1.0 + abs(m) ** 2)
