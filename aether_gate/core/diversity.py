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

  Tracker             Keeps a 2x2 noise covariance (from the empty bins just
                      outside the filter, and from the passband between
                      overs) and a 2x2 signal covariance (while someone is
                      talking), and refits m every refresh interval: the
                      smallest eigenvector of the noise covariance for
                      "null", the generalised eigenvector that maximises SNR
                      for "track". Both are closed form on a 2x2, so a refit
                      costs microseconds and can follow a new talker within
                      a syllable; a TalkerMemory makes a known talker
                      instant.

  blank_impulses      A two-channel noise blanker for the raw stream.
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

# |m| is capped at 20 dB either way so a fit can never hand the output to one
# channel alone by a large factor. The cap is not what protects a dead or
# disconnected channel B: that is NULL_MIN_COHERENCE below.
WEIGHT_MAX_ABS = 10.0

# A null is adopted only when the two channels' noise is this coherent
# (|Rn01| / sqrt(Rn00 Rn11)). Uncorrelated noise has nothing to null, and the
# smallest eigenvector of a near-diagonal Rn is simply the quieter channel:
# with tuner 2 unplugged it would put 99 % of the output on the dead input.
NULL_MIN_COHERENCE = 0.3

# A refit replaces the current weight only if it predicts at least this
# much improvement, so the weight does not chatter on estimation noise.
# The first fit of an over is cheap to accept (it is what makes a new
# talker adopted within a syllable); re-steering DURING an over needs a
# real margin, because a weak signal's 0.3 s covariance is noisy enough
# that candidates a few dB apart in predicted SNR are the same beam
# (seen live: a steady rag-chew re-steered every few syllables at 0.3 dB).
REFIT_MIN_GAIN_DB = 0.3
REFIT_HOLD_GAIN_DB = 1.0

# Block power this far above the tracked floor counts as "someone talking".
# Deliberately low: a weak DX signal only lifts the passband a couple of dB,
# and the block mean of ~1000 noise samples wanders by ~3%, so 1.5 is still
# far outside the noise's own scatter.
VAD_RATIO = 1.5                     # ~ +1.8 dB

# A quiet block is booked into the noise covariance only if its power is
# within this ratio of the noise power already learned (the mean of Rn's
# diagonal), so a signal too weak to trip the VAD is booked nowhere rather
# than learned as "noise" and then nulled: the classic signal-cancellation
# failure of adaptive nulling, which bites at exactly the SNR where
# diversity is worth having. Measured against the learned mean, not the
# tracked floor, so it does not depend on how much block powers scatter.
NOISE_MAX_RATIO = 1.15              # ~ +0.6 dB

# The signal covariance only learns from talking that has lasted this long.
# A static crash is many dB above the floor for a few milliseconds; booking
# it would steer the beam at the lightning instead of the operator.
TALK_HOLD_S = 0.05
# ...and, once an over is established, a dip shorter than this (the gap
# between words, up to half a second in unhurried speech) does not end it:
# the hold is not restarted and the talker is not "new" again. A net's
# turnaround between two operators is comfortably longer. Before an over is
# established the gap allowed is only the hold itself, so a train of static
# crashes 200 ms apart never adds up to one (each is 10 ms above the floor).
TALK_HANG_S = 0.6
TALK_ONSET_HANG_S = TALK_HOLD_S
# The first fit of an over waits for this much voiced time: on a weak
# signal a quarter-second covariance steers somewhere random (seen live),
# and the memory recall at onset already covers the known talkers.
FIRST_FIT_MIN_TALK_S = 0.4
# Fade guard: when, block after block for FADE_HOLD_S of voice, the better
# antenna alone beats the output by FADE_DROP_DB on the block's own
# covariance, switch to that antenna at once, without the hold margin. QSB
# on the two antennas is what diversity is for; waiting for a smoothed
# estimate to notice a 20 dB fade takes several time constants (seen live:
# 8 dB under antenna A for three seconds). Per-block SNRs are noisy at
# low SNR, but a 3 dB margin sustained over every block of 0.2 s is not,
# and falling back to one antenna is a low-regret move.
FADE_DROP_DB = 3.0
FADE_HOLD_S = 0.2


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
    phase_deg, ratio_db = float(phase_deg), float(ratio_db)
    if not (math.isfinite(phase_deg) and math.isfinite(ratio_db)):
        raise ValueError("phase and ratio must be finite")
    mag = min(WEIGHT_MAX_ABS, 10.0 ** (ratio_db / 20.0))
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


# The in-band noise covariance learned between overs is preferred over the
# guard-band one (it alone sees co-channel QRM) while it is this fresh.
RN_INBAND_FRESH_S = 5.0

# A talker's spatial signature is recognised again when the squared cosine
# between its steering vector and the remembered one is at least this:
# 0.75 is a 60 degree phase tolerance for equal-level antennas. Live on a
# two-station QSO the fitted phase of one station scatters by 30-50
# degrees between overs (signal coherence 0.7-0.9, phase drifting ~20
# degrees in ten seconds), and 0.9 (36 degrees) filled all eight slots
# with one station. Stations in a QSO sit 100+ degrees apart.
MEMORY_MATCH = 0.75
MEMORY_MAX = 8
# On a match the remembered signature moves towards the new one by this
# fraction, so a slowly drifting bearing is followed rather than re-added.
MEMORY_MERGE = 0.3

# Only speech-like overs are remembered: the block-power modulation index
# (std/mean over the last second) of speech is well above this, a steady
# carrier's or a noise burst's is not.
SPEECH_MIN_MOD = 0.3
MOD_WINDOW_S = 1.0
# STEADY IN-BAND ENERGY IS NOISE. A carrier, a tone or a digital signal
# parked in the passband keeps the voice detector on for as long as it
# lasts, so the in-band noise estimate never sees it and the beam is fitted
# to it (seen live: a tone broke into a QSO and the tracker had nowhere to
# put it). Once an over has run this long with no speech-like modulation
# it is folded into the in-band noise instead, and the beam nulls it;
# speech keying up on top of it is then a fresh over against that noise.
# The modulation index is taken on the power ABOVE the loaded noise floor,
# so once the carrier is in the floor, speech over it is seen as speech,
# and a held vowel has to stay flat this long before it counts as steady.
STEADY_MIN_S = 1.5
# ...and only overs with this much voiced time behind them: a memory of
# eight slots fills with syllable-long fits of one talker otherwise. An
# over is memorised once it gets here (whatever weight it has settled on)
# and again on any later re-steer.
MEMORY_MIN_TALK_S = 0.5


def combine_ramp(a, b, m0, m1):
    """combine() with the weight gliding linearly from m0 to m1 over the block,
    so a steering change never lands as a step (a click) in the audio."""
    n = min(len(a), len(b))
    if n == 0:
        return a[:0]
    if m0 == m1:
        return combine(a[:n], b[:n], m1)
    m = m0 + (m1 - m0) * (np.arange(n) / n)
    return (a[:n] + m * b[:n]) / np.sqrt(1.0 + np.abs(m) ** 2)


def blank_impulses(a, b, threshold_db, widen=2):
    """A two-channel noise blanker on the raw aligned stream.

    An impulse (lightning, powerline, an ignition) arrives on both antennas,
    so it is detected on the SUM of their powers and blanked on both, which
    keeps the pair coherent. The threshold is relative to the block's median
    power, which a strong broadcast carrier raises only a few dB.
    Returns (a, b, fraction_blanked); the inputs are not modified.
    """
    n = min(len(a), len(b))
    if n == 0:
        return a, b, 0.0
    e = np.abs(a[:n]) ** 2 + np.abs(b[:n]) ** 2
    med = float(np.median(e))
    if med <= 0.0:
        return a, b, 0.0
    hit = e > med * (10.0 ** (float(threshold_db) / 10.0))
    if not hit.any():
        return a, b, 0.0
    if widen > 0:
        idx = np.flatnonzero(hit)
        for d in range(-widen, widen + 1):
            j = idx + d
            hit[j[(j >= 0) & (j < n)]] = True
    a = np.array(a[:n], copy=True); b = np.array(b[:n], copy=True)
    a[hit] = 0; b[hit] = 0
    return a, b, float(hit.mean())


def steering_of(R):
    """Unit principal eigenvector of a (signal) covariance: where it comes from."""
    vals, vecs = np.linalg.eigh(R)
    v = vecs[:, int(np.argmax(vals))]
    ph = np.exp(-1j * np.angle(v[0])) if abs(v[0]) > 0 else 1.0
    return v * ph                                    # first element real, positive


class TalkerMemory:
    """Spatial signatures of recent talkers and the weight that suited each.

    When someone keys up, one block is enough to tell whether they are a
    known voice (a known bearing, really); if so the weight jumps straight
    to what worked last time instead of being re-learned over a refit cycle.
    """

    def __init__(self, max_n=MEMORY_MAX, match=MEMORY_MATCH):
        self.max_n = int(max_n)
        self.match = float(match)
        self.entries = []                    # dicts: id, s, m, hits, first_seen, last_seen, name
        self._next_id = 1                    # ids never reuse within a run
        self.active = None                   # id of the talker whose weight is live
        self.active_since = None

    def _activate(self, e, now):
        if self.active != e["id"]:
            self.active_since = now
        self.active = e["id"]

    def release(self):
        """The over ended: nobody's weight is live."""
        self.active = None
        self.active_since = None

    def recall(self, s, now):
        best, best_c = None, 0.0
        for e in self.entries:
            c = abs(np.vdot(e["s"], s)) ** 2
            if c > best_c:
                best, best_c = e, c
        if best is not None and best_c >= self.match:
            best["hits"] += 1
            best["last_seen"] = now
            self._activate(best, now)
            return best["m"]
        return None

    def store(self, s, m, now):
        for e in self.entries:
            c = np.vdot(e["s"], s)
            if abs(c) ** 2 >= self.match:
                # align the new vector's global phase to the stored one
                # before blending, so the blend cannot cancel
                s_al = s * (np.conj(c) / max(abs(c), 1e-12))
                v = (1.0 - MEMORY_MERGE) * e["s"] + MEMORY_MERGE * s_al
                e["s"] = v / max(np.linalg.norm(v), 1e-12)
                e["m"], e["last_seen"] = m, now
                self._activate(e, now)
                return
        e = {"id": self._next_id, "s": s, "m": m, "hits": 0,
             "first_seen": now, "last_seen": now, "name": None}
        self._next_id += 1
        self.entries.append(e)
        self._activate(e, now)
        if len(self.entries) > self.max_n:
            self.entries.sort(key=lambda e: e["last_seen"])
            dropped = self.entries.pop(0)
            if dropped["id"] == self.active:
                self.release()

    def name(self, talker_id, name):
        """Label an entry; '' or None clears. False when the id is unknown."""
        for e in self.entries:
            if e["id"] == int(talker_id):
                e["name"] = (str(name).strip() or None) if name is not None else None
                return True
        return False

    def clear(self):
        self.entries = []
        self.release()

    def talker(self, now):
        """{"id", "since_s"} for the live talker, or None."""
        if self.active is None:
            return None
        return {"id": int(self.active),
                "since_s": round(max(0.0, now - self.active_since), 1)}

    def status(self, now):
        out = []
        for e in sorted(self.entries, key=lambda e: -e["last_seen"]):
            ph, ra = weight_to_polar(e["m"])
            out.append({"id": int(e["id"]), "name": e["name"],
                        "phase_deg": round(ph, 1), "ratio_db": round(ra, 1),
                        "age_s": round(max(0.0, now - e["last_seen"]), 1),
                        "first_seen_s": round(max(0.0, now - e["first_seen"]), 1),
                        "hits": int(e["hits"])})
        return out


class Tracker:
    """Estimates the passband covariances and refits the weight on a clock.

    Feed it, block by block, two 2x2 covariances of the slice's spectrum:
    R_in over the bins the operator hears, and R_guard over the empty bins
    just outside the filter. The guard band is noise by construction, so
    the noise covariance is known from the first block without waiting for
    a pause and can never contain the wanted signal; a local noise source
    has the same spatial signature a few kHz away as in the passband. The
    in-band covariance learned between overs is preferred while fresh,
    because it alone sees co-channel QRM. "Talking" is in-band power over
    guard power, which makes the VAD independent of the band's level.

    Every refresh_s the weight is refitted for the requested mode and
    adopted only if it predicts at least REFIT_MIN_GAIN_DB of improvement.
    With a TalkerMemory attached, a recognised talker's weight is applied
    in the block they key up in.
    """

    def __init__(self, rate_hz, refresh_s=0.25, noise_tc_s=2.0, signal_tc_s=1.0,
                 memory=None, t0=0.0):
        self.rate_hz = float(rate_hz)
        self.refresh_s = float(refresh_s)
        self.noise_tc_s = float(noise_tc_s)
        self.signal_tc_s = float(signal_tc_s)
        self.memory = memory
        self.Rn_guard = None
        self.Rn_in = None
        self.Rs = None
        self.m = 0j
        self.updates = 0
        self.talking = False
        self.talk_mod = None
        self.steady = False                 # the over is a carrier, not speech
        self._low_mod_s = 0.0               # voiced time with no speech-like modulation
        self.t = float(t0)                  # tracker time: t0 + seconds of signal seen
                                            # (t0 = a shared clock, so several
                                            # trackers can stamp one TalkerMemory)
        self._rn_in_t = -1e9
        self._since_fit = 0.0
        self._talk_s = 0.0
        self._quiet_s = 0.0
        self._onset_done = False
        self._rs_n = 0
        self._rs_half = [None, None]        # Rs from even / odd blocks (see refit)
        self._fade_s = 0.0                  # voiced time the fade guard has held
        self._over_fits = 0                 # refits adopted during this over
        self._memorised = False             # this over has been stored
        self._hist = []                     # (t, p_in) over the last MOD_WINDOW_S

    def _alpha(self, n, tc_s):
        """EMA coefficient for a block of n samples against a time constant."""
        return 1.0 - math.exp(-n / (self.rate_hz * tc_s)) if tc_s > 0 else 1.0

    @property
    def rn_source(self):
        if self.Rn_in is not None and self.t - self._rn_in_t <= RN_INBAND_FRESH_S:
            return "inband"
        return "guard" if self.Rn_guard is not None else None

    @property
    def Rn(self):
        return self.Rn_in if self.rn_source == "inband" else self.Rn_guard

    def update(self, R_in, R_guard, n, mode):
        """One block: in-band and guard-band covariances, n samples of signal."""
        if n <= 0 or R_in is None:
            return
        R_in = np.asarray(R_in, dtype=np.complex128)
        dt = n / self.rate_hz
        self.t += dt
        if R_guard is not None:
            al = self._alpha(n, self.noise_tc_s)
            R_guard = np.asarray(R_guard, dtype=np.complex128)
            self.Rn_guard = R_guard if self.Rn_guard is None else (1 - al) * self.Rn_guard + al * R_guard
        if self.Rn_guard is None:
            return
        p_in = float(np.real(np.trace(R_in))) / 2.0
        p_ref = float(np.real(np.trace(self.Rn_guard))) / 2.0
        self._hist.append((self.t, p_in))
        while self._hist and self._hist[0][0] < self.t - MOD_WINDOW_S:
            self._hist.pop(0)
        self.talking = p_in > VAD_RATIO * p_ref
        if self.talking:
            self._talk_s += dt
            self._quiet_s = 0.0
            ps = np.array([p for _t, p in self._hist])
            if len(ps) >= 4 and ps.mean() > 0:
                p_floor = float(np.real(np.trace(self._loaded_noise()))) / 2.0
                excess = float(ps.mean()) - p_floor
                self.talk_mod = float(ps.std() / excess) if excess > 0.25 * p_floor else 0.0
            else:
                self.talk_mod = None
            low = self.talk_mod is not None and self.talk_mod < SPEECH_MIN_MOD
            self._low_mod_s = self._low_mod_s + dt if low else 0.0
            steady = self._low_mod_s >= STEADY_MIN_S
            if steady:
                if not self.steady:         # see STEADY_MIN_S
                    self.Rs = None
                    self._rs_half = [None, None]
                    self._rs_n = 0
                    self._over_fits = 0
                self.steady = True
                al = self._alpha(n, self.noise_tc_s)
                self.Rn_in = R_in if self.Rn_in is None else (1 - al) * self.Rn_in + al * R_in
                self._rn_in_t = self.t
            else:
                self.steady = False
            if self._talk_s >= TALK_HOLD_S and not steady:
                if not self._onset_done:
                    # A new over: its covariance starts fresh (the previous
                    # talker's is not evidence about this one) and is a
                    # running mean until a time constant's worth has been
                    # seen, so the beam can settle within a few syllables.
                    self._onset_done = True
                    self._rs_n = 0
                    self._rs_half = [None, None]
                    self._fade_s = 0.0
                    self._over_fits = 0
                    self._memorised = False
                    self.Rs = None
                    if self.memory is not None:
                        self.memory.release()   # a new over is nobody until recalled
                    self._recall(R_in, mode)
                self._rs_n += 1
                al = max(self._alpha(n, self.signal_tc_s), 1.0 / self._rs_n)
                self.Rs = R_in if self.Rs is None else (1 - al) * self.Rs + al * R_in
                # the same estimate split by block parity: two independent
                # views of the over, so a fit can be judged on data it did
                # not see
                if mode == "track" and self._over_fits > 0:
                    self._fade_guard(R_in, dt)
                h = self._rs_n % 2
                k = (self._rs_n + 1) // 2
                al_h = max(self._alpha(2 * n, self.signal_tc_s), 1.0 / k)
                old = self._rs_half[h]
                self._rs_half[h] = R_in if old is None else (1 - al_h) * old + al_h * R_in
        else:
            self._quiet_s += dt
            self.steady = False
            if self._quiet_s >= (TALK_HANG_S if self._onset_done else TALK_ONSET_HANG_S):
                self._talk_s = 0.0
                self._low_mod_s = 0.0
                self._onset_done = False
                self.talk_mod = None
                if self.memory is not None:
                    self.memory.release()
            if p_in <= NOISE_MAX_RATIO * p_ref:
                al = self._alpha(n, self.noise_tc_s)
                self.Rn_in = R_in if self.Rn_in is None else (1 - al) * self.Rn_in + al * R_in
                self._rn_in_t = self.t
        self._since_fit += dt
        if mode in ("null", "track") and self._since_fit >= self.refresh_s:
            self._since_fit = 0.0
            # a steady carrier is nulled like any other coherent noise
            self.refit("null" if (mode == "track" and self.steady) else mode)
            if mode == "track" and not self._memorised:
                self._memorise()

    def _recall(self, R_in, mode):
        if self.memory is None or mode != "track":
            return
        Rn = self._loaded_noise()
        S = R_in - Rn
        if float(np.real(np.trace(S))) <= 0:
            return
        m = self.memory.recall(steering_of(S), self.t)
        if m is not None and m != self.m:
            self.m = m
            self.updates += 1

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
            if _coherence(Rn) < NULL_MIN_COHERENCE:
                return False                    # nothing directional to null
            cand = fit_null(Rn)
            gain = _out_noise(self.m, Rn) / max(_out_noise(cand, Rn), 1e-30)
            gain_db = 10.0 * math.log10(max(gain, 1e-30))
            if self.Rs is not None and \
                    _snr_of(cand, self.Rs, Rn) < _snr_of(self.m, self.Rs, Rn):
                return False                    # quieter, but at the signal's expense
        elif mode == "track":
            if self.Rs is None or any(h is None for h in self._rs_half):
                return False
            if self._over_fits == 0 and self._talk_s < FIRST_FIT_MIN_TALK_S:
                return False
            # HELD-OUT SCORING. A max-SNR fit scored on the covariance it was
            # fitted to always looks better than the weight in use, because
            # it has also fitted that covariance's noise; at 3 dB SNR that
            # optimism is several dB, and the beam re-steers on nothing (seen
            # live: six re-steers in 30 s, none of which moved the output
            # SNR). So fit on one parity's estimate and score on the other,
            # alternating; and score each antenna alone the same way, so the
            # output is never left worse than the better antenna.
            fit_on, score_on = (self._rs_half if self.updates % 2 == 0
                                else self._rs_half[::-1])
            cand = fit_max_snr(fit_on, Rn)
            best, s_new = cand, _snr_of(cand, score_on, Rn)
            for alone in (0j, complex(WEIGHT_MAX_ABS)):
                s_alone = _snr_of(alone, score_on, Rn)
                if s_alone > s_new:
                    best, s_new = alone, s_alone
            cand = best
            s_cur = _snr_of(self.m, score_on, Rn)
            gain_db = 10.0 * math.log10(max(s_new, 1e-30) / max(s_cur, 1e-30))
        else:
            return False
        need = REFIT_MIN_GAIN_DB if (mode != "track" or self._over_fits == 0) \
            else REFIT_HOLD_GAIN_DB
        if gain_db >= need:
            self.m = cand
            self.updates += 1
            if mode == "track":
                self._over_fits += 1
                self._memorise()
            return True
        return False

    def _fade_guard(self, R_in, dt):
        """See FADE_DROP_DB: one voiced block's covariance against the noise."""
        Rn = self._loaded_noise()
        cur = _snr_of(self.m, R_in, Rn)
        alone = {a: _snr_of(a, R_in, Rn) for a in (0j, complex(WEIGHT_MAX_ABS))}
        best = max(alone, key=alone.get)
        if best != self.m and alone[best] > cur * 10 ** (FADE_DROP_DB / 10.0):
            self._fade_s += dt
            if self._fade_s >= FADE_HOLD_S:
                self.m = best
                self.updates += 1
                self._fade_s = 0.0
        else:
            self._fade_s = 0.0

    def _memorise(self):
        """Store this over's steering vector and weight, if it is speech and
        has lasted long enough to be worth remembering."""
        # _onset_done, not talking: the over is alive through a syllable's
        # trough, and the refresh tick lands in troughs as often as not
        if self.memory is None or not self._onset_done or self.Rs is None \
                or self._talk_s < MEMORY_MIN_TALK_S \
                or self.talk_mod is None or self.talk_mod < SPEECH_MIN_MOD:
            return
        S = self.Rs - self._loaded_noise()
        if float(np.real(np.trace(S))) > 0:
            self.memory.store(steering_of(S), self.m, self.t)
            self._memorised = True

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


def _coherence(R):
    """|R01| / sqrt(R00 R11): how much of the two channels' power is shared."""
    d = math.sqrt(max(float(np.real(R[0, 0])) * float(np.real(R[1, 1])), 0.0))
    return abs(complex(R[0, 1])) / d if d > 0 else 0.0


def _out_noise(m, Rn):
    """Noise power at combine()'s output for weight m, normalisation included."""
    v = np.array([1.0, m], dtype=np.complex128)
    return float(np.real(v @ Rn @ np.conj(v))) / (1.0 + abs(m) ** 2)
