#
# Aether-gate — the two-microphone coherence post-filter for a combined slice.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""What both loops heard at the talker's phase is the talker; the rest is band.

The combiner spends the pair's one spatial degree of freedom on a beam or a
null and hands up a single stream. It cannot spend it twice. But the two
loops still carry evidence the beam threw away: in every bin, the wanted
station arrives on both of them in step -- at one particular phase, the one
the weight was fitted to -- and the sky noise, the receivers' own noise and
everything arriving from somewhere else does not. So for each bin, ask how
much of the cross-spectrum between the loops points along the talker's
phase, and take that as the signal; the rest of the bin's power is noise:

    S_k = Re{Sab_k * exp(-j * expect)}          the talker, at their phase
    P_k = 0.5 * (Saa_k + Sbb_k)                 everything the loops heard
    N_k = P_k - S_k, or the pause estimate      what they did not share
    G_k = clip(S_k / (S_k + N_k), floor, 1)

which is the Wiener gain of a two-microphone post-filter in the Zelinski /
McCowan spirit, made from spatial evidence rather than a single-channel
guess about what noise sounds like. With N_k = P_k - S_k it reduces to the
plain S_k / P_k the coherence gives you for free; when a PauseGate has seen
the talker breathe, N_k is measured instead of inferred, and a coherent
interferer stops being counted as signal for as long as it is quiet in the
gaps too.

This is a POST-filter, applied to the combined stream and to nothing else:

  * it runs on the block the combiner just produced, with the same two
    aligned loop blocks that made it, over a sqrt-Hann STFT at half-frame
    hop that reconstructs exactly at G = 1 (a gain of one is a wire, to
    within 1e-6, delayed by one frame);
  * OUTSIDE the slice it does not exist. Every bin beyond [lo, hi) is
    multiplied by exactly 1, so the panadapter and the S-meter see the
    band the combiner gave them and only the audio the operator is
    actually listening to is touched;
  * the frame is a TIME, not a sample count. frame_s stays 10 ms from
    62.5 kS/s to 2.04 MS/s, so the FFT grows with the rate and a bin
    stays about 61 Hz wide whatever the driver is doing. It is also 122
    frames a second at every rate, and the work per second of audio is
    therefore n log n in the rate: measured on an M-series Mac, 10 ms of
    CPU per second at 62.5 kS/s (1 % of a core), 15 ms at 125 kS/s, and
    247 ms at 2.04 MS/s (25 % of a core, and several times that on a Pi).
    A 3 kHz slice does not need 2 MHz of FFT; if the top rates ever cost
    too much, the place to run this is the demodulator's decimated
    passband, where the same frame is a few hundred samples.

The one asymmetry worth knowing: the denominator is the MEAN of the two
auto-spectra, so a loop that hears the talker 10 dB quieter than the other
drags G down (|Sab| can only reach sqrt(Saa Sbb), which is under the mean).
That is the honest answer -- half the evidence is half the evidence -- but
it means the gain reads low on a badly unbalanced pair rather than reading
"no talker".
"""
import math

import numpy as np

FRAME_S = 0.010             # the analysis frame, in seconds, at every rate
FLOOR_DB = -12.0            # nothing is ever taken further down than this
TC_S = 0.050                # the cross- and auto-spectra's memory. The short
                            # end of the useful 50-100 ms: every millisecond of
                            # it is lag at a syllable's onset, and lag there is
                            # the gain arriving after the word (measured on a
                            # 4 Hz talker: 100 ms of memory gives up 0.9 dB of
                            # the 3.4 dB dig-out that 50 ms gets)
GAIN_TC_S = 0.030           # the gain's RELEASE, against musical noise; it
                            # rises without smoothing (see _gain)
SMOOTH_BINS = 3
NFFT_MIN = 256              # under this a "bin" is wider than the slice
NFFT_MAX = 32768            # over it the FFT costs more than the audio is worth
SNR_TC_S = 1.0              # how steady the reported SNRs are


def frame_bins(rate_hz, frame_s=FRAME_S):
    """The FFT size for frame_s of signal at rate_hz: the power of two at
    least that long, bounded to [NFFT_MIN, NFFT_MAX]."""
    want = max(1.0, float(rate_hz) * float(frame_s))
    n = 1 << max(0, int(math.ceil(math.log2(want))))
    return int(min(max(n, NFFT_MIN), NFFT_MAX))


def _smooth(v, k=SMOOTH_BINS):
    """Moving average along the bin axis, edge-padded (as core/postfilter)."""
    if k <= 1 or len(v) < k:
        return v
    pad = k // 2
    vp = np.concatenate([np.repeat(v[:1], pad), v, np.repeat(v[-1:], pad)])
    c = np.concatenate([[0.0], np.cumsum(vp)])
    return (c[k:] - c[:-k]) / k


def _db(x, floor=-100.0):
    return round(10.0 * math.log10(x), 1) if x > 0 else floor


class PauseGate:
    """The talker's syllabic gaps, and what the band sounds like inside them.

    Speech is holes: between words, between overs, the wanted signal simply
    is not there, and for those tens of milliseconds the receiver is
    measuring the noise alone. Tracked here as a floor that falls quickly
    onto whatever the quiet frames say and is only allowed to climb at
    FLOOR_RISE_DB_S while somebody talks, so a long over cannot drag the
    threshold up onto itself. A dip counts as a pause once it has lasted
    MIN_GAP_S -- a syllable's trough is not a gap -- and while it lasts the
    per-bin power is averaged into `noise_psd`, which is the noise reference
    the post-filter would otherwise have had to infer.

    A STEADY SIGNAL HAS NO GAPS. Nothing about a level alone distinguishes
    a carrier parked in the passband from a quiet band, so a gap is claimed
    only while the recent PEAK stands MARGIN_DB over the floor: something
    has to be there for its absence to mean anything. The peak is tracked
    the way the floor is and in the opposite direction -- it takes the
    loudest frame at once and forgets it at LEVEL_DECAY_DB_S -- so the two
    together are the band's crest over the last few seconds. Without that
    guard a dead-flat input reads as one long pause and its own power is
    learned as the noise it is supposed to be measured against, which is
    the signal-cancellation trap core/diversity.py's STEADY_MIN_S is
    about, arriving by a different door.

    `hold` is the AGC's copy of the same news, and it is deliberately
    EARLIER than `in_pause`: an AGC that waits 60 ms to be sure has already
    spent that 60 ms winding its gain up into the gap.
    """

    MARGIN_DB = 6.0             # over the floor, and it is not a gap
    MIN_GAP_S = 0.060           # ...and under it for less than this is a syllable
    FLOOR_TC_S = 0.15           # the floor falls with this time constant
    FLOOR_RISE_DB_S = 2.0       # ...and climbs no faster than this, never past e
    LEVEL_DECAY_DB_S = 3.0      # the peak is taken at once and let go this fast
    NOISE_TC_S = 1.0            # the pause spectrum's memory
    WINDOW_S = 10.0             # pause_fraction is over the last this long

    def __init__(self, rate_hz, hop, margin_db=MARGIN_DB, min_gap_s=MIN_GAP_S,
                 noise_tc_s=NOISE_TC_S, window_s=WINDOW_S):
        self.rate_hz = float(rate_hz)
        self.hop_s = float(hop) / self.rate_hz
        self.margin = 10.0 ** (float(margin_db) / 10.0)
        self.margin_db = float(margin_db)
        self.min_gap_s = float(min_gap_s)
        self.window_s = float(window_s)
        self._al_down = 1.0 - math.exp(-self.hop_s / self.FLOOR_TC_S)
        self._rise = 10.0 ** (self.FLOOR_RISE_DB_S * self.hop_s / 10.0)
        self._decay = 10.0 ** (-self.LEVEL_DECAY_DB_S * self.hop_s / 10.0)
        self._al_noise = 1.0 - math.exp(-self.hop_s / float(noise_tc_s))
        self.reset()

    def reset(self):
        self.floor = None               # the quiet the talker keeps falling back to
        self.level = None               # ...and the loudest they have been lately
        self.noise_psd = None           # (n,) the band's own spectrum, learned in gaps
        self.in_pause = False           # a gap, confirmed: >= min_gap_s under the bar
        self.hold = False               # under the bar at all: freeze the AGC now
        self.gaps = 0
        self.t = 0.0
        self._below_s = 0.0
        self._noise_frames = 0
        self._hist = []                 # (t, in_pause) over the last window_s

    def update(self, energy, psd=None):
        """One frame: the combined signal's in-band energy, and the per-bin
        power to learn as noise while the talker is not there. Returns
        in_pause."""
        e = float(energy)
        self.t += self.hop_s
        if self.floor is None or not math.isfinite(self.floor) or self.floor <= 0.0:
            self.floor = self.level = max(e, 1e-30)
        else:
            self.level = max(e, self.level * self._decay)
            if e < self.floor:
                self.floor += self._al_down * (e - self.floor)
            else:
                self.floor = min(self.floor * self._rise, e)
        self.hold = (e <= self.margin * self.floor
                     and self.level > self.margin * self.floor)
        if self.hold:
            self._below_s += self.hop_s
            if not self.in_pause and self._below_s >= self.min_gap_s:
                self.in_pause = True
                self.gaps += 1
        else:
            self._below_s = 0.0
            self.in_pause = False
        if self.in_pause and psd is not None:
            p = np.asarray(psd, dtype=np.float64)
            self._noise_frames += 1
            if self.noise_psd is None or self.noise_psd.shape != p.shape:
                self.noise_psd = p.copy()
            else:
                # a running mean until a time constant's worth of gaps has
                # been seen: gaps are short, and one seed frame is one
                # chi-square sample per bin
                al = max(self._al_noise, 1.0 / self._noise_frames)
                self.noise_psd += al * (p - self.noise_psd)
        self._hist.append((self.t, self.in_pause))
        while self._hist and self._hist[0][0] < self.t - self.window_s:
            self._hist.pop(0)
        return self.in_pause

    @property
    def pause_fraction(self):
        """How much of the last WINDOW_S the talker was not there."""
        if not self._hist:
            return 0.0
        return sum(1 for _t, p in self._hist if p) / len(self._hist)

    def status(self):
        n = self.noise_psd
        return {
            "in_pause": bool(self.in_pause), "hold": bool(self.hold),
            "pause_fraction": round(self.pause_fraction, 2),
            "gaps": int(self.gaps),
            "noise_db": (_db(float(np.mean(n))) if n is not None else None),
            "floor_db": (_db(self.floor) if self.floor else None),
            "dynamic_db": (_db(self.level / self.floor) if self.floor else None),
            "margin_db": self.margin_db,
        }


class CoherencePostFilter:
    """The gain above, over a perfectly-reconstructing STFT of the combined
    block. Feed it the combiner's output and the two aligned loop blocks for
    the same samples; it returns the same number of samples, delayed by one
    frame (`latency_samples`), with the slice band filtered and everything
    else untouched."""

    def __init__(self, rate_hz, slice_lo_hz, slice_hi_hz, frame_s=FRAME_S,
                 floor_db=FLOOR_DB, tc_s=TC_S, gate=None):
        self.rate_hz = float(rate_hz)
        self.frame_s = float(frame_s)
        self.n = frame_bins(self.rate_hz, self.frame_s)
        self.h = self.n // 2
        self.floor_db = float(floor_db)
        self.floor = 10.0 ** (self.floor_db / 20.0)
        self.win = np.sqrt(np.hanning(self.n + 1)[:-1])        # periodic sqrt-Hann
        self.f = np.fft.fftfreq(self.n, 1.0 / self.rate_hz)
        self._al = 1.0 - math.exp(-self.h / self.rate_hz / float(tc_s))
        self._al_g = 1.0 - math.exp(-self.h / self.rate_hz / GAIN_TC_S)
        self._al_snr = 1.0 - math.exp(-self.h / self.rate_hz / SNR_TC_S)
        self.gate = PauseGate(self.rate_hz, self.h) if gate is None else gate
        self.set_band(slice_lo_hz, slice_hi_hz)
        self.reset()

    # --- setup ------------------------------------------------------------
    def set_band(self, lo_hz, hi_hz):
        """The slice, in the block's own baseband Hz. Bins outside it are
        multiplied by one and are never looked at again."""
        self.lo_hz, self.hi_hz = float(lo_hz), float(hi_hz)
        self.band = (self.f >= self.lo_hz) & (self.f < self.hi_hz)
        if not self.band.any():                 # a slice narrower than one bin
            self.band = np.zeros(self.n, dtype=bool)
            self.band[int(np.argmin(np.abs(self.f - 0.5 * (self.lo_hz + self.hi_hz))))] = True

    def reset(self):
        self.Saa = self.Sbb = None
        self.Sab = None
        self.g = np.ones(self.n)
        self.frames = 0
        self.expect_phase = 0.0
        self.gain_mean_db = 0.0
        self.coherence_mean = 0.0
        self._snr = [0.0, 0.0, 0.0, 0.0]        # sig_in, noi_in, sig_out, noi_out
        self._snr_frames = 0
        # half a frame of silence in front of the input, so the very first
        # samples land in the window's ramp instead of the operator's audio;
        # half a frame of silence in the output queue, so a block always has
        # its own length of finished samples waiting for it
        self._in_y = np.zeros(self.h, dtype=np.complex128)
        self._in_a = np.zeros(self.h, dtype=np.complex128)
        self._in_b = np.zeros(self.h, dtype=np.complex128)
        self._ola = np.zeros(self.n, dtype=np.complex128)
        self._out = np.zeros(self.h, dtype=np.complex128)

    @property
    def latency_samples(self):
        """Output sample j is input sample j - latency_samples."""
        return self.n

    @property
    def bin_hz(self):
        return self.rate_hz / self.n

    # --- the audio path ---------------------------------------------------
    def process(self, y, a, b, expect_phase_rad=None):
        """One block: the combined signal and the two aligned loops that made
        it. `expect_phase_rad` is the phase the talker's cross-spectrum is
        expected at -- the configured weight's angle -- or None to follow the
        band's own dominant phase. Returns len(y) samples, delayed."""
        y = np.asarray(y)
        dtype = y.dtype if y.dtype in (np.complex64, np.complex128) else np.complex128
        n_in = min(len(y), len(a), len(b))
        self._in_y = np.concatenate([self._in_y, np.asarray(y[:n_in], dtype=np.complex128)])
        self._in_a = np.concatenate([self._in_a, np.asarray(a[:n_in], dtype=np.complex128)])
        self._in_b = np.concatenate([self._in_b, np.asarray(b[:n_in], dtype=np.complex128)])
        out = [self._out]
        while len(self._in_y) >= self.n:
            Fy = np.fft.fft(self._in_y[:self.n] * self.win)
            Fa = np.fft.fft(self._in_a[:self.n] * self.win)
            Fb = np.fft.fft(self._in_b[:self.n] * self.win)
            self._in_y = self._in_y[self.h:]
            self._in_a = self._in_a[self.h:]
            self._in_b = self._in_b[self.h:]
            g = self._gain(Fy, Fa, Fb, expect_phase_rad)
            self._ola += np.fft.ifft(Fy * g) * self.win
            out.append(self._ola[:self.h].copy())
            self._ola = np.concatenate([self._ola[self.h:],
                                        np.zeros(self.h, dtype=np.complex128)])
        buf = np.concatenate(out)
        self._out = buf[n_in:]
        return buf[:n_in].astype(dtype, copy=False)

    def _gain(self, Fy, Fa, Fb, expect_phase_rad):
        aa = np.abs(Fa) ** 2
        bb = np.abs(Fb) ** 2
        ab = Fa * np.conj(Fb)
        inst = 0.5 * (aa + bb)          # THIS frame, for the gate: an averaged
                                        # spectrum still holds the word that
                                        # just ended and would be learned as
                                        # noise the moment the gap is confirmed
        self.frames += 1
        if self.Saa is None:
            self.Saa, self.Sbb, self.Sab = aa, bb, ab
        else:
            al = max(self._al, 1.0 / self.frames)
            self.Saa += al * (aa - self.Saa)
            self.Sbb += al * (bb - self.Sbb)
            self.Sab += al * (ab - self.Sab)
        band = self.band
        if expect_phase_rad is None:
            # the band's own dominant phase: the cross-spectrum is already
            # averaged, so its in-band sum is the talker while there is one
            s = complex(np.sum(self.Sab[band]))
            if abs(s) > 0:
                self.expect_phase = float(np.angle(s))
        else:
            self.expect_phase = float(expect_phase_rad)
        sig = np.maximum(np.real(self.Sab * np.exp(-1j * self.expect_phase)), 0.0)
        p = 0.5 * (self.Saa + self.Sbb)
        noise = self.gate.noise_psd
        if noise is None or noise.shape != p.shape:
            noise = np.maximum(p - sig, 0.0)
        g = sig / np.maximum(sig + noise, 1e-30)
        g = np.clip(_smooth(g), 0.0, 1.0)
        # UP AT ONCE, DOWN ON THE RELEASE. Smoothing the gain both ways costs
        # the first 50 ms of every word: the bin is still held down from the
        # gap it was in when the talker starts again, and what it takes down
        # is the consonant. Smoothing only the fall keeps the reason for
        # smoothing at all -- a bin that flickers between frames is what
        # musical noise is made of -- and gives back a dB of the dig-out.
        self.g = np.where(g >= self.g, g, self.g + self._al_g * (g - self.g))
        gain = np.where(band, np.clip(self.g, self.floor, 1.0), 1.0)
        self._measure(Fy, gain, p, inst)
        return gain

    def _measure(self, Fy, gain, p, inst):
        """What status() reports: the gain and coherence in the slice, the
        pause gate fed from the combined signal, and the voice-band SNR over
        the pause noise with and without the gain applied."""
        band = self.band
        self.gain_mean_db = float(20.0 * math.log10(max(float(np.mean(gain[band])), 1e-6)))
        den = np.sqrt(np.maximum(self.Saa[band] * self.Sbb[band], 1e-60))
        self.coherence_mean = float(np.mean(np.abs(self.Sab[band]) / den))
        energy = float(np.sum(np.abs(Fy[band]) ** 2))
        self.gate.update(energy, inst)
        n_psd = self.gate.noise_psd
        if n_psd is None or self.gate.in_pause:
            return                       # a pause is not what the SNR is about
        g2 = gain[band] ** 2
        nb = n_psd[band]
        sb = np.maximum(p[band] - nb, 0.0)
        self._snr_frames += 1
        al = max(self._al_snr, 1.0 / self._snr_frames)
        for i, v in enumerate((float(np.sum(sb)), float(np.sum(nb)),
                               float(np.sum(g2 * sb)), float(np.sum(g2 * nb)))):
            self._snr[i] += al * (v - self._snr[i])

    # --- the control port -------------------------------------------------
    def _snr_db(self, sig, noi):
        if self._snr_frames == 0 or noi <= 0.0:
            return None
        return _db(sig / noi)

    def status(self):
        return {
            "nfft": int(self.n), "bin_hz": round(self.bin_hz, 1),
            "frame_s": self.frame_s, "floor_db": self.floor_db,
            "band_hz": [round(self.lo_hz, 1), round(self.hi_hz, 1)],
            "frames": int(self.frames),
            "gain_mean_db": round(self.gain_mean_db, 1),
            "coherence_mean": round(self.coherence_mean, 2),
            "expect_phase_deg": round(math.degrees(self.expect_phase) % 360.0, 1),
            "pause_fraction": round(self.gate.pause_fraction, 2),
            "in_pause": bool(self.gate.in_pause), "hold": bool(self.gate.hold),
            "snr_in_db": self._snr_db(self._snr[0], self._snr[1]),
            "snr_out_db": self._snr_db(self._snr[2], self._snr[3]),
        }
