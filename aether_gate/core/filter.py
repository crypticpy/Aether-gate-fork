#
# Aether-gate — the receive filter: passband, shape, notches, contour, APF,
# auto width, auto EQ, a noise blanker and an AGC with real time constants.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""What a transceiver's IF DSP does, done once here as ONE designed FIR.

The passband filter that follows decimation is a complex one-sided FIR.
Instead of a fixed pair of sideband taps, its frequency response is built
from the operator's settings every time one of them changes: the low and
high edges (both, so shift and width are the same thing as twin PBT), the
shape (SOFT is a short window with gentle skirts and no ringing; SHARP is a
long Kaiser window with skirts a few tens of hertz wide), any notches, a
CONTOUR bump or dip, an APF peak for CW, and the auto-EQ tilt. One
convolution per block whatever is switched on, linear phase, and the
status can say exactly what the filter is doing because the filter IS the
description.

Everything here is numpy only (no scipy on the gate hosts). Frequencies in
the spec are audio hertz as the operator sees them (positive); the design
maps them onto the signed passband of the sideband in use.
"""
import math

import numpy as np

SHAPES = {"soft": 255, "sharp": 1023}   # soft: 196 Hz transition at 25 kS/s; 127 taps ate
                                        # 400 Hz of a 2.4 k passband and sounded muffled
DESIGN_N = 4096                      # frequency-sampling grid for the design
SPEC_N = 1024                        # analysis FFT (24 Hz/bin at 25 kS/s)
SPEC_TC_S = 1.0
SPEC_WARMUP_BLOCKS = 30              # ~1 s of averaging before the automatics read it
AUTO_GAP_BINS = 4                    # a formant gap this wide does not end the occupied run
MAX_NOTCHES = 6
NOTCH_MIN_WIDTH_HZ = 100.0             # narrower than the window's main lobe cannot be deep
NOTCH_DEFAULT_WIDTH_HZ = 140.0
ANF_MAX = 3
ANF_DETECT_DB = 15.0                 # a tone stands this far above the passband median
ANF_RELEASE_DB = 6.0
ANF_WIDTH_HZ = 160.0
AUTO_SEARCH_HZ = 4000.0              # how far from the carrier the auto width looks
AUTO_EDGE_DB = 8.0                   # occupied where the spectrum clears the floor by this
AUTO_LOW_MARGIN_HZ = 50.0
AUTO_HIGH_MARGIN_HZ = 100.0
AUTO_MIN_WIDTH_HZ = 300.0
AUTO_MIN_HIGH_HZ = 2400.0            # AUTO never closes the top below this: a voice's own
                                     # spectrum falls 20 dB by 2 kHz, but the consonants that
                                     # carry intelligibility live above it
EQ_MAX_DB = 6.0
EQ_REFERENCE_TILT_DB = -6.0          # a normally set-up SSB station: highs ~6 dB under the lows
EQ_STRENGTH = 0.5                    # take half the deviation out, never all of it
EQ_LOW_CENTRE_HZ = 550.0             # the print's tilt is 1.5-2.5 k over 300-800 Hz
EQ_HIGH_CENTRE_HZ = 2000.0

AGC_MODES = {                        # attack_ms, decay_ms, hang_ms
    "fast": (2.0, 100.0, 0.0),
    "med": (5.0, 250.0, 250.0),
    "slow": (5.0, 500.0, 500.0),
    "long": (5.0, 2000.0, 1000.0),
    "off": None,
}
AGC_MAX_GAIN = 1000.0                # 60 dB
AGC_THRESHOLD_DB = 20.0              # the floor between words is held this far under the target
AGC_FLOOR_RISE_MS = 8000.0           # the floor tracker follows speech up this slowly


def _kaiser(n, beta):
    return np.kaiser(n, beta)


def _window(shape, n):
    return np.hamming(n) if shape == "soft" else _kaiser(n, 8.0)


def _bell(f, centre, width, order=2):
    """A Gaussian bump, 1 at the centre, 0.5 at +-width/2. order=4 is a
    flat-topped super-Gaussian: for a notch that keeps its floor wider than
    the window's main lobe, which is what makes it deep."""
    w = max(float(width), 1.0) / 2.0
    return np.exp(-math.log(2.0) * ((f - centre) / w) ** order)


class FilterSpec:
    """The operator's settings, in audio hertz. Plain attributes so a copy is
    a copy and the status is a straight read-out."""

    def __init__(self):
        self.low_hz = 100.0
        self.high_hz = 2900.0
        self.shape = "soft"
        self.notches = []                         # [{"hz", "width_hz"}]
        self.anf = False
        self.contour_on = False
        self.contour_hz = 1200.0
        self.contour_db = 0.0
        self.contour_width_hz = 600.0
        self.apf_on = False
        self.apf_hz = 600.0
        self.apf_width_hz = 150.0
        self.auto = False
        self.auto_eq = False
        self.nb_on = False
        self.nb_db = 12.0
        self.agc_mode = "med"
        self.agc_attack_ms, self.agc_decay_ms, self.agc_hang_ms = AGC_MODES["med"]
        self.agc_threshold_db = AGC_THRESHOLD_DB

    def copy(self):
        c = FilterSpec()
        c.__dict__.update(self.__dict__)
        c.notches = [dict(n) for n in self.notches]
        return c


def design_taps(rate_hz, low_hz, high_hz, shape="soft", notches=(), contour=None,
                apf=None, tilt_db=0.0):
    """Complex FIR taps for a passband from low_hz to high_hz (signed,
    relative to the carrier). notches: (hz, width) in the same signed sense;
    contour: (hz, db, width); apf: (hz, width); tilt_db: the print's tilt,
    which the response leans against."""
    n = SHAPES.get(shape, SHAPES["soft"])
    lo, hi = (float(low_hz), float(high_hz)) if low_hz <= high_hz else (float(high_hz), float(low_hz))
    f = np.fft.fftfreq(DESIGN_N, 1.0 / rate_hz)
    H = ((f >= lo) & (f <= hi)).astype(np.float64)
    if apf is not None:
        H = H * _bell(f, apf[0], apf[1])
    if contour is not None and contour[1]:
        H = H * 10 ** (contour[1] / 20.0 * _bell(f, contour[0], contour[2]))
    if tilt_db:
        span = EQ_HIGH_CENTRE_HZ - EQ_LOW_CENTRE_HZ
        g = -float(tilt_db) * (np.abs(f) - (EQ_LOW_CENTRE_HZ + EQ_HIGH_CENTRE_HZ) / 2.0) / span
        H = H * 10 ** (np.clip(g, -EQ_MAX_DB, EQ_MAX_DB) / 20.0)
    for hz, width in notches:
        H = H * (1.0 - _bell(f, hz, width, order=4))
    win = _window(shape, n)
    h = np.roll(np.fft.ifft(H), n // 2)[:n] * win
    # Unity in the passband. The scale comes from a PLAIN design (passband and
    # APF only, same window) read at its own centre, so a contour, a tilt or
    # a notch shapes the response relative to 0 dB rather than renormalising
    # it -- and an APF's peak, not the empty passband centre, is the 0 dB.
    plain = ((f >= lo) & (f <= hi)).astype(np.float64)
    ref_hz = (lo + hi) / 2.0
    if apf is not None:
        plain = plain * _bell(f, apf[0], apf[1])
        ref_hz = apf[0]
    hp = np.roll(np.fft.ifft(plain), n // 2)[:n] * win
    ref = np.exp(-2j * np.pi * ref_hz / rate_hz * (np.arange(n) - (n - 1) / 2.0))
    gain = abs(np.sum(hp * ref))
    return (h / gain if gain > 0 else h).astype(np.complex128)


def response_at(taps, rate_hz, hz):
    """|H| in dB of complex taps at one signed frequency."""
    k = np.arange(len(taps)) - (len(taps) - 1) / 2.0
    return float(20 * np.log10(abs(np.sum(taps * np.exp(-2j * np.pi * hz / rate_hz * k))) + 1e-12))


def blank_impulses(x, db, hold=4):
    """Zero the samples whose envelope stands `db` above the block's median,
    and `hold` samples either side of each. Returns (x, fraction blanked)."""
    env = np.abs(x)
    med = float(np.median(env[::8])) if len(env) >= 8 else float(np.median(env)) if len(env) else 0.0
    if med <= 0:
        return x, 0.0
    mask = env > med * 10 ** (db / 20.0)
    if not mask.any():
        return x, 0.0
    if hold > 0:
        mask = np.convolve(mask.astype(np.float64), np.ones(2 * hold + 1), mode="same") > 0
    out = x.copy()
    out[mask] = 0
    return out, float(mask.mean())


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


class SliceFilter:
    """The passband filter of one slice, with the analysis that drives its
    automatic parts. `apply(sig, ch)` filters one channel's decimated IQ;
    channel 0 also feeds the spectrum the auto width and auto notch read."""

    def __init__(self, rate_hz, spec=None, print_source=None):
        self.rate_hz = float(rate_hz)
        self.spec = spec if spec is not None else FilterSpec()
        self.print_source = print_source          # () -> voice print dict or None
        self.agc = Agc()
        self.lsb = False
        self.taps = None
        self.state = {}
        self.dirty = True
        self.spec_db = None                       # EMA power spectrum, dB, SPEC_N bins
        self.spec_f = np.fft.fftfreq(SPEC_N, 1.0 / self.rate_hz)
        self.auto_low = None
        self.auto_high = None
        self.auto_source = None
        self.anf_found = []                       # [(signed hz, width)]
        self.eq_tilt_db = 0.0                     # the tilt measured (print or spectrum)
        self.eq_lean_db = 0.0                     # the correction in the taps
        self.blanked_pct = 0.0
        self._blocks = 0

    # ----- settings -------------------------------------------------------
    def set(self, **kw):
        s = self.spec.copy()                      # a rejected value leaves nothing behind
        for k, v in kw.items():
            if v is None:
                continue
            if k in ("low", "low_hz"):
                s.low_hz = float(v)
            elif k in ("high", "high_hz"):
                s.high_hz = float(v)
            elif k == "shape":
                if v not in SHAPES:
                    raise ValueError(f"shape must be one of {sorted(SHAPES)}")
                s.shape = v
            elif k == "anf":
                s.anf = bool(v)
                if not s.anf:
                    self.anf_found = []
            elif k == "contour":
                s.contour_on = bool(v)
            elif k == "contour_hz":
                s.contour_hz = float(v)
            elif k == "contour_db":
                s.contour_db = max(-20.0, min(20.0, float(v)))
            elif k == "contour_width":
                s.contour_width_hz = max(50.0, float(v))
            elif k == "apf":
                s.apf_on = bool(v)
            elif k == "apf_hz":
                s.apf_hz = float(v)
            elif k == "apf_width":
                s.apf_width_hz = max(30.0, float(v))
            elif k == "auto":
                s.auto = bool(v)
                if not s.auto:
                    self.auto_low = self.auto_high = self.auto_source = None
            elif k == "auto_eq":
                s.auto_eq = bool(v)
                if not s.auto_eq:
                    self.eq_tilt_db = self.eq_lean_db = 0.0
            elif k == "nb":
                s.nb_on = bool(v)
            elif k == "nb_db":
                v = float(v)
                if not (0.0 <= v <= 40.0):
                    raise ValueError("nb_db must be 0..40")
                s.nb_db = v
            elif k == "agc":
                self.agc.set(mode=v)
                s.agc_mode = self.agc.mode
                s.agc_attack_ms, s.agc_decay_ms, s.agc_hang_ms = \
                    self.agc.attack_ms, self.agc.decay_ms, self.agc.hang_ms
            elif k in ("attack_ms", "decay_ms", "hang_ms", "threshold_db"):
                self.agc.set(**{k: v})
                setattr(s, "agc_" + k, getattr(self.agc, k))
            else:
                raise ValueError(f"unknown filter setting {k!r}")
        if s.high_hz - s.low_hz < 50.0 and s.low_hz - s.high_hz < 50.0:
            raise ValueError("passband must be at least 50 Hz wide")
        self.spec = s
        self.dirty = True

    def notch_add(self, hz, width_hz=NOTCH_DEFAULT_WIDTH_HZ):
        if len(self.spec.notches) >= MAX_NOTCHES:
            raise ValueError(f"at most {MAX_NOTCHES} notches")
        self.spec.notches.append({"hz": float(hz),
                                  "width_hz": max(NOTCH_MIN_WIDTH_HZ, float(width_hz))})
        self.dirty = True

    def notch_clear(self, hz=None):
        if hz is None:
            self.spec.notches = []
        else:
            self.spec.notches = [n for n in self.spec.notches if abs(n["hz"] - float(hz)) > 1.0]
        self.dirty = True

    # ----- the sideband's sign ------------------------------------------
    def _sign(self):
        return -1.0 if self.lsb else 1.0

    def edges(self):
        """Signed passband edges in use (auto overrides the spec's)."""
        s = self.spec
        lo, hi = (s.low_hz, s.high_hz) if s.low_hz <= s.high_hz else (s.high_hz, s.low_hz)
        if s.auto and self.auto_low is not None:
            lo, hi = self.auto_low, self.auto_high
        sgn = self._sign()
        if (lo + hi) * sgn < 0:                   # the spec came signed for the other sideband
            lo, hi = -hi, -lo
        elif lo >= 0 and sgn < 0:
            lo, hi = -hi, -lo
        return lo, hi

    def audio_edges(self):
        lo, hi = self.edges()
        return (abs(hi), abs(lo)) if hi < 0 else (lo, hi)

    # ----- design ---------------------------------------------------------
    def _redesign(self):
        s = self.spec
        lo, hi = self.edges()
        sgn = self._sign()
        notches = [(sgn * n["hz"], n["width_hz"]) for n in s.notches]
        notches += [(hz, w) for hz, w in self.anf_found]
        contour = (sgn * s.contour_hz, s.contour_db, s.contour_width_hz) if s.contour_on else None
        apf = (sgn * s.apf_hz, s.apf_width_hz) if s.apf_on else None
        taps = design_taps(self.rate_hz, lo, hi, s.shape, notches, contour, apf,
                           self.eq_lean_db if s.auto_eq else 0.0)
        if self.taps is None or len(taps) != len(self.taps):
            self.state = {}
        self.taps = taps
        self.dirty = False

    # ----- per block ------------------------------------------------------
    def apply(self, sig, ch=0, lsb=False):
        if lsb != self.lsb:
            self.lsb = lsb
            self.dirty = True
        if ch == 0 and len(sig):
            self._observe(sig)
        if self.dirty or self.taps is None:
            self._redesign()
        n = len(self.taps)
        st = self.state.get(ch)
        if st is None:
            st = np.zeros(n - 1, dtype=np.complex128)
        x = np.concatenate([st, sig])
        y = np.convolve(x, self.taps, mode="valid")
        self.state[ch] = x[len(x) - (n - 1):]
        return y

    def _observe(self, sig):
        m = min(len(sig), SPEC_N)
        x = sig[-m:] * np.hanning(m)
        X = np.fft.fft(x, SPEC_N)
        p = 10 * np.log10(np.abs(X) ** 2 / m + 1e-30)
        if self.spec_db is None:
            self.spec_db = p
        else:
            al = 1.0 - math.exp(-m / self.rate_hz / SPEC_TC_S)
            self.spec_db += al * (p - self.spec_db)
        self._blocks += 1
        if self._blocks % 4:
            return
        if self.spec.auto:
            self._auto_width()
        if self.spec.anf:
            self._auto_notch()
        if self.spec.auto_eq:
            self._auto_eq()

    def _print(self):
        try:
            return self.print_source() if self.print_source else None
        except Exception:
            return None

    def _auto_width(self):
        if self._blocks < SPEC_WARMUP_BLOCKS:
            return
        sgn = self._sign()
        # The occupied run of the 1 s spectrum around its peak, read against
        # the noise floor: where the station actually reaches. This is the
        # fit. The voice print's edges are its -20 dB points -- a fingerprint,
        # stable across signal strength, and 400-800 Hz INSIDE where a voice
        # still carries its consonants -- so the print may only ever WIDEN the
        # fit (a quiet over must not close the door on the next loud one),
        # never narrow it, and the top never closes below AUTO_MIN_HIGH_HZ.
        f = self.spec_f * sgn
        sel = (f >= 0) & (f <= AUTO_SEARCH_HZ)
        d = self.spec_db[sel]
        fr = f[sel]
        floor = float(np.percentile(d, 10))
        if float(d.max()) < floor + AUTO_EDGE_DB + 2.0:
            return                                # nobody there: hold the edges
        # the occupied run around the peak, not the first and last bin
        # anywhere above the line: one noise bin at 3.9 kHz is not a voice
        occ = d > floor + AUTO_EDGE_DB
        peak = int(np.argmax(d))
        i0 = i1 = peak
        while i0 > 0 and occ[max(0, i0 - AUTO_GAP_BINS - 1):i0].any():
            i0 -= 1
        while i1 < len(d) - 1 and occ[i1 + 1:i1 + AUTO_GAP_BINS + 2].any():
            i1 += 1
        lo = max(AUTO_LOW_MARGIN_HZ, float(fr[i0]) - AUTO_LOW_MARGIN_HZ)
        hi = float(fr[i1]) + AUTO_HIGH_MARGIN_HZ
        source = "spectrum"
        pr = self._print()
        if pr and pr.get("low_hz") is not None and pr.get("high_hz") is not None:
            lo = min(lo, max(AUTO_LOW_MARGIN_HZ, float(pr["low_hz"]) - AUTO_LOW_MARGIN_HZ))
            hi = max(hi, float(pr["high_hz"]) + AUTO_HIGH_MARGIN_HZ)
            source = "print"
        hi = max(hi, AUTO_MIN_HIGH_HZ)
        if hi - lo < AUTO_MIN_WIDTH_HZ:
            hi = lo + AUTO_MIN_WIDTH_HZ
        if self.auto_low is None:
            new_lo, new_hi = lo, hi
        else:
            # widen quickly, narrow slowly: a syllable that reaches further
            # must not be clipped while one quiet over must not close the door
            new_lo = self.auto_low + (0.5 if lo < self.auto_low else 0.05) * (lo - self.auto_low)
            new_hi = self.auto_high + (0.5 if hi > self.auto_high else 0.05) * (hi - self.auto_high)
        if (self.auto_low is None or abs(new_lo - self.auto_low) > 25.0
                or abs(new_hi - self.auto_high) > 25.0):
            self.dirty = True
        self.auto_low, self.auto_high, self.auto_source = new_lo, new_hi, source

    def _auto_notch(self):
        lo, hi = self.edges()
        f = self.spec_f
        sel = (f >= min(lo, hi)) & (f <= max(lo, hi))
        if sel.sum() < 8:
            return
        d = self.spec_db[sel]
        fr = f[sel]
        med = float(np.median(d))
        keep = []
        for hz, w in self.anf_found:              # release a tone that has gone
            i = int(np.argmin(np.abs(fr - hz)))
            if d[i] > med + ANF_RELEASE_DB:
                keep.append((hz, w))
        found = list(keep)
        order = np.argsort(d)[::-1]
        for i in order:
            if len(found) >= ANF_MAX or d[i] < med + ANF_DETECT_DB:
                break
            if i < 2 or i >= len(d) - 2:
                continue
            side = max(d[i - 2], d[i + 2])
            if d[i] - side < 8.0:                 # broad: a voice formant, not a tone
                continue
            hz = float(fr[i])
            if all(abs(hz - h) > ANF_WIDTH_HZ for h, _w in found):
                found.append((hz, ANF_WIDTH_HZ))
        if [h for h, _ in found] != [h for h, _ in self.anf_found]:
            self.anf_found = found
            self.dirty = True

    def _auto_eq(self):
        pr = self._print()
        tilt = pr.get("tilt_db") if pr else None
        if tilt is None:
            f = np.abs(self.spec_f)
            lo_b = (f >= 300) & (f <= 800)
            hi_b = (f >= 1500) & (f <= 2500)
            lo_p, hi_p = float(np.mean(self.spec_db[lo_b])), float(np.mean(self.spec_db[hi_b]))
            tilt = hi_p - lo_p
        # A voice is SUPPOSED to tilt down: flattening it to 0 dB/octave puts
        # 6-10 dB on the highs of a normal station and it comes out thin and
        # hissy. Lean against the deviation from a normal station's tilt, and
        # only half of it, so a bassy station gets a little top and a tinny
        # one a little bottom.
        tilt = float(tilt)
        lean = max(-EQ_MAX_DB, min(EQ_MAX_DB, EQ_STRENGTH * (tilt - EQ_REFERENCE_TILT_DB)))
        if abs(lean - self.eq_lean_db) > 0.5:
            self.eq_tilt_db = tilt
            self.eq_lean_db = lean
            self.dirty = True

    # ----- reporting ------------------------------------------------------
    def status(self):
        s = self.spec
        lo, hi = self.audio_edges()
        sgn = self._sign()
        if self.taps is None:
            self._redesign()
        notches = [dict(n, depth_db=round(-response_at(self.taps, self.rate_hz, sgn * n["hz"]), 1))
                   for n in s.notches]
        return {
            "low_hz": round(lo), "high_hz": round(hi), "width_hz": round(hi - lo),
            "set_low_hz": round(min(abs(s.low_hz), abs(s.high_hz))),
            "set_high_hz": round(max(abs(s.low_hz), abs(s.high_hz))),
            "shape": s.shape, "taps": SHAPES[s.shape],
            "transition_hz": round(2.0 * self.rate_hz / SHAPES[s.shape]),
            "sideband": "lsb" if self.lsb else "usb",
            "notches": notches,
            "anf": {"enabled": s.anf,
                    "found_hz": [round(abs(hz)) for hz, _w in self.anf_found],
                    "depth_db": [round(-response_at(self.taps, self.rate_hz, hz), 1)
                                 for hz, _w in self.anf_found]},
            "contour": {"enabled": s.contour_on, "hz": s.contour_hz, "db": s.contour_db,
                        "width_hz": s.contour_width_hz},
            "apf": {"enabled": s.apf_on, "hz": s.apf_hz, "width_hz": s.apf_width_hz},
            "auto": {"enabled": s.auto, "source": self.auto_source,
                     "low_hz": round(self.auto_low) if self.auto_low is not None else None,
                     "high_hz": round(self.auto_high) if self.auto_high is not None else None},
            "auto_eq": {"enabled": s.auto_eq, "tilt_db": round(self.eq_tilt_db, 1),
                        "lean_db": round(self.eq_lean_db, 1)},
            "nb": {"enabled": s.nb_on, "threshold_db": s.nb_db,
                   "blanked_pct": round(self.blanked_pct, 2)},
            "agc": self.agc.status(),
            "roofing": {"analogue_hz": None, "digital_hz": round(self.rate_hz)},
            "_sign": sgn,
        }

    def spectrum_db(self, points=128):
        """What is arriving ahead of the filter, on response_db's grid: the
        1 s spectrum the auto width and the ANF read, in dB below its peak,
        with the floor (its median) on the same scale. None until heard."""
        if self.spec_db is None:
            return None
        lo, hi = self.audio_edges()
        f_audio = np.linspace(0.0, max(hi + 500.0, 3500.0), points)
        f = self._sign() * f_audio
        order = np.argsort(self.spec_f)
        p = np.interp(f, self.spec_f[order], self.spec_db[order])
        peak = float(np.max(p))
        floor = float(np.median(p))
        return {"hz": [round(x) for x in f_audio],
                "db": [round(max(float(x) - peak, -120.0), 1) for x in p],
                "floor_db": round(floor - peak, 1)}

    def response_db(self, points=128):
        """The designed response across the audio band, for a picture."""
        if self.taps is None:
            self._redesign()
        lo, hi = self.audio_edges()
        f_audio = np.linspace(0.0, max(hi + 500.0, 3500.0), points)
        f = self._sign() * f_audio
        return {"hz": [round(x) for x in f_audio],
                "db": [round(max(response_at(self.taps, self.rate_hz, fx), -120.0), 1) for fx in f]}
