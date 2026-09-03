#
# Aether-gate — SQUEEZE: hold a null on one chosen signal, or one comb.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The tracker (core/diversity.py) steers at whoever is talking, or nulls
whatever coherent noise is loudest; the focus (core/focus.py) pins a
remembered talker and nulls everyone else's over. Neither lets the operator
point at ONE signal by ear and say "null that, whatever else happens" -- an
S9 neighbour 1.5 kHz up whose splatter sits in the passband, or a carrier
that keeps drifting into it, or a switch-mode supply's three thin lines
repeating across the band. That is what this is for.

A Squeeze holds one of two kinds of target:

  "signal"  an offset from the slice centre (like a filter edge or
            `contour_hz`: signed, hertz) and a width.
  "comb"    spacing_hz/offset_hz (core.comb), either given outright or
            auto-detected from ~2 s of channel A; teeth_in_band()
            re-derives which teeth sit in the CURRENT passband every
            refresh, so a retune costs nothing.

Either way, on the tracker's own covariance cadence (~0.25 s) the target's
own steering vector is measured -- the same closed-form principal
eigenvector as `diversity.steering_of`, but summed only over the bins the
target occupies (one region for a signal, the union of every in-band tooth
for a comb -- one source, one steering vector, every tooth) -- from the
same STFT frame the audio thread already builds for the passband/guard
covariances (observe() hands it over; nothing here talks to hardware or
takes a second FFT).

Only LEVEL genuinely refuses a signal target -- it must stand at least
MIN_LEVEL_DB over the floor measured from the OTHER passband bins in the
same frame, or there is nothing there worth a dedicated null (a comb's
teeth were already judged against their own local floor at detection
time, core.comb.TOOTH_OVER_DB, so this is not asked twice). Failing it,
or -- comb only -- finding no comb at all, the squeeze is held OFF
(`held: False`) with a `reason` ("too weak", "outside the passband", or
"no comb found"), and the weight in use is left alone.

COHERENCE does not refuse the target -- it CHOOSES THE TOOL. The two
loops hearing the target's bins in the same direction (|sum of unit
ratios| / bin count >= MIN_COHERENCE, the tracker's own NULL_MIN_COHERENCE
idea, stricter) means a combiner null will hold; not agreeing (a diffuse
or ambiguous source, or two loops that simply disagree) means a null
fitted to it would wander, so the target's own bins get a spectral notch
each in core/filter.py instead (the same machinery the operator's own
notches and the ANF use) -- one source either way, dealt with by whichever
tool actually holds on it. The choice is re-made every refresh with
hysteresis (NULL_ENTER_COHERENCE to switch to the null, back down to
NOTCH_ENTER_COHERENCE before switching away from it) so a target sitting
near the line does not flap tool every 0.25 s; `tool` and `why` report
which is in force and the coherence figure that decided it. `held` stays
True either way -- something is always dealt with once a target is
accepted; only "too weak"/"outside the passband"/"no comb found" mean
nothing is.

Once the null is in force, the steering vector is smoothed with a ~1 s
complex EMA (SMOOTH_TC_S) across refreshes, so it does not wander
syllable to syllable the way a single frame's estimate would; the null
itself, `null_m`, is `focus.null_of` on that smoothed vector -- the same
closed form the focus module uses to null an interferer.

`depth_db` is measured, not assumed, through whichever tool is in force:
under the null, the target bins' summed 2x2 covariance run through the
combiner at the squeeze's null and at whatever weight was in use a moment
before, in the same refresh that measured it; under the notch, the FIR's
own designed response at the target's frequencies (adapters.diversity_squeeze
fills this in -- core.squeeze does not know the filter's taps).
"""
import math

import numpy as np

from . import comb as _comb
from .diversity import steering_of
from .focus import null_of

# The two-channel coherence a target needs before the NULL is the tool --
# stricter than the tracker's NULL_MIN_COHERENCE (0.3): a general "null the
# loudest noise" fit costs little when it is wrong, but a squeeze the
# operator asked for by frequency should not be built on a coin flip. Below
# it the target's own bins are notched instead (see the module docstring).
NULL_ENTER_COHERENCE = 0.5
NOTCH_ENTER_COHERENCE = 0.35             # ...and back below this before giving up on the null
MIN_COHERENCE = NULL_ENTER_COHERENCE     # kept for anything still reading the old name
# ...and how far a SIGNAL target over the rest of the passband it must
# stand, in dB, before there is anything there worth a dedicated null.
MIN_LEVEL_DB = 6.0
DEFAULT_WIDTH_HZ = 300.0
# The smoothed steering vector's time constant: about four refreshes at the
# tracker's default 0.25 s cadence, so a real re-tune of the target still
# catches up within a second or two.
SMOOTH_TC_S = 1.0
# Fewer bins than this in the target span is not a measurement, it is a
# single noisy point; report "outside the passband" rather than guess.
MIN_BINS = 2


def _out_power(m, R):
    """Power at the combiner's output for weight m and covariance R (a SUM
    over bins, not diversity._out_noise's mean-per-bin -- both weights in
    any one call share the same R, so the scale cancels in the ratio)."""
    v = np.array([1.0, m], dtype=np.complex128)
    return float(np.real(v @ R @ np.conj(v))) / (1.0 + abs(m) ** 2)


class Squeeze:
    """The operator's target, its acceptance state, and the measured facts
    reported under `/diversity`'s "squeeze" key. `scope` ("bins" while the
    sub-band refinement is on, else "passband") is set by the caller each
    block -- it says nothing about the target itself, only where the null
    it earns gets applied, and status() reports it either way."""

    def __init__(self, refresh_s=0.25):
        self.refresh_s = float(refresh_s)
        self.target = "signal"          # "signal" | "comb"
        self.hz = None                  # signal target only
        self.width_hz = DEFAULT_WIDTH_HZ
        self.comb_spacing_hz = None
        self.comb_offset_hz = None
        self.teeth_in_band = []
        self.teeth_seen = 0
        self._detector = None           # comb.CombDetector, while auto-detecting
        self.held = False
        self.reason = None
        self.tool = None                # "null" | "notch", while held
        self.why = None
        self.phase_deg = None
        self.ratio_db = None
        self.coherence = None
        self.depth_db = None
        self.since = None
        self.scope = "passband"
        self.s = np.array([1.0, 0j], dtype=np.complex128)   # smoothed, unit, s[0] real >= 0
        self._since_refresh = 1e9        # measure on the first block, not after refresh_s

    @property
    def active(self):
        """A target is configured, whether or not it is currently held."""
        return self.hz is not None or self.target == "comb" and self.since is not None

    @property
    def null_m(self):
        """The weight a held, NULL-tool squeeze wants; 0j (no effect) while
        held on the notch tool instead, or not held at all -- a caller
        should also check `tool == "null"`, not just `held`, before using
        this (see `_dsq().observe`/`subband_squeeze`)."""
        return null_of(self.s) if self.held and self.tool == "null" else 0j

    def set(self, hz, width_hz, now):
        """Pin a new signal target (or move the current one). Measured
        fresh on the very next block -- an operator retuning the target
        should not wait out a stale refresh clock."""
        self.__init__(self.refresh_s)
        self.hz = float(hz)
        if width_hz is not None:
            self.width_hz = max(1.0, float(width_hz))
        self.since = float(now)
        self.reason = "not measured yet"

    def set_comb(self, spacing_hz, offset_hz, now):
        """Pin a comb target outright: spacing_hz/offset_hz already known
        (an operator's own figures, or a previous auto-detect's)."""
        self.__init__(self.refresh_s)
        self.target = "comb"
        self.comb_spacing_hz = max(1.0, float(spacing_hz))
        self.comb_offset_hz = float(offset_hz) % self.comb_spacing_hz
        self.since = float(now)
        self.reason = "not measured yet"

    def set_comb_auto(self, now):
        """squeeze=comb with no spacing/offset given: find one from the
        next ~2 s of channel A (core.comb.CombDetector)."""
        self.__init__(self.refresh_s)
        self.target = "comb"
        self.since = float(now)
        self.reason = "no comb found"
        self._detector = _comb.CombDetector()

    def off(self):
        """Release: the tracker (and the sub-band combiner) go back to
        whatever they would otherwise be doing."""
        self.__init__(self.refresh_s)

    def refresh(self, X, f, lo_hz, hi_hz, dt, m_current, now, ref_hz=0.0):
        """One audio block's STFT (X: (2, n) complex, slice at DC; f: its
        frequency axis, hertz) -- measured at most once every refresh_s.
        `lo_hz`/`hi_hz` are the demod passband's own edges; `ref_hz` is the
        slice's own absolute frequency (baseband's DC), needed only for a
        comb target's absolute spacing/offset; `m_current` is the wideband
        weight in use a moment before this call, the `without` half of
        depth_db."""
        if not self.active:
            return
        if self.target == "comb" and self._detector is not None:
            band = (f >= lo_hz) & (f < hi_hz)
            self._detector.feed(np.asarray(X[0])[band], f[band], dt)
            if not self._detector.ready:
                return
            found = self._detector.detect(ref_hz)
            if found is None:
                self._detector = _comb.CombDetector()   # try again over the next ~2 s
                self._reject("no comb found")
                return
            self._detector = None       # spacing/offset known now: the ~0.25 s cadence from here
            self.comb_spacing_hz, self.comb_offset_hz = found
        self._since_refresh += float(dt)
        if self._since_refresh < self.refresh_s:
            return
        self._since_refresh = 0.0
        if self.target == "comb":
            teeth = _comb.teeth_in_band(self.comb_spacing_hz, self.comb_offset_hz,
                                        lo_hz, hi_hz, ref_hz)
            self.teeth_in_band, self.teeth_seen = teeth, len(teeth)
            if not teeth:
                self._reject("outside the passband")
                return
            bin_hz = abs(float(f[1] - f[0])) if len(f) > 1 else _comb.TOOTH_WIDTH_HZ
            sel = _comb.teeth_mask(f, teeth, max(_comb.TOOTH_WIDTH_HZ, 3.0 * bin_hz))
        else:
            lo_t, hi_t = self.hz - self.width_hz / 2.0, self.hz + self.width_hz / 2.0
            if hi_t <= lo_hz or lo_t >= hi_hz:
                self._reject("outside the passband")
                return
            sel = (f >= lo_t) & (f < hi_t)
        n = int(np.count_nonzero(sel))
        if n < MIN_BINS:
            self._reject("no comb found" if self.target == "comb" else "outside the passband")
            return
        Xa, Xb = np.asarray(X[0])[sel], np.asarray(X[1])[sel]
        Raa = float(np.sum(np.abs(Xa) ** 2))
        Rbb = float(np.sum(np.abs(Xb) ** 2))
        Rab = complex(np.sum(Xa * np.conj(Xb)))
        R = np.array([[Raa, Rab], [np.conj(Rab), Rbb]], dtype=np.complex128)
        coh = abs(Rab) / math.sqrt(max(Raa * Rbb, 1e-30))
        self.coherence = round(float(coh), 3)
        s_new = steering_of(R)
        self.phase_deg = round(math.degrees(np.angle(s_new[1])), 1)
        self.ratio_db = round(20.0 * math.log10(max(abs(s_new[1]), 1e-9)
                                                / max(abs(s_new[0]), 1e-9)), 1)
        if self.target != "comb":
            floor_sel = (f >= lo_hz) & (f < hi_hz) & ~sel
            target_p = (Raa + Rbb) / (2.0 * n)
            if np.count_nonzero(floor_sel) >= MIN_BINS:
                fa = float(np.mean(np.abs(np.asarray(X[0])[floor_sel]) ** 2))
                fb = float(np.mean(np.abs(np.asarray(X[1])[floor_sel]) ** 2))
                floor_p = (fa + fb) / 2.0
            else:
                floor_p = target_p
            level_db = 10.0 * math.log10(max(target_p, 1e-30) / max(floor_p, 1e-30))
            if level_db < MIN_LEVEL_DB:
                self._reject("too weak")
                return
        # COHERENCE DECIDES THE TOOL, not whether the target is held (see
        # the module docstring) -- hysteresis so it does not flap near the
        # line: already null needs a drop below NOTCH_ENTER_COHERENCE to
        # give it up; already notch (or fresh) needs NULL_ENTER_COHERENCE.
        if self.tool == "null":
            self.tool = "notch" if coh < NOTCH_ENTER_COHERENCE else "null"
        else:
            self.tool = "null" if coh >= NULL_ENTER_COHERENCE else "notch"
        al = 1.0 - math.exp(-self.refresh_s / SMOOTH_TC_S)
        self.s = s_new if not self.held else _renorm((1 - al) * self.s + al * s_new)
        self.held, self.reason = True, None
        if self.tool == "null":
            m_null = null_of(self.s)
            self.depth_db = round(10.0 * math.log10(
                max(_out_power(m_current, R), 1e-30) / max(_out_power(m_null, R), 1e-30)), 1)
            self.why = f"coherence {coh:.2f} — nulled in {n} bins"
        else:
            self.depth_db = None       # adapters.diversity_squeeze fills this in from the taps
            word = "teeth" if self.target == "comb" else "bins"
            self.why = f"coherence {coh:.2f} — not one direction; notched {n} {word}"

    def _reject(self, reason):
        self.held, self.reason, self.depth_db = False, reason, None
        self.tool, self.why = None, None

    def status(self):
        return {"hz": (None if self.target != "signal" or self.hz is None else round(self.hz)),
                "width_hz": (round(self.width_hz) if self.target == "signal" else None),
                "held": bool(self.held), "reason": self.reason,
                "tool": self.tool, "why": self.why,
                "phase_deg": self.phase_deg, "ratio_db": self.ratio_db,
                "coherence": self.coherence, "depth_db": self.depth_db,
                "scope": self.scope, "target": self.target,
                "comb": self._comb_status() if self.target == "comb" else None,
                "since": self.since}

    def _comb_status(self):
        return {"spacing_hz": (None if self.comb_spacing_hz is None
                               else round(self.comb_spacing_hz, 1)),
                "offset_hz": (None if self.comb_offset_hz is None
                             else round(self.comb_offset_hz, 1)),
                "teeth_in_band": [round(x, 1) for x in self.teeth_in_band],
                "teeth_seen": int(self.teeth_seen), "coherence": self.coherence}


def _renorm(v):
    n = np.linalg.norm(v)
    if n <= 0:
        return v
    v = v / n
    ph = np.exp(-1j * np.angle(v[0])) if abs(v[0]) > 0 else 1.0
    return v * ph
