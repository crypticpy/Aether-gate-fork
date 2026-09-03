#
# Aether-gate — finding a comb: a switch-mode supply's teeth, for SQUEEZE.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""A switch-mode supply, an LED driver, a device clock: local noise with a
clock behind it repeats at the clock's rate, so it shows up as several thin,
steady lines evenly spaced across the band -- a comb. It is one source, so
the pair hears every tooth from the same direction: ONE steering vector
nulls all of them, which is what makes a comb worth its own SQUEEZE target
rather than nulling each tooth by hand.

Two jobs, kept apart:

  CombDetector accumulates channel A's power spectrum, restricted to the
    demod passband, over WINDOW_S (~2 s -- one block is too noisy to
    peak-pick reliably) and then calls fit() once.

  fit() picks the peaks (TOOTH_OVER_DB over a per-segment local median, the
    same running-median-by-segment idea as core.noisebearing.station_mask,
    aimed the opposite way: THERE a loud narrow run is excluded as a
    station, HERE it is exactly what is wanted) and, from at least
    MIN_TEETH of them sharing a common spacing within TOOTH_TOL, fits
    spacing and offset by least squares against their integer multiples.

Both spacing_hz and offset_hz are ABSOLUTE (offset_hz = a tooth's frequency
modulo spacing_hz, in [0, spacing_hz)) so they mean the same thing whatever
the slice is tuned to: teeth_in_band() re-derives which teeth fall in the
CURRENT passband from the slice's own centre frequency every time it is
called, rather than re-detecting after every retune.
"""
import math

import numpy as np

MIN_TEETH = 3
TOOTH_TOL = 0.02              # spacing/offset match, as a fraction of the spacing
TOOTH_OVER_DB = 6.0           # a tooth must stand this far over its segment's median
TOOTH_WIDTH_HZ = 60.0         # bins counted as "at" a tooth, either side of its centre
SEGMENT_HZ = 2_000.0          # the local-floor neighbourhood a peak is judged against
WINDOW_S = 2.0                 # channel A accumulated this long before a fit is tried


def _peaks(level_db, f_abs, over_db=TOOTH_OVER_DB, segment_hz=SEGMENT_HZ):
    """The centre frequency of every run of bins standing over_db over the
    median of its own segment_hz neighbourhood -- f_abs must already be
    ascending. One peak per run (its loudest bin), not one per bin."""
    n = len(level_db)
    if n < 4:
        return np.zeros(0)
    span = float(f_abs[-1] - f_abs[0]) or 1.0
    segs = int(min(n, max(1, round(span / segment_hz))))
    while segs > 1 and n % segs:
        segs -= 1
    med = np.repeat(np.median(level_db.reshape(segs, n // segs), axis=1), n // segs)
    loud = level_db > med + over_db
    edge = np.flatnonzero(np.diff(np.concatenate(([0], loud.astype(np.int8), [0]))))
    out = []
    for lo, hi in zip(edge[::2], edge[1::2]):
        out.append(float(f_abs[lo + int(np.argmax(level_db[lo:hi]))]))
    return np.array(out)


def fit(level_db, f_abs, min_teeth=MIN_TEETH, tol=TOOTH_TOL):
    """(spacing_hz, offset_hz, teeth_abs) from an ascending-frequency power
    spectrum in dB, or None -- "no comb found" -- when fewer than min_teeth
    peaks share a common spacing. offset_hz is in [0, spacing_hz)."""
    peaks = np.sort(_peaks(level_db, f_abs))
    if len(peaks) < min_teeth:
        return None
    gaps = np.diff(peaks)
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return None
    spacing = float(np.median(gaps))
    if spacing <= 0:
        return None
    offset = float(np.median(peaks % spacing))
    resid = np.abs(((peaks - offset + spacing / 2.0) % spacing) - spacing / 2.0)
    kept = peaks[resid <= tol * spacing]
    if len(kept) < min_teeth:
        return None
    # refine by least squares against each kept peak's own integer multiple
    k = np.round((kept - offset) / spacing)
    A = np.stack([k, np.ones_like(k)], axis=1)
    sol, *_ = np.linalg.lstsq(A, kept, rcond=None)
    spacing, offset = float(sol[0]), float(sol[1]) % float(sol[0])
    return spacing, offset, kept.tolist()


def teeth_in_band(spacing_hz, offset_hz, lo_hz, hi_hz, ref_hz):
    """Every tooth's BASEBAND offset (matching Squeeze.hz's own convention)
    that falls inside [lo_hz, hi_hz) of the slice's own passband right now,
    from spacing_hz/offset_hz's absolute frame and the slice's current
    centre frequency ref_hz. Re-derived fresh every call -- no state, so a
    retune needs nothing done to it."""
    lo_abs, hi_abs = ref_hz + lo_hz, ref_hz + hi_hz
    k0 = int(math.floor((lo_abs - offset_hz) / spacing_hz))
    k1 = int(math.ceil((hi_abs - offset_hz) / spacing_hz))
    out = []
    for k in range(k0, k1 + 1):
        abs_hz = k * spacing_hz + offset_hz
        if lo_abs <= abs_hz < hi_abs:
            out.append(abs_hz - ref_hz)
    return out


def teeth_mask(f, teeth_hz, width_hz=TOOTH_WIDTH_HZ):
    """The union of narrow windows around each baseband tooth frequency,
    for pooling their bins into one covariance -- one steering vector fits
    every tooth, being the one source."""
    sel = np.zeros(len(f), dtype=bool)
    half = width_hz / 2.0
    for hz in teeth_hz:
        sel |= np.abs(f - hz) <= half
    return sel


class CombDetector:
    """Channel A's power spectrum, restricted to the passband and summed
    block by block, until WINDOW_S has gone by and a fit can be tried."""

    def __init__(self, window_s=WINDOW_S):
        self.window_s = float(window_s)
        self._acc = None
        self._f = None
        self._t = 0.0

    @property
    def ready(self):
        return self._t >= self.window_s

    def feed(self, Xa, f, dt):
        p = np.abs(np.asarray(Xa)) ** 2
        if self._acc is None or len(self._acc) != len(p):
            self._acc, self._f, self._t = p.copy(), np.asarray(f), 0.0
        else:
            self._acc = self._acc + p
        self._t += float(dt)

    def detect(self, ref_hz):
        """A (spacing_hz, offset_hz) fit against the accumulated spectrum's
        peaks (ref_hz: the slice's own absolute frequency, baseband's DC),
        once ready() -- None ("no comb found") otherwise."""
        order = np.argsort(self._f)
        f_abs = self._f[order] + float(ref_hz)
        level_db = 10.0 * np.log10(np.maximum(self._acc[order], 1e-30))
        found = fit(level_db, f_abs)
        return None if found is None else (found[0], found[1])
