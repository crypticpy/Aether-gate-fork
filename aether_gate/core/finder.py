#
# Aether-gate — the band as the pair hears it, live: spatial rows and a finder.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Two things the floor-tracked SpatialMap deliberately refuses to show.

  LiveSpatial   a fast (quarter-second) cross-spectrum per bin: the inter-loop
                phase, coherence and level of whatever is there RIGHT NOW,
                stations included. Decimated to a row for a waterfall whose
                hue is arrival phase, so stations from different directions
                paint in different colours across the whole span and a local
                noise source is one flat colour.

  Finder        where people are talking. Voice on SSB is a ~2.7 kHz patch
                whose energy is modulated at syllable rate (2-8 Hz) with a
                large modulation depth; a carrier is steady, band noise is
                shallow chi-square scatter. Every ~30 ms frame the finder
                keeps both loops' decimated power spectra; every second it
                scores each 2.7 kHz window over the last ~8 s (SNR over the
                band's floor, modulation depth, syllabic share of the
                modulation spectrum) and keeps ten minutes of those scores,
                so the answer to "where was someone in the last ten minutes"
                comes from stored data rather than from sitting on each
                frequency in turn. Every scored window also gets a verdict
                on what it IS (voice, cw, data, carrier, noise) from the
                same frames -- see kinds.py -- because "there is something
                here" and "there is somebody talking here" are different
                pieces of news. Each candidate carries the pair's phase,
                coherence and level ratio there, and the diversity gain the
                pair could earn on it (signal coherent, noise not).

Both are fed from the reader thread with the same windowed FFT frame the
SpatialMap gets; everything else runs on demand from the control port.
"""
import math

import numpy as np

from . import finder_floor as ffloor
from . import finder_report
from . import kinds
# the read side's constants, re-exported: /diversity/finder is assembled in
# finder_report.py and this is still the module everything imports it from
from .finder_report import (CANDIDATE_MAX, CANDIDATE_MIN_S, CANDIDATE_RECENT_S,
                            DIAL_GRID_HZ, EDGE_MARGIN_HZ, USB_ABOVE_HZ, VOICE_SCORE)

SPATIAL_TC_S = 0.25
SPATIAL_POINTS = 512

FAST_FRAMES = 256            # slots of SLOT_S: ~8.5 s of modulation history
SLOT_S = 0.030               # frames are averaged into slots at least this
                             # long before the modulation analysis sees them.
                             # The reader hands the finder one frame per raw
                             # block, so frames arrive at rate/CHUNK a second
                             # -- 30 a second at 125 kS/s, but 500 a second at
                             # 2.04 MS/s, where 256 raw frames would be half a
                             # second of history and SYLLABIC_HZ (2-8 Hz) would
                             # have no resolution to be measured in at all. A
                             # slot is a frame at 62.5 k and 125 kS/s (32.8 ms
                             # and 65.5 ms, the spans this was calibrated on)
                             # and a group of frames above that, so the ring is
                             # ~8.5 s of syllables at every span.
SLOW_ROWS = 600              # one scored row per second: ten minutes
SLOW_PERIOD_S = 1.0
KIND_HOLD_ROWS = 3           # consecutive rows a new verdict has to win before
                             # it is shown, and how the shown one gives way: see
                             # Finder._hold
KIND_CONF_RISE = 0.5         # how fast the shown confidence follows rows that
                             # agree with it -- half the gap per row
VOICE_WIDTH_HZ = 2700.0
WINDOW_STEP_POINTS = 2
SYLLABIC_HZ = (2.0, 8.0)     # syllable-rate band of the modulation spectrum
MOD_HZ = (0.25, 15.0)        # the modulation band it is measured against

# The voice score, calibrated on "can the operator copy it", which on SSB is
# about 3 dB in a 2.4 kHz passband with the envelope moving at syllable rate.
# The old ramps were (2, 8) dB, (0.15, 0.60) depth and (0.40, 0.70) syllabic
# under a cube root, which put the 0.5 gate at 4 dB and syllabic 0.51 -- and
# that is exactly where the live gate sat on 2026-09-03: the weakest candidate
# it would admit scored 0.496 at 4.0 dB, and the talker the operator was
# copying by ear at 14178 kHz, 1-3 dB over the floor, never appeared at all.
VOICE_SNR_DB = (1.5, 4.5)      # ...and read from the SNR WHILE SOMEBODY IS
VOICE_DEPTH = (0.10, 0.45)     # TALKING (ON_PCTL), not the ring average: a
VOICE_SYLLABIC = (0.25, 0.55)  # talker holds the frequency about 40% of the
ON_PCTL = 75.0                 # time and averages ~4 dB below what you hear
OCCUPANCY_GATE = (0.08, 0.20)  # a single crash of static is not a conversation

# ...and the detection score, which is how everything that is NOT a
# conversation is ranked: how far the strongest point of the window stands
# over its own local floor, and how much of the ring it was there for.
DETECT_DB = (3.0, 15.0)
DETECT_PRESENT_FRAC = 0.5
DETECT_MAX = 0.9             # a perfect carrier ranks below a perfect talker:
                             # the list still answers "where is somebody" first,
                             # it simply no longer answers ONLY that


def _decimate(x, order, points):
    """Mean of x (natural FFT order) over `points` equal groups, low freq first."""
    n = len(x)
    edges = np.linspace(0, n, points + 1).astype(int)
    c = np.concatenate([[0.0 if not np.iscomplexobj(x) else 0j], np.cumsum(x[order])])
    return (c[edges[1:]] - c[edges[:-1]]) / np.maximum(edges[1:] - edges[:-1], 1)


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _ramp(x, bounds):
    """0 below the pair's first value, 1 above its second, linear between."""
    lo, hi = bounds
    return np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)


def _shift_bins(arr, shift, fill, axis=-1):
    """arr's `axis`, in ASCENDING (low-to-high frequency) order, translated
    by `shift` positions: new[i] = old[i + shift]. Positions that would come
    from off the end (the span moved past what was ever measured there) get
    `fill` instead. shift == 0 is a no-op; |shift| >= the axis length is
    nothing but fill. Used to retune Finder's point/window history, which
    is already decimated into ascending order on the way in."""
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
    upper half negative frequency): used for LiveSpatial's cross-spectrum,
    which — unlike Finder's decimated history — is kept in the FFT's own
    order."""
    if shift == 0:
        return arr
    s = np.fft.fftshift(arr, axes=axis)
    return np.fft.ifftshift(_shift_bins(s, shift, fill, axis=axis), axes=axis)


class LiveSpatial:
    def __init__(self, nbins, rate_hz, tc_s=SPATIAL_TC_S):
        self.nbins = int(nbins)
        self.rate_hz = float(rate_hz)
        self.tc_s = float(tc_s)
        self.Saa = self.Sbb = self.Sab = None
        self.frames = 0
        self._order = np.fft.fftshift(np.arange(self.nbins))

    # --- reader thread -----------------------------------------------------
    def retune(self, delta_hz):
        """The hardware centre moved by delta_hz at the same sample rate:
        slide the cross-spectrum estimate along frequency so it stays with
        the bins it was measured on. There is no staleness gate here (update
        always accepts), so bins entering from off-span just start at zero
        (no power, no coherence) and pick back up on the next frame."""
        if self.Saa is None:
            return
        bin_hz = self.rate_hz / self.nbins
        shift = int(round(float(delta_hz) / bin_hz))
        if shift == 0:
            return
        if abs(shift) >= self.nbins:
            self.Saa = self.Sbb = self.Sab = None
            self.frames = 0
            return
        self.Saa = _shift_natural(self.Saa, shift, 0.0, axis=0)
        self.Sbb = _shift_natural(self.Sbb, shift, 0.0, axis=0)
        self.Sab = _shift_natural(self.Sab, shift, 0j, axis=0)

    def update(self, X, frame_s):
        Xa, Xb = np.asarray(X[0]), np.asarray(X[1])
        saa = np.abs(Xa) ** 2
        sbb = np.abs(Xb) ** 2
        sab = Xa * np.conj(Xb)
        self.frames += 1
        if self.Saa is None:
            self.Saa, self.Sbb, self.Sab = saa, sbb, sab
            return
        al = max(1.0 - math.exp(-frame_s / self.tc_s), 1.0 / self.frames)
        self.Saa = self.Saa + al * (saa - self.Saa)
        self.Sbb = self.Sbb + al * (sbb - self.Sbb)
        self.Sab = self.Sab + al * (sab - self.Sab)

    # --- on demand -----------------------------------------------------------
    def decimated(self, points=SPATIAL_POINTS):
        """(saa, sbb, sab) averaged into `points` groups, low frequency first."""
        if self.Saa is None:
            return None
        points = max(1, min(int(points), self.nbins))
        return (_decimate(self.Saa, self._order, points),
                _decimate(self.Sbb, self._order, points),
                _decimate(self.Sab, self._order, points))

    def rows(self, center_hz=0.0, points=SPATIAL_POINTS):
        d = self.decimated(points)
        if d is None:
            return None
        saa, sbb, sab = d
        points = len(saa)
        coh = _clip01(np.abs(sab) ** 2 / np.maximum(saa * sbb, 1e-30))
        phase = np.degrees(np.angle(sab))
        level = 10.0 * np.log10(np.maximum(0.5 * (saa + sbb), 1e-30))
        return {
            "start_hz": float(center_hz - self.rate_hz / 2),
            "step_hz": float(self.rate_hz / points),
            "points": int(points),
            "phase_deg": [round(float(x), 1) for x in phase],
            "coherence": [round(float(x), 3) for x in coh],
            "level_db": [round(float(x), 1) for x in level],
        }


WINDOW_POINTS_TARGET = 11    # map points across one voice window: what 125 kS/s
                             # (VOICE_WIDTH_HZ / (125e3 / 512)) gives, and what
                             # every width threshold in kinds.py is calibrated
                             # against. Holding SPATIAL_POINTS = 512 fixed instead
                             # let step_hz grow with the span: the window blew out
                             # to 5.9 kHz at 1.02 MS/s and 12 kHz at 2.04 MS/s, and
                             # even where its total width was right (250-500 kS/s)
                             # it was only 3-6 points across -- too coarse to tell a
                             # keyed tone from a conversation by shape. Points are
                             # therefore scaled to hold the spacing near
                             # VOICE_WIDTH_HZ / WINDOW_POINTS_TARGET, floored at
                             # SPATIAL_POINTS and capped at nbins: the raw FFT's own
                             # resolution, which is what runs out at 1.02 and
                             # 2.04 MS/s and cannot be bought back here.
POINT_HZ_TARGET = VOICE_WIDTH_HZ / WINDOW_POINTS_TARGET


def _default_points(rate_hz, nbins):
    scaled = int(round(float(rate_hz) / POINT_HZ_TARGET))
    return max(SPATIAL_POINTS, min(int(nbins), scaled))


class Finder:
    def __init__(self, nbins, rate_hz, points=None):
        self.nbins = int(nbins)
        self.rate_hz = float(rate_hz)
        if points is None:
            points = _default_points(self.rate_hz, self.nbins)
        self.points = max(8, min(int(points), self.nbins))
        self.step_hz = self.rate_hz / self.points
        self.win = max(3, int(round(VOICE_WIDTH_HZ / self.step_hz)))
        self.nwin = max(1, (self.points - self.win) // WINDOW_STEP_POINTS + 1)
        self._order = np.fft.fftshift(np.arange(self.nbins))
        # the read side (finder_report) asks the finder for its own geometry
        # rather than importing it back, which would be a circle
        self.window_step = WINDOW_STEP_POINTS
        self.slow_period_s = SLOW_PERIOD_S
        self.slow_rows = SLOW_ROWS
        self.min_present_frac = min(1.0, CANDIDATE_MIN_S / (FAST_FRAMES * SLOT_S))
        self._reset_history()

    def _reset_history(self):
        """The fast ring and the ten-minute score history, empty: what a
        fresh Finder starts with, and what retune() collapses to when the
        centre has moved past the whole span."""
        self.fast = np.zeros((FAST_FRAMES, 2, self.points), dtype=np.float32)
        self.fast_i = 0
        self.fast_n = 0
        self.slot = np.zeros((2, self.points), dtype=np.float64)   # part-built slot
        self.slot_s = 0.0
        self.slot_frames = 0
        self.frame_s = None
        self.elapsed = 0.0
        self._since_slow = 0.0
        self.slow = np.zeros((SLOW_ROWS, self.nwin), dtype=np.float32)
        # the voice score on its own, because the ranking score is now the
        # better of "somebody is talking here" and "something is here"
        self.slow_voice = np.zeros((SLOW_ROWS, self.nwin), dtype=np.float32)
        # how much of each row's ring the window stood over its LOCAL floor:
        # per window, for the candidate gate, and per map point, for the strip
        self.slow_wpres = np.zeros((SLOW_ROWS, self.nwin), dtype=np.float32)
        self.slow_pres = np.zeros((SLOW_ROWS, self.points), dtype=np.float32)
        # what the score was made of, per row, so a candidate can be described
        # as it was at its best rather than as it is right now
        self.slow_terms = np.zeros((SLOW_ROWS, 3, self.nwin), dtype=np.float32)
        # and what each window looked like: voice, cw, data, carrier or noise
        self.slow_kind = np.full((SLOW_ROWS, self.nwin), kinds.NOISE, dtype=np.int8)
        self.slow_kconf = np.zeros((SLOW_ROWS, self.nwin), dtype=np.float32)
        self.slow_t = np.zeros(SLOW_ROWS)
        self.slow_i = 0
        self.slow_n = 0
        # the verdict the finder SHOWS, which is not any one row's -- see _hold
        self.held_kind = np.full(self.nwin, kinds.NOISE, dtype=np.int8)
        self.held_conf = np.zeros(self.nwin, dtype=np.float32)
        self.held_any = False        # has any row established a verdict yet
        self.pend_kind = np.full(self.nwin, kinds.NOISE, dtype=np.int8)
        self.pend_n = np.zeros(self.nwin, dtype=np.int16)
        self._last = None            # the latest per-window analysis

    # --- reader thread -----------------------------------------------------
    def retune(self, delta_hz):
        """The hardware centre moved by delta_hz at the same sample rate:
        slide the fast ring and the ten-minute score history along frequency
        so a candidate found before the retune is still found at the same
        absolute frequency after it, without waiting for new frames.

        A point or window that enters from off-span gets exactly what a
        fresh Finder starts with there (zero power, zero score, a 'noise'
        verdict at zero confidence) — it must be scored again, not inherit
        whatever used to sit at that index. slow_t (when each ROW was
        scored, not which window) is per-row time, not per-bin, and is
        left alone. A shift past the whole span is nothing left to slide,
        so it resets."""
        if self.fast_n == 0 and self.slow_n == 0 and self._last is None:
            return
        point_shift = int(round(float(delta_hz) / self.step_hz))
        window_shift = int(round(float(delta_hz) / (self.step_hz * WINDOW_STEP_POINTS)))
        if point_shift == 0 and window_shift == 0:
            return
        if abs(point_shift) >= self.points or abs(window_shift) >= self.nwin:
            self._reset_history()
            return
        self.fast = _shift_bins(self.fast, point_shift, 0.0)
        self.slot = _shift_bins(self.slot, point_shift, 0.0)
        self.slow = _shift_bins(self.slow, window_shift, 0.0)
        self.slow_voice = _shift_bins(self.slow_voice, window_shift, 0.0)
        self.slow_wpres = _shift_bins(self.slow_wpres, window_shift, 0.0)
        self.slow_pres = _shift_bins(self.slow_pres, point_shift, 0.0)
        self.slow_terms = _shift_bins(self.slow_terms, window_shift, 0.0)
        self.slow_kind = _shift_bins(self.slow_kind, window_shift, kinds.NOISE)
        self.slow_kconf = _shift_bins(self.slow_kconf, window_shift, 0.0)
        self.held_kind = _shift_bins(self.held_kind, window_shift, kinds.NOISE)
        self.held_conf = _shift_bins(self.held_conf, window_shift, 0.0)
        self.pend_kind = _shift_bins(self.pend_kind, window_shift, kinds.NOISE)
        self.pend_n = _shift_bins(self.pend_n, window_shift, 0)
        if self._last is not None:
            last = self._last
            self._last = {
                "score": _shift_bins(last["score"], window_shift, 0.0),
                "snr_db": _shift_bins(last["snr_db"], window_shift, 0.0),
                "depth": _shift_bins(last["depth"], window_shift, 0.0),
                "syllabic": _shift_bins(last["syllabic"], window_shift, 0.0),
                "pa": _shift_bins(last["pa"], window_shift, 0.0),
                "pb": _shift_bins(last["pb"], window_shift, 0.0),
                "na": last["na"], "nb": last["nb"],
                "kind": _shift_bins(last["kind"], window_shift, kinds.NOISE),
                "kind_conf": _shift_bins(last["kind_conf"], window_shift, 0.0),
                "mean_points": _shift_bins(last["mean_points"], point_shift, 0.0),
                "floor_pts": _shift_bins(last["floor_pts"], point_shift, last["floor"]),
                "bw_hz": _shift_bins(last["bw_hz"], window_shift, 0.0),
                "peak_db": _shift_bins(last["peak_db"], window_shift, 0.0),
                "peak_off": _shift_bins(last["peak_off"], window_shift, 0),
                "present": _shift_bins(last["present"], window_shift, 0.0),
                "wpres": _shift_bins(last["wpres"], window_shift, 0.0),
                "voice": _shift_bins(last["voice"], window_shift, 0.0),
                "floor": last["floor"],
            }

    def update(self, X, frame_s):
        pa = _decimate(np.abs(np.asarray(X[0])) ** 2, self._order, self.points)
        pb = _decimate(np.abs(np.asarray(X[1])) ** 2, self._order, self.points)
        self.elapsed += frame_s
        self._since_slow += frame_s
        # frames go into the ring a slot at a time, so "a row of the ring" is
        # the same length of TIME at every span (see SLOT_S)
        self.slot[0] += pa
        self.slot[1] += pb
        self.slot_s += frame_s
        self.slot_frames += 1
        if self.slot_s < SLOT_S:
            return
        self.fast[self.fast_i] = self.slot / self.slot_frames
        self.fast_i = (self.fast_i + 1) % FAST_FRAMES
        self.fast_n = min(self.fast_n + 1, FAST_FRAMES)
        slot_s = self.slot_s
        self.frame_s = slot_s if self.frame_s is None else self.frame_s + 0.05 * (slot_s - self.frame_s)
        self.slot[:] = 0.0
        self.slot_s = 0.0
        self.slot_frames = 0
        if self._since_slow >= SLOW_PERIOD_S and self.fast_n >= FAST_FRAMES // 2:
            self._since_slow = 0.0
            self._analyse()

    def _frames(self):
        """The fast ring in time order: (n, 2, points)."""
        if self.fast_n < FAST_FRAMES:
            return self.fast[:self.fast_n]
        return np.concatenate([self.fast[self.fast_i:], self.fast[:self.fast_i]])

    def _window_sums(self, p):
        """Sum of p (..., points) over each 2.7 kHz window -> (..., nwin)."""
        c = np.cumsum(p, axis=-1)
        c = np.concatenate([np.zeros(c.shape[:-1] + (1,), dtype=c.dtype), c], axis=-1)
        lo = np.arange(self.nwin) * WINDOW_STEP_POINTS
        return c[..., lo + self.win] - c[..., lo]

    def _window_max(self, p):
        """Max of p (points,) over each window -> (nwin,). A 200 Hz tone is
        present in its window even where it lifts the window's total by under
        3 dB, and that -- not the classifier -- is why no CW column was ever
        a candidate."""
        seg = np.lib.stride_tricks.sliding_window_view(p, self.win)[::WINDOW_STEP_POINTS]
        if len(seg) < self.nwin:
            seg = np.concatenate([seg, np.repeat(seg[-1:], self.nwin - len(seg), axis=0)])
        return np.max(seg[:self.nwin], axis=1)

    def _analyse(self):
        F = self._frames().astype(np.float64)                  # (n, 2, points)
        n = F.shape[0]
        both = F[:, 0] + F[:, 1]
        frame_s = self.frame_s or SLOT_S
        self.min_present_frac = min(1.0, CANDIDATE_MIN_S / max(n * frame_s, 1e-6))
        # The whole span's median per frame -- what kinds.py correlates a
        # window against to recognise weather rather than a station...
        floor = np.median(both, axis=1)                         # (n,)
        # ...and the floor a signal is actually weak against, which is the one
        # under IT: per ~10 kHz, robust, following the span's own tilt. See
        # finder_floor.py. Everything below is measured against this one.
        floor_pts = ffloor.local_floor(both, self.step_hz)      # (points,)
        W = self._window_sums(both)                             # (n, nwin)
        mean_w = np.mean(W, axis=0)
        floor_w = np.maximum(self._window_sums(floor_pts), 1e-30)
        snr_db = 10.0 * np.log10(np.maximum(mean_w, 1e-30) / floor_w)
        # ...and the SNR while somebody is actually TALKING, which is what
        # "copyable" is a statement about: a talker holds the frequency about
        # 40% of the time, so his ring average is some 4 dB below what you hear.
        on_w = np.percentile(W, ON_PCTL, axis=0)
        snr_on_db = 10.0 * np.log10(np.maximum(on_w, 1e-30) / floor_w)
        x = W / np.maximum(mean_w, 1e-30) - 1.0                 # (n, nwin) modulation
        depth = np.std(x, axis=0)
        taper = np.hanning(n)[:, None]
        M = np.abs(np.fft.rfft(x * taper, axis=0)) ** 2
        f = np.fft.rfftfreq(n, frame_s)
        syl = (f >= SYLLABIC_HZ[0]) & (f <= SYLLABIC_HZ[1])
        band = (f >= MOD_HZ[0]) & (f <= MOD_HZ[1])
        syllabic = np.sum(M[syl], axis=0) / np.maximum(np.sum(M[band], axis=0), 1e-30)
        # geometric mean of the three verdicts: one weak term (a modest SNR,
        # a real voice's broader modulation spectrum) does not veto the other
        # two, while a term at zero (a steady carrier's depth, a single burst's
        # flat modulation spectrum) still does
        voice = np.cbrt(_ramp(snr_on_db, VOICE_SNR_DB) * _ramp(depth, VOICE_DEPTH)
                        * _ramp(syllabic, VOICE_SYLLABIC))
        # a single crash of static is deep, loud and broad in modulation too;
        # what it is not is THERE: voice occupies a third to a half of the
        # frames, a burst a few percent. A gate, not a grade.
        occupancy = np.mean(W > 0.5 * mean_w, axis=0)
        voice = voice * _ramp(occupancy, OCCUPANCY_GATE)
        # What is HERE, whatever it is: the share of the ring each point spent
        # over its own floor, and how far the best point of each window stands
        # over its. A finder that ranks only by voice can only find voice.
        mean_points = np.mean(both, axis=0)
        pres_pts = ffloor.presence(both, floor_pts, self.step_hz, frame_s)
        wpres = np.maximum(self._window_max(pres_pts),
                           ffloor.presence_wide(W, floor_w, frame_s))
        peak_db, peak_off = ffloor.peak_excess(mean_points, floor_pts, self.win,
                                               WINDOW_STEP_POINTS, self.nwin,
                                               self.step_hz)
        # how far the window stands over its floor, measured the way its own
        # shape asks to be measured: a keyed tone by its strongest point (a
        # 2.7 kHz window can only ever see 2.8 dB of it), a filled sub-band by
        # the whole window (its peak point is no higher than the rest of it)
        excess = np.maximum(peak_db, snr_db)
        detect = (DETECT_MAX * _ramp(excess, DETECT_DB)
                  * _clip01(wpres / DETECT_PRESENT_FRAC))
        score = np.maximum(voice, detect)
        # per-loop window power against each loop's own floor, for the gain
        pa_w = np.mean(self._window_sums(F[:, 0]), axis=0)
        pb_w = np.mean(self._window_sums(F[:, 1]), axis=0)
        na_w = np.mean(np.median(F[:, 0], axis=1)) * self.win
        nb_w = np.mean(np.median(F[:, 1], axis=1)) * self.win
        # what each window is, from the same frames the score came from, held
        # steady across rows so the answer is the band's and not the second's
        feat = kinds.features(W, floor, mean_points, snr_db, depth, syllabic,
                              occupancy, self.win, WINDOW_STEP_POINTS, self.step_hz,
                              floor_points=floor_pts, peak_db=peak_db)
        kind, kconf = self._hold(*kinds.verdict(feat))
        self.slow[self.slow_i] = score
        self.slow_voice[self.slow_i] = voice
        self.slow_wpres[self.slow_i] = wpres
        self.slow_pres[self.slow_i] = pres_pts
        self.slow_terms[self.slow_i] = np.stack([snr_db, depth, syllabic])
        self.slow_kind[self.slow_i] = kind
        self.slow_kconf[self.slow_i] = kconf
        self.slow_t[self.slow_i] = self.elapsed
        self.slow_i = (self.slow_i + 1) % SLOW_ROWS
        self.slow_n = min(self.slow_n + 1, SLOW_ROWS)
        self._last = {
            "score": score, "voice": voice, "snr_db": snr_db, "depth": depth,
            "syllabic": syllabic, "wpres": wpres,
            "present": np.asarray(kinds.present(feat), dtype=np.float64),
            "bw_hz": np.asarray(feat["bw_hz"], dtype=np.float64),
            "peak_db": peak_db, "peak_off": peak_off,
            "pa": pa_w, "pb": pb_w, "na": na_w, "nb": nb_w,
            "kind": kind, "kind_conf": kconf,
            "mean_points": mean_points, "floor_pts": floor_pts,
            "floor": float(np.mean(floor)),
        }

    def _hold(self, kind, kconf):
        """One row's verdict turned into the verdict the finder will show.

        A row is eight and a half seconds of a band that changes: a talker
        pauses between overs, a signal fades, the window the finder centred on
        a conversation holds nothing but its skirt for a moment. Shown raw, the
        per-row verdict flapped -- on 2026-09-03 the same eight 80 m phone
        candidates came back "cw 1.0" on one read of /diversity/finder and
        "voice" on the next, thirty seconds later, with nobody sending Morse.
        So a verdict has to earn the display:

          * the first analysis sets it, because there is nothing yet to protect;
          * a row that AGREES eases the shown confidence towards its own;
          * a row that disagrees does not flip anything. It spends its own
            confidence against the held one, so a verdict held at 1.0 survives
            two or three confident contradictions and a verdict barely held at
            0.2 gives way at once -- decay, not a coin toss;
          * and the challenger has to win KIND_HOLD_ROWS rows in a row, and be
            at least as sure as what is left of the verdict it displaces,
            before it takes the window.

        Rows are SLOW_PERIOD_S apart over a ring of FAST_FRAMES slots, so
        consecutive rows share most of their frames: three of them is three
        seconds of genuinely new evidence, which is about one over.
        """
        kind = np.asarray(kind, dtype=np.int8)
        kconf = np.asarray(kconf, dtype=np.float32)
        if not self.held_any:
            self.held_any = True
            self.held_kind = kind.copy()
            self.held_conf = kconf.copy()
            self.pend_kind = kind.copy()
            self.pend_n = np.zeros(self.nwin, dtype=np.int16)
            return self.held_kind.copy(), self.held_conf.copy()
        agree = kind == self.held_kind
        # "signal" is not a verdict, it is an admission that nothing named the
        # window this second -- so against a window that HAS a name it is not
        # evidence either way: it spends no confidence, it accrues no rows
        # towards taking over, and it cannot displace anything. (Measured on
        # the 2026-09-03 80 m recording: voice->signal->noise round trips were
        # 53 of the 99 verdict changes the hold still let through.) A named
        # kind still displaces a held "signal" on the usual terms.
        mute = (~agree) & (kind == kinds.SIGNAL)
        conf = np.where(agree,
                        self.held_conf + KIND_CONF_RISE * (kconf - self.held_conf),
                        np.where(mute, self.held_conf,
                                 np.maximum(self.held_conf - kconf, 0.0)))
        again = (~agree) & (~mute) & (kind == self.pend_kind)
        self.pend_n = np.where(agree, 0,
                               np.where(mute, self.pend_n,
                                        np.where(again, self.pend_n + 1, 1))).astype(np.int16)
        self.pend_kind = np.where(agree | mute, self.pend_kind, kind).astype(np.int8)
        take = (~agree) & (~mute) & (self.pend_n >= KIND_HOLD_ROWS) & (kconf >= conf)
        self.held_kind = np.where(take, kind, self.held_kind).astype(np.int8)
        self.held_conf = np.where(take, kconf, conf).astype(np.float32)
        self.pend_n = np.where(take, 0, self.pend_n).astype(np.int16)
        return self.held_kind.copy(), self.held_conf.copy()

    # --- on demand -----------------------------------------------------------
    def _slow_idx(self):
        if self.slow_n < SLOW_ROWS:
            return np.arange(self.slow_n)
        return np.concatenate([np.arange(self.slow_i, SLOW_ROWS), np.arange(self.slow_i)])

    def _slow_rows(self):
        """The slow ring in time order: scores (n, nwin), their terms
        (n, 3, nwin) = snr_db/depth/syllabic, the kind code and its
        confidence (n, nwin) each, times (n,), the voice score alone
        (n, nwin), and the share of each row's ring the window stood over its
        local floor (n, nwin)."""
        idx = self._slow_idx()
        return (self.slow[idx], self.slow_terms[idx], self.slow_kind[idx],
                self.slow_kconf[idx], self.slow_t[idx], self.slow_voice[idx],
                self.slow_wpres[idx])

    def _slow_points(self):
        """The same history per MAP POINT: (n, points) of presence share,
        which is what the activity strip is the mean of."""
        return self.slow_pres[self._slow_idx()]

    def window_kinds(self):
        """The latest verdict per window: (codes into kinds.KINDS,
        confidences), or None before the first analysis."""
        if self._last is None:
            return None
        return self._last["kind"], self._last["kind_conf"]

    def candidates(self, center_hz=0.0, live=None, tuned_hz=None):
        """/diversity/finder, whole: see finder_report.payload, which builds it.

        `tuned_hz` is what the operator is listening to. Its column is always
        in the list, flagged `tuned`, however it scored -- what the finder
        thinks of what they can hear is the one row they can check.
        """
        return finder_report.payload(self, center_hz, live, tuned_hz)
