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

from . import kinds

SPATIAL_TC_S = 0.25
SPATIAL_POINTS = 512

FAST_FRAMES = 256            # ~8.5 s of frames at the reader's ~30 frames/s
SLOW_ROWS = 600              # one scored row per second: ten minutes
SLOW_PERIOD_S = 1.0
VOICE_WIDTH_HZ = 2700.0
WINDOW_STEP_POINTS = 2
SYLLABIC_HZ = (2.0, 8.0)     # syllable-rate band of the modulation spectrum
MOD_HZ = (0.25, 15.0)        # the modulation band it is measured against
VOICE_SCORE = 0.5            # a window at or above this is "voice"
CANDIDATE_MAX = 12
CANDIDATE_RECENT_S = 30.0    # a candidate must have scored within this long
EDGE_MARGIN_HZ = 150.0       # dial sits this far outside the voice energy
DIAL_GRID_HZ = 500.0         # phone sits on whole and half kilohertz; the map's
                             # points are ~244 Hz apart, so the raw dial estimate
                             # is snapped to the grid (hz) and kept beside it (hz_raw)
USB_ABOVE_HZ = 10_000_000.0  # band convention: USB above 10 MHz, LSB below


def _decimate(x, order, points):
    """Mean of x (natural FFT order) over `points` equal groups, low freq first."""
    n = len(x)
    edges = np.linspace(0, n, points + 1).astype(int)
    c = np.concatenate([[0.0 if not np.iscomplexobj(x) else 0j], np.cumsum(x[order])])
    return (c[edges[1:]] - c[edges[:-1]]) / np.maximum(edges[1:] - edges[:-1], 1)


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


class LiveSpatial:
    def __init__(self, nbins, rate_hz, tc_s=SPATIAL_TC_S):
        self.nbins = int(nbins)
        self.rate_hz = float(rate_hz)
        self.tc_s = float(tc_s)
        self.Saa = self.Sbb = self.Sab = None
        self.frames = 0
        self._order = np.fft.fftshift(np.arange(self.nbins))

    # --- reader thread -----------------------------------------------------
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


class Finder:
    def __init__(self, nbins, rate_hz, points=SPATIAL_POINTS):
        self.nbins = int(nbins)
        self.rate_hz = float(rate_hz)
        self.points = max(8, min(int(points), self.nbins))
        self.step_hz = self.rate_hz / self.points
        self.win = max(3, int(round(VOICE_WIDTH_HZ / self.step_hz)))
        self.nwin = max(1, (self.points - self.win) // WINDOW_STEP_POINTS + 1)
        self._order = np.fft.fftshift(np.arange(self.nbins))
        self.fast = np.zeros((FAST_FRAMES, 2, self.points), dtype=np.float32)
        self.fast_i = 0
        self.fast_n = 0
        self.frame_s = None
        self.elapsed = 0.0
        self._since_slow = 0.0
        self.slow = np.zeros((SLOW_ROWS, self.nwin), dtype=np.float32)
        # what the score was made of, per row, so a candidate can be described
        # as it was at its best rather than as it is right now
        self.slow_terms = np.zeros((SLOW_ROWS, 3, self.nwin), dtype=np.float32)
        # and what each window looked like: voice, cw, data, carrier or noise
        self.slow_kind = np.full((SLOW_ROWS, self.nwin), kinds.NOISE, dtype=np.int8)
        self.slow_kconf = np.zeros((SLOW_ROWS, self.nwin), dtype=np.float32)
        self.slow_t = np.zeros(SLOW_ROWS)
        self.slow_i = 0
        self.slow_n = 0
        self._last = None            # the latest per-window analysis

    # --- reader thread -----------------------------------------------------
    def update(self, X, frame_s):
        pa = _decimate(np.abs(np.asarray(X[0])) ** 2, self._order, self.points)
        pb = _decimate(np.abs(np.asarray(X[1])) ** 2, self._order, self.points)
        self.fast[self.fast_i, 0] = pa
        self.fast[self.fast_i, 1] = pb
        self.fast_i = (self.fast_i + 1) % FAST_FRAMES
        self.fast_n = min(self.fast_n + 1, FAST_FRAMES)
        self.frame_s = frame_s if self.frame_s is None else self.frame_s + 0.05 * (frame_s - self.frame_s)
        self.elapsed += frame_s
        self._since_slow += frame_s
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

    def _analyse(self):
        F = self._frames().astype(np.float64)                  # (n, 2, points)
        n = F.shape[0]
        both = F[:, 0] + F[:, 1]
        # the band's floor per point, per frame: the median point of the span
        floor = np.median(both, axis=1)                         # (n,)
        W = self._window_sums(both)                             # (n, nwin)
        mean_w = np.mean(W, axis=0)
        floor_w = np.mean(floor) * self.win
        snr_db = 10.0 * np.log10(np.maximum(mean_w, 1e-30) / max(floor_w, 1e-30))
        x = W / np.maximum(mean_w, 1e-30) - 1.0                 # (n, nwin) modulation
        depth = np.std(x, axis=0)
        taper = np.hanning(n)[:, None]
        M = np.abs(np.fft.rfft(x * taper, axis=0)) ** 2
        f = np.fft.rfftfreq(n, self.frame_s or SLOW_PERIOD_S / 30.0)
        syl = (f >= SYLLABIC_HZ[0]) & (f <= SYLLABIC_HZ[1])
        band = (f >= MOD_HZ[0]) & (f <= MOD_HZ[1])
        syllabic = np.sum(M[syl], axis=0) / np.maximum(np.sum(M[band], axis=0), 1e-30)
        # geometric mean of the three verdicts: one weak term (a modest SNR,
        # a real voice's broader modulation spectrum) does not veto the other
        # two, while a term at zero (a steady carrier's depth, a single burst's
        # flat modulation spectrum) still does
        score = np.cbrt(_clip01((snr_db - 2.0) / 6.0) * _clip01((depth - 0.15) / 0.45)
                        * _clip01((syllabic - 0.4) / 0.3))
        # a single crash of static is deep, loud and broad in modulation too;
        # what it is not is THERE: voice occupies a third to a half of the
        # frames, a burst a few percent. A gate, not a grade.
        occupancy = np.mean(W > 0.5 * mean_w, axis=0)
        score = score * _clip01((occupancy - 0.08) / 0.12)
        # per-loop window power against each loop's own floor, for the gain
        pa_w = np.mean(self._window_sums(F[:, 0]), axis=0)
        pb_w = np.mean(self._window_sums(F[:, 1]), axis=0)
        na_w = np.mean(np.median(F[:, 0], axis=1)) * self.win
        nb_w = np.mean(np.median(F[:, 1], axis=1)) * self.win
        mean_points = np.mean(both, axis=0)
        # what each window is, from the same frames the score came from
        kind, kconf = kinds.classify(W, floor, mean_points, snr_db, depth, syllabic,
                                     occupancy, self.win, WINDOW_STEP_POINTS,
                                     self.step_hz)
        self.slow[self.slow_i] = score
        self.slow_terms[self.slow_i] = np.stack([snr_db, depth, syllabic])
        self.slow_kind[self.slow_i] = kind
        self.slow_kconf[self.slow_i] = kconf
        self.slow_t[self.slow_i] = self.elapsed
        self.slow_i = (self.slow_i + 1) % SLOW_ROWS
        self.slow_n = min(self.slow_n + 1, SLOW_ROWS)
        self._last = {
            "score": score, "snr_db": snr_db, "depth": depth, "syllabic": syllabic,
            "pa": pa_w, "pb": pb_w, "na": na_w, "nb": nb_w,
            "kind": kind, "kind_conf": kconf,
            "mean_points": mean_points, "floor": float(np.mean(floor)),
        }

    # --- on demand -----------------------------------------------------------
    def _slow_rows(self):
        """The slow ring in time order: scores (n, nwin), their terms
        (n, 3, nwin) = snr_db/depth/syllabic, the kind code and its
        confidence (n, nwin) each, times (n,)."""
        if self.slow_n < SLOW_ROWS:
            idx = np.arange(self.slow_n)
        else:
            idx = np.concatenate([np.arange(self.slow_i, SLOW_ROWS), np.arange(self.slow_i)])
        return (self.slow[idx], self.slow_terms[idx], self.slow_kind[idx],
                self.slow_kconf[idx], self.slow_t[idx])

    def window_kinds(self):
        """The latest verdict per window: (codes into kinds.KINDS,
        confidences), or None before the first analysis."""
        if self._last is None:
            return None
        return self._last["kind"], self._last["kind_conf"]

    def _point_hz(self, i, center_hz):
        return center_hz - self.rate_hz / 2 + (i + 0.5) * self.step_hz

    def _dial_hz(self, w, center_hz):
        """Where to put the dial for window w: just outside the voice energy
        on the carrier side (USB below the energy, LSB above)."""
        last = self._last
        lo = w * WINDOW_STEP_POINTS
        seg = last["mean_points"][lo:lo + self.win]
        above = np.nonzero(seg > 2.0 * last["floor"])[0]
        usb = center_hz >= USB_ABOVE_HZ
        if len(above) == 0:
            edge = lo if usb else lo + self.win - 1
        else:
            edge = lo + (above[0] if usb else above[-1])
        if usb:
            raw = self._point_hz(edge, center_hz) - self.step_hz / 2 - EDGE_MARGIN_HZ
        else:
            raw = self._point_hz(edge, center_hz) + self.step_hz / 2 + EDGE_MARGIN_HZ
        return DIAL_GRID_HZ * round(raw / DIAL_GRID_HZ), ("USB" if usb else "LSB"), raw

    def candidates(self, center_hz=0.0, live=None):
        last = self._last
        if last is None:
            return {"available": False}
        rows, terms, kind_rows, kconf_rows, times = self._slow_rows()
        is_recent = (self.elapsed - times) <= CANDIDATE_RECENT_S
        recent = rows[is_recent]
        if len(recent):
            rec = np.max(recent, axis=0)
            best = np.nonzero(is_recent)[0][np.argmax(recent, axis=0)]   # row of each max
        else:
            rec = last["score"]
            best = None
        voiced = rows >= VOICE_SCORE
        activity = np.mean(voiced, axis=0) if len(rows) else np.zeros(self.nwin)
        active_s = np.sum(voiced, axis=0) * SLOW_PERIOD_S
        span = max(1, self.win // WINDOW_STEP_POINTS)
        dec = live.decimated(self.points) if live is not None else None
        out = []
        for w in np.argsort(-rec):
            if rec[w] < VOICE_SCORE or len(out) >= CANDIDATE_MAX:
                break
            a, b = max(0, w - span), min(self.nwin, w + span + 1)
            if rec[w] < np.max(rec[a:b]) or any(abs(o["_w"] - w) <= span for o in out):
                continue
            hit = np.nonzero(voiced[:, w])[0]
            last_s = float(self.elapsed - times[hit[-1]]) if len(hit) else None
            hz, mode, hz_raw = self._dial_hz(w, center_hz)
            # the terms as they were when the window scored best, not now:
            # a row must describe the conversation it lists, and 20 s after
            # the last over "now" is the floor
            if best is not None:
                snr_w, depth_w, syl_w = (float(x) for x in terms[best[w], :, w])
                kind_w, kconf_w = int(kind_rows[best[w], w]), float(kconf_rows[best[w], w])
            else:
                snr_w, depth_w, syl_w = (float(last[k][w]) for k in ("snr_db", "depth", "syllabic"))
                kind_w, kconf_w = int(last["kind"][w]), float(last["kind_conf"][w])
            c = {
                "_w": int(w), "hz": round(float(hz), 1), "hz_raw": round(float(hz_raw), 1),
                "mode": mode,
                "width_hz": round(self.win * self.step_hz, 1),
                "score": round(float(rec[w]), 2),
                # what the gate thinks it is, and how sure: a row that says
                # "cw 0.9" saves the operator the trip
                "kind": kinds.name(kind_w),
                "kind_conf": round(kconf_w, 2),
                "snr_db": round(snr_w, 1),
                "syllabic": round(syl_w, 2),
                "depth": round(depth_w, 2),
                "active_s": round(float(active_s[w]), 1),
                "last_s": None if last_s is None else round(last_s, 1),
            }
            sa = max(float(last["pa"][w] - last["na"]), 0.0)
            sb = max(float(last["pb"][w] - last["nb"]), 0.0)
            if sa > 0 and sb > 0:
                r = min(sa / sb, sb / sa)
                c["gain_db"] = round(10.0 * math.log10(1.0 + r), 1)
            else:
                c["gain_db"] = 0.0
            if dec is not None:
                lo = w * WINDOW_STEP_POINTS
                saa = float(np.sum(dec[0][lo:lo + self.win]))
                sbb = float(np.sum(dec[1][lo:lo + self.win]))
                sab = complex(np.sum(dec[2][lo:lo + self.win]))
                c["phase_deg"] = round(math.degrees(math.atan2(sab.imag, sab.real)), 1)
                c["coherence"] = round(min(1.0, abs(sab) ** 2 / max(saa * sbb, 1e-30)), 2)
                c["ratio_db"] = round(10.0 * math.log10(max(sbb, 1e-30) / max(saa, 1e-30)), 1)
            out.append(c)
        for c in out:
            del c["_w"]
        # activity per point for a strip under the waterfall: the best window
        # covering each point
        act_pts = np.zeros(self.points)
        for w in range(self.nwin):
            lo = w * WINDOW_STEP_POINTS
            act_pts[lo:lo + self.win] = np.maximum(act_pts[lo:lo + self.win], activity[w])
        return {
            "available": True,
            "span_hz": [float(center_hz - self.rate_hz / 2), float(center_hz + self.rate_hz / 2)],
            "history_s": float(min(self.elapsed, SLOW_ROWS * SLOW_PERIOD_S)),
            "points": int(self.points),
            "activity": [round(float(x), 3) for x in act_pts],
            "candidates": out,
        }
