#
# Aether-gate — the pair's optional stages: the v2 post-filter, sub-band
# weights, time-signal stations, and a cached compass.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""What _DiversityState hands its blocks through when the operator asks.
Everything here is off by default and says so in status().

  post=v2   core.cohpost on the demod passband: a per-bin gain from the
            loops' cross-spectrum with a pause gate that learns the noise
            between words. It takes the place of the sub-band combiner AND
            its own post-filter (core.postfilter, "v1") for the audio, so
            the two never run in series and an A/B is one switch: v1 is
            sub-band + post-filter, v2 is the wideband weight + cohpost.
  mrc=on    core.binweights on the panadapter block: one MVDR weight per
            bin from the spatial map's floor covariance. Measured +0.15 dB
            over the broadband weight on a real 80 m capture, so it stays
            a lab switch until a scene shows more. The pan only; the
            audio's per-bin refinement is the sub-band combiner.
  time signals ride along once aligned, like the beacons: WWV, WWVH, CHU,
            RWM and BPM windows scored into the site log for the compass,
            which needs a band the NCDXF beacons never use to break the
            14.1/21.15/28.2 MHz alias (see core.compass).
  the compass is fitted from the site log on every call: nothing while the
            log is short, 0.4 s at a month of beacons. It is cached for a
            few seconds keyed on the phase and frequency it was asked
            about, so a polling window never pays twice.
  the noise bearing rides on that same fit (core.noisebearing): the spatial
            map's coherent floor bins averaged into one phase, put through
            the compass. Cached the same way -- and the fit it needs is kept
            for a minute, because the status poll asks several times a
            second and beacons arrive one every three minutes.
"""
import time

import numpy as np

COMPASS_TTL_S = 5.0
GLOBAL_FIT_TTL_S = 60.0      # the search behind the bearing, which no new beacon outruns
MRC_REFRESH_S = 1.0          # how often the map's covariance is handed to the weights


def _cpf():
    from ..core import cohpost
    return cohpost


def _bwm():
    from ..core import binweights
    return binweights


def _ts():
    from ..core import timesignals
    return timesignals


def _cp():
    from ..core import compass
    return compass


def _nb():
    from ..core import noisebearing
    return noisebearing


class Enhancers:
    def __init__(self):
        self.post_v2 = False
        self.mrc_on = False
        self._pf = None                     # CoherencePostFilter
        self._pf_key = None                 # (rate, lo, hi) it was built for
        self._bw = None                     # BinWeights
        self._bw_rate = None
        self._bw_at = 0.0                   # when the covariance was last handed over
        self.timesignals = None             # TimeSignalWatch
        self._last_ts = None
        self._compass = None                # (key, at, answer)
        self._gfit = None                   # (at, GlobalFit) behind the noise bearing
        self._noise = None                  # (at, answer)
        self._noise_hist = None             # BearingHistory, for its `since`

    # --- post=v2: the coherence post-filter on the passband ------------------
    def post_audio(self, y, pa, pb, rate_hz, lo_hz, hi_hz):
        """The combined passband block y, made from the aligned pair (pa, pb),
        taken through cohpost. Delayed by one frame. The gain follows the
        band's own dominant phase rather than the weight's: on a real 80 m
        capture that was 0.4 dB better than pinning it (11.5 vs 11.1 dB
        out), and the weight's sign convention never enters into it."""
        key = (float(rate_hz), float(lo_hz), float(hi_hz))
        if self._pf is None or self._pf_key[0] != key[0]:
            self._pf = _cpf().CoherencePostFilter(rate_hz, lo_hz, hi_hz)
        elif self._pf_key != key:
            self._pf.set_band(lo_hz, hi_hz)
        self._pf_key = key
        return self._pf.process(y, pa, pb, None)

    def post_reset(self):
        self._pf, self._pf_key = None, None

    def post_status(self):
        out = {"enabled": self.post_v2, "version": 2}
        if self._pf is not None:
            out.update(self._pf.status())
            out["gate"] = self._pf.gate.status()
        return out

    # --- mrc=on: per-bin weights on the pan ------------------------------------
    def mrc_pan(self, a, b, spmap, rate_hz, center_hz, band_hz, m, now):
        """The pan block with the map's per-bin weights, or None when the
        weights have nothing to say yet (no covariance, or the STFT buffer
        is still filling) -- the caller falls back to the broadband weight."""
        if spmap is None or spmap.R is None:
            return None
        if self._bw is None or self._bw_rate != float(rate_hz):
            self._bw = _bwm().BinWeights(rate_hz, spmap.nbins, center_hz=center_hz)
            self._bw_rate, self._bw_at = float(rate_hz), 0.0
        bw = self._bw
        bw.set_center(center_hz)
        if band_hz is not None and (bw.lo_hz, bw.hi_hz) != (band_hz[0], band_hz[1]):
            bw.set_band(band_hz[0], band_hz[1])
        bw.set_weight(m)
        if now - self._bw_at >= MRC_REFRESH_S:
            step = float(rate_hz) / spmap.nbins
            bw.set_covariance(np.fft.fftshift(spmap.R, axes=0),
                              np.fft.fftshift(spmap.stale_mask()),
                              start_hz=float(center_hz) - float(rate_hz) / 2.0, step_hz=step)
            self._bw_at = now
        out = bw.apply(a, b)
        if len(out) < len(a):
            return None
        return out

    def mrc_reset(self):
        self._bw, self._bw_rate = None, None

    def mrc_status(self):
        out = {"enabled": self.mrc_on}
        if self._bw is not None and self._bw.R is not None:
            out.update(self._bw.status())
        return out

    # --- the time-signal stations, beside the beacons --------------------------
    def timesignals_update(self, a, b, rate_hz, center_hz, t, sitelog, grid):
        w = self.timesignals
        if w is None or w.rate_hz != float(rate_hz):
            w = self.timesignals = _ts().TimeSignalWatch(rate_hz)
            if grid:
                w.set_station(grid)
        w.update(a, b, float(center_hz), t)
        if w.last is not self._last_ts:          # one window scored: keep it
            self._last_ts = w.last
            sitelog.beacon_result(w.last)

    def timesignals_json(self, t):
        if self.timesignals is None:
            return {"available": False}
        return self.timesignals.status(t)

    def set_station(self, grid):
        if self.timesignals is not None:
            self.timesignals.set_station(grid)

    def set_assumed(self, f_hz, call):
        if self.timesignals is None:
            self.timesignals = _ts().TimeSignalWatch(1.0)      # rebuilt at the real rate on ingest
        self.timesignals.set_assumed(f_hz, call)      # ValueError when the call is not there

    # --- the compass, cached ----------------------------------------------------
    def compass(self, sitelog, phase_deg, f_hz, now=None):
        now = time.time() if now is None else now
        key = (None if phase_deg is None else round(float(phase_deg)), f_hz)
        c = self._compass
        if c is not None and c[0] == key and now - c[1] < COMPASS_TTL_S:
            return c[2]
        out = _cp().compass_json(sitelog, phase_deg=phase_deg, f_hz=f_hz)
        self._compass = (key, now, out)
        return out

    def global_fit(self, sitelog, now):
        """The pair itself, fitted from every band the log has heard, kept
        for a minute (see GLOBAL_FIT_TTL_S)."""
        g = self._gfit
        if g is not None and now - g[0] < GLOBAL_FIT_TTL_S:
            return g[1]
        fit = _cp().fit_global_from_log(sitelog)
        self._gfit = (now, fit)
        return fit

    # --- the noise bearing, cached ----------------------------------------------
    def noise_bearing(self, spmap, profile_status, sitelog, rate_hz, center_hz,
                      now=None):
        """Which way the noise the profile just described is coming from:
        the map's coherent floor bins as one phase, that phase as a bearing.
        See core.noisebearing for what is averaged and what is left out."""
        now = time.time() if now is None else now
        c = self._noise
        if c is not None and now - c[0] < COMPASS_TTL_S:
            return c[1]
        if self._noise_hist is None:
            self._noise_hist = _nb().BearingHistory()
        out = _nb().noise_bearing(spmap, profile_status,
                                  self.global_fit(sitelog, now), center_hz,
                                  rate_hz, now=now, history=self._noise_hist)
        self._noise = (now, out)
        return out

    def status(self):
        return {"post_v2": self.post_v2, "mrc": self.mrc_status()}
