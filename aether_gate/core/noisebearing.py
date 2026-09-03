#
# Aether-gate -- which way the noise is coming from.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The noise profile says WHAT the interference is -- a mains comb, a line,
an impulse train, or just the band. The spatial map says, bin by bin, what
phase the two loops hear each bin at. The compass turns a phase into a
bearing. This is the wire between the three: one number the operator can
walk in the direction of.

The bearing is measured on the FLOOR, not on a signal. Every bin of the
spatial map is already floor-tracked (it only learns a bin while that bin
sits near the level it has learned there, so a passing station never
becomes "noise"), so the bins to average are the coherent ones with no
station in them:

  * coherence at or over the map's own SOURCE_MIN_COHERENCE -- an
    incoherent bin is the sky and the two receivers, and has no direction;
  * not stale -- a bin whose floor has not been accepted for half a minute
    is describing a band that has moved on;
  * not within a few bins of DC, where the tuner's own leakage is perfectly
    coherent and points nowhere;
  * not a station. A station is NARROW: three kilohertz of voice, one bin
    of carrier. Local noise is BROAD -- tens or hundreds of kilohertz of
    hash. So a bin is dropped when it stands STATION_OVER_DB over the
    median of its own SEGMENT_HZ-wide neighbourhood AND it is part of a run
    narrower than STATION_MAX_HZ -- wider than any one HF transmission, so
    a loud hump of hash keeps its bins however far over the floor it sits.
    The cost, said out loud: a loud noise source under 20 kHz wide that is
    not the widest thing in its neighbourhood goes with the stations.
    Saying nothing is the safer mistake.

What is left is averaged as unit complex ratios weighted by each bin's
coherence -- phase wraps, so a mean of degrees would put a bin at +179 and
one at -179 in the middle of nowhere. The length of that sum, over the bin
count, is the number reported as `coherence`: it falls both when the bins
are individually incoherent and when they disagree with each other, which
is exactly when the answer should not be trusted. Under MIN_COHERENCE
there is a phase but no bearing, and `reason` says so.

Two things the phase does NOT need correcting for. The map's phase is the
same measurement the site log writes for a beacon (loop B relative to loop
A, which is the opposite sign to the map's own steering angle), so the
compass reads it directly. And the phase a pair measures grows with
frequency -- 2 pi f times the total delay -- so across a wide span it
rotates; the bearing is therefore asked at the coherence-weighted MEAN
frequency of the bins that were averaged, not at the dial. What rotation is
left across the span shows up honestly as a shorter sum, i.e. a lower
coherence.

Every answer is one of four kinds, taken from the profile's verdict: "hum"
(the rectifier comb), "lines" (something else periodic), "impulse" (an
electric fence, a thermostat, PLT) or "floor" (no verdict -- the band
itself). Impulses are broadband and brief, so the map's floor tracker
rejects the frames they land in: an "impulse" bearing is still the FLOOR's
bearing, and it is the impulses' only when they are what is holding the
floor up. Say that to the operator, not to the model.

`since` is when this bearing first held, unix epoch seconds. A bearing
counts as the same source while it stays within HOLD_DEG of the one before
it; a jump, or a gap longer than HOLD_GAP_S with nobody asking, starts the
clock again. It is a short memory on purpose: it answers "has that been
there all evening or did it just switch on", and nothing more.
"""
import math
import time

import numpy as np

from .noiseprofile import LINE_MIN_DB
from .spatial import SMOOTH_BINS, SOURCE_MIN_COHERENCE

# Fewer bins than the map's own smoothing width is less than one independent
# neighbourhood: a mean of nine smoothed bins is a mean of one measurement.
MIN_BINS = SMOOTH_BINS
# The same 0.4 the adapter calls nullable: under it the noise has no
# direction to null, so it has none worth printing either.
MIN_COHERENCE = 0.4
# A bin this far over the floor of its own neighbourhood is somebody
# transmitting, not the noise (see the module docstring).
STATION_OVER_DB = 10.0
SEGMENT_HZ = 50_000.0        # ... and that neighbourhood is this wide
SEGMENT_MAX = 64
# ... but a loud run WIDER than this is a hump of hash, not a transmission:
# nothing on HF is 20 kHz wide (SSB is 3, NFM 8, a broadcast channel 9-10).
STATION_MAX_HZ = 20_000.0
DC_GUARD_BINS = SMOOTH_BINS // 2 + 1
IMPULSE_MIN_PER_S = 1.0      # under this, impulses are not what is heard
HOLD_DEG = 15.0              # a bearing within this of the last one is the same source
HOLD_GAP_S = 60.0            # ... unless nobody asked for this long


def _wrap180(d):
    return (float(d) + 180.0) % 360.0 - 180.0


def _blank(reason):
    """No bearing and why, in the shape every caller reads."""
    return {"available": False, "kind": None, "phase_deg": None, "coherence": None,
            "bearing_deg": None, "mirror_deg": None, "bins": 0, "since": None,
            "reason": reason}


def kind_of(profile_status):
    """What the profile says this noise IS, named in the order an operator
    would say it: the mains comb first because it is the one with an address,
    then any other periodic line, then impulses, then the band itself."""
    st = profile_status or {}
    hum = float(st.get("hum_db") or 0.0)
    if st.get("mains_hz") and int(st.get("harmonics") or 0) >= 1 and hum >= LINE_MIN_DB:
        return "hum"
    if st.get("periodic"):
        return "lines"
    if float(st.get("impulses_per_s") or 0.0) >= IMPULSE_MIN_PER_S:
        return "impulse"
    return "floor"


def station_mask(level_db, rate_hz):
    """True for the bins that are somebody's transmission rather than noise:
    those standing STATION_OVER_DB over the median of their own SEGMENT_HZ
    of band, in a run under STATION_MAX_HZ wide. Natural FFT order in and
    out. The segment count comes from the span in hertz, so 62.5 kS/s (one
    segment, the whole span) and 2.04 MS/s (forty of them) both compare a
    bin against the same 50 kHz of band."""
    n = len(level_db)
    segs = int(min(SEGMENT_MAX, max(1, round(float(rate_hz) / SEGMENT_HZ))))
    while segs > 1 and n % segs:
        segs -= 1
    asc = np.fft.fftshift(np.asarray(level_db, dtype=np.float64))
    med = np.repeat(np.median(asc.reshape(segs, n // segs), axis=1), n // segs)
    loud = asc > med + STATION_OVER_DB
    bin_hz = float(rate_hz) / n
    edge = np.flatnonzero(np.diff(np.concatenate(
        ([0], loud.astype(np.int8), [0]))))
    for lo, hi in zip(edge[::2], edge[1::2]):          # runs of loud bins
        if (hi - lo) * bin_hz > STATION_MAX_HZ:
            loud[lo:hi] = False
    return np.fft.ifftshift(loud)


class BearingHistory:
    """How long the noise has pointed the same way. One bearing deep: the
    question it answers is 'since when', not 'what has the evening done'."""

    def __init__(self, hold_deg=HOLD_DEG, gap_s=HOLD_GAP_S):
        self.hold_deg = float(hold_deg)
        self.gap_s = float(gap_s)
        self.since = None            # when the bearing being held first held
        self.last = None             # ... and what it was, and when it was asked
        self.at = None

    def hold(self, bearing_deg, now):
        """Stamp this bearing and return when it first held (epoch seconds),
        or None when there is no bearing to hold."""
        if bearing_deg is None:
            self.since = self.last = self.at = None
            return None
        now = float(now)
        moved = (self.last is None
                 or abs(_wrap180(float(bearing_deg) - self.last)) > self.hold_deg)
        if self.since is None or moved or now - self.at > self.gap_s:
            self.since = now
        self.last, self.at = float(bearing_deg), now
        return self.since


def noise_bearing(spmap, profile_status, fit, center_hz, rate_hz=None,
                  now=None, history=None):
    """Where the noise the profile just described is coming from.

    `spmap` is the live core.spatial.SpatialMap, `profile_status` a
    core.noiseprofile.NoiseProfile.status() (or None -- the kind is then
    "floor"), `fit` a fitted core.compass.GlobalFit (or None, or an
    unfitted one: there is then a phase but no bearing), `center_hz` the
    map's centre and `rate_hz` its span, defaulting to the map's own.
    `history` is a BearingHistory the caller keeps between polls, for
    `since`.

    Returns the one dict the SITE page reads: available, kind, phase_deg,
    coherence, bearing_deg, mirror_deg, bins, since, reason. bearing_deg
    and mirror_deg are None -- with the reason -- when the compass has no
    fit yet or when the bins disagree too much to mean anything."""
    if spmap is None or getattr(spmap, "R", None) is None:
        return _blank("no spatial map yet")
    rate = float(spmap.rate_hz if rate_hz is None else rate_hz)
    if not (rate > 0.0):
        return _blank("no sample rate yet")
    now = time.time() if now is None else float(now)
    coh, steer, _m, level = spmap._analyse()      # the tuple sources() reads, cached
    nbins = len(coh)
    idx = np.arange(nbins)
    kk = np.where(idx < nbins / 2, idx, idx - nbins)          # natural order -> offset
    f_hz = float(center_hz) + kk * rate / nbins
    stale = spmap.stale_mask()
    sel = ((coh >= SOURCE_MIN_COHERENCE) & np.isfinite(level)
           & (np.abs(kk) > DC_GUARD_BINS) & ~station_mask(level, rate))
    if stale is not None:
        sel &= ~stale
    n = int(np.count_nonzero(sel))
    if n < MIN_BINS:
        return _blank(f"{n} coherent floor bin(s) with no station in them, "
                      f"{MIN_BINS} needed")

    # the log's convention, B relative to A: the map's steering angle is
    # angle(A conj(B)), the opposite sign to the ratio a beacon is scored at
    w = coh[sel]
    z = complex(np.sum(w * np.exp(-1j * steer[sel])))
    phase = math.degrees(math.atan2(z.imag, z.real))
    quality = float(abs(z) / n)
    at_hz = float(np.sum(w * f_hz[sel]) / max(float(np.sum(w)), 1e-30))
    kind = kind_of(profile_status)

    bearing = mirror = None
    if fit is None or not getattr(fit, "available", False):
        why = "no beacon fit yet" if fit is None else (fit.reason or "no beacon fit yet")
        reason = f"{kind} on {n} bins at {quality:.2f}, phase only: {why}"
    elif quality < MIN_COHERENCE:
        reason = (f"{kind} on {n} bins, but they point {quality:.2f} the same way "
                  f"and {MIN_COHERENCE:.2f} is needed: no one direction")
    else:
        ans = fit.bearing_from_phase(phase, at_hz)
        seen = ans.get("bearings_deg") or []
        bearing = round(float(seen[0]), 1) if seen else None
        if bearing is not None:
            mirror = round((2.0 * fit.baseline_deg - bearing) % 360.0, 1)
        reason = (f"{kind} on {n} bins at {quality:.2f} coherence, "
                  f"measured at {at_hz / 1e6:.3f} MHz")
        if ans.get("outside_model"):
            reason += ("; the phase is further than this spacing can turn, so the "
                       "bearing is the nearest end of the baseline")
        if len(seen) > 2:
            reason += ("; the pair is over half a wavelength apart here, so there "
                       "are grating lobes beyond the mirror")
    since = None if history is None else history.hold(bearing, now)
    return {"available": True, "kind": kind, "phase_deg": round(phase, 1),
            "coherence": round(quality, 3), "bearing_deg": bearing,
            "mirror_deg": mirror, "bins": n, "since": since, "reason": reason}
