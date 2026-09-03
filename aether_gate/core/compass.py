#
# Aether-gate -- fitting the pair's array response from the beacons it heard.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Eighteen beacons is eighteen known directions. The site log keeps what the
pair measured towards each of them; this turns that into the one thing the
pair has never known about itself -- where its baseline points and how far
apart, in wavelengths, the two loops actually are -- and then runs the
measurement backwards: an unknown signal's phase becomes a bearing.

The model, for two elements:

    phi(theta) = phi0 + A cos(theta) + B sin(theta)
               = phi0 + k cos(theta - theta_b)

with theta the bearing in degrees TRUE, k = hypot(A, B) and
theta_b = atan2(B, A). Physically k = 2 pi (d / lambda) cos(elevation) and
theta_b is the baseline's bearing; phi0 is everything that is not geometry
-- cable length difference, the tuners' fixed offset, the residual of the
sample alignment. Because k scales with frequency the fit is PER BAND, and k
is reported so 21 MHz can later be checked against 14 MHz: the ratio of the
two k's should be the ratio of the two frequencies, and if it is not, one of
the fits is a wrap.

Phase wraps, so plain least squares on phi is wrong: a beacon at +179 and
one at -179 are two degrees apart and the normal equations think they are
358. The fit is done on the unit COMPLEX ratios r_i = exp(i phi_i) instead.
For a candidate (A, B) the best phi0 is free in closed form -- it is the
angle of

    z(A, B) = sum_i w_i r_i exp(-i (A cos theta_i + B sin theta_i))

-- and |z| / sum_i w_i is the fit quality, 1.0 when every beacon agrees
exactly and 0 when they cancel. That leaves a two-dimensional search, done
on a grid over [-2 pi, 2 pi]^2 (a pair further apart than a wavelength on HF
is not a pair, it is two stations) and then refined locally. Weights are the
coherence times a soft SNR term: a beacon heard at 3 dB has a phase, but not
one worth a degree.

What it needs: five beacons, spread. Three unknowns and three or four
wrapped measurements are not a fit but a family -- a wider pair whose extra
turn of phase lands on the same few beacons matches them EXACTLY, and the
quality is 1.0 for both. Every basin within a whisker of the best is
therefore refined, the smallest spacing wins (two loops on one roof are a
fraction of a wavelength apart; the alias is nearly always the wider pair),
and the others are reported: `unique` is False and `alias_k` lists them.
From five well-spread beacons up the ambiguity is gone. The cross-band check
on k is the second opinion.

What it cannot do: a two-element line array is symmetric about its baseline,
so theta and its reflection about theta_b give the same phase and always
will. bearing_from_phase returns BOTH. Only a third element that is not on
that line breaks the mirror -- which is why the log keeps complex ratios,
and why nothing here is written as "the phase" of "the pair".
"""
import math

import numpy as np

MIN_BEACONS = 3              # three unknowns: phi0, A, B
MIN_SPREAD_DEG = 60.0        # ... spread over at least this much of the compass
DISTINCT_DEG = 5.0           # bearings closer than this are one direction
K_MAX = 2.0 * math.pi        # half a wavelength of spacing at the top band
GRID = 65                    # the coarse (A, B) search
REFINE_ROUNDS = 6
REFINE_N = 9
COARSE_QUALITY = 0.10        # basins this close to the best are all refined
TIE_QUALITY = 0.005          # ... and among those, the closest loops win
MAX_BASINS = 16
SNR_KNEE_DB = 6.0            # below this a beacon's phase is discounted


def _wrap180(d):
    return (float(d) + 180.0) % 360.0 - 180.0


def _spread_deg(bearings):
    """How much of the compass a set of bearings covers: 360 minus the widest
    gap between neighbours. Two clusters 180 apart cover 180."""
    b = sorted(float(x) % 360.0 for x in bearings)
    if len(b) < 2:
        return 0.0
    gaps = [b[i + 1] - b[i] for i in range(len(b) - 1)] + [b[0] + 360.0 - b[-1]]
    return 360.0 - max(gaps)


def _distinct(bearings, tol=DISTINCT_DEG):
    """How many genuinely different directions are in the set."""
    out = []
    for x in sorted(float(v) % 360.0 for v in bearings):
        if not out or min(abs(x - out[-1]), 360.0 - abs(x - out[-1])) > tol:
            out.append(x)
    if len(out) > 1 and min(abs(out[-1] - out[0]), 360.0 - abs(out[-1] - out[0])) <= tol:
        out.pop()
    return len(out)


class BandFit:
    """One band's array response, or the reason there is not one yet."""

    def __init__(self, available, reason=None, band_hz=None, phi0_deg=None, k=None,
                 baseline_deg=None, quality=None, residuals=None, a=None, b=None,
                 alias_k=None):
        self.available = bool(available)
        self.reason = reason
        self.band_hz = band_hz
        self.phi0_deg = phi0_deg
        self.k = k
        self.baseline_deg = baseline_deg
        self.quality = quality
        self.a = a                       # the A of A cos theta, radians
        self.b = b
        self.residuals = residuals or []
        self.n_beacons = len(self.residuals)
        # spacings that fit these beacons just as well (see _best): more than
        # one and the geometry is a guess, however good the quality looks
        self.alias_k = alias_k or ([] if k is None else [k])
        self.unique = len(self.alias_k) <= 1

    MIRROR = ("a two-element line array cannot tell a bearing from its "
              "reflection about the baseline: both are given")

    def bearing_from_phase(self, phase_deg):
        """The bearings an unknown signal's inter-loop phase is consistent
        with. Two, mirrored about the baseline -- more when k > pi, where the
        phase wraps within the pattern and the pair has grating lobes."""
        if not self.available:
            return {"available": False, "reason": self.reason, "bearings_deg": []}
        d = math.radians(_wrap180(float(phase_deg) - self.phi0_deg))
        k = max(float(self.k), 1e-9)
        xs, outside = [], True
        for branch in (d, d + 2.0 * math.pi, d - 2.0 * math.pi):
            x = branch / k
            if abs(x) <= 1.0:
                xs.append(x)
                outside = False
        if outside:
            xs = [max(-1.0, min(1.0, d / k))]
        out = []
        for x in xs:
            ac = math.degrees(math.acos(max(-1.0, min(1.0, x))))
            for th in ((self.baseline_deg + ac) % 360.0, (self.baseline_deg - ac) % 360.0):
                if all(min(abs(th - o), 360.0 - abs(th - o)) > 0.05 for o in out):
                    out.append(th)
        return {"available": True, "phase_deg": round(_wrap180(phase_deg), 1),
                "bearings_deg": [round(t, 1) for t in sorted(out)],
                "outside_model": bool(outside), "mirror": self.MIRROR}

    def phase_at(self, bearing_deg):
        """The model's own phase towards a bearing (degrees, wrapped)."""
        if not self.available:
            return None
        th = math.radians(float(bearing_deg))
        return _wrap180(self.phi0_deg + math.degrees(self.a * math.cos(th)
                                                     + self.b * math.sin(th)))

    def as_dict(self):
        if not self.available:
            return {"available": False, "reason": self.reason, "band_hz": self.band_hz,
                    "n_beacons": self.n_beacons}
        return {"available": True, "band_hz": self.band_hz,
                "phi0_deg": round(self.phi0_deg, 1), "k": round(self.k, 4),
                "baseline_deg": round(self.baseline_deg, 1),
                "spacing_wavelengths": round(self.k / (2.0 * math.pi), 4),
                "quality": round(self.quality, 3), "n_beacons": self.n_beacons,
                "unique": self.unique, "alias_k": [round(x, 4) for x in self.alias_k],
                "beacons": self.residuals, "mirror": self.MIRROR}


def _refine(z_of, A0, B0, step):
    """A local grid, quartered each round, until (A, B) is exact enough that
    the residuals are the measurement's and not the search's."""
    zb = complex(z_of([A0], [B0])[0])
    for _ in range(REFINE_ROUNDS):
        ga = A0 + np.linspace(-step, step, REFINE_N)
        gb = B0 + np.linspace(-step, step, REFINE_N)
        AA, BB = np.meshgrid(ga, gb, indexing="ij")
        z = z_of(AA.ravel(), BB.ravel())
        j = int(np.argmax(np.abs(z)))
        A0, B0, zb = float(AA.ravel()[j]), float(BB.ravel()[j]), complex(z[j])
        step /= 4.0
    return A0, B0, zb


def _best(z_of, Ag, Bg, coarse, step):
    """The maximum of |z| over the grid -- but with the CLOSEST loops that fit.

    Phase wraps, so with only a handful of beacons the surface has several
    maxima of near-equal height: a larger spacing whose extra turns land on
    the same measured phases fits exactly as well. Every basin within
    TIE_QUALITY of the best is refined and the SMALLEST k among them wins,
    because two loops on one roof are a fraction of a wavelength apart and
    the alias is always the wider pair. This is a prior, not a measurement:
    with only three or four beacons it is the only thing separating the
    answers, which is why n_beacons and the cross-band k check matter."""
    order = np.argsort(-coarse)
    cut = float(coarse[order[0]]) - COARSE_QUALITY
    seeds = []
    for i in order[:512]:
        if coarse[i] < cut:
            break
        a, b = float(Ag[i]), float(Bg[i])
        if all(math.hypot(a - sa, b - sb) > 2.0 * step for sa, sb, _ in seeds):
            seeds.append((a, b, float(coarse[i])))
        if len(seeds) >= MAX_BASINS:
            break
    out = [_refine(z_of, a, b, step) for a, b, _ in seeds]
    top = max(abs(z) for _, _, z in out)
    close = [(A, B, z) for A, B, z in out if abs(z) >= top - TIE_QUALITY * top]
    ks = []
    for A, B, _ in close:                          # the spacings that fit as well
        k = math.hypot(A, B)
        if all(abs(k - o) > 1e-3 for o in ks):
            ks.append(k)
    A0, B0, zb = min(close, key=lambda e: math.hypot(e[0], e[1]))
    return A0, B0, zb, sorted(ks)


def fit(bearings_deg, ratios, weights=None, band_hz=None, calls=None):
    """Fit phi(theta) = phi0 + A cos theta + B sin theta to complex inter-loop
    ratios measured towards known bearings. `ratios` are complex (only their
    angle is used; give the magnitude anyway, an N-element fit will want it),
    `weights` the confidence in each -- coherence times an SNR term is what
    fit_from_log passes."""
    th, r, w, cl = [], [], [], []
    for i, (bd, z) in enumerate(zip(bearings_deg, ratios)):
        c = complex(z)
        if bd is None or not (math.isfinite(c.real) and math.isfinite(c.imag)) or c == 0:
            continue
        wt = 1.0 if weights is None else float(weights[i])
        if not (wt > 0.0):
            continue
        th.append(float(bd) % 360.0)
        r.append(c / abs(c))
        w.append(wt)
        cl.append(None if calls is None else calls[i])
    n = len(th)
    if n < MIN_BEACONS:
        return BandFit(False, f"{n} beacon(s) with a bearing and a ratio, "
                              f"{MIN_BEACONS} needed", band_hz)
    if _distinct(th) < MIN_BEACONS:
        return BandFit(False, f"only {_distinct(th)} distinct bearing(s) among "
                              f"{n} beacons, {MIN_BEACONS} needed", band_hz)
    spread = _spread_deg(th)
    if spread <= MIN_SPREAD_DEG:
        return BandFit(False, f"bearings span {spread:.0f} deg, more than "
                              f"{MIN_SPREAD_DEG:.0f} needed", band_hz)

    thr = np.radians(np.asarray(th))
    cos_t, sin_t = np.cos(thr), np.sin(thr)
    wr = np.asarray(w, dtype=np.float64) * np.asarray(r, dtype=np.complex128)
    wsum = float(np.sum(w))

    def z_of(A, B):
        """sum_i w_i r_i exp(-i(A cos th_i + B sin th_i)) for grids of A, B."""
        ph = np.outer(np.asarray(A, dtype=np.float64), cos_t) \
            + np.outer(np.asarray(B, dtype=np.float64), sin_t)
        return (wr[None, :] * np.exp(-1j * ph)).sum(axis=1)

    g = np.linspace(-K_MAX, K_MAX, GRID)
    AA, BB = np.meshgrid(g, g, indexing="ij")
    coarse = np.abs(z_of(AA.ravel(), BB.ravel()))
    step0 = float(g[1] - g[0])
    A0, B0, zb, alias_k = _best(z_of, AA.ravel(), BB.ravel(), coarse, step0)
    phi0 = math.degrees(math.atan2(zb.imag, zb.real))
    quality = abs(zb) / max(wsum, 1e-30)
    model = np.degrees(A0 * cos_t + B0 * sin_t) + phi0
    meas = np.degrees(np.angle(np.asarray(r, dtype=np.complex128)))
    rows = []
    for i in range(n):
        rows.append({"call": cl[i], "bearing_deg": round(th[i], 1),
                     "residual_deg": round(_wrap180(meas[i] - model[i]), 1),
                     "weight": round(float(w[i]), 3)})
    rows.sort(key=lambda e: e["bearing_deg"])
    return BandFit(True, None, band_hz, phi0_deg=phi0, k=math.hypot(A0, B0),
                   baseline_deg=math.degrees(math.atan2(B0, A0)) % 360.0,
                   quality=quality, residuals=rows, a=A0, b=B0, alias_k=alias_k)


def _weight(rec):
    """Coherence times a soft SNR term on the WEAKER loop: a beacon the pair
    only half agreed about, or one barely out of the noise, still counts --
    just less. Bounded, so one loud beacon cannot become the whole fit."""
    coh = rec.get("coherence")
    if coh is None or not (float(coh) > 0.0):
        return 0.0
    snrs = [s for s in (rec.get("snr_a_db"), rec.get("snr_b_db")) if s is not None]
    if snrs:
        lin = 10.0 ** (min(snrs) / 10.0)
        knee = 10.0 ** (SNR_KNEE_DB / 10.0)
        soft = lin / (lin + knee)
    else:
        soft = 1.0
    return float(coh) * soft


def latest_beacons(log, band_hz, since=None):
    """The most recent usable beacon line per callsign on one band."""
    out = {}
    for rec in log.read(kind="beacon", since=since):
        if band_hz is not None and rec.get("band_hz") != float(band_hz):
            continue
        if rec.get("callsign"):
            out[rec["callsign"]] = rec                 # the file is in time order
    return list(out.values())


def fit_from_log(log, band_hz, since=None):
    """The band's array response from what the site log has heard."""
    rows = [r for r in latest_beacons(log, band_hz, since)
            if r.get("ratio") and r.get("bearing_deg") is not None]
    return fit([r["bearing_deg"] for r in rows],
               [complex(r["ratio"][0], r["ratio"][1]) for r in rows],
               [_weight(r) for r in rows], band_hz=band_hz,
               calls=[r["callsign"] for r in rows])


def pattern_from_log(log, band_hz, since=None):
    """Loop A against loop B by bearing: each loop's INSTALLED gain pattern,
    which is the number that says which way to physically turn one. Note the
    sign -- positive means A hears that direction better. (core.beacons'
    own pattern() reports b_minus_a_db, the other way round.)"""
    rows = []
    for r in latest_beacons(log, band_hz, since):
        if (r.get("bearing_deg") is None or r.get("snr_a_db") is None
                or r.get("snr_b_db") is None):
            continue
        rows.append({"call": r.get("callsign"), "bearing_deg": r["bearing_deg"],
                     "distance_km": r.get("distance_km"),
                     "snr_a_db": r["snr_a_db"], "snr_b_db": r["snr_b_db"],
                     "a_minus_b_db": round(r["snr_a_db"] - r["snr_b_db"], 1),
                     "floor_a_db": r.get("floor_a_db"), "floor_b_db": r.get("floor_b_db")})
    return sorted(rows, key=lambda e: e["bearing_deg"])


def compass_json(log, bands_hz=None, since=None, phase_deg=None):
    """A JSON-ready compass for every band the log has beacons on. Give
    `phase_deg` (an unknown signal's measured inter-loop phase, e.g. a
    tracker's) and each band also answers what bearings it could be."""
    if bands_hz is None:
        seen = []
        for rec in log.read(kind="beacon", since=since):
            b = rec.get("band_hz")
            if b is not None and b not in seen:
                seen.append(b)
        bands_hz = sorted(seen)
    bands = []
    for b in bands_hz:
        f = fit_from_log(log, b, since=since)
        row = f.as_dict()
        row["pattern"] = pattern_from_log(log, b, since=since)
        if phase_deg is not None:
            row["bearing_from_phase"] = f.bearing_from_phase(phase_deg)
        bands.append(row)
    fitted = [b for b in bands if b.get("available")]
    # one sentence for the window when nothing fits: the best band's own
    # reason, or that no band has been heard at all
    reason = (None if fitted
              else (max(bands, key=lambda r: r.get("n_beacons", 0)).get("reason")
                    if bands else "no beacon band heard yet, %d needed" % MIN_BEACONS))
    return {"available": bool(fitted), "reason": reason, "bands": bands,
            "fitted": len(fitted), "model": "phi0 + k cos(theta - theta_b)",
            "mirror": BandFit.MIRROR}
