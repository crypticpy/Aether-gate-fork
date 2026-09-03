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

--- and then the same pair, on every band -------------------------------

The per-band fit has one flaw the operator meets on the first evening: the
beacons live on five frequencies from 14 to 28 MHz, and 80 m is not one of
them. A compass earned on 20 m has to serve on 3.8 MHz or it is a toy.

It can, because phi0 and k are not free numbers. Two loops separated by d
metres on a bearing theta_b, fed through cables that differ by dtau
seconds, measure

    phase(f, theta) = 2 pi f dtau + (2 pi f d / c) cos(theta - theta_b)

-- THREE numbers, dtau, d and theta_b, for every band at once (elevation is
ignored for now; a high-angle signal reads as a slightly closer pair). Each
band's phi0 is 2 pi f dtau and each band's k is 2 pi f d / c, so a beacon
heard on 24.930 constrains the same three unknowns as one heard on 14.100,
just with a longer lever. That is also what kills the per-band alias: a
wider pair that lands on the same four phases at 14.100 lands somewhere
else entirely at 28.200, and the alias the one-band fit had to report as
unique:false is simply gone.

Written as a delay the model is LINEAR in three delays -- dtau, and the two
components dx/c, dy/c of the baseline -- which is why the search box is the
same +-200 ns on all three axes (60 m is 200 ns of light). It is still a
wrapped objective, so the search is still a grid and a local refine, and
the quality is still the weighted mean of cos(residual). One difference
from the per-band fit: there is NO free constant here. The per-band phi0
absorbs any fixed phase offset; the global fit has to explain it as a
delay, and a constant that is not a delay shows up as a poor global quality
under good per-band ones. That comparison is the point of keeping both --
compass_json reports, per band, how far that band's own baseline sits from
the global one.
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

C_M_S = 299792458.0
MIN_GLOBAL_BEACONS = 4       # ... spread over MIN_GLOBAL_BANDS bands
MIN_GLOBAL_BANDS = 2
MIN_ONE_BAND_BEACONS = 5     # ... or this many on a single band, well spread
D_MIN_M, D_MAX_M = 0.5, 60.0
DTAU_MAX_S = 200e-9          # 200 ns is 60 m of cable difference
GLOBAL_OVERSAMPLE = 6.0      # grid steps per cycle of the highest frequency
GLOBAL_GRID_MAX = 96         # ... but never more axis points than this
GLOBAL_SEEDS = 48            # a wrapped objective has a ladder of basins
GLOBAL_REFINE_ROUNDS = 8
GLOBAL_REFINE_N = 5
GLOBAL_TIE = 0.005           # basins whose quality is this close all get told
GLOBAL_CHUNK = 20_000        # parameter rows evaluated at a time
GLOBAL_MODEL = "2 pi f (dtau + (d/c) cos(theta - theta_b))"


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


class GlobalFit:
    """The pair itself -- a cable delay, a spacing and a bearing -- or the
    reason the bands heard so far cannot say. Unlike BandFit this answers on
    ANY frequency, including the ones no beacon lives on."""

    def __init__(self, available, reason=None, dtau_s=None, d_m=None,
                 baseline_deg=None, quality=None, residuals=None, bands=None,
                 alternatives=None, n_beacons=0, n_bands=0):
        self.available = bool(available)
        self.reason = reason
        self.dtau_s = dtau_s
        self.d_m = d_m
        self.baseline_deg = baseline_deg
        self.quality = quality
        self.residuals = residuals or []
        self.bands = bands or []
        # solutions that fit these beacons as well as the chosen one does:
        # more than none and the geometry is still a guess
        self.alternatives = alternatives or []
        self.unique = not self.alternatives
        self.n_beacons = int(n_beacons)
        self.n_bands = int(n_bands)

    MIRROR = BandFit.MIRROR

    @property
    def dtau_ns(self):
        return None if self.dtau_s is None else self.dtau_s * 1e9

    def k_at(self, f_hz):
        """The per-band k this pair has at a frequency, radians."""
        return 2.0 * math.pi * float(f_hz) * self.d_m / C_M_S

    def phi0_at(self, f_hz):
        """The per-band phi0 this pair has at a frequency, degrees."""
        return math.degrees(2.0 * math.pi * float(f_hz) * self.dtau_s)

    def phase_at(self, bearing_deg, f_hz):
        """The model's phase towards a bearing at a frequency (deg, wrapped)."""
        if not self.available:
            return None
        th = math.radians(float(bearing_deg) - self.baseline_deg)
        return _wrap180(self.phi0_at(f_hz)
                        + math.degrees(self.k_at(f_hz) * math.cos(th)))

    def bearing_from_phase(self, phase_deg, f_hz):
        """The bearings an unknown signal's inter-loop phase is consistent
        with AT THAT FREQUENCY. Two, mirrored about the baseline -- more once
        the pair is over half a wavelength apart, where the phase wraps
        inside the pattern and the extra answers are grating lobes."""
        if not self.available:
            return {"available": False, "reason": self.reason, "bearings_deg": []}
        f = float(f_hz)
        k = max(self.k_at(f), 1e-9)
        lam = C_M_S / max(abs(f), 1e-9)
        d_over_lam = self.d_m / lam
        rest = math.radians(_wrap180(float(phase_deg) - self.phi0_at(f)))
        xs, outside = [], True
        m_lo = int(math.ceil((-k - rest) / (2.0 * math.pi)))
        m_hi = int(math.floor((k - rest) / (2.0 * math.pi)))
        for m in range(m_lo, min(m_hi, m_lo + 31) + 1):
            xs.append((rest + 2.0 * math.pi * m) / k)
            outside = False
        if outside:
            xs = [max(-1.0, min(1.0, rest / k))]
        out = []
        for x in xs:
            ac = math.degrees(math.acos(max(-1.0, min(1.0, x))))
            for th in ((self.baseline_deg + ac) % 360.0,
                       (self.baseline_deg - ac) % 360.0):
                if all(min(abs(th - o), 360.0 - abs(th - o)) > 0.05 for o in out):
                    out.append(th)
        return {"available": True, "phase_deg": round(_wrap180(phase_deg), 1),
                "f_hz": f, "bearings_deg": [round(t, 1) for t in sorted(out)],
                "outside_model": bool(outside),
                "d_over_lambda": round(d_over_lam, 3),
                "grating_lobes": bool(d_over_lam > 0.5), "mirror": self.MIRROR}

    def as_dict(self):
        if not self.available:
            return {"available": False, "reason": self.reason,
                    "n_beacons": self.n_beacons, "n_bands": self.n_bands,
                    "model": GLOBAL_MODEL}
        return {"available": True, "dtau_ns": round(self.dtau_ns, 2),
                "d_m": round(self.d_m, 3), "baseline_deg": round(self.baseline_deg, 1),
                "quality": round(self.quality, 3), "n_beacons": self.n_beacons,
                "n_bands": self.n_bands, "unique": self.unique,
                "alternatives": self.alternatives, "bands": self.bands,
                "beacons": self.residuals, "model": GLOBAL_MODEL,
                "mirror": self.MIRROR}


def _global_rows(events):
    """(bearing, frequency, unit ratio, weight, call) from site-log beacon
    lines -- the NCDXF slots and the time-signal windows alike, since both
    are written by SiteLog.beacon_result and both carry a known bearing."""
    rows = []
    for rec in events:
        bd, f, z = rec.get("bearing_deg"), rec.get("band_hz"), rec.get("ratio")
        if bd is None or f is None or not z:
            continue
        c = z if isinstance(z, complex) else complex(float(z[0]), float(z[1]))
        if not (math.isfinite(c.real) and math.isfinite(c.imag)) or c == 0:
            continue
        w = _weight(rec)
        if not (w > 0.0) or not (float(f) > 0.0):
            continue
        rows.append((float(bd) % 360.0, float(f), c / abs(c), w,
                     rec.get("callsign") or rec.get("call")))
    return rows


def _global_refine(j_of, p0, step):
    """A local grid on (dtau, dx/c, dy/c), thirded each round."""
    p0 = np.asarray(p0, dtype=np.float64)
    best = float(j_of(p0[None, :])[0])
    for _ in range(GLOBAL_REFINE_ROUNDS):
        axes = [p0[i] + np.linspace(-step, step, GLOBAL_REFINE_N) for i in range(3)]
        grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        j = j_of(grid)
        i = int(np.argmax(j))
        p0, best = grid[i].copy(), float(j[i])
        step /= 3.0
    return p0, best


def _global_reason(n, n_bands, spread):
    """Why this handful of beacons is not yet a pair, in one sentence."""
    if n < MIN_GLOBAL_BEACONS:
        return (f"{n} beacon(s) with a bearing and a ratio, "
                f"{MIN_GLOBAL_BEACONS} needed over {MIN_GLOBAL_BANDS} bands")
    if n_bands < MIN_GLOBAL_BANDS and n < MIN_ONE_BAND_BEACONS:
        return (f"{n} beacon(s) on one band, {MIN_ONE_BAND_BEACONS} needed "
                f"unless a second band is heard")
    if spread <= MIN_SPREAD_DEG:
        return (f"bearings span {spread:.0f} deg, more than "
                f"{MIN_SPREAD_DEG:.0f} needed")
    return None


def fit_global(events):
    """Fit one pair -- dtau, d, theta_b -- to beacon lines from EVERY band.

    `events` are site-log beacon records (band_hz, bearing_deg, ratio,
    coherence, snr_a_db, snr_b_db); the weights are the per-band fit's, and
    the residuals are on the unit circle, so wraps cost nothing. The search
    is a coarse grid over the three delays followed by a local refine of
    every basin that came close, because a wrapped objective has many peaks:
    the best is taken, and any other within GLOBAL_TIE of it is REPORTED
    rather than hidden -- that is what `unique` means.

    One alias is structural rather than statistical. If every band heard is
    a multiple of some f0 -- 14.100, 21.150 and 28.200 are all multiples of
    7.050 MHz -- then adding 1/f0 to dtau adds a whole turn on every one of
    them and cannot be detected, however many beacons there are. It comes
    back as an alternative with the same d and baseline and a dtau 141.8 ns
    away. A band that is not commensurate with the others (18.110, 24.930,
    or any time-signal carrier) removes it; the shortest cable difference is
    the tie-break until one is heard."""
    rows = _global_rows(events)
    n = len(rows)
    freqs = sorted({r[1] for r in rows})
    spread = _spread_deg([r[0] for r in rows]) if n else 0.0
    why = _global_reason(n, len(freqs), spread)
    if why is not None:
        return GlobalFit(False, why, n_beacons=n, n_bands=len(freqs))

    th = np.radians(np.asarray([r[0] for r in rows]))
    f = np.asarray([r[1] for r in rows])
    wr = (np.asarray([r[3] for r in rows], dtype=np.float64)
          * np.asarray([r[2] for r in rows], dtype=np.complex128))
    wsum = float(sum(r[3] for r in rows))
    u, v = np.cos(th), np.sin(th)
    two_pi_f = 2.0 * math.pi * f

    def j_of(P):
        """The weighted mean of cos(residual) for rows of (dtau, dx/c, dy/c).
        Real, not |.|: nothing here is free to absorb a constant."""
        P = np.asarray(P, dtype=np.float64).reshape(-1, 3)
        out = np.empty(len(P), dtype=np.float64)
        for s in range(0, len(P), GLOBAL_CHUNK):
            q = P[s:s + GLOBAL_CHUNK]
            tau = q[:, 0:1] + q[:, 1:2] * u[None, :] + q[:, 2:3] * v[None, :]
            out[s:s + GLOBAL_CHUNK] = (wr[None, :]
                                       * np.exp(-1j * two_pi_f[None, :] * tau)
                                       ).sum(axis=1).real
        return out / max(wsum, 1e-30)

    # the grid has to sample the fastest wrap: a whole turn of phase at the
    # top frequency is 1/f of delay, and GLOBAL_OVERSAMPLE steps of it
    lim = D_MAX_M / C_M_S
    npts = int(min(GLOBAL_GRID_MAX,
                   max(9, math.ceil(2.0 * lim * GLOBAL_OVERSAMPLE * float(f.max())))))
    ax_t = np.linspace(-DTAU_MAX_S, DTAU_MAX_S, npts)
    ax_d = np.linspace(-lim, lim, npts)
    G = np.stack(np.meshgrid(ax_t, ax_d, ax_d, indexing="ij"), axis=-1).reshape(-1, 3)
    d_grid = np.hypot(G[:, 1], G[:, 2]) * C_M_S
    G = G[(d_grid >= D_MIN_M) & (d_grid <= D_MAX_M)]
    coarse = j_of(G)
    step = float(max(ax_t[1] - ax_t[0], ax_d[1] - ax_d[0]))

    seeds = []
    for i in np.argsort(-coarse):
        p = G[i]
        if all(float(np.max(np.abs(p - s))) > 1.5 * step for s in seeds):
            seeds.append(p)
        if len(seeds) >= GLOBAL_SEEDS:
            break
    sols = [_global_refine(j_of, p, step) for p in seeds]
    sols = [(p, q) for p, q in sols
            if D_MIN_M <= math.hypot(p[1], p[2]) * C_M_S <= D_MAX_M]
    if not sols:
        return GlobalFit(False, "no spacing in [%.1f, %.0f] m fits these beacons"
                                % (D_MIN_M, D_MAX_M), n_beacons=n, n_bands=len(freqs))
    top = max(q for _p, q in sols)
    close = [(p, q) for p, q in sols if q >= top - GLOBAL_TIE]
    # the same prior the per-band fit uses: two loops on one roof are the
    # close pair, so among equals the smallest spacing is the answer, and
    # among those the shortest cable difference -- and every equal that is
    # NOT the answer is listed
    dmin = min(math.hypot(p[1], p[2]) for p, _q in close) * C_M_S
    close.sort(key=lambda e: (math.hypot(e[0][1], e[0][2]) * C_M_S > dmin + 0.05,
                              abs(e[0][0]), -e[1]))
    p_best, quality = close[0]
    seen = [_global_solution(p_best, quality)]
    alts = []
    for p, q in close[1:]:
        row = _global_solution(p, q)
        if all(abs(row["d_m"] - o["d_m"]) > 0.05
               or abs(row["dtau_ns"] - o["dtau_ns"]) > 1.0
               or abs(_wrap180(row["baseline_deg"] - o["baseline_deg"])) > 1.0
               for o in seen):
            seen.append(row)
            alts.append(row)

    dtau = float(p_best[0])
    d_m = math.hypot(p_best[1], p_best[2]) * C_M_S
    baseline = math.degrees(math.atan2(p_best[2], p_best[1])) % 360.0
    tau = p_best[0] + p_best[1] * u + p_best[2] * v
    model = np.degrees(2.0 * math.pi * f * tau)
    meas = np.degrees(np.angle(np.asarray([r[2] for r in rows])))
    resid = [_wrap180(meas[i] - model[i]) for i in range(n)]
    beacons = [{"call": rows[i][4], "band_hz": rows[i][1],
                "bearing_deg": round(rows[i][0], 1),
                "residual_deg": round(resid[i], 1),
                "weight": round(rows[i][3], 3)} for i in range(n)]
    beacons.sort(key=lambda e: (e["band_hz"], e["bearing_deg"]))
    bands = []
    for band in freqs:
        rs = [resid[i] for i in range(n) if rows[i][1] == band]
        ws = [rows[i][3] for i in range(n) if rows[i][1] == band]
        q = (sum(w * math.cos(math.radians(r)) for w, r in zip(ws, rs))
             / max(sum(ws), 1e-30))
        bands.append({"band_hz": band, "n_beacons": len(rs),
                      "quality": round(q, 3),
                      "rms_residual_deg": round(math.sqrt(sum(r * r for r in rs)
                                                          / len(rs)), 1),
                      "max_residual_deg": round(max(abs(r) for r in rs), 1)})
    return GlobalFit(True, None, dtau_s=dtau, d_m=d_m, baseline_deg=baseline,
                     quality=quality, residuals=beacons, bands=bands,
                     alternatives=alts, n_beacons=n, n_bands=len(freqs))


def _global_solution(p, q):
    """One point of the search as the three numbers a person reads."""
    return {"dtau_ns": round(float(p[0]) * 1e9, 2),
            "d_m": round(math.hypot(p[1], p[2]) * C_M_S, 3),
            "baseline_deg": round(math.degrees(math.atan2(p[2], p[1])) % 360.0, 1),
            "quality": round(float(q), 3)}


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


def fit_global_from_log(log, since=None):
    """The pair itself, from every band the log has heard -- the latest line
    per callsign PER BAND, so one beacon worked on three bands is three
    measurements of the same three unknowns."""
    latest = {}
    for rec in log.read(kind="beacon", since=since):
        if rec.get("callsign") and rec.get("band_hz") is not None:
            latest[(rec["band_hz"], rec["callsign"])] = rec
    return fit_global(latest.values())


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


def compass_json(log, bands_hz=None, since=None, phase_deg=None, f_hz=None):
    """A JSON-ready compass: the pair fitted GLOBALLY from every band, and
    the per-band fits that are the check on it.

    Give `phase_deg` (an unknown signal's measured inter-loop phase, e.g. a
    tracker's) and each band answers what bearings it could be; give `f_hz`
    as well -- the frequency that phase was measured AT, which is the active
    slice's -- and the global fit answers for that frequency whether or not
    a beacon has ever been heard on it. That is the 80 m answer."""
    if bands_hz is None:
        seen = []
        for rec in log.read(kind="beacon", since=since):
            b = rec.get("band_hz")
            if b is not None and b not in seen:
                seen.append(b)
        bands_hz = sorted(seen)
    g = fit_global_from_log(log, since=since)
    bands = []
    for b in bands_hz:
        f = fit_from_log(log, b, since=since)
        row = f.as_dict()
        row["pattern"] = pattern_from_log(log, b, since=since)
        if phase_deg is not None:
            row["bearing_from_phase"] = f.bearing_from_phase(phase_deg)
        # the second opinion, both ways: what the global pair says this band's
        # k should be, and how far this band's own baseline sits from it
        if g.available:
            row["vs_global"] = {
                "k_global": round(g.k_at(b), 4),
                "baseline_global_deg": round(g.baseline_deg, 1),
                "disagreement_deg": (None if not f.available else
                                     round(_wrap180(f.baseline_deg - g.baseline_deg), 1)),
            }
        bands.append(row)
    fitted = [b for b in bands if b.get("available")]
    # one sentence for the window when nothing fits: the best band's own
    # reason, or that no band has been heard at all
    reason = (None if fitted
              else (max(bands, key=lambda r: r.get("n_beacons", 0)).get("reason")
                    if bands else "no beacon band heard yet, %d needed" % MIN_BEACONS))
    out = {"available": bool(fitted), "reason": reason, "bands": bands,
           "fitted": len(fitted), "model": "phi0 + k cos(theta - theta_b)",
           "global": g.as_dict(), "mirror": BandFit.MIRROR}
    if g.available and phase_deg is not None and f_hz is not None:
        out["bearing"] = g.bearing_from_phase(phase_deg, f_hz)
    return out
