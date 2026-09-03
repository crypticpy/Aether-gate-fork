#
# Aether-gate -- the time-signal stations as known directions on the low bands.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The eighteen NCDXF beacons live on five frequencies, all of them above 14
MHz. A compass fitted from them is a compass for the high bands, and the
operator who works 160 through 40 metres has been handed a calibration
source that never once appears in his passband.

The low bands have their own known directions, and they have had them for
sixty years: the standard-frequency stations. WWV in Fort Collins, WWVH on
Kauai, CHU in Ottawa, RWM outside Moscow, BPM at Lintong -- fixed
transmitters at published locations sending a carrier on published
frequencies from 2.5 to 25 MHz, all day, every day. Each one is a beacon in
everything but name: a locator gives the bearing, the carrier gives the
pair a phase, and the global fit in core.compass turns any two of them,
heard on different frequencies, into the same three numbers.

They differ from the NCDXF slots in two ways.

The first is easy: there is no schedule. Nobody has to be told who is on
air, because they all are, always. So instead of scoring a ten-second slot
when it ends, this scores a ten-second WINDOW once a minute, on whichever
carrier is in the span. Nothing else changes -- the same NarrowbandTap
mixes and decimates it and the same measure_window reads the carrier out,
which is why the result dict has the same shape as a beacon's and
SiteLog.beacon_result writes it without knowing the difference. The two
power-step fields a time signal has no equivalent of, steps_heard and
lowest_w, are None.

The second is not easy, and is the reason a `shared_with` key exists.
2.5, 5, 10 and 15 MHz carry WWV, WWVH and BPM AT THE SAME TIME. They are
not merely on the same band: they are on the same nominal carrier, within a
few tenths of a hertz of each other, and a 100 Hz search bin cannot tell
Colorado from Kauai from Shaanxi. A phase measured on 10 MHz is therefore
NOT a known direction, and a compass fitted from it would be fitted from a
bearing that is one of three. Those windows are still scored -- the SNRs
and the floors are honest propagation data -- but they are marked
ambiguous, `bearing_deg` is None, and core.compass drops them from the fit
on exactly that ground.

What is left unambiguous is enough to fit with: CHU on 3.330, 7.850 and
14.670, RWM on 4.996, 9.996 and 14.996 (four kilohertz clear of WWV, so
the search bin never confuses them), and WWV alone on 20 and 25 MHz. Three
of those are below 10 MHz, which is the whole point.

When the operator KNOWS which one he is hearing -- the wrong side of the
world is closed, the announcement was a woman's voice, the tone was 600 Hz
and not 440 -- set_assumed(10e6, "WWVH") says so, and from then on that
frequency is a known direction too. It is his judgement, not a
measurement, and the result says so with "assumed": True.
"""
from .beacons import (NarrowbandTap, bearing_distance, grid_to_latlon,
                      measure_window, CHUNK_S, EDGE_MARGIN_HZ)

WINDOW_S = 10.0              # a scored window is as long as an NCDXF slot
PERIOD_S = 60.0              # ... and one of them a minute is plenty

# call -> (name, locator, frequencies). Four-character squares, the form
# these stations are published in: half a square is at most fifty-odd
# kilometres, which at five thousand kilometres is well under a degree of
# bearing -- less than the fit's own residuals.
STATIONS = {
    "WWV": ("Fort Collins, Colorado", "DN70",
            (2_500_000.0, 5_000_000.0, 10_000_000.0, 15_000_000.0,
             20_000_000.0, 25_000_000.0)),
    "WWVH": ("Kekaha, Kauai, Hawaii", "BL01",
             (2_500_000.0, 5_000_000.0, 10_000_000.0, 15_000_000.0)),
    "CHU": ("Barrhaven, Ottawa, Ontario", "FN25",
            (3_330_000.0, 7_850_000.0, 14_670_000.0)),
    "RWM": ("Taldom, near Moscow", "KO85",
            (4_996_000.0, 9_996_000.0, 14_996_000.0)),
    "BPM": ("Lintong, Shaanxi, China", "OM94",
            (2_500_000.0, 5_000_000.0, 10_000_000.0, 15_000_000.0)),
}
# frequency -> the calls that are on it. More than one and the direction is
# not known, whatever the SNR says.
CARRIERS = {}
for _call, (_name, _grid, _freqs) in STATIONS.items():
    for _f in _freqs:
        CARRIERS.setdefault(_f, []).append(_call)
CARRIERS = {f: tuple(sorted(c)) for f, c in sorted(CARRIERS.items())}
FREQS_HZ = tuple(CARRIERS)
UNAMBIGUOUS_HZ = tuple(f for f, c in CARRIERS.items() if len(c) == 1)


def shared_with(f_hz, call=None):
    """The OTHER stations on a frequency: empty when it is one station's
    alone, which is the same test the compass fit applies."""
    calls = CARRIERS.get(float(f_hz), ())
    return [c for c in calls if c != call]


def station_table(grid=None):
    """Every station, with the bearing and distance from a locator when one
    is given -- the low-band half of the compass's known directions."""
    here = grid_to_latlon(grid) if grid else None
    rows = []
    for call, (name, sgrid, freqs) in STATIONS.items():
        brg = km = None
        if here is not None:
            b, k = bearing_distance(*here, *grid_to_latlon(sgrid))
            brg, km = round(b), round(k)
        rows.append({"call": call, "location": name, "grid": sgrid,
                     "bearing_deg": brg, "distance_km": km,
                     "frequencies_hz": list(freqs),
                     "unambiguous_hz": [f for f in freqs
                                        if len(CARRIERS[f]) == 1]})
    return sorted(rows, key=lambda r: r["call"])


class TimeSignalWatch:
    """One ten-second window a minute on whichever standard-frequency
    carrier is in the span, scored into the same shape a beacon slot is.

    Drive it exactly like BeaconWatch: update(a, b, center_hz, t_utc) with
    the aligned pair, and when `last` changes hand it to
    SiteLog.beacon_result. Give it the station's own locator with
    set_station() or every result is a measurement of an unknown
    direction."""

    def __init__(self, rate_hz, period_s=PERIOD_S, window_s=WINDOW_S, assume=None):
        self.rate_hz = float(rate_hz)
        self.period_s = float(period_s)
        self.window_s = float(window_s)
        self.tap = NarrowbandTap(self.rate_hz)
        self.station_grid = None
        self._station = None
        self.freq_hz = None                  # the carrier in the span, if any
        self._window = None                  # (period index, freq, t start)
        self._scored = None                  # the last period index scored
        self.results = {}                    # (freq, call) -> dict
        self.last = None
        self.assume = {}
        for f, call in (assume or {}).items():
            self.set_assumed(f, call)

    # --- the station, and the operator's judgement ---------------------------
    def set_station(self, grid):
        """The station's own locator ('' forgets); every result gains a
        bearing, except the ones no locator can help -- see the docstring."""
        g = str(grid or "").strip()
        if g:
            self._station = grid_to_latlon(g)
            self.station_grid = g[:2].upper() + g[2:4] + g[4:].lower()
        else:
            self._station, self.station_grid = None, None
        for r in self.results.values():
            self._geometry(r)

    def set_assumed(self, f_hz, call):
        """Name the station the operator believes he is hearing on a shared
        frequency (None forgets). Not a measurement: the result says
        'assumed' and the fit trusts it as far as he does."""
        f = float(f_hz)
        if call is None:
            self.assume.pop(f, None)
            return
        if call not in CARRIERS.get(f, ()):
            raise ValueError(f"{call} is not on {f / 1e6:.3f} MHz")
        self.assume[f] = call

    def _who(self, f_hz):
        """(call, location, grid, shared_with, ambiguous, assumed) for a
        carrier: one station, or the operator's pick, or all of them."""
        calls = CARRIERS[f_hz]
        pick = calls[0] if len(calls) == 1 else self.assume.get(f_hz)
        if pick is None:
            return ("/".join(calls), "one of %s" % ", ".join(calls), None,
                    list(calls), True, False)
        name, grid, _f = STATIONS[pick]
        return (pick, name, grid, shared_with(f_hz, pick), False, len(calls) > 1)

    def _geometry(self, r):
        grid = r.get("grid")
        if self._station is None or not grid:
            r["bearing_deg"], r["distance_km"] = None, None
            return
        brg, km = bearing_distance(*self._station, *grid_to_latlon(grid))
        r["bearing_deg"], r["distance_km"] = round(brg), round(km)

    # --- driving --------------------------------------------------------------
    def _carrier_in_span(self, center_hz):
        """The carrier nearest the centre that is safely inside the span."""
        half = self.rate_hz / 2.0 - EDGE_MARGIN_HZ
        near = [f for f in FREQS_HZ if abs(f - center_hz) <= half]
        return min(near, key=lambda f: abs(f - center_hz)) if near else None

    def update(self, a, b, center_hz, t_utc):
        f = self._carrier_in_span(center_hz)
        if f is None:
            self.freq_hz, self._window = None, None
            self.tap.reset()
            return
        moved = self.tap.set_offset(f - center_hz)   # a retune voids the window
        if moved or f != self.freq_hz:
            self.freq_hz = f
            self._window = None
        idx = int(t_utc // self.period_s)
        inside = (t_utc % self.period_s) < self.window_s
        if self._window is not None and (self._window[0] != idx or not inside):
            self._score(*self._window)
            self._window = None
            self.tap.reset()
        if inside and self._window is None and idx != self._scored:
            self._window = (idx, f, float(t_utc))
            self.tap.reset()
        if self._window is not None:
            self.tap.feed(a, b)

    # --- scoring a finished window -------------------------------------------
    def _score(self, idx, f_hz, t0):
        self._scored = idx
        if len(self.tap.spec) < int(0.8 * self.window_s / CHUNK_S):
            return                           # a partial window: a retune, or a start
        got = measure_window(self.tap.spec, self.rate_hz)
        if got is None:
            return
        fields, _dashes, _ref = got
        call, loc, grid, shared, ambiguous, assumed = self._who(f_hz)
        res = {"call": call, "location": loc, "grid": grid, "band_hz": f_hz,
               "at": float(t0 - (t0 % self.period_s)), "source": "time_signal",
               "shared_with": shared, "ambiguous": ambiguous, "assumed": assumed}
        res.update(fields)
        # the two fields only a four-step beacon has; the log keeps the
        # columns so one table holds both kinds
        res["steps_heard"] = None
        res["lowest_w"] = None
        prev = self.results.get((f_hz, call))
        heard_n = int(prev.get("heard_n", 0)) if prev else 0
        res["samples"] = (int(prev.get("samples", 1)) if prev else 0) + 1
        res["heard_n"] = heard_n + (1 if res["heard"] else 0)
        if res["heard"]:
            m = prev.get("snr_mean_db") if prev else None
            res["snr_mean_db"] = (round(res["snr_db"], 1) if m is None or heard_n == 0
                                  else round((m * heard_n + res["snr_db"])
                                             / (heard_n + 1), 1))
            res["last_heard"] = res["at"]
        else:
            res["snr_mean_db"] = prev.get("snr_mean_db") if prev else None
            res["last_heard"] = prev.get("last_heard") if prev else None
        self._geometry(res)
        self.results[(f_hz, call)] = res
        self.last = res

    # --- what the windows add up to ------------------------------------------
    def status(self, t_utc):
        left = None
        if self._window is not None:
            left = round(self.window_s - (t_utc % self.period_s), 1)
        rows = sorted(self.results.values(), key=lambda r: (r["band_hz"], -r["at"]))
        return {"available": True, "freq_hz": self.freq_hz,
                "in_window": self._window is not None, "seconds_left": left,
                "next_window_s": round(self.period_s - (t_utc % self.period_s), 1),
                "results": rows, "last": self.last,
                "station_grid": self.station_grid,
                "assumed": {str(int(f)): c for f, c in sorted(self.assume.items())},
                "stations": station_table(self.station_grid)}
