#
# Aether-gate — the amateur bands, and which one the dial is on.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Which band a frequency is on, and whether two frequencies are on the same one.

The band is the unit the station's memory is keyed by. A loop pair's phase for a
given bearing is a function of wavelength, so a weight, a talker fingerprint or
a passband earned on 20 m is the wrong answer on 40 m: anything the gate keeps
across a retune has to be able to say which band it was measured on (AGENTS.md,
"Keep what the station learned" -- key by the world, not by the widget).

EDGES ARE IARU REGION 2, where this station is. Region 1 and Region 3 differ
(40 m ends at 7.200 in Region 1, 6 m at 52.000, 160 m starts at 1.810), so a
later `region` flag picks the table; nothing outside this module knows a number.

A band is IDENTIFIED BY ITS CENTRE in Hz: one integer, stable across runs, and
readable as a frequency. The names in the table are for people and for the
region flag to key off; nothing keys memory by them.

Pure: no clock, no IO, no thread. BandWatch is the little bit of state that goes
with it -- where the dial is, when it last moved, when it last changed band --
and the caller passes the wall clock in, because /diversity publishes those two
stamps as epoch seconds so the app can say "3 h ago" after a restart.
"""

# name, low Hz, high Hz -- IARU Region 2. 60 m is the ARRL channel span
# 5330.5-5406.5 kHz, which holds both the five US channels and the WRC-15
# 5351.5-5366.5 allocation inside it.
BANDS = (
    ("160 m", 1_800_000, 2_000_000),
    ("80 m", 3_500_000, 4_000_000),
    ("60 m", 5_330_500, 5_406_500),
    ("40 m", 7_000_000, 7_300_000),
    ("30 m", 10_100_000, 10_150_000),
    ("20 m", 14_000_000, 14_350_000),
    ("17 m", 18_068_000, 18_168_000),
    ("15 m", 21_000_000, 21_450_000),
    ("12 m", 24_890_000, 24_990_000),
    ("10 m", 28_000_000, 29_700_000),
    ("6 m", 50_000_000, 54_000_000),
)


def band_of(hz):
    """The centre Hz of the amateur band `hz` is on, or None off the bands."""
    try:
        f = float(hz)
    except (TypeError, ValueError):
        return None
    if f != f:                                   # NaN is nowhere
        return None
    for _name, lo, hi in BANDS:
        if lo <= f <= hi:
            return (lo + hi) // 2
    return None


def band_name(hz):
    """"20 m", or None off the bands. For sentences, never for a key."""
    try:
        f = float(hz)
    except (TypeError, ValueError):
        return None
    for name, lo, hi in BANDS:
        if lo <= f <= hi:
            return name
    return None


def band_changed(a, b):
    """True when `a` and `b` are not on the same amateur band.

    Two frequencies that are both off the bands count as the same band (None):
    out-of-band tuning is not a reason to put a tool back or drop a fingerprint,
    and a station that lives outside the table would otherwise churn on every
    dial move.
    """
    return band_of(a) != band_of(b)


class BandWatch:
    """Where the dial is, when it last moved, and when it last changed band."""

    def __init__(self, on_band=None):
        self.retuned_at = None          # epoch s of the last hardware retune
        self.band_hz = None             # band_of the tuned frequency
        self.band_changed_at = None     # epoch s of the last band change
        self.tuned_hz = None
        self._on_band = on_band         # called (band_hz, tuned_hz) on a change
        self._seeded = False

    def retuned(self, old_hz, new_hz, now):
        """The hardware centre moved. Returns True when that left the band.

        The stamp is the event the app watches; the band itself comes from
        `tune`, because the centre is offset a quarter-span from what the
        operator is listening to (soapy's _dc_offset_hz).
        """
        self.retuned_at = float(now)
        return band_changed(old_hz, new_hz)

    def tune(self, hz, now):
        """Where the operator is now. Returns True when the band changed.

        The FIRST call seeds the band and is not a change: coming up on 20 m is
        not tuning away from anything.
        """
        self.tuned_hz = None if hz is None else float(hz)
        band = band_of(hz)
        if self._seeded and band == self.band_hz:
            return False
        first, self._seeded = not self._seeded, True
        self.band_hz = band
        if not first:
            self.band_changed_at = float(now)
        if self._on_band is not None:
            self._on_band(band, self.tuned_hz)
        return not first

    def status(self):
        """The three additive keys /diversity publishes."""
        return {"retuned_at": self.retuned_at, "band_hz": self.band_hz,
                "band_changed_at": self.band_changed_at}
