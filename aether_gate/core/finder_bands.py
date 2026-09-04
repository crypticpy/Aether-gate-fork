#
# Aether-gate — the three sub-bands where the mode is decided by the band plan.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Where a signal IS tells you what it is, when nothing else can.

FT8 keys 8 tones 6.25 Hz apart inside a 50 Hz channel and changes them every
0.16 s; the finder's map is 244 Hz per point and its ring is 8.5 s long, so it
can see neither the tones nor the 15 s frame the mode is built on. What it CAN
see is that the block is wide, filled edge to edge, on all the time and not
paced by syllables -- and that it sits in the three kilohertz above 14.074 MHz,
which on an amateur band is not a coincidence. On 2026-09-03 the live gate
called 14074.0 "voice 0.41" and 14080.5 "voice 0.65": the FT8 and the FT4
window on 20 m, both of them, at four in the afternoon.

So the structural verdict is refined by the band plan, and only in the
direction the band plan can be trusted: a window over an FT8/FT4/PSK31
allocation may not be called speech unless it is REALLY paced like speech
(SPOKEN_KEEP below), and is otherwise named for the mode that lives there.
Nothing here invents a signal: refine() only relabels a window the finder has
already decided something is in.

The dial frequencies are the standard ones; the occupied stretch is the three
kilohertz above the dial for FT8/FT4 (the whole USB passband is used) and one
kilohertz for PSK31, which is where the operators actually are.
"""

FT8_DIALS_HZ = (1_840_000.0, 3_573_000.0, 5_357_000.0, 7_074_000.0, 10_136_000.0,
                14_074_000.0, 18_100_000.0, 21_074_000.0, 24_915_000.0,
                28_074_000.0, 50_313_000.0, 50_323_000.0)
FT4_DIALS_HZ = (3_575_000.0, 7_047_500.0, 10_140_000.0, 14_080_000.0,
                18_104_000.0, 21_140_000.0, 24_919_000.0, 28_180_000.0,
                50_318_000.0)
PSK31_DIALS_HZ = (1_838_000.0, 3_580_000.0, 7_040_000.0, 7_070_000.0,
                  10_142_000.0, 14_070_000.0, 18_100_000.0, 21_070_000.0,
                  24_920_000.0, 28_120_000.0)

FT_WIDTH_HZ = 3_000.0
PSK_WIDTH_HZ = 1_000.0
OVERLAP_FRAC = 0.5        # of the narrower of (window, allocation)
SPOKEN_KEEP = 0.80        # syllabic share above which speech keeps its name
                          # even inside a digital allocation: somebody working
                          # phone on the edge of the FT8 window is rare but
                          # real, and 0.80 is where measured phone sits
                          # (0.81-0.91) and measured FT8 does not (0.52).

_ALLOCATIONS = tuple(
    [("ft8", d, d + FT_WIDTH_HZ) for d in FT8_DIALS_HZ]
    + [("ft4", d, d + FT_WIDTH_HZ) for d in FT4_DIALS_HZ]
    + [("psk31", d, d + PSK_WIDTH_HZ) for d in PSK31_DIALS_HZ])

# A window may be relabelled from these; a window the finder called "cw" is
# left alone (a CW operator inside the FT8 window is a real, if rude, thing,
# and keying is measured evidence rather than a guess). Nor "noise", which
# since 2026-09-03 means the finder decided there is no station here -- the
# band plan is a prior on what a signal IS and may not be the whole reason a
# signal exists, which is what naming a dead window "ft8" amounts to.
REFINABLE = ("voice", "data", "carrier", "signal", "rtty")


def sub_band(lo_hz, hi_hz, overlap=OVERLAP_FRAC):
    """(mode, dial) the band plan puts between lo_hz and hi_hz, or None.

    The overlap is measured against the NARROWER of the stretch and the
    allocation, so a 2.7 kHz window inside a 3 kHz FT8 stretch and a 12 kHz
    window straddling a 1 kHz PSK stretch are both judged on how much of the
    small one is covered. Pass the stretch the energy is actually IN rather
    than the whole window: a window that catches the top kilohertz of the FT8
    block has that kilohertz wholly inside the allocation, and calling it
    two-thirds bare band is how the finder ends up naming it something else.
    """
    lo, hi = float(min(lo_hz, hi_hz)), float(max(lo_hz, hi_hz))
    span = max(hi - lo, 1e-9)
    best, best_frac = None, 0.0
    for name, a, b in _ALLOCATIONS:
        cover = min(hi, b) - max(lo, a)
        if cover <= 0.0:
            continue
        frac = cover / min(span, b - a)
        if frac >= overlap and frac > best_frac:
            best, best_frac = (name, a), frac
    return best


def refine(kind, conf, lo_hz, hi_hz, syllabic=0.0, present=1.0):
    """(kind, conf, dial) after the band plan has had its say.

    `dial` is the allocation's own dial frequency when the band plan named the
    signal, and None otherwise: the row for a block of FT8 says 14074.0, which
    is where an operator would set the radio, not the edge of whichever
    2.7 kHz window happened to score best inside it.

    `present` is how sure the finder is that anything is there at all: the
    band plan names a signal, it does not conjure one.
    """
    if kind not in REFINABLE or float(present) < 0.5:
        return kind, conf, None
    band = sub_band(lo_hz, hi_hz)
    if band is None:
        return kind, conf, None
    if kind == "voice" and float(syllabic) >= SPOKEN_KEEP:
        return kind, conf, None
    name, dial = band
    return name, max(float(conf), 0.6), dial
