#
# Aether-gate — core/filter.py's response/spectrum reporting, split out.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Pure functions behind the FILTER page's picture: the designed response
curve, the live spectrum ahead of it, and where the operator's own notches
land on top of a SQUEEZE one. Split out of core/filter.py only because that
module is at its own 800-line budget (see its header) -- every function
here takes the state it needs as arguments and hands back a status dict or
a plain value; nothing here holds state of its own, the same "no hardware,
no adapter state, no clock" rule chainstatus.py's own docstring states for
the same reason.
"""
import numpy as np


def response_at(taps, rate_hz, hz):
    """|H| in dB of complex taps at one signed frequency."""
    k = np.arange(len(taps)) - (len(taps) - 1) / 2.0
    return float(20 * np.log10(abs(np.sum(taps * np.exp(-2j * np.pi * hz / rate_hz * k))) + 1e-12))


def notch_overlaps(notches, squeeze_notches, sgn):
    """Operator notches (audio Hz, unsigned, `width_hz`) that land within
    combined width of a live SQUEEZE notch target (`squeeze_notches`:
    [(signed hz, width_hz), ...], core/filter.py's SliceFilter.
    squeeze_notches) -- the two tables stack silently at the same Hz
    otherwise (the operator's own IF NOTCH is folded into the FIR design;
    SQUEEZE's own table is kept OUT of it, see SliceFilter.
    set_squeeze_notches' own docstring for why). No behaviour change: this
    is the flag only, in the signed-hertz frame both tables already share."""
    if not squeeze_notches:
        return []
    out = []
    for n in notches:
        op_hz, op_w = sgn * n["hz"], n["width_hz"]
        for sq_hz, sq_w in squeeze_notches:
            if abs(op_hz - sq_hz) <= (op_w + sq_w) / 2.0:
                out.append({"hz": n["hz"], "width_hz": op_w, "squeeze_hz": round(sq_hz)})
                break
    return out


def spectrum_snapshot(spec_db, spec_f, spec_f_order, audio_edges, sign, points=128,
                      floor_pctl=20.0):
    """What is arriving ahead of the filter, on response_snapshot's own
    grid: the 1 s spectrum the auto width and the ANF read, in dB below its
    peak, with the floor on the same scale. None until heard (spec_db is).

    The floor is the waterfall's own convention -- the `floor_pctl`th
    percentile of the full raw spectrum (spec_db, every bin), not the
    median of these `points` DISPLAYED ones: a busy passband (several
    strong carriers, a loud band) can occupy more than half of the
    displayed range, and that median is then a strong signal's own level,
    not a floor -- everyone weaker collapses to "below the floor" on the
    display.
    """
    if spec_db is None:
        return None
    lo, hi = audio_edges
    f_audio = np.linspace(0.0, max(hi + 500.0, 3500.0), points)
    f = sign * f_audio
    p = np.interp(f, spec_f[spec_f_order], spec_db[spec_f_order])
    peak = float(np.max(p))
    floor = float(np.percentile(spec_db, floor_pctl))
    return {"hz": [round(x) for x in f_audio],
            "db": [round(max(float(x) - peak, -120.0), 1) for x in p],
            "floor_db": round(floor - peak, 1)}


def response_snapshot(taps, rate_hz, squeeze_bank, audio_edges, sign, bypass, points=128):
    """The designed response across the audio band, for a picture -- FIR
    and SQUEEZE notch bank together (they are in series on the signal, so
    their dB simply add: the true depth SQUEEZE's own notch tool reaches,
    not just the FIR's), except while HEAR RAW holds (bypass=True): flat
    0 dB, since nothing is in circuit."""
    lo, hi = audio_edges
    f_audio = np.linspace(0.0, max(hi + 500.0, 3500.0), points)
    if bypass:
        return {"hz": [round(x) for x in f_audio], "db": [0.0] * points}
    f = sign * f_audio
    db = [round(max(response_at(taps, rate_hz, fx) + squeeze_bank.response_db(fx), -120.0), 1)
          for fx in f]
    return {"hz": [round(x) for x in f_audio], "db": db}
