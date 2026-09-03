#
# Aether-gate — what the finder found, not only where it is.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Five answers to "yes, but what IS that?" for every window the finder scores.

The finder ranks 2.7 kHz windows by how much they look like somebody talking,
and it is honest about how far that gets you: a keyed CW tone swings its
envelope at a few hertz too, an RTTY pair sits in the middle of a phone-width
window, and a static crash is loud, deep and broad. So the same frames that
make the score also make a verdict, from features that separate the five
things actually on an 80 m evening:

  voice     a phone-wide patch (1.5-2.7 kHz) whose envelope swings at
            syllable rate, 2-8 Hz, and swings deeply
  cw        a few hundred hertz wide at most, envelope hard on and off at
            a keying rate rather than continuously varying
  data      a fixed width and a constant envelope: PSK, RTTY, FT8 mid-burst
  carrier   one bin of the map, no modulation at all — a heterodyne, an
            unattended beacon, somebody's switch-mode supply, not a station
  noise     nothing standing above the band's own floor here, or something
            impulsive, or hash that fills the window without any structure

What this cannot do, said plainly so a confidence of 0.4 reads as the honest
number it is. The map's points are ~244 Hz apart and the frames arrive ~30 a
second, so: a single-bin data mode (PSK31 at 60 Hz wide, its symbol rate far
above the frame rate) is indistinguishable from a bare carrier and will be
called one; CW faster than about 35 wpm keys faster than the frames can
follow and softens towards "data"; and a mode nobody on the band is using is
whichever of these five it most resembles. Every verdict therefore ships with
`kind_conf`, and a low one means what it says.
"""
import numpy as np

# The order here is the wire order of the codes stored in the finder's ring.
KINDS = ("voice", "cw", "data", "carrier", "noise")
NOISE = KINDS.index("noise")

# Every threshold below is a (soft-off, soft-on) pair rather than a step: a
# window that sits between the two gets a partial verdict and a confidence to
# match, because the band does not sort itself into five bins for us.
PRESENT_DB = (1.0, 4.0)        # SNR over the band floor: is anything here at all
NARROW_HZ = (500.0, 1100.0)    # occupied width: a tone, not a conversation
WIDE_HZ = (900.0, 1600.0)      # occupied width: a conversation, not a tone
FILLED_FRAC = (0.85, 0.98)     # share of the window's own width that is occupied
DEPTH_STEADY = (0.12, 0.40)    # envelope swing: below this it is a constant one
DEPTH_SWUNG = (0.20, 0.50)     # envelope swing: above this something is keying it
SYLLABIC_VOICE = (0.35, 0.60)  # syllabic share of the modulation spectrum
PEAKY = (0.45, 0.80)           # share of the excess energy in the strongest point
BIMODAL = (0.35, 0.65)         # 1 - the share of frames caught between on and off
DUTY_ON = (0.10, 0.25)         # keying that is never off, or never on, is not keying
DUTY_OFF = (0.80, 0.95)
CREST_IMPULSE = (4.0, 10.0)    # loudest frame over the average one: a crash, a spark
OCCUPANCY_HERE = (0.10, 0.30)  # how much of the window's time anything was there
FLOOR_TRACK = (0.35, 0.70)     # correlation with the whole band's floor: weather

BW_THRESHOLD = 0.25            # a point counts as occupied at a quarter of the peak


def _ramp(x, bounds):
    """0 below the pair's first value, 1 above its second, linear between."""
    lo, hi = bounds
    return np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)


def name(code):
    """The wire name for a stored code, defaulting to the humblest answer."""
    i = int(code)
    return KINDS[i] if 0 <= i < len(KINDS) else KINDS[NOISE]


def features(W, floor, mean_points, snr_db, depth, syllabic, occupancy,
             win, window_step, step_hz):
    """The per-window features the five verdicts are made of.

    W (frames, windows) window sums, floor (frames,) the band's own floor per
    frame, mean_points (points,) the mean spectrum over the same frames, and
    the finder's existing snr_db/depth/syllabic/occupancy terms per window.
    """
    W = np.asarray(W, dtype=np.float64)
    nwin = W.shape[1]
    depth = np.asarray(depth, dtype=np.float64)
    mean_w = np.maximum(np.mean(W, axis=0), 1e-30)
    e = W / mean_w                                        # envelope, mean 1

    # On, off, and neither -- all measured against the window's own average
    # rather than against its extremes, so a window that is silent 96% of the
    # time and enormous for one frame reads as one loud frame (which is what
    # it is) instead of as a key held down half the time.
    duty = np.mean(e > 0.5, axis=0)
    # Frames caught between on and off: speech spends most of its time there,
    # a keyed tone almost none, which is what tells the two swings apart.
    mid = np.mean((e > 0.35) & (e < 0.8), axis=0)
    crest = np.max(e, axis=0)                             # loudest frame / average

    # Does this window rise and fall with the WHOLE band? That is the weather,
    # not a station: QRN lifts the floor and every window with it.
    f = np.asarray(floor, dtype=np.float64)
    f = f / max(float(np.mean(f)), 1e-30) - 1.0
    a = e - 1.0
    den = np.std(a, axis=0) * float(np.std(f))
    fc = np.where(den > 1e-12, np.mean(a * f[:, None], axis=0) / np.maximum(den, 1e-12), 0.0)

    # Occupied width INSIDE the window, from the mean spectrum: how many of the
    # window's own points carry a quarter of its strongest point's excess over
    # the floor. A carrier is one point; phone is ten or eleven.
    floor_level = float(np.mean(np.asarray(floor, dtype=np.float64)))
    p = np.asarray(mean_points, dtype=np.float64)
    seg = np.lib.stride_tricks.sliding_window_view(p, win)[::window_step]
    if len(seg) < nwin:                                   # short map: hold the last
        seg = np.concatenate([seg, np.repeat(seg[-1:], nwin - len(seg), axis=0)])
    seg = seg[:nwin]
    exc = np.maximum(seg - floor_level, 0.0)
    peak = np.max(exc, axis=1)
    total = np.sum(exc, axis=1)
    occupied = np.sum(exc > BW_THRESHOLD * peak[:, None], axis=1)
    bw_hz = np.where(peak > 0.0, occupied * step_hz, 0.0)
    peak_frac = np.where(total > 0.0, peak / np.maximum(total, 1e-30), 0.0)

    return {
        "bw_hz": bw_hz,
        "filled": bw_hz / max(win * step_hz, 1e-9),
        "peak_frac": peak_frac,
        "depth": depth,
        "syllabic": np.asarray(syllabic, dtype=np.float64),
        "occupancy": np.asarray(occupancy, dtype=np.float64),
        "snr_db": np.asarray(snr_db, dtype=np.float64),
        "mid": mid, "duty": duty, "crest": crest, "floor_corr": fc,
    }


def scores(feat):
    """A 0..1 verdict per kind. They do not sum to one; the winner takes it."""
    present = _ramp(feat["snr_db"], PRESENT_DB)
    absent = 1.0 - present
    narrow = 1.0 - _ramp(feat["bw_hz"], NARROW_HZ)
    wide = _ramp(feat["bw_hz"], WIDE_HZ)
    filled = _ramp(feat["filled"], FILLED_FRAC)
    steady = 1.0 - _ramp(feat["depth"], DEPTH_STEADY)
    swung = _ramp(feat["depth"], DEPTH_SWUNG)
    syl = _ramp(feat["syllabic"], SYLLABIC_VOICE)
    peaky = _ramp(feat["peak_frac"], PEAKY)
    # keying is a swing that is nearly all on or all off, and that is sometimes
    # both: a tone left down for the whole window is a carrier, not a station
    # working somebody.
    keyed = (swung * _ramp(1.0 - feat["mid"], BIMODAL)
             * _ramp(feat["duty"], DUTY_ON) * (1.0 - _ramp(feat["duty"], DUTY_OFF)))
    impulsive = _ramp(feat["crest"], CREST_IMPULSE) * (1.0 - _ramp(feat["occupancy"],
                                                                  OCCUPANCY_HERE))
    weather = _ramp(feat["floor_corr"], FLOOR_TRACK)
    return {
        "voice": present * wide * syl * swung,
        "cw": present * narrow * keyed,
        # a fixed width and a flat envelope, and not the single point that
        # would make it a carrier
        "data": present * steady * (1.0 - peaky) * (1.0 - filled) * (0.5 + 0.5 * narrow),
        "carrier": present * narrow * steady * peaky,
        # nothing above the floor, or a crash, or hash filling the window with
        # no syllables and no keying in it
        "noise": np.maximum.reduce([absent, present * impulsive, present * weather,
                                    present * filled * (1.0 - syl) * (1.0 - keyed)]),
    }


def classify(W, floor, mean_points, snr_db, depth, syllabic, occupancy,
             win, window_step, step_hz):
    """(codes, confidences) per window, ready for the finder's ring.

    The confidence is the winner's own verdict docked by half the runner-up's:
    two kinds that both half-fit describe a window nobody can name from a
    quarter-kilohertz map and thirty frames a second, and the number says so.
    """
    s = scores(features(W, floor, mean_points, snr_db, depth, syllabic, occupancy,
                        win, window_step, step_hz))
    S = np.stack([np.asarray(s[k], dtype=np.float64) for k in KINDS])
    code = np.argmax(S, axis=0).astype(np.int8)
    ranked = np.sort(S, axis=0)
    conf = np.clip(ranked[-1] - 0.5 * ranked[-2], 0.0, 1.0)
    return code, conf.astype(np.float32)
