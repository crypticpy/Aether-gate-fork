#
# Aether-gate — what the finder found, not only where it is.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The answer to "yes, but what IS that?" for every window the finder scores.

The finder ranks 2.7 kHz windows by what is standing over the floor in them,
and it is honest about how far a shape gets you: a keyed CW tone swings its
envelope at a few hertz too, an RTTY pair sits in the middle of a phone-width
window, and a static crash is loud, deep and broad. So the same frames that
make the score also make a verdict, from features that separate the things
actually on a band:

  voice     a phone-wide patch (1.5-2.7 kHz) whose envelope swings at
            syllable rate, 2-8 Hz, and swings deeply
  cw        a few hundred hertz wide at most, envelope hard on and off at
            a keying rate rather than continuously varying
  data      a fixed width and a constant envelope: PSK, RTTY, FT8 mid-burst
  rtty      two columns of it, about 170 Hz apart, where the map can resolve
            the shift at all (~90 Hz a point or finer)
  carrier   one bin of the map, no modulation at all — a heterodyne, an
            unattended beacon, somebody's switch-mode supply, not a station
  ft8/ft4/  named by finder_bands from the band plan, not from the shape:
  psk31     what the map can see of them is a filled block on all the time,
            and where that block SITS is the rest of the evidence
  signal    something is standing over the local floor and nothing above
            fitted it well enough to name. An honest row an operator can go
            and listen to, and the reason a CW column is no longer dropped
            for failing to look like speech
  noise     nothing standing above the band's own floor here, or something
            impulsive, or hash that fills the window without any structure

What this cannot do, said plainly so a confidence of 0.4 reads as the honest
number it is. The map's points are ~244 Hz apart and the frames arrive ~30 a
second, so: a single-bin data mode (PSK31 at 60 Hz wide, its symbol rate far
above the frame rate) is indistinguishable from a bare carrier and will be
called one unless the band plan knows better; CW faster than about 35 wpm
keys faster than the frames can follow and softens towards "data"; the 15 s
frame of FT8 is slower than the whole 8.5 s ring and cannot be seen at all;
and a mode nobody on the band is using comes back as `signal` rather than as
whichever of these it least badly resembles. Every verdict therefore ships
with `kind_conf`, and a low one means what it says.
"""
import numpy as np

# The order here is the wire order of the codes stored in the finder's ring:
# the first five are the original ones and keep their codes, the rest are
# appended so that a ring written by an older build still reads correctly.
#
#   voice/cw/data/carrier/noise   as they always were
#   signal                        something is here and nothing named it: the
#                                 finder must not drop a detection just because
#                                 it cannot say what it is (see classify)
#   ft8/ft4/psk31                 named by the band plan, in finder_bands.refine
#   rtty                          a two-tone shift, where the map can resolve one
KINDS = ("voice", "cw", "data", "carrier", "noise",
         "signal", "ft8", "ft4", "psk31", "rtty")
NOISE = KINDS.index("noise")
SIGNAL = KINDS.index("signal")
RTTY = KINDS.index("rtty")
DIGITAL = ("data", "ft8", "ft4", "psk31", "rtty")

# Every threshold below is a (soft-off, soft-on) pair rather than a step: a
# window that sits between the two gets a partial verdict and a confidence to
# match, because the band does not sort itself into five bins for us.
PRESENT_DB = (1.0, 4.0)        # SNR over the band floor: is anything here at all
NARROW_HZ = (900.0, 1400.0)    # occupied width: a tone, not a conversation.
WIDE_HZ = (1100.0, 1800.0)     # occupied width: a conversation, not a tone.
                               # Measured against DETECT_FLOOR_FRAC below: a
                               # keyed tone at 25 wpm reads 330-1180 Hz across
                               # the spans (its keying sidebands are real width
                               # and grow with signal strength), off-air 80 m
                               # and 40 m phone reads 1400-2700 Hz.
FILLED_FRAC = (0.85, 0.98)     # share of the window's own width that is occupied
DEPTH_STEADY = (0.12, 0.40)    # envelope swing: below this it is a constant one
DEPTH_SWUNG = (0.20, 0.50)     # envelope swing: above this something is keying it
SYLLABIC_VOICE = (0.35, 0.60)  # syllabic share of the modulation spectrum
SYLLABIC_KEYED = (0.50, 0.80)  # ...and the sharper reading of the same number that
                               # stands in for width where there is no width to be
                               # had: below it something is being keyed, above it
                               # somebody is talking. It only bites on the two
                               # coarsest spans (1.02 and 2.04 MS/s, where a
                               # window is 5 and 3 map points across and there is
                               # no width verdict to be had); measured there over
                               # four noise seeds and 3-20 dB, a keyed tone reads
                               # 0.43-0.57 and phone reads 0.81-0.91, so the
                               # crossing has to clear 0.57 -- at 0.60 a keyer
                               # that noise has lifted to 0.57 reads as speech.
PEAKY = (0.45, 0.80)           # share of the excess energy in the strongest point
BIMODAL = (0.35, 0.65)         # 1 - the share of frames caught between on and
                               # off. Loose on purpose: a keyed tone reads
                               # 0.81-1.00 here at 6 dB and up, but only
                               # 0.61-0.65 at 3 dB, where the noise fills the
                               # gaps in. It is a guard against an envelope
                               # that is continuously varied rather than
                               # switched, not a keying detector.
DUTY_ON = (0.10, 0.25)         # keying that is never off, or never on, is not
DUTY_OFF = (0.70, 0.85)        # keying. PARIS timing is key-down half the time
                               # within a character and less across words;
                               # measured over the ring at 8-35 wpm and every
                               # span it comes out 0.43-0.69, so a window
                               # occupied more than about 70% of the time is
                               # not being keyed by an operator -- it is a
                               # signal that is simply on.
CREST_IMPULSE = (4.0, 10.0)    # loudest frame over the average one: a crash, a spark
OCCUPANCY_HERE = (0.10, 0.30)  # how much of the window's time anything was there
FLOOR_TRACK = (0.35, 0.70)     # correlation with the whole band's floor: weather

PEAK_PRESENT_DB = (3.0, 6.0)   # the STRONGEST point's excess over the LOCAL floor.
                               # A 200 Hz tone in a 2.7 kHz window lifts the
                               # window by 2.8 dB however loud it is, so the
                               # window SNR alone can never say a CW column is
                               # present. The ring averages 128-256 slots, so a
                               # point of bare noise sits within ~0.5 dB of its
                               # own floor and 3 dB is a long way out.
ONTIME = DUTY_OFF              # occupied more than 70-85% of its time: not being
                               # keyed by an operator, simply on. A digital block
                               # is on; a conversation is not.
SPOKEN_BLOCK = (0.50, 0.80)    # the syllabic reading a filled block has to beat
                               # before it may be called speech rather than data
RTTY_SHIFT_HZ = 170.0          # the standard amateur shift...
RTTY_SHIFT_TOL_HZ = 60.0       # ...and how far off it may read
RTTY_RESOLVE_HZ = 90.0         # ...but only where the map has two points across
                               # it. Above this step size an RTTY pair is one
                               # blob and the honest answer is "data".
SIGNAL_TOP = 0.35              # no verdict scored better than this...
SIGNAL_CONF = 0.20             # ...nor won by more than this over the next...
SIGNAL_PRESENT = 0.5           # ...and something is certainly here: "signal".
                               # All three, so that a modest but CLEAR verdict
                               # ("voice 0.30, nothing else near it") keeps its
                               # name and only a genuine tie is renamed.

RESOLVED_POINTS = (5.0, 8.0)   # map points across one window: fewer than this
                               # and no width verdict has been earned at all.
                               # Measured: at five points (1.02 MS/s) a keyed
                               # tone occupies 1.5-2.0 kHz of them and phone
                               # 2.0 kHz; at three (2.04 MS/s) both fill the
                               # window. The map cannot tell them apart there
                               # and must not pretend to.

BW_ENERGY_FRAC = 0.90          # the share of a window's excess energy its
                               # ENERGY width has to account for -- the width
                               # `filled` is measured with, which asks where a
                               # signal's bulk sits, not how far it reaches
DETECT_FLOOR_FRAC = 0.5        # ...and the level, as a fraction of the band's
                               # own floor, at which a point counts as OCCUPIED
                               # for the narrow/wide verdict: 1.8 dB over the
                               # floor, which 256 averaged slots resolve
                               # comfortably against the floor's own scatter


def _ramp(x, bounds):
    """0 below the pair's first value, 1 above its second, linear between."""
    lo, hi = bounds
    return np.clip((np.asarray(x, dtype=np.float64) - lo) / (hi - lo), 0.0, 1.0)


def name(code):
    """The wire name for a stored code, defaulting to the humblest answer."""
    i = int(code)
    return KINDS[i] if 0 <= i < len(KINDS) else KINDS[NOISE]


def _two_tone_shift(exc, step_hz):
    """The spacing of the two strongest points in each window, where there ARE
    two: an RTTY pair is 170 Hz apart and constant, which is the one thing that
    separates it from a carrier -- when the map is fine enough to see it, and
    zero when it is not."""
    nwin, win = exc.shape
    if win < 3:
        return np.zeros(nwin)
    i1 = np.argmax(exc, axis=1)
    v1 = exc[np.arange(nwin), i1]
    idx = np.arange(win)[None, :]
    masked = np.where(np.abs(idx - i1[:, None]) <= 1, -1.0, exc)
    i2 = np.argmax(masked, axis=1)
    v2 = masked[np.arange(nwin), i2]
    pair = (v2 >= 0.4 * np.maximum(v1, 1e-30)) & (v1 > 0.0)
    return np.where(pair, np.abs(i1 - i2) * float(step_hz), 0.0)


def features(W, floor, mean_points, snr_db, depth, syllabic, occupancy,
             win, window_step, step_hz, floor_points=None, peak_db=None):
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
    # Frames caught between on and off. Measured off-air, this is a small
    # number for BOTH a talker (3-17%) and a keyed tone (1-19%) -- it is kept
    # because a signal that spends most of its time there is being varied
    # rather than switched, which is neither -- but see scores(): it is not
    # what tells speech and keying apart, and it must not be asked to be.
    mid = np.mean((e > 0.35) & (e < 0.8), axis=0)
    crest = np.max(e, axis=0)                             # loudest frame / average

    # Does this window rise and fall with the WHOLE band? That is the weather,
    # not a station: QRN lifts the floor and every window with it.
    f = np.asarray(floor, dtype=np.float64)
    f = f / max(float(np.mean(f)), 1e-30) - 1.0
    a = e - 1.0
    den = np.std(a, axis=0) * float(np.std(f))
    fc = np.where(den > 1e-12, np.mean(a * f[:, None], axis=0) / np.maximum(den, 1e-12), 0.0)

    # Two widths INSIDE the window, from the mean spectrum, because "how far
    # does this signal reach" and "where does its energy sit" are different
    # questions and one number cannot answer both.
    #
    # bw_hz: how much of the window stands DETECTABLY over the band's own
    # floor, at DETECT_FLOOR_FRAC of it, counted strongest point first and
    # carried a fraction of a point past the last whole one so that a talker
    # fading does not step the width a whole map point at a time. This is the
    # number narrow/wide are read from.
    #
    # Two earlier measures got this wrong in ways worth naming, because both
    # look reasonable on paper. Counting the points within some decibels of the
    # STRONGEST one measures a tone correctly and speech not at all: the
    # long-term spectrum of an SSB signal falls 15-20 dB from its strongest
    # formant region to the top of the passband, so a -6 dB width of real phone
    # came out 500-1000 Hz -- narrower than a keyed tone -- and on 2026-09-03
    # the live gate called every conversation on 80 m "cw". Counting instead
    # the points holding a SHARE of the energy (BW_ENERGY_FRAC, below) fixed
    # that but read phone as 730-1470 Hz for the same reason of slope: 90% of a
    # talker's energy is in the bottom kilohertz of the passband. Those numbers
    # straddle the narrow/wide bounds, so one point of drift flipped the verdict
    # and the live gate flapped between "voice 1.0" and "cw 1.0" every few
    # seconds. Against the FLOOR the same signals separate cleanly -- a tone is
    # a tone all the way down, phone is 2 kHz wide all the way down -- and the
    # measurement no longer depends on how the energy is distributed within it.
    # ...measured against the LOCAL floor where one was given (finder_floor),
    # so that a tilted span and a dense sub-band do not decide the width of a
    # signal 40 kHz away. Everything below is in units of the floor under each
    # point, which is exactly what it used to be when the floor was one number.
    p = np.asarray(mean_points, dtype=np.float64)
    if floor_points is None:
        fl = np.full(len(p), max(float(np.mean(np.asarray(floor, dtype=np.float64))),
                                 1e-30))
    else:
        fl = np.maximum(np.asarray(floor_points, dtype=np.float64), 1e-30)
    rel = p / fl
    seg = np.lib.stride_tricks.sliding_window_view(rel, win)[::window_step]
    if len(seg) < nwin:                                   # short map: hold the last
        seg = np.concatenate([seg, np.repeat(seg[-1:], nwin - len(seg), axis=0)])
    seg = seg[:nwin]
    exc = np.maximum(seg - 1.0, 0.0)
    peak = np.max(exc, axis=1)
    total = np.sum(exc, axis=1)
    ranked = np.sort(exc, axis=1)[:, ::-1]                 # strongest point first
    detect = DETECT_FLOOR_FRAC
    whole = np.sum(ranked > detect, axis=1)
    part = np.where(whole < win,
                    ranked[np.arange(nwin), np.minimum(whole, win - 1)] / detect, 0.0)
    bw_hz = (whole + np.clip(part, 0.0, 1.0)) * step_hz
    # ...unless the signal runs off the end of the window, in which case what
    # was measured is a LOWER BOUND and not a width. Windows overlap by
    # WINDOW_STEP_POINTS, so the finder puts one on the skirt of every strong
    # talker: it holds the top point of a conversation and nothing else, reads
    # a few hundred hertz wide, and was called "cw" at 0.6 on off-air 80 m
    # phone. A signal that leaves the window is at least the window wide.
    #
    # "Runs off the end" is a boundary point occupied AND the point just
    # outside it occupied too -- the occupancy crosses the line -- which takes
    # no threshold beyond the one already chosen and holds at any SNR. A
    # boundary point occupied on its own is a signal that happens to end there,
    # which is what a tone parked on the edge of a window looks like, and it
    # keeps its measured width.
    occ = np.maximum(rel - 1.0, 0.0) > detect
    lo = np.arange(nwin) * window_step
    left = np.where(lo > 0, occ[np.maximum(lo - 1, 0)], False)
    right_i = np.minimum(lo + win, len(p) - 1)
    right = np.where(lo + win < len(p), occ[right_i], False)
    crosses = ((exc[:, 0] > detect) & left) | ((exc[:, -1] > detect) & right)
    bw_hz = np.where(crosses, win * step_hz, bw_hz)
    # ...and the energy width, which is what `filled` means: hash with no
    # structure needs nearly the whole window to account for its energy, a
    # station needs a fraction of it.
    cum = np.cumsum(ranked, axis=1)
    energy_pts = np.sum(cum < BW_ENERGY_FRAC * total[:, None], axis=1) + 1
    peak_frac = np.where(total > 0.0, peak / np.maximum(total, 1e-30), 0.0)

    return {
        "bw_hz": bw_hz,
        # the strongest point over its own floor, in dB: what says a narrow
        # signal is there at all, since the window SNR cannot
        "peak_db": (10.0 * np.log10(1.0 + peak) if peak_db is None
                    else np.asarray(peak_db, dtype=np.float64)),
        "shift_hz": _two_tone_shift(exc, step_hz),
        "resolves_shift": float(step_hz <= RTTY_RESOLVE_HZ),
        "filled": np.where(total > 0.0, energy_pts / max(win, 1), 0.0),
        # how much of a width verdict this map has earned here: a window only
        # three points across cannot tell a tone from a conversation by shape
        "resolved": float(_ramp(win, RESOLVED_POINTS)),
        "peak_frac": peak_frac,
        "depth": depth,
        "syllabic": np.asarray(syllabic, dtype=np.float64),
        "occupancy": np.asarray(occupancy, dtype=np.float64),
        "snr_db": np.asarray(snr_db, dtype=np.float64),
        "mid": mid, "duty": duty, "crest": crest, "floor_corr": fc,
    }


def present(feat):
    """How sure a window holds anything at all, from whichever of the two
    measures can see it: a conversation fills the window and is read by its
    SNR, a keyed tone or a carrier is one point of eleven and is read by its
    peak. Before this, a 15 dB CW column raised its 2.7 kHz window by 2.8 dB,
    scored 0.6 here, and every narrow verdict was docked for it."""
    return np.maximum(_ramp(feat["snr_db"], PRESENT_DB),
                      _ramp(feat.get("peak_db", feat["snr_db"]), PEAK_PRESENT_DB))


def scores(feat):
    """A 0..1 verdict per kind. They do not sum to one; the winner takes it."""
    here = present(feat)                  # not shadowed: scores() calls it
    absent = 1.0 - here
    syl = _ramp(feat["syllabic"], SYLLABIC_VOICE)
    # Where the map is fine enough, width says which of a tone and a
    # conversation this is. Where it is not -- a 2 MHz span puts three
    # kilohertz-wide points across the whole window, and a keyed tone and a
    # phone signal both land on two of them -- the width terms stand aside
    # and the envelope's own modulation carries the verdict instead, rather
    # than a measurement that cannot separate them casting a vote anyway.
    res = np.asarray(feat.get("resolved", 1.0), dtype=np.float64)
    spoken = _ramp(feat["syllabic"], SYLLABIC_KEYED)
    narrow = res * (1.0 - _ramp(feat["bw_hz"], NARROW_HZ)) + (1.0 - res) * (1.0 - spoken)
    wide = res * _ramp(feat["bw_hz"], WIDE_HZ) + (1.0 - res) * spoken
    filled = _ramp(feat["filled"], FILLED_FRAC)
    steady = 1.0 - _ramp(feat["depth"], DEPTH_STEADY)
    swung = _ramp(feat["depth"], DEPTH_SWUNG)
    peaky = _ramp(feat["peak_frac"], PEAKY)
    # Keying is a swing that is nearly all on or all off, and that is sometimes
    # both: a tone left down for the whole window is a carrier, not a station
    # working somebody. Where the map has no width to offer, it must also not be
    # paced by syllables -- that is the whole of the evidence left there.
    #
    # What this deliberately does NOT claim is that the on/off structure alone
    # separates a keyer from a talker. It does not, and 2026-09-03 measured it
    # both ways: off-air 80 m phone is `swung` (depth ~1.0), sits inside the duty
    # band (0.5), and spends 3-17% of its frames between 0.35 and 0.8 of its own
    # mean, against a keyed tone's 1-19% -- so `keyed` read 1.0 on every
    # conversation on the band. Nor can it be repaired by looking harder at the
    # timing: eight seconds of 30 ms slots resolve modulation to about 15 Hz,
    # 12-35 wpm Morse keys elements at 5-15 Hz, and the run lengths of Morse
    # (0.10-0.49 s) and of speech (0.07-0.82 s) overlap outright at that
    # resolution. Keying is a corroborating term. WIDTH is the discriminator,
    # and where the width is not resolved the modulation spectrum is.
    keyed = (swung * _ramp(1.0 - feat["mid"], BIMODAL)
             * (res + (1.0 - res) * (1.0 - spoken))
             * _ramp(feat["duty"], DUTY_ON) * (1.0 - _ramp(feat["duty"], DUTY_OFF)))
    impulsive = _ramp(feat["crest"], CREST_IMPULSE) * (1.0 - _ramp(feat["occupancy"],
                                                                  OCCUPANCY_HERE))
    weather = _ramp(feat["floor_corr"], FLOOR_TRACK)
    # On all the time. A block of FT8 is; an operator is not, whatever else
    # the two have in common.
    ontime = _ramp(feat["duty"], ONTIME)
    spoken_block = _ramp(feat["syllabic"], SPOKEN_BLOCK)
    # A whole sub-band of digital signals fills its window edge to edge, stays
    # on, and is not paced by syllables. Measured on the live gate 2026-09-03,
    # the FT8 window on 20 m read depth 0.39 and syllabic 0.52: `steady` was
    # 0.04, so the old data term could not claim it, `swung` was 0.63 and
    # `syl` 0.68, so voice did -- at 14074.0 and again at 14080.5, the FT8 and
    # FT4 windows, both called "voice". What separates them from a talker is
    # not the envelope at all: it is that a talker's energy is in the bottom
    # kilohertz of his passband (`filled` ~0.4) and a block's is everywhere.
    block = here * filled * (1.0 - spoken_block) * ontime
    # Two tones a standard shift apart, held on: RTTY, where the map can
    # resolve the shift at all. Where it cannot, this is zero and the same
    # signal lands on `data`, which is the honest answer at 244 Hz a point.
    shift_fit = np.exp(-((np.asarray(feat.get("shift_hz", 0.0), dtype=np.float64)
                          - RTTY_SHIFT_HZ) / RTTY_SHIFT_TOL_HZ) ** 2)
    rtty = (float(feat.get("resolves_shift", 0.0)) * here * shift_fit
            * ontime * steady * narrow)
    return {
        "voice": here * wide * syl * swung,
        "cw": here * narrow * keyed,
        # a fixed width and a flat envelope, and not the single point that
        # would make it a carrier -- or a whole sub-band of them
        "data": np.maximum(
            here * steady * (1.0 - peaky) * (1.0 - filled) * (0.5 + 0.5 * narrow),
            block),
        "rtty": rtty,
        "carrier": here * narrow * steady * peaky,
        # nothing above the floor, or a crash, or hash filling the window with
        # no syllables and no keying in it
        "noise": np.maximum.reduce([absent, here * impulsive, here * weather,
                                    here * filled * (1.0 - syl) * (1.0 - keyed)]),
        # not a kind: what classify() needs to know before it may say "signal"
        "_present": here,
    }


def classify(W, floor, mean_points, snr_db, depth, syllabic, occupancy,
             win, window_step, step_hz, floor_points=None):
    """(codes, confidences) per window, ready for the finder's ring.

    The confidence is the winner's own verdict docked by half the runner-up's:
    two kinds that both half-fit describe a window nobody can name from a
    quarter-kilohertz map and thirty frames a second, and the number says so.

    Where nothing names a window that certainly HAS something in it, the answer
    is "signal" at the confidence that something is there -- never "noise", and
    never the least-bad of five names. A finder that drops what it cannot name
    is worse than one that admits it: the operator can hear it either way.
    """
    return verdict(features(W, floor, mean_points, snr_db, depth, syllabic,
                            occupancy, win, window_step, step_hz,
                            floor_points=floor_points))


def verdict(feat):
    """(codes, confidences) from an already-measured feature dict.

    Split out of classify() so the finder can keep the features it paid for --
    the occupied width decides how far a candidate suppresses its neighbours,
    and measuring it twice would be measuring it differently.
    """
    s = scores(feat)
    present = np.asarray(s["_present"], dtype=np.float64)
    S = np.stack([np.asarray(np.broadcast_to(s.get(k, 0.0), present.shape),
                             dtype=np.float64) for k in KINDS])
    ranked = np.sort(S, axis=0)
    # a window nothing scored at all is not the first kind in KINDS, it is the
    # last one: argmax has to answer something even when every verdict is zero
    code = np.where(ranked[-1] > 0.0, np.argmax(S, axis=0), NOISE).astype(np.int8)
    conf = np.clip(ranked[-1] - 0.5 * ranked[-2], 0.0, 1.0)
    unsure = ((ranked[-1] < SIGNAL_TOP) & (conf < SIGNAL_CONF)
              & (present >= SIGNAL_PRESENT))
    code = np.where(unsure, SIGNAL, code).astype(np.int8)
    conf = np.where(unsure, present, conf)
    return code, conf.astype(np.float32)
