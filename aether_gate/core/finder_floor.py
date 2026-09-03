#
# Aether-gate — the floor a weak signal is weak against, measured where it is.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""A LOCAL noise floor, and what stands over it.

The finder used to measure every window against one number: the median point
of the whole span, averaged over the ring. On a band that is flat and empty
either side of one conversation that is the right number. On 20 m at four in
the afternoon it is not: the span is tilted (the 2026-09-03 capture ran 2.4 dB
from 14.07 to 14.15 MHz), and a third of it is a solid block of FT8 and PSK
that drags the median of the WHOLE span up while the quiet stretch a talker
is actually sitting in stays where it was. A signal 3 dB over the floor beside
it -- copyable by ear, which is the only threshold that matters -- read 1 dB
over the span's median and scored below every gate the finder has.

So the floor is measured per ~10 kHz and robustly:

  local_floor   the mean spectrum over the ring per point (which averages the
                chi-square scatter away), then the 25th percentile of that
                across each block of ~10 kHz, interpolated between block
                centres. A percentile rather than a mean because a block can
                be half signal; the 25th rather than the 50th so that a block
                which IS a dense digital sub-band still reports the noise
                between the signals rather than the signals.

  presence      the share of the ring each point spent above its own local
                floor by PRESENT_DB. Measured on ~0.24 s chunks of the ring
                smoothed over ~700 Hz, so that a 200 Hz CW tone is detected
                (it is 10 dB up in a three-point average even where it is
                2.8 dB up in a 2.7 kHz window) without band noise tripping the
                test: 24 averaged samples put +3 dB about five sigma out.

Both are computed once a second from the same ring the scores come from.
"""
import numpy as np

FLOOR_BLOCK_HZ = 10_000.0      # the neighbourhood a floor is local to
FLOOR_BLOCK_POINTS_MIN = 12    # ...but never fewer points than a percentile needs
FLOOR_PCTL = 25.0              # the percentile of a block that IS its floor
PRESENT_DB = 3.0               # over the local floor: "something is here"
PRESENT_SMOOTH_HZ = 700.0      # points averaged together before the test
PRESENT_CHUNK_S = 0.24         # ...and time, so noise cannot trip it


def _blocks(n, block_points):
    """(centre, lo, hi) for overlapping blocks covering 0..n, half-block hop."""
    nb = int(min(max(block_points, 1), n))
    hop = max(1, nb // 2)
    out = []
    lo = 0
    while True:
        hi = min(lo + nb, n)
        lo = max(0, hi - nb)
        out.append((0.5 * (lo + hi - 1), lo, hi))
        if hi >= n:
            break
        lo += hop
    return out


def block_points(step_hz, n, block_hz=FLOOR_BLOCK_HZ):
    """How many points a `block_hz` neighbourhood is, at this resolution."""
    want = int(round(float(block_hz) / max(float(step_hz), 1e-9)))
    return int(min(max(want, FLOOR_BLOCK_POINTS_MIN), max(int(n), 1)))


def local_floor(P, step_hz, pctl=FLOOR_PCTL, block_hz=FLOOR_BLOCK_HZ):
    """The local noise floor per point, from P (frames, points) of power.

    Returns (points,) of power, strictly positive. A one-frame P is accepted
    (it is then the frequency percentile alone).
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim == 1:
        P = P[None, :]
    base = np.mean(P, axis=0)
    n = base.shape[0]
    if n == 0:
        return np.zeros(0)
    b = _blocks(n, block_points(step_hz, n, block_hz))
    xs = np.array([c for c, _lo, _hi in b], dtype=np.float64)
    ys = np.array([np.percentile(base[lo:hi], pctl) for _c, lo, hi in b], dtype=np.float64)
    if len(xs) == 1:
        f = np.full(n, ys[0])
    else:
        f = np.interp(np.arange(n, dtype=np.float64), xs, ys)
    return np.maximum(f, 1e-30)


def _smooth(x, k, axis=-1):
    """Moving average of x over k points along `axis`, edges included."""
    k = int(max(1, k))
    if k == 1:
        return np.asarray(x, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[axis]
    k = min(k, n)
    ker = np.ones(k) / k
    return np.apply_along_axis(lambda v: np.convolve(v, ker, mode="same"), axis, x)


def _chunks(P, frame_s, chunk_s=PRESENT_CHUNK_S):
    """P (frames, points) averaged into chunks of at least chunk_s."""
    n = P.shape[0]
    k = int(max(1, round(float(chunk_s) / max(float(frame_s), 1e-6))))
    k = min(k, n)
    m = n // k
    if m < 1:
        return P.mean(axis=0, keepdims=True)
    return P[:m * k].reshape(m, k, P.shape[1]).mean(axis=1)


def presence(P, floor_pts, step_hz, frame_s, db=PRESENT_DB,
             smooth_hz=PRESENT_SMOOTH_HZ, chunk_s=PRESENT_CHUNK_S):
    """Share of the ring each point stood `db` over its own local floor.

    P (frames, points) power, floor_pts (points,) from local_floor. Returns
    (points,) in 0..1.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.ndim == 1:
        P = P[None, :]
    if P.shape[0] == 0:
        return np.zeros(P.shape[-1])
    k = max(1, int(round(float(smooth_hz) / max(float(step_hz), 1e-9))))
    C = _smooth(_chunks(P, frame_s, chunk_s), k, axis=-1)
    f = _smooth(np.asarray(floor_pts, dtype=np.float64), k, axis=-1)
    thr = np.maximum(f, 1e-30) * (10.0 ** (float(db) / 10.0))
    return np.mean(C > thr[None, :], axis=0)


def presence_wide(W, floor_w, frame_s, db=PRESENT_DB, chunk_s=PRESENT_CHUNK_S):
    """Share of the ring each WINDOW stood `db` over its own local floor.

    presence() asks the question of a single map point, which is the right
    question for a keyed tone and the wrong one for a conversation: 3 dB of
    speech spread across a 2.4 kHz passband is 3 dB in the window and, between
    syllables, under 2 dB in any one point of it. A talker is present as a
    whole, so he is measured as one -- over the same chunks, with no smoothing,
    the window's own eleven points being the average already.

    W (frames, nwin) window sums, floor_w (nwin,) the same windows of the local
    floor. Returns (nwin,) in 0..1.
    """
    W = np.asarray(W, dtype=np.float64)
    if W.ndim == 1:
        W = W[None, :]
    if W.shape[0] == 0:
        return np.zeros(W.shape[-1])
    C = _chunks(W, frame_s, chunk_s)
    thr = np.maximum(np.asarray(floor_w, dtype=np.float64), 1e-30) * (
        10.0 ** (float(db) / 10.0))
    return np.mean(C > thr[None, :], axis=0)


def peak_excess(mean_points, floor_pts, win, window_step, nwin, step_hz):
    """Per window: (excess of the strongest point over its local floor in dB,
    which point it was, counted from the window's own first point).

    The window SNR of a 200 Hz tone in a 2.7 kHz window is 2.8 dB however loud
    the tone is -- eleven points of floor drown one point of signal -- so a
    narrow signal needs its own measure or it is never "present" at all. Where
    the peak SITS is what tells two windows holding the same signal from two
    windows holding two signals, so it comes back with it; as an offset INTO
    the window, which a retune does not move.

    Read RAW, unlike presence(): a hundred and twenty-eight averaged slots put
    a point of bare noise within half a decibel of its own floor, so there is
    nothing here to smooth away -- and smoothing costs the position, which is
    the point. A three-point average spreads one carrier equally over three
    points, and the window that ENDS one point below a carrier then peaks
    exactly as high as the window the carrier is in.
    """
    r = (np.asarray(mean_points, dtype=np.float64)
         / np.maximum(np.asarray(floor_pts, dtype=np.float64), 1e-30))
    nwin = int(nwin)
    win = int(min(max(int(win), 1), len(r)))
    seg = np.lib.stride_tricks.sliding_window_view(r, win)[::int(window_step)]
    if len(seg) < nwin:                                   # short map: hold the last
        seg = np.concatenate([seg, np.repeat(seg[-1:], nwin - len(seg), axis=0)])
    seg = seg[:nwin]
    off = np.argmax(seg, axis=1)
    return (10.0 * np.log10(np.maximum(seg[np.arange(nwin), off], 1e-30)),
            off.astype(np.int32))
