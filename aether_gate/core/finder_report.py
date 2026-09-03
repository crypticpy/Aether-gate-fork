#
# Aether-gate — the finder's read side: what /diversity/finder actually says.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Turning ten minutes of scored rows into a list an operator can act on.

The rule this file exists to enforce, after the 2026-09-03 report: the finder
lists what is THERE, not only what it can name. On 20 m that afternoon the
spatial waterfall showed a dense data block at 14070-14082, CW columns at
14085, 14090, 14093, 14100, 14110, 14153, 14160, a strong talker near 14170
and a weak one at 14178 the operator was copying by ear -- and /diversity/finder
returned three candidates, all of them called "voice", two of them digital
sub-bands, and not the one being listened to.

So a candidate is now any window that stood over its own LOCAL floor for at
least CANDIDATE_MIN_S of the ring, whatever kind it turned out to be, ranked
by score, capped at CANDIDATE_MAX. The tuned slice's own column is always in
the list, flagged `tuned`, even when it scored nothing: what the finder thinks
of what the operator is listening to is the one row they can check for
themselves.

Two strips come back with it: `activity`, the share of the ten minutes each
column held anything at all, and `voice_share`, the share it held somebody
talking -- which is what `activity` used to mean on its own, and why the strip
was near-black at a maximum of 0.09 on a busy band.
"""
import math

import numpy as np

from . import finder_bands, kinds

VOICE_SCORE = 0.5            # a window's VOICE score at or above this is
                             # "somebody is talking here" -- the voice_share
                             # strip and active_s, no longer the way in
CANDIDATE_MAX = 40           # the list is a band, not a top ten
CANDIDATE_RECENT_S = 30.0    # a candidate must have scored within this long
CANDIDATE_MIN_S = 2.0        # ...and stood over its local floor for this much
                             # of the ring: two seconds of presence is a
                             # signal, a quarter of a second is a click
CANDIDATE_MIN_SCORE = 0.05   # ...and scored SOMETHING. The presence test is
                             # deliberately sensitive (a 3 dB smoothed chunk),
                             # and a site with five impulses a second trips it
                             # here and there on bare band: measured on the
                             # 2026-09-03 80 m capture it let four windows at
                             # 0.1-0.7 dB into the list. A row the operator can
                             # act on has to have a level as well as a history.
SUPPRESS_MIN_HZ = 500.0      # a candidate hides its neighbours within the
SUPPRESS_WIDTH_FRAC = 0.6    # greater of these -- windows overlap by half a
                             # window, so a 2.4 kHz talker must swallow the two
                             # windows either side of it, while two CW columns
                             # 3 kHz apart are two signals and must both be
                             # listed. The old rule suppressed everything
                             # within a full window width of a peak and lost
                             # every narrow signal beside a strong one.
EDGE_MARGIN_HZ = 150.0       # dial sits this far outside the voice energy
DIAL_GRID_HZ = 500.0         # phone sits on whole and half kilohertz; the map's
                             # points are ~244 Hz apart, so the raw dial estimate
                             # is snapped to the grid (hz) and kept beside it (hz_raw)
DIAL_GRID_NARROW_HZ = 100.0  # ...but a CW tone is not on the phone grid, and
                             # snapping it to the nearest half kilohertz moves
                             # the dial off the signal
NARROW_DIAL_HZ = 900.0       # occupied width at or under which the dial goes ON
                             # the signal rather than beside its passband
USB_ABOVE_HZ = 10_000_000.0  # band convention: USB above 10 MHz, LSB below
NARROW_KINDS = ("cw", "carrier", "psk31", "rtty")
SNR_TIEBREAK = 1e-3          # of a score, to choose between rows that tie


def point_hz(fd, i, center_hz):
    return center_hz - fd.rate_hz / 2 + (i + 0.5) * fd.step_hz


def occupied(fd, w, center_hz):
    """(lo, hi) of the stretch of window w that has energy in it, or None.

    Two points either side of the floor is not a signal's width; what is over
    twice its own local floor is, and it is what the band plan has to be asked
    about.
    """
    last = fd._last
    lo = w * fd.window_step
    seg = last["mean_points"][lo:lo + fd.win]
    fl = last["floor_pts"][lo:lo + fd.win]
    above = np.nonzero(seg > 2.0 * fl)[0]
    if len(above) == 0:
        return None
    return (point_hz(fd, lo + int(above[0]), center_hz) - fd.step_hz / 2,
            point_hz(fd, lo + int(above[-1]), center_hz) + fd.step_hz / 2)


def dial_hz(fd, w, center_hz, narrow=False):
    """(snapped, mode, raw) for window w: for a phone-wide signal the dial sits
    just outside the energy on the carrier side (USB below it, LSB above); for
    a narrow one it sits on the signal, where a CW operator would put it."""
    last = fd._last
    lo = w * fd.window_step
    seg = last["mean_points"][lo:lo + fd.win]
    fl = last["floor_pts"][lo:lo + fd.win]
    usb = center_hz >= USB_ABOVE_HZ
    if narrow:
        rel = seg / np.maximum(fl, 1e-30)
        raw = point_hz(fd, lo + int(np.argmax(rel)), center_hz)
        grid = DIAL_GRID_NARROW_HZ
        return grid * round(raw / grid), ("USB" if usb else "LSB"), raw
    above = np.nonzero(seg > 2.0 * fl)[0]
    if len(above) == 0:
        edge = lo if usb else lo + fd.win - 1
    else:
        edge = lo + (above[0] if usb else above[-1])
    if usb:
        raw = point_hz(fd, edge, center_hz) - fd.step_hz / 2 - EDGE_MARGIN_HZ
    else:
        raw = point_hz(fd, edge, center_hz) + fd.step_hz / 2 + EDGE_MARGIN_HZ
    return DIAL_GRID_HZ * round(raw / DIAL_GRID_HZ), ("USB" if usb else "LSB"), raw


def _window_of(fd, hz, center_hz):
    """The window whose middle is nearest `hz`, clipped into the span."""
    point = (float(hz) - (center_hz - fd.rate_hz / 2)) / fd.step_hz - 0.5
    mid = int(round((point - fd.win / 2.0) / fd.window_step))
    return int(min(max(mid, 0), fd.nwin - 1))


def _pair_terms(fd, w, live_dec, c):
    """The pair's own numbers at a candidate: the diversity gain it could earn,
    and the inter-loop phase, coherence and level ratio there."""
    last = fd._last
    sa = max(float(last["pa"][w] - last["na"]), 0.0)
    sb = max(float(last["pb"][w] - last["nb"]), 0.0)
    if sa > 0 and sb > 0:
        r = min(sa / sb, sb / sa)
        c["gain_db"] = round(10.0 * math.log10(1.0 + r), 1)
    else:
        c["gain_db"] = 0.0
    if live_dec is None:
        return
    lo = w * fd.window_step
    saa = float(np.sum(live_dec[0][lo:lo + fd.win]))
    sbb = float(np.sum(live_dec[1][lo:lo + fd.win]))
    sab = complex(np.sum(live_dec[2][lo:lo + fd.win]))
    c["phase_deg"] = round(math.degrees(math.atan2(sab.imag, sab.real)), 1)
    c["coherence"] = round(min(1.0, abs(sab) ** 2 / max(saa * sbb, 1e-30)), 2)
    c["ratio_db"] = round(10.0 * math.log10(max(sbb, 1e-30) / max(saa, 1e-30)), 1)


def _row(fd, w, center_hz, score, snapshot, live_dec, active_s, last_s):
    """One candidate, described as it was when it scored best.

    `snapshot` is (snr_db, depth, syllabic, kind code, kind confidence) from
    that row: twenty seconds after the last over, "now" is the floor, and a
    row has to describe the conversation it lists.
    """
    snr_w, depth_w, syl_w, kind_w, kconf_w = snapshot
    kind = kinds.name(kind_w)
    bw = float(fd._last["bw_hz"][w])
    narrow = kind in NARROW_KINDS or bw <= NARROW_DIAL_HZ
    hz, mode, hz_raw = dial_hz(fd, w, center_hz, narrow=narrow)
    lo_hz = point_hz(fd, w * fd.window_step, center_hz) - fd.step_hz / 2
    span = occupied(fd, w, center_hz) or (lo_hz, lo_hz + fd.win * fd.step_hz)
    kind, kconf_w, dial = finder_bands.refine(kind, kconf_w, span[0], span[1],
                                              syllabic=syl_w,
                                              present=float(fd._last["present"][w]))
    if dial is not None:      # the band plan's own dial, and its own sideband
        hz, hz_raw, mode = dial, dial, "USB"
    c = {
        "_w": int(w), "hz": round(float(hz), 1), "hz_raw": round(float(hz_raw), 1),
        "mode": mode,
        "width_hz": round(fd.win * fd.step_hz, 1),
        "score": round(float(score), 2),
        # what the gate thinks it is, and how sure: a row that says
        # "cw 0.9" saves the operator the trip
        "kind": kind,
        "kind_conf": round(float(kconf_w), 2),
        "snr_db": round(float(snr_w), 1),
        "syllabic": round(float(syl_w), 2),
        "depth": round(float(depth_w), 2),
        "active_s": round(float(active_s), 1),
        "last_s": None if last_s is None else round(float(last_s), 1),
        "occupied_hz": round(bw, 1),
        "tuned": False,
    }
    _pair_terms(fd, w, live_dec, c)
    return c


def peak_hz(fd, w, center_hz):
    """Where the strongest point of window w actually is: what tells two
    windows holding one signal from two windows holding two."""
    return point_hz(fd, w * fd.window_step + int(fd._last["peak_off"][w]), center_hz)


def _same_signal(fd, w, kept):
    """Is window w the signal `kept` already lists?

    Their strongest points within half the sum of their occupied widths (never
    less than SUPPRESS_MIN_HZ). Windows overlap by half a window, so a talker
    is seen by five of them and every one scores about the same; a keyed column
    3 kHz from a strong one is a different station and must survive. The rule
    this replaced -- suppress everything within a whole window width of a peak
    -- could only do the first of those.
    """
    bw = float(fd._last["bw_hz"][w])
    for o in kept:
        if abs(o["_peak_hz"] - peak_hz(fd, w, o["_center"])) <= max(
                SUPPRESS_MIN_HZ, 0.5 * (bw + o["_bw"])):
            return True
    return False


def _strips(fd, rows_voice, pres_rows):
    """(activity, voice_share) per map point over the whole history."""
    act = (np.mean(pres_rows, axis=0) if len(pres_rows)
           else np.zeros(fd.points))
    voiced = (np.mean(rows_voice >= VOICE_SCORE, axis=0) if len(rows_voice)
              else np.zeros(fd.nwin))
    vs = np.zeros(fd.points)
    for w in range(fd.nwin):
        lo = w * fd.window_step
        vs[lo:lo + fd.win] = np.maximum(vs[lo:lo + fd.win], voiced[w])
    return act, vs


def payload(fd, center_hz=0.0, live=None, tuned_hz=None):
    """/diversity/finder, whole."""
    last = fd._last
    if last is None:
        return {"available": False}
    rows, terms, kind_rows, kconf_rows, times, voice_rows, wpres = fd._slow_rows()
    pres_rows = fd._slow_points()
    is_recent = (fd.elapsed - times) <= CANDIDATE_RECENT_S
    ridx = np.nonzero(is_recent)[0]
    if len(ridx):
        # The best row per window: by score, and among rows that scored the
        # same -- a strong signal saturates the score for its whole over --
        # by SNR, so that a candidate is described from the second it sounded
        # best rather than from the first second it was good enough.
        key = (rows[ridx] + SNR_TIEBREAK
               * np.clip((terms[ridx, 0, :] + 20.0) / 60.0, 0.0, 1.0))
        pick = np.argmax(key, axis=0)
        cols = np.arange(rows.shape[1])
        rec = rows[ridx][pick, cols]
        snr_best = terms[ridx][pick, 0, cols]
        best = ridx[pick]
        rec_pres = np.max(wpres[ridx], axis=0)
    else:                                  # nothing scored lately: what is there now
        rec, best = last["score"], None
        snr_best = last["snr_db"]
        rec_pres = last["wpres"]
    min_frac = float(fd.min_present_frac)
    here = wpres >= min_frac
    active_s = np.sum(here, axis=0) * fd.slow_period_s
    dec = live.decimated(fd.points) if live is not None else None

    def snapshot(w):
        if best is None:
            return (float(last["snr_db"][w]), float(last["depth"][w]),
                    float(last["syllabic"][w]), int(last["kind"][w]),
                    float(last["kind_conf"][w]))
        r = best[w]
        return (float(terms[r, 0, w]), float(terms[r, 1, w]), float(terms[r, 2, w]),
                int(kind_rows[r, w]), float(kconf_rows[r, w]))

    def age(w):
        hit = np.nonzero(here[:, w])[0]
        return float(fd.elapsed - times[hit[-1]]) if len(hit) else None

    def keep(w, score):
        c = _row(fd, w, center_hz, score, snapshot(w), dec, active_s[w], age(w))
        c["_peak_hz"] = peak_hz(fd, w, center_hz)
        c["_bw"] = float(fd._last["bw_hz"][w])
        c["_center"] = center_hz
        return c

    out = []
    # by score, and among the ties -- a plateau of windows all saturated on the
    # same conversation -- by how strong the window was WHEN it scored that, so
    # the row that represents a signal is the one sitting on it and not the one
    # on its skirt that happens to hold more noise right now
    for w in np.lexsort((-np.asarray(snr_best, dtype=np.float64), -rec)):
        if len(out) >= CANDIDATE_MAX:
            break
        if rec_pres[w] < min_frac or rec[w] < CANDIDATE_MIN_SCORE:
            continue                       # never there long enough, or not there
        if _same_signal(fd, w, out):
            continue
        out.append(keep(w, rec[w]))
    if tuned_hz is not None:
        w = _window_of(fd, tuned_hz, center_hz)
        near = [o for o in out if abs(o["_peak_hz"] - float(tuned_hz))
                <= fd.win * fd.step_hz / 2]
        if near:
            near[0]["tuned"] = True
        else:
            c = keep(w, rec[w])
            c["tuned"] = True
            out.append(c)
    for c in out:
        for k in ("_w", "_peak_hz", "_bw", "_center"):
            c.pop(k, None)
    act, voice_share = _strips(fd, voice_rows, pres_rows)
    return {
        "available": True,
        "span_hz": [float(center_hz - fd.rate_hz / 2), float(center_hz + fd.rate_hz / 2)],
        "history_s": float(min(fd.elapsed, fd.slow_rows * fd.slow_period_s)),
        "points": int(fd.points),
        "activity": [round(float(x), 3) for x in act],
        "activity_max": round(float(np.max(act)) if len(act) else 0.0, 3),
        "voice_share": [round(float(x), 3) for x in voice_share],
        "candidates": out,
    }
