#
# Aether-gate — the adapter's side of SQUEEZE (core/squeeze.py).
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""What core.squeeze cannot know about on its own: the operator's set()
call, the target's absolute reference frequency (core.squeeze works in the
slice's own baseband; the adapter is what knows `_slice_hz`), the notch
tool's own bins in core/filter.py (core.squeeze picks the tool; it does not
touch a filter), and the two numbers that need something outside the
target itself -- `talker_cost_db` (the tracker's talker, which the target
has never seen) and `bearing_deg`/`mirror_deg` (the site's beacon-fitted
compass, which lives in the site log). Kept out of _DiversityState, which
is at its own 800-line budget, the same way diversity_enhance.py holds
post/mrc/compass.
"""
import time

import numpy as np


def set_squeeze(state, hz, width_hz, spacing_hz=None, offset_hz=None, now=None):
    """hz: a signed offset in Hz (a "signal" target), the literal "comb"
    (spacing_hz/offset_hz given: that pair outright; neither given:
    auto-detect from the next ~2 s), "off"/""/"none" to release, or None to
    leave the target alone and only move squeeze_width on one already held.
    width_hz, given, replaces the default/last signal width."""
    now = time.time() if now is None else now
    if hz is not None and str(hz).strip().lower() in ("off", "none", ""):
        state.squeeze.off()
        return
    if hz is not None and str(hz).strip().lower() == "comb":
        if spacing_hz is not None:
            state.squeeze.set_comb(float(spacing_hz), float(offset_hz or 0.0), now)
        else:
            state.squeeze.set_comb_auto(now)
        return
    if hz is None:
        if not state.squeeze.active or state.squeeze.target != "signal":
            return
        hz = state.squeeze.hz
    state.squeeze.set(float(hz), None if width_hz is None else float(width_hz), now)


def _ref_hz(state):
    return float(getattr(state.a, "_slice_hz", None) or getattr(state.a, "center_hz", 0.0) or 0.0)


def observe(state, t, X, f, n, rate, now):
    """One audio block's STFT, measured for the squeeze target; returns it
    for Tracker.update() only while it should hold the NULL over the WHOLE
    passband (scope "passband", tool "null") -- coherence chose the notch
    tool instead, or the sub-band refinement is on, and the tracker's own
    fit runs untouched (see core.squeeze's module docstring)."""
    lo, hi = state._pass_edges(getattr(state.a, "_mode", "USB"))
    sq = state.squeeze
    sq.scope = "bins" if state.subband_on else "passband"
    sq.refresh(X, f, lo, hi, n / rate, t.m, now, ref_hz=_ref_hz(state))
    return sq if sq.held and sq.tool == "null" and sq.scope == "passband" else None


def _notch_targets(sq):
    """(hz, width) pairs the notch tool owns right now -- one region for a
    signal target, every in-band tooth for a comb (core.comb.TOOTH_WIDTH_HZ:
    the same window core.squeeze itself pools their bins with)."""
    if sq.target == "comb":
        from ..core.comb import TOOTH_WIDTH_HZ
        return [(hz, TOOTH_WIDTH_HZ) for hz in sq.teeth_in_band]
    return [(sq.hz, sq.width_hz)]


def subband_squeeze(sq, sb):
    """Wire a held, NULL-tool, "bins"-scope squeeze into the sub-band
    combiner's forced per-bin null -- one target region for a signal, every
    in-band tooth for a comb, all forced to the SAME null (one source, one
    steering vector). See core.subband.SubbandCombiner.set_squeeze."""
    on = sq.held and sq.tool == "null" and sq.scope == "bins"
    sb.set_squeeze(on, _notch_targets(sq) if on else [], sq.null_m)


def sync_notches(state, sq):
    """Wire a held, NOTCH-tool squeeze into core.filter's own notch
    machinery (see SliceFilter.set_squeeze_notches); released the moment
    the tool is not "notch" -- the null tool, or nothing held at all."""
    filt = getattr(state.a, "_filt", None)
    if filt is None:
        return
    filt.set_squeeze_notches(_notch_targets(sq) if sq.held and sq.tool == "notch" else [])


def _talker_cost_db(t, m_null):
    """What the whole-passband null costs the over in progress: the
    talker's own steering vector (from the same Rs/Rn the tracker fits
    against) run through the squeeze's null and through the tracker's own
    beam, compared. None while nobody is talking -- there is no talker
    steering vector to cost."""
    if t is None or t.Rs is None or t.Rn is None:
        return None
    from ..core.diversity import steering_of
    S = t.Rs - t.Rn
    if not (float(np.real(np.trace(S))) > 0):
        return None
    s = steering_of(S)
    g_beam = abs(s[0] + t.m * s[1])
    g_squeeze = abs(s[0] + m_null * s[1])
    if g_squeeze <= 1e-12:
        return None
    return round(20.0 * np.log10(max(g_beam, 1e-12) / g_squeeze), 1)


def _bearing(state, now):
    """Where the held target's own phase points, the same compass the site
    log's beacons fit (core.compass): None, None while there is nothing
    held or the pair has no fit yet."""
    sq = state.squeeze
    if not sq.held:
        return None, None
    fit = state.enh.global_fit(state.sitelog, now)
    if fit is None or not getattr(fit, "available", False):
        return None, None
    slice_hz = getattr(state.a, "_slice_hz", None)
    if slice_hz is None:
        return None, None
    # a signal target's own offset; a comb's nearest in-band tooth, since
    # the comb itself has no single "hz" -- either way, one bin the target
    # is actually in right now.
    offset = sq.hz if sq.target == "signal" else (sq.teeth_in_band[0] if sq.teeth_in_band else None)
    if offset is None:
        return None, None
    f_hz = float(slice_hz) + float(offset)
    # phase_deg is already s[1]'s angle with s[0] real positive, i.e. "B
    # relative to A" -- the log's own convention (see core.noisebearing);
    # unlike the tracker's weight, which is the OPPOSITE sign, this needs
    # no negation.
    ans = fit.bearing_from_phase(sq.phase_deg, f_hz)
    seen = ans.get("bearings_deg") or []
    if not seen:
        return None, None
    bearing = round(float(seen[0]), 1)
    mirror = round((2.0 * fit.baseline_deg - bearing) % 360.0, 1)
    return bearing, mirror


def _notch_depth_db(state, sq):
    """core.squeeze cannot measure this -- it has no taps, and no bank
    either -- so the slice filter's own COMBINED response (the FIR and its
    dedicated SQUEEZE notch bank, core/notchbank.py, together) at the
    target's frequencies: the depth actually delivered, not the FIR's own
    design alone (see SliceFilter.combined_response_db)."""
    filt = getattr(state.a, "_filt", None)
    if filt is None or filt.taps is None:
        return None
    hzs = [hz for hz, _w in _notch_targets(sq)]
    if not hzs:
        return None
    vals = [-filt.combined_response_db(hz) for hz in hzs]
    return round(float(np.mean(vals)), 1)


def status(state, sid, now=None):
    """The full "squeeze" status row: core.squeeze's own measured facts
    (depth_db filled in from the taps when the tool is the notch), plus the
    talker cost (passband-scope null only -- see core.squeeze's module
    docstring) and the bearing, if the site has a compass fit."""
    now = time.time() if now is None else now
    sq = state.squeeze
    out = sq.status()
    if out["held"] and out["tool"] == "notch":
        out["depth_db"] = _notch_depth_db(state, sq)
    t = state.trackers.get(sid)
    out["talker_cost_db"] = (_talker_cost_db(t, sq.null_m)
                             if sq.held and sq.tool == "null" and sq.scope == "passband" else None)
    out["bearing_deg"], out["mirror_deg"] = _bearing(state, now)
    return out
