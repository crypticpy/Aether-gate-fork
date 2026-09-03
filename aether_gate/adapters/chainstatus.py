#
# Aether-gate — the receive chain as one array of rows, in signal order.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""What `chain` in /filter is: the whole receive chain written out as one
row per stage, from the antenna port to the point where the gate hands the
audio to the app, in the order the signal actually travels.

The operator's ask was "I'd like to see which roofing filters we're adding,
or turn all the different filters on and off, simulating all the filters in
a really high-end radio" — so every row names the stage the way a top-end
radio's manual names it (PRE/ATT, RF GAIN, ROOFING, NB, twin PBT, CONTOUR,
APF, AGC-T), says in one line what it is doing right now, and carries either
the one control-port call that changes it or the reason it cannot be
changed. That is the same `{"label", "route", "query"}` + `"why"` shape the
noise profile's `kinds` rows have carried since the SITE page shipped
(noise_kinds.py), so the app renders a stage it has never heard of without
an app release, and a new stage here is a new row there for free.

NOTHING IS RECOMPUTED AND NOTHING IS INVENTED. Every number in every row is
quoted from a status dict the gate had already built: the slice filter's
status(), the pair's diversity status(), the device's own read-back
settings. `measured` appears only where the gate genuinely measures a level
at that stage — the combiner's SNR in and out, the post-filter's mean gain,
the depth of a notch it designed — and is absent everywhere else rather than
carrying a plausible-looking zero.

Pure functions of those dicts: no hardware, no adapter state, no clock.
"""

CHAIN_KINDS = ("toggle", "select", "value", "fixed")

SET_ON_THE_SETUP_PAGE = "set on the setup page"
COMBINER_MODES = ["off", "manual", "null", "track"]
AGC_ORDER = ["fast", "med", "slow", "long", "off"]
SHAPE_ORDER = ["soft", "sharp"]


def _row(row_id, name, kind, detail, enabled=True, fixed=False,
         value=None, options=None, action=None, why=None, measured=None):
    """One chain row. `action` and `why` are both always present with one of
    them None — the SITE page's rows have that shape and the app already
    reads it."""
    out = {"id": row_id, "name": name, "kind": kind, "fixed": bool(fixed),
           "enabled": bool(enabled), "detail": detail,
           "action": action, "why": why}
    if value is not None:
        out["value"] = value
    if options is not None:
        out["options"] = list(options)
    if measured is not None:
        out["measured"] = measured
    return out


def _set(route, query, label="SET"):
    return {"label": label, "route": route, "query": query}


def _toggle(route, key, on, label_on="ON", label_off="OFF"):
    """The one call that flips this stage: OFF while it is on, ON while it is
    off. The label is the verb, never the state."""
    return {"label": label_off if on else label_on,
            "route": route, "query": f"{key}=" + ("off" if on else "on")}


def _khz(hz):
    return f"{float(hz) / 1000.0:g} kHz"


def _db(x, nd=1):
    return "-" if x is None else f"{float(x):.{nd}f}"


def _dot(*parts):
    return " · ".join(p for p in parts if p)


# ----- the front end, from device_controls() ---------------------------------

def _settings(device):
    if not device:
        return {}
    return {s["key"]: s for s in device.get("settings") or [] if "key" in s}


def _onoff(item):
    return bool(item) and str(item.get("value", "")).lower() in ("true", "1", "on")


def _frontend_rows(device, frontend):
    """ANTENNA, the broadcast traps, LNA, IFGR and the hardware AGC: what the
    rest of the chain hears is decided here, and all of it is set on the setup
    page, so every row is `fixed` and carries `why` instead of an action."""
    rows = []
    ant = (device or {}).get("antenna")
    if ant:
        ports = ant.get("options") or []
        rows.append(_row(
            "antenna", "ANTENNA", "fixed",
            _dot(str(ant.get("value") or "?"),
                 "the only port this mode offers" if len(ports) == 1 else
                 f"1 of {len(ports)} ports"),
            fixed=True, why=SET_ON_THE_SETUP_PAGE))
    st = _settings(device)
    mw, dab = st.get("rfnotch_ctrl"), st.get("dabnotch_ctrl")
    if mw is not None or dab is not None:
        rows.append(_row(
            "traps", "BC / DAB TRAP", "fixed",
            _dot("MW/FM " + ("on" if _onoff(mw) else "off") if mw else None,
                 "DAB " + ("on" if _onoff(dab) else "off") if dab else None),
            enabled=_onoff(mw) or _onoff(dab), fixed=True,
            why=SET_ON_THE_SETUP_PAGE))
    lna = st.get("rfgain_sel")
    if lna is not None:
        opts = lna.get("options") or []
        rows.append(_row(
            "lna", "PRE / ATT · LNA", "fixed",
            f"state {lna.get('value')}" + (f" of 0-{opts[-1]}" if opts else ""),
            fixed=True, why=SET_ON_THE_SETUP_PAGE))
    fe = frontend or {}
    if fe.get("gain_db") is not None:
        lo, hi = (fe.get("gain_range") or (None, None))[:2]
        bounded = lo is not None and hi is not None and float(hi) > float(lo)
        rows.append(_row(
            "ifgr", "RF GAIN · IFGR", "fixed",
            f"{float(fe['gain_db']):.0f} dB"
            + (f" of {float(lo):.0f}-{float(hi):.0f}" if bounded else ""),
            fixed=True, why=SET_ON_THE_SETUP_PAGE))
    setp = st.get("agc_setpoint")
    if "agc" in fe or setp is not None:
        on = bool(fe.get("agc"))
        rows.append(_row(
            "rf_agc", "RF AGC", "fixed",
            _dot("on" if on else "off",
                 f"set-point {setp['value']} dBfs" if setp else None),
            enabled=on, fixed=True, why=SET_ON_THE_SETUP_PAGE))
    return rows


# ----- roofing ---------------------------------------------------------------

def _roof_rf_row(roofing):
    hz, opts = roofing.get("analogue_hz"), roofing.get("analogue_options")
    if hz is None and not opts:
        return None
    detail = _khz(hz) if hz else "unknown"
    if opts and hz is not None and float(hz) <= min(float(o) for o in opts):
        detail = _dot(detail, "the narrowest this hardware has")
    if roofing.get("analogue_source") == "rate":
        detail = _dot(detail, "following the sample rate")
    if not opts:
        return _row("roof_rf", "ROOFING · RF", "value", detail, value=hz,
                    why="this driver does not offer its IF bandwidths")
    return _row("roof_rf", "ROOFING · RF", "select", detail, value=hz,
                options=opts, action=_set("/filter/set", "roof_hz="))


def _roof_digital_row(roofing):
    hz, opts = roofing.get("digital_hz"), roofing.get("digital_options")
    full, taps = roofing.get("digital_full_hz"), roofing.get("digital_taps")
    on = roofing.get("digital_active")
    if on is None:                       # a gate that has no digital roof stage
        on = bool(hz) and bool(full) and float(hz) * 2.0 < float(full)
    detail = (_dot(_khz(hz), f"{taps} taps" if taps else None) if on
              else _dot("off", f"the full {_khz(full)}" if full else None))
    if not opts:
        return _row("roof_digital", "ROOFING · DIGITAL", "value", detail,
                    enabled=on, value=hz,
                    why="this gate has no digital roofing filter")
    return _row("roof_digital", "ROOFING · DIGITAL", "select", detail,
                enabled=on, value=hz, options=opts,
                action=_set("/filter/set", "digital_roof_hz="))


# ----- the pair (dual tuner only) --------------------------------------------

def _pair_rows(div):
    """ALIGN, COMBINER, SUB-BAND and POST-FILTER, quoting /diversity's own
    numbers. Absent entirely on a single-tuner device, which is the point of
    a gate-authored array: the app renders what it is sent."""
    rows = []
    lag, peak = div.get("lag_samples"), div.get("corr_peak")
    aligned = bool(div.get("aligned"))
    rows.append(_row(
        "align", "ALIGN", "value",
        _dot(f"lag {lag} samples" if lag is not None else None,
             "realigning" if div.get("realigning") else
             ("locked" if aligned else "searching"),
             f"peak {peak:g}" if peak is not None else None),
        enabled=aligned, value=lag,
        action=_set("/diversity/align", "", label="REALIGN")))
    return rows


def _combiner_rows(div):
    rows = []
    mode = div.get("mode")
    snr = div.get("snr_db") or {}
    loops = [snr.get("a"), snr.get("b")]
    best = max([v for v in loops if v is not None], default=None)
    measured = ({"in_db": round(float(best), 1), "out_db": round(float(snr["out"]), 1)}
                if best is not None and snr.get("out") is not None else None)
    rows.append(_row(
        "combiner", "COMBINER", "select",
        _dot(str(mode),
             f"φ {div['phase_deg']:g}°" if div.get("phase_deg") is not None else None,
             f"{div['ratio_db']:g} dB" if div.get("ratio_db") is not None else None,
             f"SNR a {_db(snr.get('a'))} / b {_db(snr.get('b'))} → "
             f"{_db(snr.get('out'))} dB" if measured else None),
        enabled=mode not in (None, "off"), value=mode, options=COMBINER_MODES,
        action=_set("/diversity/set", "mode="), measured=measured))
    sb = div.get("subband") or {}
    rows.append(_row(
        "subband", "SUB-BAND NULL", "toggle",
        _dot(f"{sb.get('bins', 0)} bins",
             f"{float(sb.get('extra_db') or 0.0):+.1f} dB"),
        enabled=bool(sb.get("enabled")),
        action=_toggle("/diversity/set", "subband", bool(sb.get("enabled")))))
    pf = div.get("post") or {}
    mean = pf.get("mean_db")
    rows.append(_row(
        "post", "POST-FILTER", "toggle",
        _dot(f"floor {_db(pf.get('floor_db'))} dB",
             f"mean {_db(mean)} dB" if mean is not None else None),
        enabled=bool(pf.get("enabled")),
        action=_toggle("/diversity/set", "post", bool(pf.get("enabled"))),
        measured=None if mean is None else {"in_db": None,
                                            "out_db": round(float(mean), 1)}))
    return rows


# ----- the slice filter ------------------------------------------------------

def _nb_row(filt, div):
    nb = (div or {}).get("nb") or filt.get("nb") or {}
    on = bool(nb.get("enabled"))
    auto = nb.get("auto") or {}
    reason = auto.get("reason")
    if reason is None and auto.get("mode") == "auto":
        reason = "idle"
    return _row(
        "nb", "NB", "toggle",
        _dot(f"{_db(nb.get('threshold_db'))} dB",
             f"{float(nb.get('blanked_pct') or 0.0):.2f} % blanked",
             f"auto: {reason}" if reason else None),
        enabled=on, action=_toggle("/filter/set", "nb", on))


def _slice_rows(filt):
    rows = []
    bypassed = bool(filt.get("bypass"))
    rows.append(_row(
        "slice", "SLICE FILTER", "toggle",
        "bypassed, nothing below is in circuit" if bypassed else
        _dot(f"{filt.get('taps')} taps", str(filt.get("shape"))),
        enabled=not bypassed,
        action={"label": "IN" if bypassed else "BYPASS", "route": "/filter/set",
                "query": "bypass=off" if bypassed else "bypass=on"}))
    rows.append(_row(
        "passband", "PASSBAND (twin PBT)", "value",
        _dot(f"{filt.get('low_hz')}-{filt.get('high_hz')} Hz",
             f"asked {filt.get('set_low_hz')}-{filt.get('set_high_hz')}",
             str(filt.get("sideband") or "").upper() or None),
        value=filt.get("width_hz"),
        why="both edges move on the curve, or /filter/set?low=&high="))
    au = filt.get("auto") or {}
    rows.append(_row(
        "auto", "AUTO WIDTH", "toggle",
        _dot(au.get("source") or ("no fit yet" if au.get("enabled") else "off"),
             f"{au['low_hz']}-{au['high_hz']} Hz" if au.get("low_hz") is not None else None),
        enabled=bool(au.get("enabled")),
        action=_toggle("/filter/set", "auto", bool(au.get("enabled")))))
    rows.append(_row(
        "shape", "SHAPE", "select",
        _dot(str(filt.get("shape") or "").upper(), f"{filt.get('taps')} taps",
             f"{filt.get('transition_hz')} Hz skirt"),
        value=filt.get("shape"), options=SHAPE_ORDER,
        action=_set("/filter/set", "shape=")))
    notches = filt.get("notches") or []
    depths = [n.get("depth_db") for n in notches if n.get("depth_db") is not None]
    on = bool(filt.get("notches_on", True))
    rows.append(_row(
        "notch", "IF NOTCH", "toggle",
        _dot(f"{len(notches)} set" if notches else "none set",
             ", ".join(f"{n['hz']:g} Hz" for n in notches) or None,
             "held out of the taps" if notches and not on else None),
        enabled=on, action=_toggle("/filter/set", "notches", on),
        measured=None if not (on and depths) else
        {"in_db": None, "out_db": round(-max(depths), 1)}))
    anf = filt.get("anf") or {}
    found, adepth = anf.get("found_hz") or [], anf.get("depth_db") or []
    rows.append(_row(
        "anf", "ANF · DNF", "toggle",
        _dot(", ".join(f"{hz:g} Hz" for hz in found) or
             ("no tones found" if anf.get("enabled") else "off")),
        enabled=bool(anf.get("enabled")),
        action=_toggle("/filter/set", "anf", bool(anf.get("enabled"))),
        measured=None if not adepth else
        {"in_db": None, "out_db": round(-max(abs(d) for d in adepth), 1)}))
    co = filt.get("contour") or {}
    # AUTO IS THE SWITCH WHILE AUTO IS ON. With auto_contour set, the bell in
    # force is the one fitted from the talker's voice print and `contour` does
    # not reach it (filter.py _contour), so a row that sent contour=off there
    # would be a switch that did nothing.
    auto = bool(co.get("auto"))
    armed = auto or bool(co.get("enabled"))
    rows.append(_row(
        "contour", "CONTOUR", "toggle",
        _dot("auto" if auto else "manual",
             f"{co['hz']:g} Hz {float(co.get('db') or 0.0):+.1f} dB, "
             f"{co['width_hz']:g} Hz wide" if co.get("enabled") and co.get("hz")
             else "no bell fitted" if auto else "off"),
        enabled=armed,
        action=_toggle("/filter/set", "auto_contour" if auto else "contour", armed)))
    apf = filt.get("apf") or {}
    rows.append(_row(
        "apf", "APF", "toggle",
        _dot(f"{float(apf.get('hz') or 0.0):g} Hz",
             f"{float(apf.get('width_hz') or 0.0):g} Hz wide"),
        enabled=bool(apf.get("enabled")),
        action=_toggle("/filter/set", "apf", bool(apf.get("enabled")))))
    eq = filt.get("auto_eq") or {}
    rows.append(_row(
        "auto_eq", "RX EQ (auto tilt)", "toggle",
        _dot(f"tilt {float(eq.get('tilt_db') or 0.0):+.1f} dB",
             f"lean {float(eq.get('lean_db') or 0.0):+.1f} dB"),
        enabled=bool(eq.get("enabled")),
        action=_toggle("/filter/set", "auto_eq", bool(eq.get("enabled")))))
    return rows


def _agc_row(filt):
    ag = filt.get("agc") or {}
    return _row(
        "agc", "AGC", "select",
        _dot(str(ag.get("mode")),
             f"{ag['attack_ms']:g}/{ag['decay_ms']:g}/{ag['hang_ms']:g} ms"
             if ag.get("attack_ms") is not None else None,
             f"AGC-T {float(ag.get('threshold_db') or 0.0):g}",
             f"{float(ag['gain_db']):+.1f} dB" if ag.get("gain_db") is not None else None),
        enabled=ag.get("mode") != "off", value=ag.get("mode"), options=AGC_ORDER,
        action=_set("/filter/set", "agc="))


# ----- the whole chain -------------------------------------------------------

def chain_rows(filt, div=None, device=None, frontend=None):
    """The chain, in signal order.

    filt      the slice filter's status (SliceFilter.status() plus the
              adapter's `mode` and `roofing`) -- i.e. /filter itself.
    div       /diversity's status dict, or None on a single-tuner device:
              the ALIGN, COMBINER, SUB-BAND and POST-FILTER rows simply do
              not appear when there is no pair.
    device    device_controls() (antenna + the driver's own settings), or
              None when the adapter has no device surface.
    frontend  {"gain_db", "gain_range": (lo, hi), "agc": bool} -- the two
              front-end numbers Soapy carries as a gain element rather than
              a setting.
    """
    roofing = filt.get("roofing") or {}
    rows = _frontend_rows(device, frontend)
    roof_rf = _roof_rf_row(roofing)
    if roof_rf is not None:
        rows.append(roof_rf)
    rate = roofing.get("samp_rate_hz")
    if rate:
        rows.append(_row(
            "adc", "ADC · SAMPLE RATE", "fixed", f"{float(rate) / 1000.0:g} kS/s",
            fixed=True, value=rate,
            why="the resolution control sets it (/resolution?rate=)"))
    if div:
        rows.extend(_pair_rows(div))
    rows.append(_nb_row(filt, div))
    rows.append(_roof_digital_row(roofing))
    if div:
        rows.extend(_combiner_rows(div))
    rows.extend(_slice_rows(filt))
    rows.append(_row(
        "detect", "DETECTOR", "fixed",
        "FM discriminator" if str(filt.get("mode") or "").upper() in
        ("FM", "FM-N", "NFM", "DFM") else
        f"{str(filt.get('mode') or 'SSB').upper()} product detector",
        fixed=True, why="the mode decides it"))
    rows.append(_agc_row(filt))
    rows.append(_row(
        "app", "→ AETHER VOICE", "fixed",
        "noise reduction and compression run in the app",
        fixed=True, why="the app's own chain, downstream of the gate"))
    return rows
