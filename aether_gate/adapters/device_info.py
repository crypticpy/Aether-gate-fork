#
# Aether-gate — which device is actually plugged in (operator complaint B13).
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""device_block(): the one dict that answers the three questions the control
panel could not — which device this is (RSPduo vs RSPdx vs anything else
Soapy fronts), whether diversity is actually running, and, on a duo, which
tuner the audio and pan are following right now.

NO DEVICE REOPEN. "Diversity off" on a duo does not stop the pair: an RSPduo
in mode=DT keeps both tuners on one clock and one LO regardless of what the
combiner is doing (see diversity_state.py), and /diversity/set already has
every verb the app needs to point the audio and pan at a single tuner —
this module adds nothing there, it only reads back what is already true:

    diversity off, tuner A:  GET /diversity/set?mode=off&source=a&pan=a
    diversity off, tuner B:  GET /diversity/set?mode=off&source=b&pan=b
    diversity on (track):    GET /diversity/set?mode=track&source=combined&pan=combined

device_block() is built from sdr.getHardwareKey(), sdr.getHardwareInfo()
(driver-specific and best-effort — SoapySDRPlay3 has been seen to leave it
sparse, relying on --soapy-args for the serial instead), the --soapy-args
string itself, the channel count the adapter actually opened, and the
adapter's _DiversityState when it has one. Every SoapySDR call is optional:
a quirky or half-open driver must never take down /diagnostics, so nothing
here raises — the caller (soapy.py's diagnostics()) wraps the call in
try/except anyway, but each helper below already catches its own.
"""


def _parse_args(args):
    """"serial=2405055D34,mode=DT" -> {"serial": "2405055D34", "mode": "DT"}.

    The same split soapy.py's _open_hw() does on device_args; kept local so
    this module carries no import of soapy.py (soapy.py imports this one).
    """
    out = {}
    for kv in str(args or "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _hardware_info(sdr):
    """sdr.getHardwareInfo() as a plain str -> str dict, or {} on anything —
    no such method, a driver that raises, a return value that turns out not
    to be dict-like after all. Never raises."""
    if sdr is None:
        return {}
    try:
        info = sdr.getHardwareInfo()
    except Exception:
        return {}
    try:
        return {str(k): str(info[k]) for k in info.keys()}
    except Exception:
        return {}


def _hardware_key(sdr):
    if sdr is None:
        return None
    try:
        key = str(sdr.getHardwareKey())
    except Exception:
        return None
    return key or None


def _tuner_count(channels):
    """`channels` is normally the adapter's list of stream channel indices
    ([0] or [0, 1]), but a bare count is accepted too so a caller need not
    fake a list just to report how many tuners are streaming."""
    if channels is None:
        return 0
    try:
        return len(channels)
    except TypeError:
        return int(channels)


def _diversity_block(tuners, div_state):
    capable = tuners >= 2 and div_state is not None
    mode = getattr(div_state, "mode", None) if div_state is not None else None
    running = capable and mode not in (None, "off")
    hear = getattr(div_state, "hear", None) if div_state is not None else None
    if running or hear == "combined":
        tuner = "both"
    elif hear == "a":
        tuner = "A"
    elif hear == "b":
        tuner = "B"
    elif hear == "stereo":
        tuner = "both"
    else:
        tuner = None
    return {"capable": capable, "running": running, "mode": mode, "tuner": tuner}


def _label(model, serial, driver, tuners, diversity):
    base = model or driver or "radio"
    name = f"{base} {serial}" if serial else base
    if tuners < 2 or not diversity["capable"]:
        tag = "single tuner" if tuners < 2 else f"{tuners} tuners"
        return f"{name} - {tag}"
    if not diversity["running"]:
        return f"{name} - diversity off, tuner {diversity['tuner'] or '?'}"
    return f"{name} - diversity ({diversity['mode']})"


def device_block(sdr, driver, args, channels, div_state):
    """The JSON-ready dict soapy.py's diagnostics() hangs off "device_block".

    sdr        the live SoapySDR Device, or None (closed / never opened).
    driver     the --soapy-driver string ("sdrplay", "rtlsdr", ...).
    args       the --soapy-args string ("serial=2405055D34,mode=DT").
    channels   the adapter's stream channel list ([0] or [0, 1]), or a bare
               tuner count.
    div_state  the adapter's _DiversityState, or None on anything that is
               not a dual-tuner RSPduo.
    """
    info = _hardware_info(sdr)
    parsed = _parse_args(args)
    hw_key = _hardware_key(sdr)
    model = info.get("label") or info.get("model") or hw_key
    serial = info.get("serial") or parsed.get("serial")
    tuners = _tuner_count(channels)
    diversity = _diversity_block(tuners, div_state)
    return {
        "driver": str(driver or ""),
        "model": model,
        "serial": serial,
        "hardware_key": hw_key,
        "tuners": tuners,
        "diversity": diversity,
        "label": _label(model, serial, driver, tuners, diversity),
    }
