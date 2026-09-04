#
# Aether-gate — B25: the governor's readings, and what scores a move with no talker.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""core/governor.py is the state machine and the rules; this is everything it
reads out of ONE snapshot, and the numbers it scores a move by when the usual
one is missing. Pure, no state, no clock: every function here takes a dict.

THE PROXY OBJECTIVES. The governor scores every move against the speech-band
objective the dig uses; on a slice with nobody on it there is no such number
(`snr_db.out` is None, so `digout.objective` is None) and it used to stand still
and say so. Wrong answer: a carrier squeezed at 3 a.m. is still a carrier
squeezed, and nothing measuring the result is a reason to measure it
differently. So each tool that can be judged without a talker gets its own
objective, and the move says which one judged it:

  squeeze  the squeeze's OWN measured depth. core/squeeze.py takes a signal
           target only if it stands MIN_LEVEL_DB over the rest of the passband,
           so a null NULL_DEPTH_KEEP_DB deep has put it back into the floor it
           was picked out of -- the whole job it was accepted for. The notch is
           asked for more: its figure is the FIR's DESIGNED response, not a
           covariance measured through the combiner, and it takes whatever else
           shares those bins with the carrier.
  nb       blanked_pct under digout's BLANK_FREE_PCT, where the real objective
           starts charging for the blanker eating the signal.
  mode     the passband floor, which is the whole point of an idle null.
  guard    the clips and the headroom, EITHER of which moving the right way
           keeps it. Never both together: putting the guard back because only
           one of its two numbers improved hands the ADC its clipping again,
           and the guard is the one stage no later one undoes.

The squeeze and the blanker also watch that floor: a move that LIFTS it more
than FLOOR_RISE_DB bought its target with the band and goes back, whatever it
scored. The floor is /diversity/spatial's level strip, median over the passband.

THE DIG is not one of these -- it needs a talker and owns its own A/B -- but its
report is read here too: a run that called its own result tentative must not be
banked, and blocks the next hand-off for as long as that verdict is fresh.
"""
import statistics

# Quoted from whichever module owns each; test_governor.py asserts they match.
NULL_COHERENCE = 0.5            # core.squeeze.NULL_ENTER_COHERENCE
NULLABLE_COHERENCE = 0.4        # _DiversityState.NULLABLE_COHERENCE
CARRIER_MIN_DB = 6.0            # core.squeeze.MIN_LEVEL_DB
HEADROOM_LOW_DB = 3.0           # adapters.frontend_guard.HEADROOM_LOW_DB
IMPULSE_RATE_ON = 1.0           # adapters.noise_kinds' own "is this impulsive"
HUM_MIN_DB = 8.0                # core.noiseprofile.LINE_MIN_DB
# ...and the proxies', by the same rule: borrowed, never invented.
NULL_DEPTH_KEEP_DB = 6.0        # core.squeeze.MIN_LEVEL_DB: back down to the floor
NOTCH_DEPTH_KEEP_DB = 10.0      # ...and a DESIGNED response is asked for more
BLANKED_MAX_PCT = 5.0           # core.digout.BLANK_FREE_PCT
FLOOR_RISE_DB = 1.0             # a move that lifts the passband floor by this
FLOOR_DROP_DB = 1.0             # ...and what an idle null has to take off it
HEADROOM_GAIN_DB = 1.0          # ...and what the guard has to buy back at the ADC

TAIL = "(no talker to score by)"
TENTATIVE = "tentative"
NAMES = {"squeeze": "depth", "nb": "blanking", "mode": "floor", "guard": "clips"}
# The tools by the names the operator sees on the chain, for every sentence
# and label the app shows a person -- never the key ("nb", "dig") itself.
WORDS = {"guard": "the front-end guard", "nb": "the blanker", "mode": "a null",
         "squeeze": "a squeeze", "dig": "DIG OUT"}
HELD_WORDS = dict(WORDS, dig="what DIG OUT found")   # ...as a thing being held


def _num(x, default=None):
    try:
        f = float(x)
    except (TypeError, ValueError):
        return default
    return f if f == f and abs(f) != float("inf") else default


def blank_threshold_db(impulse_db):
    """noise_kinds.py's own recommendation: 6..30 dB to the half-decibel, so the
    governor and the SITE page's BLANK button ask for the same number."""
    return min(30.0, max(6.0, round((_num(impulse_db, 12.0) - 3.0) * 2) / 2))


def tool_state(snap):
    """The tools' settings, one comparable value each: the whole of the drift
    detection -- what moves here that we did not write is somebody's hand."""
    sq, nb = snap.get("squeeze") or {}, snap.get("nb") or {}
    return {
        "squeeze": (sq.get("target") if sq.get("configured") else None,
                    _num(sq.get("hz"))),
        "nb": (bool(nb.get("on")), _num(nb.get("db")), nb.get("auto")),
        "mode": snap.get("mode"),
        "guard": bool(snap.get("guard")),
    }


def strongest_carrier(snap):
    """The loudest carrier the finder named that is worth a tool at all."""
    best = None
    for c in (snap.get("carriers") or []):
        hz, db = _num(c.get("hz")), _num(c.get("db"), 0.0)
        if hz is None or db < CARRIER_MIN_DB:
            continue
        if best is None or db > best["db"]:
            best = {"hz": hz, "db": db}
    return best


def inband_floor_db(spatial):
    """The median of /diversity/spatial's level strip over the passband bins:
    where the floor sits, needing nobody to be talking. The MEDIAN, so the
    carrier a squeeze is aimed at cannot drag the reading that judges it."""
    spatial = spatial or {}
    lv = spatial.get("level_db")
    start, step = _num(spatial.get("start_hz")), _num(spatial.get("step_hz"))
    if not lv or start is None or not step:
        return None
    band, seg = spatial.get("passband_hz"), lv
    if isinstance(band, (list, tuple)) and len(band) == 2:
        lo, hi = _num(band[0]), _num(band[1])
        if lo is not None and hi is not None:
            a, b = max(0, int((lo - start) / step)), min(len(lv),
                                                         int((hi - start) / step) + 1)
            seg = lv[a:b] if b > a else lv
    return round(statistics.median(seg), 1)


# ----- the proxy objectives -------------------------------------------------

def scorer(tool):
    """The number this tool's move is judged by with no objective: the app
    prints it as "scored by <proxy>"."""
    return "proxy:" + NAMES.get(tool, "floor")


def readings(snap):
    """The proxy quantities, one snapshot's worth: what a move is compared
    against when there is no objective to compare it against."""
    sq = snap.get("squeeze") or {}
    return {"floor_db": _num(snap.get("floor_db")),
            "depth_db": _num(sq.get("depth_db")),
            "blanked_pct": _num(snap.get("blanked_pct"), 0.0),
            "impulses_per_s": _num(snap.get("impulses_per_s"), 0.0),
            "clips_1s": _num(snap.get("clips_1s")),
            "headroom_db": _num(snap.get("headroom_db"))}


def verdict(tool, kind, before, snap):
    """Keep or put back, and the sentence naming the number that decided it."""
    after, before = readings(snap), before or {}
    a, b = before.get("floor_db"), after["floor_db"]
    rise = None if a is None or b is None else round(b - a, 1)
    if tool == "squeeze":
        return _squeeze(kind, snap, after, rise)
    if tool == "nb":
        return _nb(before, after, rise)
    if tool == "mode":
        return _mode(rise)
    if tool == "guard":
        return _guard(before, after)
    return True, f"kept {WORDS.get(tool, tool)}: nothing to score it by {TAIL}"


def _squeeze(kind, snap, after, rise):
    name = (snap.get("squeeze") or {}).get("tool") or "null"
    want = NOTCH_DEPTH_KEEP_DB if name == "notch" else NULL_DEPTH_KEEP_DB
    depth = after["depth_db"]
    if depth is None:
        sq = snap.get("squeeze") or {}
        if not sq.get("held"):
            reason = sq.get("reason") or "never took hold"
            return False, f"put the squeeze back: {reason} on the {kind} {TAIL}"
        return False, f"put the squeeze back: no {name} depth measured on the {kind} {TAIL}"
    if depth < want:
        return False, (f"put the squeeze back: only {depth:.1f} dB of {name} depth on "
                       f"the {kind}, under the {want:g} it needs {TAIL}")
    if rise is not None and rise > FLOOR_RISE_DB:
        return False, f"put the squeeze back: the passband floor rose {rise:.1f} dB {TAIL}"
    return True, f"kept the squeeze: {depth:.1f} dB of {name} depth on the {kind} {TAIL}"


def _nb(before, after, rise):
    pct = after["blanked_pct"] or 0.0
    if pct > BLANKED_MAX_PCT:
        return False, (f"put the blanker back: blanking {pct:.1f} % of the samples, "
                       f"over {BLANKED_MAX_PCT:g} % {TAIL}")
    if rise is not None and rise > FLOOR_RISE_DB:
        return False, f"put the blanker back: the passband floor rose {rise:.1f} dB {TAIL}"
    d = after["impulses_per_s"] - (before.get("impulses_per_s") or 0.0)
    return True, f"kept the blanker: blanking {pct:.1f} %, impulses {d:+.1f}/s {TAIL}"


def _mode(rise):
    if rise is None:
        return True, f"kept the null: no passband floor to measure it by {TAIL}"
    if rise > -FLOOR_DROP_DB:
        return False, (f"put the null back: it moved the passband floor "
                       f"{rise:+.1f} dB, not the {FLOOR_DROP_DB:g} dB off it {TAIL}")
    return True, f"kept the null: {-rise:.1f} dB off the passband floor {TAIL}"


def _guard(before, after):
    """EITHER number moving the right way keeps it, because the alternative is
    handing the ADC back its clipping over the one that did not."""
    was, has = before.get("clips_1s"), after["clips_1s"]
    hr0, hr = before.get("headroom_db"), after["headroom_db"]
    gain = None if hr0 is None or hr is None else round(hr - hr0, 1)
    if has is not None and (not has or (was is not None and has < was)):
        return True, f"kept the guard: clips {was or 0:g}/s -> {has:g} {TAIL}"
    if gain is not None and gain >= HEADROOM_GAIN_DB:
        return True, f"kept the guard: headroom {gain:+.1f} dB {TAIL}"
    if has is None and gain is None:
        return True, f"kept the guard: no front-end reading to score it by {TAIL}"
    return False, (f"put the guard back: clips {was or 0:g}/s -> {has or 0:g} and "
                   f"headroom {gain or 0.0:+.1f} dB {TAIL}")


# ----- the dig's own report -------------------------------------------------

def dig_key(snap):
    """(slice frequency, whoever is on it): the pair a hand-off happens once
    for, ever. The dig turns every knob for a minute, and doing that again for
    the same talker on the same frequency is the cycle that ran all evening."""
    hz = _num(snap.get("slice_hz"))
    if hz is None:
        return None
    who = snap.get("focus") or snap.get("talker")
    return (round(hz), None if who is None else str(who))


def tentative_age_s(snap, seen, now):
    """How long the dig's last word has stood, and the (note, t) pair to keep:
    from the dig's OWN end where the status timestamps it, else from when the
    runner first saw this note. Returns (age, seen)."""
    note = snap.get("dig_note")
    if seen is None or seen[0] != note:
        seen = (note, now)
    age = _num(snap.get("dig_age_s"))
    return (now - seen[1] if age is None else age), seen


def dig_blocked(snap, age_s=0.0, window_s=float("inf")):
    """Why this slice must not be handed to the dig, or None. A run the dig
    itself called tentative is not a measurement to plan the next hour on --
    for `window_s` from its own end, which is long enough for the band to have
    become a different band and short enough not to be a permanent lockout."""
    if snap.get("dig_running"):
        return "a dig is already running"
    if age_s >= window_s:
        return None
    if snap.get("dig_unsteady"):
        return "the last dig found the band too unsteady to conclude anything"
    if TENTATIVE in str(snap.get("dig_note") or "").lower():
        return "the last dig's own note calls its result tentative"
    return None


def dig_delta_db(snap):
    """What a finished dig earned, and the note when that is not its own
    gain_db: banked only where the dig stands behind it, so a tentative run, a
    cancelled one or a "worse" verdict all score 0."""
    gain = _num(snap.get("dig_gain_db"), 0.0)
    note = str(snap.get("dig_note") or "").strip()
    if snap.get("dig_cancelled"):
        return 0.0, "the dig was cancelled: nothing banked"
    if snap.get("dig_unsteady") or TENTATIVE in note.lower():
        return 0.0, f"scored 0, not {gain:+.1f} dB: {note or TENTATIVE}"
    if snap.get("dig_verdict") == "worse":
        return 0.0, "the dig was called worse and put back: nothing banked"
    if snap.get("dig_verdict") == "moved":
        return 0.0, "the dial moved and the dig put it all back: nothing banked"
    if not gain:
        return 0.0, "the dig tried every knob and found nothing here"
    return round(gain, 2), ""
