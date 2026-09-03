#
# Aether-gate — dig this out: a timed search over the knobs, scored by ear-proxy.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The operator has tuned something weak and pressed a button: spend the next
minute (or three, or five) trying settings and keep what actually helped.

This module is the search and nothing else. It never touches the adapter and
it never sleeps. It hands out one instruction at a time —

    {"op": "set",     ...}   put this knob here
    {"op": "measure", ...}   wait settle_s, then read the status and feed me
    {"op": "done"}           stop

— and takes back a single number for each measure. That makes the whole
strategy testable against a made-up landscape with a made-up clock; the
runner in adapters/diversity_dig.py owns the thread, the hardware and the
snapshot.

THE OBJECTIVE (`objective`) is one float, higher is better, in dB:

    J = snr_db.out                       the combined output SNR, always there
      + 3.0 * min(1, talk_mod / 2)       voice only: syllabic swing, not level
      + 2.0 * passband.flatness          one weight fitting the whole passband
      - 2.0 * (blanked_pct over 5%)      the blanker eating the signal too

`snr_db.out` carries the term because it is the only SNR the chain reports
the same way whichever post filter is running — `post.snr_out_db` exists
only under post v2, so scoring with it would make "turn v2 on" look like a
step change in the objective rather than a change in what we can hear. It is
used only as a stand-in when `snr_db.out` is missing altogether.

`talk_mod` is the modulation depth of whoever is talking. SNR alone will
happily reward a setting that makes a steady carrier louder; the syllabic
term keeps the search pointed at speech. It is dropped when the finder says
the candidate is CW or data, which have no syllabic content — that is what
the finder is for here. Its own numbers are averaged over minutes of history
and would smear across a three-second hold, so the search reads its VERDICT
and never its levels.

`passband.flatness` (|sum S| / sum |S|, from core/passband.py) is 1.0 when a
single weight fits the whole passband and falls when the combine is doing
something different in every bin. The gate publishes no residual spectral
flatness; this is the closest thing to it and it moves the right way.

The blanker penalty exists so the search cannot win by turning the noise
blanker into a gate — under 5% blanked is free, 30% costs the full 2 dB.

THE SCHEDULE. Every hold is 3 s: the tracker, the sub-band model and the AGC
all settle inside about 2.5 s, and the status numbers are running averages
over roughly that. A trial that is rejected costs two holds (candidate, then
back), one that is kept costs one, so the budget is planned at two:

    60 s  -> 8 trials    the first pass only: post, subband, mrc, nb,
                         width, contour (apf on CW), anf, auto_eq
    180 s -> 28 trials   a full cycle (16: one pass of best-guess values,
                         then every knob's remaining values) and most of a
                         second
    300 s -> 48 trials   three cycles

The cycle repeats on purpose rather than stopping: the chain a knob is
measured against has changed by the end of a pass, so one that lost early
can win late, and on a five-minute run the band itself has moved.

THE A/B RULE. The run opens with three measurements of the settings the
operator already had — that is both the baseline and the noise floor of the
measurement. The margin is the spread of those three, never less than
0.5 dB. A candidate is kept only when it beats the incumbent by the margin;
a tie goes to the operator's setting. Every rejected trial is put back and
re-measured, so the incumbent tracks the band's own drift rather than
holding a number from four minutes ago.
"""

HOLD_S = 3.0                  # settle before every read
SAMPLE_HOLDS = 3              # baseline reads before the search starts
MIN_MARGIN_DB = 0.5           # a candidate must beat the incumbent by this
TAIL_S = 3.0                  # slack kept back so a run ends on time

WEIGHTS = {"talk": 3.0, "flat": 2.0, "blank": 2.0}
TALK_FULL = 2.0               # talk_mod that earns the whole voice term
BLANK_FREE_PCT = 5.0          # blanking under this costs nothing
BLANK_SPAN_PCT = 25.0         # ... and this much more costs the whole penalty

VOICE_WIDTHS = ((300.0, 2700.0), (200.0, 2200.0), (400.0, 1800.0))
CW_WIDTHS = ((400.0, 900.0), (500.0, 750.0))
AGC_TRY = ("slow", "med", "fast")

# knob -> (target, set-kwarg builder); the runner dispatches on target
_DIVERSITY = "diversity"
_FILTER = "filter"


def _num(v, default=None):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if f == f and abs(f) != float("inf") else default


def objective(div, kind=None, weights=None):
    """One number for how good this sounds, from the status dicts alone.

    `div` is adapter.diversity_status(); `kind` is the finder's verdict on
    the candidate the operator is sitting on ("voice", "cw", ...) or None.
    Returns None when there is no SNR to score at all — the runner treats
    that as "the chain is not running" and stops.
    """
    w = dict(WEIGHTS)
    w.update(weights or {})
    div = div or {}
    snr = _num((div.get("snr_db") or {}).get("out"))
    if snr is None:
        snr = _num((div.get("post") or {}).get("snr_out_db"))
    if snr is None:
        return None
    j = snr
    if kind not in ("cw", "data") and div.get("talking"):
        j += w["talk"] * min(1.0, max(0.0, _num(div.get("talk_mod"), 0.0) / TALK_FULL))
    flat = _num(((div.get("passband") or {}).get("flatness")))
    if flat is not None:
        j += w["flat"] * min(1.0, max(0.0, flat))
    blanked = _num((div.get("nb") or {}).get("blanked_pct"), 0.0)
    if blanked > BLANK_FREE_PCT:
        j -= w["blank"] * min(1.0, (blanked - BLANK_FREE_PCT) / BLANK_SPAN_PCT)
    return round(j, 3)


def finder_kind(finder, hz, min_score=0.2):
    """What the finder thinks is on `hz` — its verdict only, never its levels.

    Returns None when nothing scores well enough there, which makes the
    objective fall back to its voice weighting (the common case on HF).
    """
    if not finder or hz is None:
        return None
    best, best_d = None, None
    for c in (finder.get("candidates") or []):
        chz, score = _num(c.get("hz")), _num(c.get("score"), 0.0)
        if chz is None or score < min_score:
            continue
        d = abs(chz - float(hz))
        if best_d is None or d < best_d:
            best, best_d = c, d
    if best is None or best_d > _num(best.get("width_hz"), 3000.0):
        return None
    return best.get("kind")


def set_kwargs(knob, value):
    """The **kwargs one knob's value becomes at set_diversity / filter_set."""
    if knob == "width":
        return {"low_hz": float(value[0]), "high_hz": float(value[1])}
    return {knob: value}


def _candidates(knob, cur, kind):
    """The values worth trying for one knob, best guess first."""
    if knob == "post":
        order = ("v2", True, False)
    elif knob == "nb":
        order = ("auto", True, False)
    elif knob == "agc":
        order = AGC_TRY
    elif knob == "width":
        order = CW_WIDTHS if kind == "cw" else VOICE_WIDTHS
        cur = tuple(float(x) for x in cur) if cur is not None else None
    elif knob == "nb_db":
        v = _num(cur, 12.0)
        order = (max(0.0, v - 4.0), min(40.0, v + 4.0))
    else:                                   # every on/off knob
        order = (not bool(cur),)
    return [v for v in order if v != cur]


def knob_order(kind):
    """Biggest lever first, and the shaping knobs that suit what is there."""
    if kind == "cw":
        return ("post", "subband", "mrc", "nb", "width", "apf", "anf",
                "agc", "contour", "nb_db")
    return ("post", "subband", "mrc", "nb", "width", "contour", "anf",
            "auto_eq", "agc", "nb_db")


def _target(knob):
    return _DIVERSITY if knob in ("post", "subband", "mrc", "nb", "nb_db") else _FILTER


def build_plan(snapshot, kind=None, trials=None):
    """The ordered trial list: one cycle is a pass of best-guess values
    (biggest lever first) followed by every knob's remaining values. The
    cycle repeats until `trials` is filled, so a long run keeps re-testing
    against a chain that has changed under it.

    Only knobs the snapshot actually carries are planned, so a runner that
    could not read a knob simply never tries it.
    """
    per = {}
    for knob in knob_order(kind):
        if knob not in snapshot:
            continue
        vals = _candidates(knob, snapshot.get(knob), kind)
        if vals:
            per[knob] = vals
    def trial(knob, value):
        return {"knob": knob, "target": _target(knob), "to": value,
                "kwargs": set_kwargs(knob, value)}
    base = ([trial(k, v[0]) for k, v in per.items()]
            + [trial(k, v) for k, vs in per.items() for v in vs[1:]])
    if not base or trials is None:
        return base
    out = []
    while len(out) < trials:
        out.extend(base)
    return out[:trials]


class DigSearch:
    """The search itself: hand out ops, take back numbers, keep a report.

    Nothing here knows what a radio is. `begin` takes the settings as they
    stand, every `next_op` is an instruction, every `feed` is one measured
    objective, and `report` is what the app draws.
    """

    def __init__(self, seconds, kind=None, hold_s=HOLD_S,
                 sample_holds=SAMPLE_HOLDS, min_margin_db=MIN_MARGIN_DB):
        self.seconds = float(seconds)
        self.kind = kind
        self.hold_s = float(hold_s)
        self.sample_holds = int(sample_holds)
        self.min_margin_db = float(min_margin_db)
        self.phase = "idle"
        self.started = self.ends = None
        self.steps = []
        self.snapshot = {}
        self.current = {}
        self.baseline = self.incumbent = None
        self.margin_db = self.min_margin_db
        self.trials_planned = 0
        self.trials_done = 0
        self._plan = []
        self._samples = []
        self._ops = []
        self._pending = None          # the measure we are waiting on

    # ----- driving it ---------------------------------------------------

    def begin(self, snapshot, now=0.0):
        snapshot = dict(snapshot)
        if snapshot.get("width") is not None:      # a list from JSON, a tuple here
            snapshot["width"] = tuple(float(x) for x in snapshot["width"])
        self.snapshot = dict(snapshot)
        self.current = dict(snapshot)
        self.started = float(now)
        self.ends = self.started + self.seconds
        room = self.seconds - self.sample_holds * self.hold_s - TAIL_S
        fits = max(0, int(room // (2.0 * self.hold_s)))
        self._plan = build_plan(self.snapshot, self.kind, fits)
        self.trials_planned = len(self._plan)
        self.phase = "sampling"
        self._ops = [{"op": "measure", "settle_s": self.hold_s, "why": "baseline"}
                     for _ in range(self.sample_holds)]

    def next_op(self, now):
        """The next instruction. Returns the same measure until it is fed."""
        if self._pending is not None:
            return dict(self._pending)
        if not self._ops:
            self._queue_trial(now)
        op = self._ops.pop(0)
        if op["op"] == "measure":
            self._pending = op
        elif op["op"] == "set":
            self.current[op["knob"]] = op["to"]
        return dict(op)

    def feed(self, value, now):
        """One measured objective for the measure we handed out."""
        op, self._pending = self._pending, None
        if op is None:
            raise RuntimeError("feed() with no measurement outstanding")
        if value is None:                    # chain stopped answering
            self.phase = "done"
            self._ops = [{"op": "done"}]
            return
        value = float(value)
        why = op.get("why")
        if why == "baseline":
            self._samples.append(value)
            if len(self._samples) >= self.sample_holds:
                s = sorted(self._samples)
                self.baseline = self.incumbent = s[len(s) // 2]
                self.margin_db = round(max(self.min_margin_db, s[-1] - s[0]), 3)
                self.phase = "searching"
        elif why == "revert":
            self.incumbent = value           # let the incumbent follow the band
        else:                                # a candidate
            self._judge(op, value, now)

    def _judge(self, op, value, now):
        delta = round(value - self.incumbent, 3)
        kept = delta >= self.margin_db
        self.trials_done += 1
        self.steps.append({"knob": op["knob"], "from": op["from"], "to": op["to"],
                           "delta_db": delta, "kept": bool(kept),
                           "at_s": round(now - self.started, 1)})
        if kept:
            self.incumbent = value
        else:
            self.current[op["knob"]] = op["from"]
            self._ops = [
                {"op": "set", "target": op["target"], "knob": op["knob"],
                 "from": op["to"], "to": op["from"],
                 "kwargs": set_kwargs(op["knob"], op["from"]), "revert": True},
                {"op": "measure", "settle_s": self.hold_s, "why": "revert"},
            ]

    def _queue_trial(self, now):
        while self._plan:
            t = self._plan.pop(0)
            cur = self.current.get(t["knob"])
            if cur == t["to"]:               # already there, nothing to learn
                continue
            if now + 2.0 * self.hold_s > self.ends:
                break
            self._ops = [
                {"op": "set", "target": t["target"], "knob": t["knob"],
                 "from": cur, "to": t["to"], "kwargs": t["kwargs"], "revert": False},
                {"op": "measure", "settle_s": self.hold_s, "why": "trial",
                 "knob": t["knob"], "from": cur, "to": t["to"], "target": t["target"]},
            ]
            return
        self.phase = "done"
        self._ops = [{"op": "done"}]

    # ----- what it found ------------------------------------------------

    @property
    def gain_db(self):
        if self.baseline is None or self.incumbent is None:
            return 0.0
        return round(self.incumbent - self.baseline, 2)

    def changed(self):
        """The knobs that ended up somewhere other than where they started."""
        return {k: v for k, v in self.current.items() if v != self.snapshot.get(k)}

    def report(self, now=None):
        elapsed = None
        if self.started is not None:
            elapsed = round((self.ends if now is None else now) - self.started, 1)
        return {
            "phase": self.phase,
            "seconds": self.seconds,
            "gain_db": self.gain_db,
            "steps": list(self.steps),
            "best": dict(self.current),
            "changed": self.changed(),
            "started": self.started,
            "ends": self.ends,
            "elapsed_s": elapsed,
            "objective_before": self.baseline,
            "objective_after": self.incumbent,
            "margin_db": self.margin_db,
            "trials_planned": self.trials_planned,
            "trials_done": self.trials_done,
            "kind": self.kind,
        }
