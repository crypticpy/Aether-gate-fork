#
# Aether-gate — B25: the governor. What to reach for, in what order, and undo.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The gate already owns every tool. What it has never owned is the decision.

An operator with a squeeze, a blanker, a combiner null, a front-end guard and a
dig-out has five things to reach for and no measurement telling them which one
THIS noise wants. AUTO is that measurement and nothing else: it maps what the
profile FOUND onto the one tool that can do something about it, applies one move
at a time, scores it against the same speech-band objective the dig uses, and
PUTS IT BACK if the audio got worse. Each rule, and the reasoning behind its
thresholds, is written out at its own `_rule_*` method below.

Pure: no adapter, no clock of its own, no thread, no I/O. `tick(snap)` takes one
dict of readings and hands back at most one proposed action; the caller applies
it through the gate's own public setters and reports back with `applied()`. All
the policy remembers lives here, so a test drives the whole of it from written
snapshots and an explicit `t`, as B23's guard is in test_frontend_guard.py.

THE STACK ORDER is the chain's order, not a preference (chainstatus.py writes the
chain out from the antenna port forwards), so each stage acts on the residual of
the one before it: guard, nb, the combiner's null, ONE squeeze target, the dig.

THE UNDO. The objective before, the move, `settle_s`, the objective after. A move
that cost more than the margin is put back and its (kind, tool) pair sits in
BACKOFF_S of silence -- the band said no, and asking again in ten seconds is not
a new measurement. The margin is digout's own (half the spread of the recent
reads, MIN_MARGIN_DB..MARGIN_MAX_DB), for the reason core/digout.py sets out.

NON-COLLIDING: one pending action at a time, because two moves in flight cannot
be told apart by one objective; a tool the operator moved inside OPERATOR_GRACE_S
is not touched, and one the governor was HOLDING that changes under it is
RELEASED -- detected by DRIFT, so a write from any route releases it without that
route knowing this file exists; and a tool already where a rule wants it is left.

`auto` starts False and holds nothing. Turning it off releases everything on the
next tick and leaves the settings where they stand: releasing is not reverting,
and what the governor kept it kept because it measured better.
"""
import collections

from .digout import MARGIN_MAX_DB, MIN_MARGIN_DB

SETTLE_S = 2.0                  # after a move, before the objective is read again
OPERATOR_GRACE_S = 60.0         # a knob the operator moved is theirs for this long
BACKOFF_S = 300.0               # a (kind, tool) that hurt is not retried for this
DIG_BACKOFF_S = 600.0           # ...and the dig is expensive, so longer
MAX_EVENTS = 50
SPREAD_N = 6                    # objective reads the margin's spread is taken over

# Quoted from whichever module owns each; test_governor.py asserts they match.
NULL_COHERENCE = 0.5            # core.squeeze.NULL_ENTER_COHERENCE
NULLABLE_COHERENCE = 0.4        # _DiversityState.NULLABLE_COHERENCE
CARRIER_MIN_DB = 6.0            # core.squeeze.MIN_LEVEL_DB
HEADROOM_LOW_DB = 3.0           # adapters.frontend_guard.HEADROOM_LOW_DB
IMPULSE_RATE_ON = 1.0           # adapters.noise_kinds' own "is this impulsive"
IMPULSE_CLEAR_S = 30.0          # ...and how long it must be gone before unblanking
HUM_COVERED_DB = 6.0            # a null this deep already has the hum
WEAK_OBJECTIVE_DB = 6.0         # under this a talker is worth a dig
STEADY_SPREAD_DB = 1.5          # ...on a band no jumpier than this
DIG_SECONDS = 60

RULES = ("guard", "nb", "mode", "squeeze", "dig")     # the chain's own order
# "released"/"error": not measurement outcomes, but how a move stops being ours.
RESULTS = ("pending", "kept", "undone", "released", "error")
_HELD = ("tool", "params", "kind", "why", "since", "delta_db")   # a holding row


def _act(tool, params, undo, kind, why, revert=False):
    return {"tool": tool, "params": params, "undo": undo, "kind": kind,
            "why": why, "revert": bool(revert)}


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


class Governor:
    """The whole policy. See the module docstring; `tick` is the only way in."""

    def __init__(self, settle_s=SETTLE_S):
        self.auto = False
        self.settle_s = float(settle_s)
        self.state = "idle"             # idle|measuring|applying|settling|backoff
        self.pending = None
        self.holding = {}               # tool -> the move being held
        self.backoff = {}               # (kind, tool) -> deadline
        self.events = collections.deque(maxlen=MAX_EVENTS)
        self.why = "auto is off"
        self._obj = collections.deque(maxlen=SPREAD_N)
        self._seen = {}                 # tool -> last observed state
        self._mine_until = {}           # tool -> deadline for "this change is ours"
        self._operator_at = {}          # tool -> when the operator last moved it
        self._clear_since = None        # impulses last went quiet at
        self._seeded = False

    def tick(self, snap):
        """One snapshot in, zero or one actions out."""
        now = _num(snap.get("t"), 0.0)
        if not self.auto:
            self._release_all(now)
            self.state, self.why = "idle", "auto is off; nothing is held"
            return []
        if not snap.get("available"):
            self.state, self.why = "measuring", "no dual-tuner stream to govern"
            return []
        self._observe(snap, now)
        if snap.get("objective") is None:
            self.state, self.why = "measuring", "no combined SNR to score against yet"
            return []
        if self.pending is not None:
            return self._resolve(snap, now)
        return self._propose(snap, now)

    def applied(self, action, now, before=None):
        """The caller made the write land. A revert closes the books; anything
        else opens a pending that the next ticks will score."""
        now = _num(now, 0.0)
        tool = action["tool"]
        self._mine_until[tool] = now + self.settle_s + 1.0
        if action.get("revert"):
            self.state = "backoff"
            return
        if tool == "dig":
            self.backoff[(action["kind"], tool)] = now + DIG_BACKOFF_S
        # the same dict goes into events: result/delta update it in place
        self.pending = self._event(now, action, "pending", action["why"], before)
        self.state = "settling"

    def failed(self, action, error, now):
        """The write threw: nothing held, nothing pending, and the pair backs
        off -- a tool that will not take a write is not a measurement."""
        now = _num(now, 0.0)
        self.pending = None
        self.backoff[(action["kind"], action["tool"])] = now + BACKOFF_S
        self._event(now, action, "error", str(error))
        self.state, self.why = "idle", f"{action['tool']}: {error}"

    def _event(self, now, action, result, why, before=None):
        e = {"t": now, "tool": action["tool"], "kind": action["kind"],
             "params": action["params"], "undo": action.get("undo"), "why": why,
             "before": _num(before), "result": result, "delta_db": None}
        self.events.append(e)
        return e

    def _observe(self, snap, now):
        j = _num(snap.get("objective"))
        if j is not None:
            self._obj.append(j)
        state = tool_state(snap)
        if not self._seeded:
            self._seen, self._seeded = state, True   # as we found them: nobody's fault
            return
        for tool, cur in state.items():
            if cur == self._seen.get(tool):
                continue
            self._seen[tool] = cur
            if now < self._mine_until.get(tool, float("-inf")) or (
                    self.pending is not None and self.pending["tool"] == tool):
                continue                             # our own write landing
            self._operator_at[tool] = now
            held = self.holding.pop(tool, None)
            if held is not None:
                held.update(result="released",
                            why=f"the operator moved {tool}: released to them")
                self.events.append(dict(held, t=now))

    def margin_db(self):
        """digout's margin, on the governor's own recent reads."""
        return (round(min(MARGIN_MAX_DB, max(MIN_MARGIN_DB, self.spread_db() / 2.0)), 2)
                if len(self._obj) >= 2 else MIN_MARGIN_DB)

    def spread_db(self):
        return round(max(self._obj) - min(self._obj), 2) if len(self._obj) >= 2 else 0.0

    def _resolve(self, snap, now):
        p = self.pending
        if p["tool"] == "dig":
            return self._resolve_dig(snap, now)
        if now - p["t"] < self.settle_s:
            self.state, self.why = "settling", f"{p['tool']}: measuring what it did"
            return []
        before, after = p["before"], _num(snap.get("objective"))
        if before is None or after is None:
            return self._keep(p)                     # nothing to compare it against
        p["delta_db"] = round(after - before, 2)
        if p["delta_db"] >= -self.margin_db():
            return self._keep(p)
        p["result"] = "undone"
        self.pending = None
        self.holding.pop(p["tool"], None)
        self.backoff[(p["kind"], p["tool"])] = now + BACKOFF_S
        self.state = "backoff"
        self.why = f"{p['tool']} cost {-p['delta_db']:.1f} dB on the {p['kind']}: put back"
        if p["undo"] is None:
            return []
        return [_act(p["tool"], p["undo"], p["params"], p["kind"], self.why, revert=True)]

    def _resolve_dig(self, snap, now):
        """The dig owns its own A/B rule and its own revert, so this scores it
        and never reverts it: what the dig kept, the dig measured."""
        if snap.get("dig_running"):
            self.state, self.why = "settling", "dig-out is working on this talker"
            return []
        self.pending["delta_db"] = _num(snap.get("dig_gain_db"), 0.0)
        return self._keep(self.pending)

    def _keep(self, p):
        p["result"] = "kept"
        self.pending = None
        self.holding[p["tool"]] = dict(p, since=p["t"])
        self.state = "idle"
        d = p["delta_db"]
        self.why = (f"{p['tool']} on the {p['kind']}: kept"
                    + (f", {d:+.1f} dB" if d is not None else ""))
        return []

    def _release_all(self, now):
        for held in self.holding.values():
            held.update(result="released",
                        why="auto off: released, settings left where they stand")
            self.events.append(dict(held, t=now))
        self.holding.clear()
        self.pending, self._seeded, self._clear_since = None, False, None

    # ----- the rules, in the chain's order --------------------------------
    def _propose(self, snap, now):
        self.state = "measuring"
        for name in RULES:
            act = getattr(self, "_rule_" + name)(snap, now)
            if act is None:
                continue
            moved = self._operator_at.get(act["tool"])
            if moved is not None and now - moved < OPERATOR_GRACE_S:
                continue
            key = (act["kind"], act["tool"])
            until = self.backoff.get(key)
            if until is not None:
                if now < until:
                    continue
                del self.backoff[key]
            self.state, self.why = "applying", act["why"]
            return [act]
        self.why = (("holding " + ", ".join(sorted(self.holding))
                     + "; nothing else on the band is asking for a tool") if self.holding
                    else "nothing the band is doing has a tool that would help it")
        return []

    def _rule_guard(self, snap, now):
        """Front end, first: a clipped ADC is not a noise problem and no later
        stage undoes it. The guard steps the LNA itself from here."""
        hr = _num(snap.get("headroom_db"))
        if (snap.get("guard") or not snap.get("frontend_available")
                or hr is None or hr >= HEADROOM_LOW_DB):
            return None
        return _act("guard", {"guard": True}, {"guard": False}, "neighbour",
                    f"only {hr:.1f} dB of ADC headroom left: the front-end guard "
                    "takes the LNA down until it is clear")

    def _rule_nb(self, snap, now):
        """Impulses, before anything downstream measures a covariance off them.
        `nb=auto` means core/nbarm.py owns the knob, so stand aside; and the
        blanker only goes OFF again if the governor was the one that set it."""
        nb = snap.get("nb") or {}
        if nb.get("auto") == "auto":
            return None
        rate = _num(snap.get("impulses_per_s"), 0.0)
        if rate >= IMPULSE_RATE_ON:
            self._clear_since = None
            if nb.get("on"):
                return None
            db = blank_threshold_db(snap.get("impulse_db"))
            return _act("nb", {"nb": True, "nb_db": db},
                        {"nb": False, "nb_db": _num(nb.get("db"))}, "impulse",
                        f"{rate:g} impulses/s at {_num(snap.get('impulse_db'), 0.0):.0f} dB "
                        f"over the floor: blanker on at {db:g} dB")
        if "nb" not in self.holding or not nb.get("on"):
            self._clear_since = None
            return None
        if self._clear_since is None:
            self._clear_since = now
            return None
        if now - self._clear_since < IMPULSE_CLEAR_S:
            return None
        self._clear_since = None
        return _act("nb", {"nb": False}, {"nb": True, "nb_db": _num(nb.get("db"))},
                    "impulse",
                    f"the impulses have been gone {IMPULSE_CLEAR_S:.0f} s: blanker off")

    def _rule_mode(self, snap, now):
        """A directional noise floor is a spatial problem, so the combiner's own
        idle null is the tool -- but only from a mode nobody chose for a talker:
        focus and track are the operator's and are never overridden."""
        if snap.get("mode") not in ("off", "manual") or snap.get("focus"):
            return None
        coh = _num(snap.get("coherence"))
        if coh is None or coh < NULLABLE_COHERENCE:
            return None
        return _act("mode", {"mode": "null"}, {"mode": snap.get("mode")}, "floor",
                    f"the noise floor has a direction (coherence {coh:.2f}): "
                    "idle-nulling it")

    def _rule_squeeze(self, snap, now):
        """One target at a time, on whatever the null above left behind: the
        carrier the finder named, strongest first, standing CARRIER_MIN_DB over
        the floor (core/squeeze.py's own MIN_LEVEL_DB, so a target handed to it
        is one it will accept). Failing that a mains comb, but only when the loops
        agree on a direction (NULLABLE_COHERENCE) and never when a null already held
        here is deeper than HUM_COVERED_DB -- notching a tone the null already took
        out is the collision this exists to avoid. WHICH tool is squeeze.py's."""
        sq = snap.get("squeeze") or {}
        if sq.get("configured") and "squeeze" not in self.holding:
            return None                  # the operator's own target: leave it alone
        coh = _num(snap.get("coherence"))
        tool = "a null" if coh is not None and coh >= NULL_COHERENCE else "a notch"
        car = self._strongest_carrier(snap)
        if car is not None:
            hz = int(round(car["hz"]))
            if sq.get("target") == "signal" and abs(_num(sq.get("hz"), 1e9) - hz) < 1.0:
                return None
            return _act("squeeze", {"squeeze": hz}, {"squeeze": ""}, "carrier",
                        f"a carrier {hz:+d} Hz in the passband at {car['db']:.0f} dB: "
                        f"squeezing it with {tool}")
        if not snap.get("mains_hz") or int(_num(snap.get("harmonics"), 0)) < 1:
            return None
        if sq.get("held") and sq.get("tool") == "null" \
                and _num(sq.get("depth_db"), 0.0) >= HUM_COVERED_DB:
            return None
        if coh is None or coh < NULLABLE_COHERENCE:
            return None                  # hum with no direction: not ours to reach
        if sq.get("target") == "comb" and sq.get("configured"):
            return None
        return _act("squeeze", {"squeeze": "comb"}, {"squeeze": ""}, "mains",
                    f"a {int(_num(snap.get('mains_hz'), 0.0))} Hz mains comb the loops "
                    f"agree on (coherence {coh:.2f}): squeezing the comb with {tool}")

    def _rule_dig(self, snap, now):
        """Last: the dig turns the same knobs and already scores every one, so this
        hands over rather than duplicating the search. It wants a full ring of reads
        (SPREAD_N) on a band no jumpier than STEADY_SPREAD_DB -- a run on a moving
        band concludes nothing (see core/digout.py)."""
        if snap.get("dig_running") or not snap.get("talking"):
            return None
        j = _num(snap.get("objective"))
        if j is None or j >= WEAK_OBJECTIVE_DB:
            return None
        if len(self._obj) < SPREAD_N or self.spread_db() > STEADY_SPREAD_DB:
            return None
        return _act("dig", {"seconds": DIG_SECONDS}, None, "weak",
                    f"a talker at {j:.1f} dB on a band steady to "
                    f"{self.spread_db():.1f} dB: handing it to dig-out")

    @staticmethod
    def _strongest_carrier(snap):
        """The loudest carrier the finder named that is worth a tool at all."""
        best = None
        for c in (snap.get("carriers") or []):
            hz, db = _num(c.get("hz")), _num(c.get("db"), 0.0)
            if hz is None or db < CARRIER_MIN_DB:
                continue
            if best is None or db > best["db"]:
                best = {"hz": hz, "db": db}
        return best

    def status(self):
        return {
            "auto": bool(self.auto), "state": self.state, "why": self.why,
            "settle_s": self.settle_s, "margin_db": self.margin_db(),
            "spread_db": self.spread_db(),
            "holding": [{k: h[k] for k in _HELD}
                        for h in sorted(self.holding.values(), key=lambda h: h["since"])],
            "pending": None if self.pending is None else dict(self.pending),
            "events": [dict(e) for e in self.events],
            "backoff": [{"kind": k, "tool": t, "until": u}
                        for (k, t), u in sorted(self.backoff.items())],
        }
