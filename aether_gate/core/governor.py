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

NO TALKER IS NOT NO DECISION. With no speech-band objective the move is still
made and still scored, by the PROXY its tool carries its own measurement for
(core/governor_proxy.py): the squeeze's depth, the blanker's blanked_pct, the
passband floor, the ADC's clips. Every event and holding row names its `scorer`
("snr" or "proxy:<name>") and the status carries `objective_source`, so nothing
is kept without saying what kept it; a proxy-kept move is NOT re-litigated when
a talker turns up. The dig is the exception: it needs a talker and is never
started without one. The guard gets its own GUARD_SETTLE_S: it steps the LNA.

ONCE PER TALKER. A dig hand-off costs a minute of knob-turning, so it happens
once for a (slice frequency, talker) pair and never again, and not at all for
DIG_BACKOFF_S after a run the dig itself called tentative: that concluded nothing.
"""
import collections

from . import governor_proxy as proxy
from .digout import MARGIN_MAX_DB, MIN_MARGIN_DB
from .governor_proxy import (CARRIER_MIN_DB, HEADROOM_LOW_DB, HUM_MIN_DB,
                             IMPULSE_RATE_ON, NULLABLE_COHERENCE, NULL_COHERENCE,
                             blank_threshold_db, tool_state)

_num = proxy._num                       # the adapter reads it through this name

SETTLE_S = 2.0                  # after a move, before the objective is read again
GUARD_SETTLE_S = 10.0           # ...but the guard steps the LNA on its own loop
SQUEEZE_SETTLE_S = 8.0          # ...and a squeeze takes hold over blocks: the comb
                                # detector alone wants ~2 s and retries; a proxy-
                                # scored squeeze is read as soon as it is held,
                                # and put back if it never is by this
OPERATOR_GRACE_S = 60.0         # a knob the operator moved is theirs for this long
BACKOFF_S = 300.0               # a (kind, tool) that hurt is not retried for this
DIG_BACKOFF_S = 1800.0          # ...and a dig is a minute of knob-turning: half
                                # an hour, after eight in eighty minutes on air
MAX_EVENTS = 50
SPREAD_N = 6                    # objective reads the margin's spread is taken over

IMPULSE_CLEAR_S = 30.0          # how long the impulses must be gone before unblanking
HUM_COVERED_DB = 6.0            # a null this deep already has the hum
WEAK_OBJECTIVE_DB = 6.0         # under this a talker is worth a dig
STEADY_SPREAD_DB = 1.5          # ...on a band no jumpier than this
DIG_SECONDS = 60

RULES = ("guard", "nb", "mode", "squeeze", "dig")     # the chain's own order
# WHICH LOOP A TOOL BELONGS TO (G4). The guard and the blanker are answers to
# this ADDRESS -- the ADC's headroom and the neighbour's impulses are the same
# on every band -- so a band change leaves them, and their backoffs, alone. A
# null on the floor and a squeeze on a carrier were measured against a signal
# that is not in the window any more, so a band change puts them back. The dig
# holds nothing, but its (kind, tool) backoff is NOT keyed by frequency: live it
# held DIG OUT off for half an hour across a band change while the operator
# waited, so that goes too. Only `_dug` survives, because that IS keyed by
# frequency and talker (governor_proxy.dig_key).
SITE_TOOLS = ("guard", "nb")
SPAN_TOOLS = ("mode", "squeeze")
SPAN_WORDS = {"mode": "the null", "squeeze": "the squeeze"}
# "released"/"error": not measurement outcomes, but how a move stops being ours.
RESULTS = ("pending", "kept", "undone", "released", "error")
_HELD = ("tool", "params", "kind", "why", "since", "delta_db", "scorer")  # a row


def _act(tool, params, undo, kind, why, revert=False, label=None):
    """`why` is the sentence (events, the AUTO CLEAN card, hover); `label` is
    the few plain words a switch shows while this move is the pending one."""
    return {"tool": tool, "params": params, "undo": undo, "kind": kind,
            "why": why, "revert": bool(revert), "scorer": "snr",
            "label": label or f"trying {proxy.WORDS.get(tool, tool)}"}


def _word(tool):
    return proxy.WORDS.get(tool, tool)


class Governor:
    """The whole policy. See the module docstring; `tick` is the only way in."""

    def __init__(self, settle_s=SETTLE_S):
        self.auto = False
        self.settle_s = float(settle_s)
        self.state = "idle"             # idle|measuring|applying|settling|backoff
        self.label = "off"              # state_label: the switch's few plain words
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
        self._source = "snr"            # is there an objective to score by?
        self._before = {}               # the proxy readings the pending is judged from
        self._dug = set()               # (slice hz, talker) pairs already handed over
        self._band = None               # the band the holdings were earned on
        self._band_seen = False         # ...once a snapshot has said which
        self._out = []                  # what the last pass RULED OUT, in chain order
        self._wall_off = None           # monotonic -> wall clock, from the snapshot
        self._note_at = None            # (the dig's last note, when we first saw it)
        self._pending_label = ""        # the pending move's own label, while it settles

    def tick(self, snap):
        """One snapshot in, zero or one actions out."""
        now = _num(snap.get("t"), 0.0)
        wall = _num(snap.get("wall"))
        if wall is not None:
            self._wall_off = wall - now
        if not self.auto:
            self._release_all(now)
            self.state, self.why, self.label = "idle", "off: nothing is held", "off"
            return []
        if not snap.get("available"):
            self.state, self.why = "measuring", "waiting for both tuners to stream"
            self.label = "waiting for the stream"
            return []
        self._observe(snap, now)
        self._source = "snr" if _num(snap.get("objective")) is not None else "none"
        moved = self._band_moved(snap, now)
        if moved:
            return moved
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
            self.state, self.label = "backoff", "put back"
            return
        if tool == "dig":
            self.backoff[(action["kind"], tool)] = now + DIG_BACKOFF_S
            self._dug.add(action.get("key"))     # this pair has now had its minute
        # the same dict goes into events: result/delta update it in place
        self.pending = self._event(now, action, "pending", action["why"], before)
        self._pending_label = action.get("label") or f"trying {_word(tool)}"
        self.state, self.label = "settling", self._pending_label

    def failed(self, action, error, now):
        """The write threw: nothing held, nothing pending, and the pair backs
        off -- a tool that will not take a write is not a measurement."""
        now = _num(now, 0.0)
        self.pending = None
        self.backoff[(action["kind"], action["tool"])] = now + BACKOFF_S
        self._event(now, action, "error", str(error))
        self.state, self.why = "idle", f"{_word(action['tool'])} refused the move: {error}"
        self.label = "failed"

    def _event(self, now, action, result, why, before=None):
        e = {"t": now, "wall": self._wall_at(now),
             "tool": action["tool"], "kind": action["kind"],
             "params": action["params"], "undo": action.get("undo"), "why": why,
             "before": _num(before), "result": result, "delta_db": None,
             "scorer": action.get("scorer", "snr")}
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
                self.events.append(dict(held, t=now, wall=self._wall_at(now)))

    def _wall_at(self, t):
        """R9: one monotonic stamp as a wall clock. `t` and `until` are uptime
        and mean nothing to a person -- the app was rendering an event from
        22:20 as 11:23 -- so every event carries `wall` and every backoff row
        `until_wall` beside them. The offset comes from the snapshot; with no
        snapshot to take it from (a pure unit test) the answer is None."""
        off = self._wall_off
        return None if off is None or t is None else round(t + off, 3)

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
        settle = GUARD_SETTLE_S if p["tool"] == "guard" else self.settle_s
        waiting = (p["tool"] == "squeeze" and p["scorer"] != "snr"
                   and not (snap.get("squeeze") or {}).get("held"))
        if waiting:
            settle = SQUEEZE_SETTLE_S            # held: scored; not yet: wait for it
        if now - p["t"] < settle:
            self.state, self.why = "settling", ("waiting for the squeeze to take hold"
                                                if waiting else
                                                f"measuring what {_word(p['tool'])} did")
            self.label = self._pending_label
            return []
        if p["scorer"] != "snr":                     # no talker: its own proxy scores it
            keep, why = proxy.verdict(p["tool"], p["kind"], self._before, snap)
            p["why"] = f"{p['why']}; {why}"
            return self._keep(p, why) if keep else self._undo(p, now, why)
        before, after = p["before"], _num(snap.get("objective"))
        if before is None or after is None:
            return self._keep(p)                     # nothing to compare it against
        p["delta_db"] = round(after - before, 2)
        if p["delta_db"] >= -self.margin_db():
            return self._keep(p)
        return self._undo(p, now, f"put it back: {_word(p['tool'])} cost the talker "
                                  f"{-p['delta_db']:.1f} dB")

    def _undo(self, p, now, why):
        p["result"] = "undone"
        self.pending = None
        self.holding.pop(p["tool"], None)
        self.backoff[(p["kind"], p["tool"])] = now + BACKOFF_S
        self.state, self.why, self.label = "backoff", why, "put back"
        if p["undo"] is None:
            return []
        return [_act(p["tool"], p["undo"], p["params"], p["kind"], why, revert=True)]

    def _resolve_dig(self, snap, now):
        """The dig owns its own A/B rule and its own revert, so this scores it
        and never reverts it -- but it banks only what the dig itself stands
        behind, and a run that concluded nothing backs the pair off again."""
        if snap.get("dig_running"):
            self.state, self.why = "settling", "DIG OUT is working on the talker"
            self.label = "DIG OUT running"
            return []
        p = self.pending
        p["delta_db"], note = proxy.dig_delta_db(snap)
        if note:
            p["why"] = f"{p['why']}; {note}"
            self.backoff[(p["kind"], "dig")] = now + DIG_BACKOFF_S
        return self._keep(p)

    def _keep(self, p, why=None):
        p["result"] = "kept"
        self.pending = None
        self.holding[p["tool"]] = dict(p, since=p["t"])
        self.state, self.label = "idle", "kept"
        d = p["delta_db"]
        self.why = why or (f"kept {_word(p['tool'])}"
                           + (f": {d:+.1f} dB on the talker" if d is not None else ""))
        return []

    def _band_moved(self, snap, now):
        """The dial changed band: put the span's tools back, keep the site's.

        A step lower in the station's order never resets a step above it
        (AGENTS.md, "Keep what the station learned"): the SITE keeps the guard
        and the blanker, and the SPAN gives back the null and the squeeze --
        PUT BACK, not released, because what kept them was a measurement of a
        signal that is no longer in the window. Their (kind, tool) backoffs go
        with them, and so does the DIG's: five minutes of silence about a
        squeeze on 20 m says nothing about 40 m, and half an hour of it about
        the dig is half an hour of the operator waiting on the new band. So
        does the objective ring, which is one band's spread. `_dug` stays: it
        is keyed by frequency and talker, so it says nothing about this band.

        A centre move inside the band changes nothing at all.
        """
        band = snap.get("band_hz")
        if not self._band_seen:
            self._band, self._band_seen = band, True     # as we found it
            return []
        if band == self._band:
            return []
        self._band = band
        # only the SITE's backoffs survive: the span's and the dig's were
        # measured on a band that is not in the window any more
        self.backoff = {k: u for k, u in self.backoff.items() if k[1] in SITE_TOOLS}
        self._obj.clear()
        acts = []
        p = self.pending
        if p is not None and p["tool"] in SPAN_TOOLS:
            self.pending = None          # it was being scored against the old band
            acts += self._put_back(p, now)
        for tool in SPAN_TOOLS:
            held = self.holding.pop(tool, None)
            if held is not None:
                acts += self._put_back(held, now)
        if acts:
            self.state, self.label = "backoff", "put back"
            self.why = ("the band changed: the span's tools are back where they were, "
                        "the site's own are untouched")
        return acts

    def _put_back(self, row, now):
        """One span tool, back where the band change found it. Nothing backs
        off: the band did not say no to this tool, the operator moved away."""
        why = f"put back {SPAN_WORDS.get(row['tool'], _word(row['tool']))}: band changed"
        self.events.append(dict(row, t=now, wall=self._wall_at(now),
                                result="undone", why=why, delta_db=None))
        if row.get("undo") is None:
            return []
        return [_act(row["tool"], row["undo"], row["params"], row["kind"], why,
                     revert=True, label="putting it back")]

    def _release_all(self, now):
        for held in self.holding.values():
            held.update(result="released",
                        why="auto off: released, settings left where they stand")
            self.events.append(dict(held, t=now, wall=self._wall_at(now)))
        self.holding.clear()
        self.pending, self._seeded, self._clear_since = None, False, None

    # ----- the rules, in the chain's order --------------------------------
    def _no(self, tool, why):
        """R7: one rule's rejection, in the operator's words. "Nothing on the
        band needs a tool right now" is true and says nothing; what a person
        wants back from a measurement that proposed no move is WHAT IT RULED
        OUT and on what number. Kept short: the app shows the joined line in
        one row and elides it, and the list is in `ruled_out` beside it."""
        self._out.append({"tool": tool, "why": why})
        return None

    def _why_idle(self):
        """R7/R8: the ruled-out line, and never a repeat of the held list --
        `state_label` already carries that, and the CHAIN banner was reading
        "holding what DIG OUT found . holding what DIG OUT found; nothing
        else...". While holding, this is the REMAINDER only."""
        if self._out:
            return " \u00b7 ".join(o["why"] for o in self._out)
        return ("nothing else on the band needs a tool" if self.holding
                else "nothing on the band needs a tool right now")

    def _propose(self, snap, now):
        self.state, self.label = "measuring", "listening"
        self._out = []
        for name in RULES:
            act = getattr(self, "_rule_" + name)(snap, now)
            if act is None:
                continue
            moved = self._operator_at.get(act["tool"])
            if moved is not None and now - moved < OPERATOR_GRACE_S:
                self._no(act["tool"], f"{_word(act['tool'])} is the operator's for "
                                      f"{OPERATOR_GRACE_S - (now - moved):.0f}s more")
                continue
            key = (act["kind"], act["tool"])
            until = self.backoff.get(key)
            if until is not None:
                if now < until:
                    self._no(act["tool"], f"{_word(act['tool'])} is backing off for "
                                          f"{until - now:.0f}s more")
                    continue
                del self.backoff[key]
            self.state, self.why, self.label = "applying", act["why"], act["label"]
            if self._source != "snr":
                act["scorer"] = proxy.scorer(act["tool"])
            self._before = proxy.readings(snap)
            return [act]
        if self.holding:
            held = ", ".join(proxy.HELD_WORDS.get(t, t) for t in sorted(self.holding))
            self.label = f"holding {held}"
        self.why = self._why_idle()
        return []

    def _rule_guard(self, snap, now):
        """Front end, first: a clipped ADC is not a noise problem and no later
        stage undoes it. The guard steps the LNA itself from here."""
        hr = _num(snap.get("headroom_db"))
        if snap.get("guard"):
            return self._no("guard", "the guard is already on")
        if not snap.get("frontend_available") or hr is None:
            return None                  # no front end here: nothing to say
        if hr >= HEADROOM_LOW_DB:
            return self._no("guard", f"{hr:.0f} dB of ADC headroom, no clipping")
        return _act("guard", {"guard": True}, {"guard": False}, "neighbour",
                    f"only {hr:.1f} dB of ADC headroom left: the front-end guard "
                    "takes the LNA down until it is clear",
                    label="trying the front-end guard")

    def _rule_nb(self, snap, now):
        """Impulses, before anything downstream measures a covariance off them.
        `nb=auto` means core/nbarm.py owns the knob, so stand aside; and the
        blanker only goes OFF again if the governor was the one that set it."""
        nb = snap.get("nb") or {}
        if nb.get("auto") == "auto":
            return self._no("nb", "the blanker is on auto")
        rate = _num(snap.get("impulses_per_s"), 0.0)
        if rate >= IMPULSE_RATE_ON:
            self._clear_since = None
            if nb.get("on"):
                return self._no("nb", f"blanker already on ({rate:g}/s)")
            db = blank_threshold_db(snap.get("impulse_db"))
            return _act("nb", {"nb": True, "nb_db": db},
                        {"nb": False, "nb_db": _num(nb.get("db"))}, "impulse",
                        f"{rate:g} impulses/s at {_num(snap.get('impulse_db'), 0.0):.0f} dB "
                        f"over the floor: blanker on at {db:g} dB",
                        label="trying the blanker")
        if "nb" not in self.holding or not nb.get("on"):
            self._clear_since = None
            return self._no("nb", f"{rate:g} impulses/s: nothing to blank")
        if self._clear_since is None:
            self._clear_since = now
            return None
        if now - self._clear_since < IMPULSE_CLEAR_S:
            return self._no("nb", f"impulses gone {now - self._clear_since:.0f}s of "
                                  f"{IMPULSE_CLEAR_S:.0f}")
        self._clear_since = None
        return _act("nb", {"nb": False}, {"nb": True, "nb_db": _num(nb.get("db"))},
                    "impulse",
                    f"the impulses have been gone {IMPULSE_CLEAR_S:.0f} s: blanker off",
                    label="taking the blanker off")

    def _rule_mode(self, snap, now):
        """A directional noise floor is a spatial problem, so the combiner's own
        idle null is the tool -- but only from a mode nobody chose for a talker:
        focus and track are the operator's and are never overridden."""
        if snap.get("focus"):
            return self._no("mode", "the combiner is on the operator's talker")
        if snap.get("mode") not in ("off", "manual"):
            return self._no("mode", f"the combiner is the operator's ({snap.get('mode')})")
        coh = _num(snap.get("coherence"))
        if coh is None:
            return None                  # no reading yet: not a rejection
        if coh < NULLABLE_COHERENCE:
            return self._no("mode", f"the floor is not directional (coh {coh:.2f})")
        return _act("mode", {"mode": "null"}, {"mode": snap.get("mode")}, "floor",
                    f"the noise floor comes from one direction (coherence {coh:.2f}): "
                    "nulling it", label="trying a null on the floor")

    def _rule_squeeze(self, snap, now):
        """One target at a time, on whatever the null above left behind: the
        carrier the finder named, strongest first, standing CARRIER_MIN_DB over
        the floor (core/squeeze.py's own MIN_LEVEL_DB, so a target handed to it
        is one it will accept). Failing that a mains comb, standing HUM_MIN_DB
        over the floor -- core/noiseprofile.py's own LINE_MIN_DB, the bar its
        harmonic count is already taken at -- and never when a null already held
        here is deeper than HUM_COVERED_DB, because notching a tone the null
        already took out is the collision this exists to avoid.

        THE COMB DOES NOT NEED A DIRECTION (R6). A comb notch is a spectral
        tool: it takes out 50 or 60 Hz and its harmonics wherever they came
        from. The null does need one, and keeps NULLABLE_COHERENCE. Live: a
        60 Hz comb standing 11.8 dB over the floor at coherence 0.02 -- mains
        wiring all round a house is not one bearing -- and AUTO CLEAN said
        nothing was needed while the hum was plainly there. WHICH tool it ends
        up using is core/squeeze.py's own call, on its own hysteresis."""
        sq = snap.get("squeeze") or {}
        if sq.get("configured") and "squeeze" not in self.holding:
            return self._no("squeeze", "the squeeze target is the operator's")
        coh = _num(snap.get("coherence"))
        tool = "a null" if coh is not None and coh >= NULL_COHERENCE else "a notch"
        car = proxy.strongest_carrier(snap)
        if car is not None:
            hz = int(round(car["hz"]))
            if sq.get("target") == "signal" and abs(_num(sq.get("hz"), 1e9) - hz) < 1.0:
                return self._no("squeeze", f"already squeezing {hz:+d} Hz")
            return _act("squeeze", {"squeeze": hz}, {"squeeze": ""}, "carrier",
                        f"a carrier {hz:+d} Hz in the passband at {car['db']:.0f} dB: "
                        f"squeezing it with {tool}", label=f"trying {tool} on a carrier")
        hum = _num(snap.get("hum_db"))
        if not snap.get("mains_hz") or int(_num(snap.get("harmonics"), 0)) < 1:
            return self._no("squeeze", "no carrier and no mains comb over the floor")
        if hum is not None and hum < HUM_MIN_DB:
            return self._no("squeeze", f"hum {hum:.1f} dB, under {HUM_MIN_DB:.0f}")
        depth = _num(sq.get("depth_db"), 0.0)
        if sq.get("held") and sq.get("tool") == "null" and depth >= HUM_COVERED_DB:
            return self._no("squeeze", f"the null already has the hum ({depth:.0f} dB)")
        if sq.get("target") == "comb" and sq.get("configured"):
            return self._no("squeeze", "already squeezing the mains comb")
        level = (f"{hum:.1f} dB over the floor" if hum is not None
                 else f"{int(_num(snap.get('harmonics'), 0))} harmonics")
        return _act("squeeze", {"squeeze": "comb"}, {"squeeze": ""}, "mains",
                    f"a {int(_num(snap.get('mains_hz'), 0.0))} Hz mains comb at "
                    f"{level}: squeezing the comb with {tool}",
                    label=f"trying {tool} on the mains hum")

    def _rule_dig(self, snap, now):
        """Last: the dig turns the same knobs and already scores every one, so this
        hands over rather than duplicating the search. It needs a TALKER -- there is
        no proxy for it, the dig's own A/B is the objective -- a full ring of reads
        (SPREAD_N) on a band no jumpier than STEADY_SPREAD_DB, and a pair this
        governor has not already spent a minute on (proxy.dig_key)."""
        if not snap.get("talking"):
            return self._no("dig", "no talker to dig for")
        j = _num(snap.get("objective"))
        if j is None:
            return None                  # no objective: the dig has no answer to give
        if j >= WEAK_OBJECTIVE_DB:
            return self._no("dig", f"talker at {j:.1f} dB: strong enough")
        if len(self._obj) < SPREAD_N:
            return self._no("dig", f"talker {j:.1f} dB: waiting for steady")
        if self.spread_db() > STEADY_SPREAD_DB:
            return self._no("dig", f"talker {j:.1f} dB: the band moves "
                                   f"{self.spread_db():.1f} dB")
        key = proxy.dig_key(snap)
        age, self._note_at = proxy.tentative_age_s(snap, self._note_at, now)
        if key is None:
            return None
        if key in self._dug:
            return self._no("dig", "DIG OUT has already had this talker")
        if proxy.dig_blocked(snap, age, DIG_BACKOFF_S):
            return self._no("dig", "DIG OUT concluded nothing here just now")
        return dict(_act("dig", {"seconds": DIG_SECONDS}, None, "weak",
                         f"a talker at {j:.1f} dB on a band steady to "
                         f"{self.spread_db():.1f} dB: handing it to DIG OUT",
                         label="handing the talker to DIG OUT"), key=key)

    def status(self):
        return {
            "auto": bool(self.auto), "state": self.state, "why": self.why,
            "state_label": self.label,
            "settle_s": self.settle_s, "margin_db": self.margin_db(),
            "spread_db": self.spread_db(), "objective_source": self._source,
            "holding": [{k: h[k] for k in _HELD}
                        for h in sorted(self.holding.values(), key=lambda h: h["since"])],
            "pending": None if self.pending is None else dict(self.pending),
            "events": [dict(e) for e in self.events],
            "ruled_out": [dict(o) for o in self._out],
            "backoff": [{"kind": k, "tool": t, "until": u,
                         "until_wall": self._wall_at(u)}
                        for (k, t), u in sorted(self.backoff.items())],
        }
