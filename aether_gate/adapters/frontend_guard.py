#
# Aether-gate — B23: the front-end linearity guard.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Yaesu's VC-Tune and Icom's Digi-Sel are analogue preselectors: their real
job is keeping the front end LINEAR when a strong signal sits tens of kHz
from a weak one, not keeping the S-meter pretty. An RSPduo has no tracking
preselector — the 200 kHz IF roofing filter already protects the ADC from
anything outside that window, so the piece that is actually missing is
automatic gain headroom: back the LNA state off before the ADC clips, and
give it back once the strong signal is gone.

This module is the decision only, same split as core/digout.py: it takes
readings (headroom in dB, a clip count, the LNA state the device reports, a
clock) and hands back a state to move to or nothing. It never touches Soapy —
no `import SoapySDR` anywhere in this file, so it can be driven by a fake
clock in a test with no device, no thread, no numpy even. adapters/soapy.py
owns the numpy (peak-per-block, the ~1 s hold, the clip counter) and the one
`writeSetting("rfgain_sel", ...)` this class's answer turns into.

RSPduo rfgain_sel is a STRING enum 0..9, and on this device HIGHER means MORE
attenuation (opposite of what "gain" suggests) — Soapy exposes it as
`options`, not a numeric `range` (checked live against the running gate,
2026-09-03: `/device`'s rfgain_sel entry carries `"options": ["0".."9"]` and
no `range` key at all, unlike `agc_setpoint` which is a real ranged control).
So `max_state` here is a string this module never interprets past `int()`,
and the caller is the one that has to derive it from whichever ArgInfo shape
the driver actually gave — see soapy.py's `_open_hw`.

THE POLICY, spelled out because the thresholds are policy, not physics:

  step UP (more attenuation) when headroom < HEADROOM_LOW_DB (3 dB) or any
  sample clipped, for LOW_TICKS_REQUIRED (2) consecutive ticks. One step,
  never past max_state.

  step DOWN (back toward the operator's floor) when headroom has stayed
  above HEADROOM_CLEAR_DB (15 dB) for CLEAR_S (30) seconds STRAIGHT — any
  tick that drops back under 15 dB resets the clock, so a signal that comes
  and goes cannot accumulate credit across the gaps. One step, never past
  floor_state.

  HOLD_S (2 s) after every step, up or down: nothing moves. This is the
  hysteresis — without it, the 15 dB/3 dB bands would let a signal sitting
  near a threshold walk the state back and forth every tick.

`enabled` starts False. The operator turns it on because the dBm caveat is
real: moving rfgain_sel is a real front-end gain change of tens of dB that
Soapy's own getGain() reports with the wrong sign AND the wrong magnitude
(soapy.py's `_setting_to` handling has the swept numbers), so every step this
guard takes away from `floor_state` is a step away from wherever the dBm
scale was last trimmed. soapy.py surfaces that as `dbm_calibrated: false`;
this module only supplies the LNA state that comparison is made against.
"""
import collections

HEADROOM_LOW_DB = 3.0        # step up below this, or on any clip
HEADROOM_CLEAR_DB = 15.0     # step down once headroom has held above this
LOW_TICKS_REQUIRED = 2       # consecutive low ticks before stepping up
CLEAR_S = 30.0                # seconds headroom must hold above HEADROOM_CLEAR_DB
HOLD_S = 2.0                  # cooldown after any step; nothing moves
MAX_EVENTS = 20


class FrontEndGuard:
    """Pure state machine. See the module docstring for the policy.

    `tick()` is the only way in: hand it the current reading and the clock,
    get back the new rfgain_sel state to write (a string) or None if nothing
    should change. Call it as often as you like — the hysteresis lives in
    `t`, the caller's clock, not in call count, so a caller that ticks once a
    second and a test that ticks once a millisecond see the same policy.
    """

    def __init__(self, floor_state="0", max_state=None, enabled=False):
        self.enabled = bool(enabled)
        self.floor_state = str(floor_state)
        self.max_state = str(max_state) if max_state is not None else None
        self.lna_state = str(floor_state)   # the device's last-reported state
        self.state = "idle"                 # idle|stepping_up|holding|stepping_down
        self.hold_until = None              # monotonic deadline, or None
        self.events = collections.deque(maxlen=MAX_EVENTS)
        self._low_ticks = 0                 # consecutive ticks under HEADROOM_LOW_DB (or clipping)
        self._clear_since = None            # monotonic stamp headroom last WENT above HEADROOM_CLEAR_DB

    def tick(self, t, headroom_db, clip_count, lna_state):
        """One reading in, at most one rfgain_sel write out (as a string).

        `lna_state` is always trusted over our own memory of it — it is what
        the device actually reported after the last write landed (soapy.py
        reads every setting back; see the "setters lie" note there), so a
        write that got clamped or ignored by the driver is reflected here on
        the very next tick rather than silently assumed to have taken.
        """
        self.lna_state = str(lna_state)
        if not self.enabled:
            # Disabled: report the measurement (state stays "idle") but never
            # touch the counters, so turning it on later does not inherit a
            # stale streak from while nobody was watching.
            self.state = "idle"
            return None

        low = headroom_db < HEADROOM_LOW_DB or clip_count > 0
        if low:
            self._low_ticks += 1
            self._clear_since = None
        else:
            self._low_ticks = 0
            if headroom_db > HEADROOM_CLEAR_DB:
                if self._clear_since is None:
                    self._clear_since = t
            else:
                self._clear_since = None

        if self.hold_until is not None and t < self.hold_until:
            self.state = "holding"
            return None

        try:
            cur = int(self.lna_state)
        except (TypeError, ValueError):
            self.state = "idle"             # a non-numeric state: nothing we can reason about
            return None

        if low and self._low_ticks >= LOW_TICKS_REQUIRED:
            top = int(self.max_state) if self.max_state is not None else cur
            if cur < top:
                reason = "clipping" if clip_count > 0 else f"headroom {headroom_db:.1f} dB"
                return self._step(t, cur, cur + 1, "stepping_up", reason, headroom_db)
            self.state = "idle"             # already at the ceiling — nothing left to do
            return None

        if self._clear_since is not None and (t - self._clear_since) >= CLEAR_S:
            floor = int(self.floor_state)
            if cur > floor:
                new_state = self._step(t, cur, cur - 1, "stepping_down",
                                        "clear for 30 s", headroom_db)
                # Needs another full CLEAR_S before the NEXT step down — a
                # single accumulated streak must not cash out one state per
                # tick once it crosses 30 s.
                self._clear_since = t
                return new_state
            self.state = "idle"             # already at the operator's floor
            return None

        self.state = "idle"
        return None

    def _step(self, t, frm, to, state, reason, headroom_db):
        self.hold_until = t + HOLD_S
        self.lna_state = str(to)
        self.state = state
        self._low_ticks = 0
        self.events.append({"t": t, "from": str(frm), "to": str(to),
                             "reason": reason, "headroom_db": round(float(headroom_db), 1)})
        return str(to)

    def snapshot(self):
        """The guard-owned slice of the /frontend status JSON — soapy.py's
        frontend_status() merges this with the measurement fields it owns
        (headroom_db, peak_dbfs, per_channel, ...)."""
        return {
            "guard": self.enabled,
            "floor_state": self.floor_state,
            "max_state": self.max_state,
            "lna_state": self.lna_state,
            "state": self.state,
            "hold_until": self.hold_until,
            "events": list(self.events),
        }
