#
# Aether-gate — the noise profile arms the blanker, so the operator does not have to.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""NbArm turns NoiseProfile.status() into a blanker on/off/threshold decision.

core/noiseprofile.py only DESCRIBES the impulses it sees each second:
impulses_per_s (a smoothed rate) and impulse_db (their median excess over
the block's own median power, EMA-smoothed — see its IMPULSE_TC_S). The
blanker itself (core/diversity.blank_impulses, wired through
adapters/diversity_state.py's nb_on / nb_db) knows nothing about the
profile; today an operator reads the numbers and flips nb by hand, or
clicks the BLANK suggestion adapters/noise_kinds.py already computes
(impulse_db - 3 dB, clamped to 6..30 dB) for that one over.

NbArm closes that loop for an operator who would rather set it and
forget it. "auto" mode watches the same status() dict every profile
period and arms/disarms the blanker itself, using the same 3 dB margin
noise_kinds.py already recommends, so the auto threshold and the manual
suggestion agree. "on"/"off" are the operator's own manual call and
always win outright — auto only decides anything while the mode is
"auto".

This module is a pure policy object: no I/O, no threads, no numpy. It is
handed a status() dict and a monotonic-ish time once a period and returns
a Decision; the caller (adapters/diversity_state.py) is the one that
actually pokes nb_on / nb_db on the real blanker.

Hysteresis, so a noise floor that dances around ARM_RATE does not
chatter the blanker on and off every second:

  * arm once impulses_per_s stays at or above ARM_RATE for ARM_HOLD_S
    straight (see ARM_RATE below for why 3/s and not lower);
  * disarm once it drops below the lower DISARM_RATE and stays there
    for DISARM_HOLD_S — a slower reaction going quiet than going noisy,
    so one dropout in a burst does not unblank mid-burst;
  * once armed, the threshold tracks impulse_db - MARGIN_DB, clamped to
    the blanker's own valid range (diversity_state.py's nb_db: 0..40 dB
    over the block's median power — the same units blank_impulses()
    takes), and only actually moves when the change exceeds
    THRESH_STEP_DB, so a wobble of a few tenths of a dB in the EMA does
    not retune the blanker every second.

A mains-locked comb never arms it on its own: hum is periodic, not
impulsive, and never shows up in impulses_per_s (see noiseprofile.py's
_analyse — the comb and the impulse count come from separate passes over
the block), so the arm rate simply never crosses ARM_RATE from hum alone.
"""
from dataclasses import dataclass

MODES = ("auto", "on", "off")

ARM_RATE = 3.0        # impulses/s to start arming. noiseprofile.py's own
                       # docstring says Gaussian noise alone crosses its
                       # IMPULSE_DB gate "a few times a second" by chance
                       # (chi-square, 4 dof) -- so 1/s is not evidence of
                       # anything impulsive. 3/s sits comfortably above
                       # that chance rate without waiting for a source
                       # loud enough to be obvious by ear first.
ARM_HOLD_S = 3.0       # straight seconds at/above ARM_RATE before arming
DISARM_RATE = 1.0      # impulses/s to start disarming: below ARM_RATE,
                       # so a rate hovering between the two never chatters
DISARM_HOLD_S = 20.0   # straight seconds below DISARM_RATE before disarming
MARGIN_DB = 3.0        # matches noise_kinds.py's own BLANK suggestion
                       # (impulse_db - 3 dB), so auto and manual agree
THRESH_STEP_DB = 2.0   # minimum move before the threshold is retuned
NB_DB_MIN = 0.0        # diversity_state.py's set(nb_db=...) valid range
NB_DB_MAX = 40.0       # (also blank_impulses()'s threshold_db range)
_ROUND_DB = 0.5        # noise_kinds.py's own rounding grid for a threshold


@dataclass
class Decision:
    nb_on: bool
    threshold: "float | None"   # the blanker's own dB-over-median units; None when off
    reason: str
    changed: bool                # nb_on differs from the previous update()'s decision


class NbArm:
    """Auto-arm policy for the pair's noise blanker. No I/O: update() is
    pure given (status, t); the caller applies the Decision to the real
    nb_on / nb_db."""

    def __init__(self, mode="auto"):
        self.mode = None
        self._armed = False
        self._threshold = None
        self._above_since = None    # t the rate first reached ARM_RATE, unbroken
        self._below_since = None    # t the rate first fell under DISARM_RATE, unbroken
        self._stopped_at = None     # t impulses were last seen active, once disarmed
        self._prev_on = False       # last Decision.nb_on, for `changed`
        self._changed_t = None      # t of the last nb_on flip (any mode)
        self._last_t = None         # t of the most recent update(), for status()'s since_s
        self._reason = "no impulses: blanker off"
        self.set_mode(mode)

    # --- control port -----------------------------------------------------
    def set_mode(self, mode):
        mode = str(mode).lower()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "auto" and self.mode != "auto":
            # a clean re-entry: no stale hold timer from before the excursion
            # into manual decides anything about the world right now
            self._armed = False
            self._above_since = None
            self._below_since = None
            self._stopped_at = None
        self.mode = mode

    # --- the profile period -------------------------------------------------
    def update(self, status, t):
        """status: NoiseProfile.status(). t: seconds, any monotonic clock the
        caller already has (the profile period works fine)."""
        t = float(t)
        if self.mode == "off":
            decision = self._settle(False, None, "manual: off", t)
        elif self.mode == "on":
            decision = self._settle(True, None, "manual: on", t)
        else:
            decision = self._auto(status, t)
        self._last_t = t
        return decision

    def _auto(self, status, t):
        rate = float(status["impulses_per_s"])
        imp_db = status["impulse_db"]
        if not self._armed:
            self._below_since = None
            if rate >= ARM_RATE:
                if self._above_since is None:
                    self._above_since = t
                elapsed = t - self._above_since
                if imp_db is not None and elapsed >= ARM_HOLD_S:
                    self._armed = True
                    self._stopped_at = None
                else:
                    reason = f"{rate:g} impulses/s for {elapsed:.0f} s: arming (needs {ARM_HOLD_S:g} s)"
                    return self._settle(False, None, reason, t)
            else:
                self._above_since = None
        else:
            self._above_since = None
            if rate < DISARM_RATE:
                if self._below_since is None:
                    self._below_since = t
                elif t - self._below_since >= DISARM_HOLD_S:
                    self._armed = False
                    self._stopped_at = self._below_since   # when they actually stopped
            else:
                self._below_since = None

        if self._armed:
            if imp_db is not None:
                target = max(NB_DB_MIN, min(NB_DB_MAX, imp_db - MARGIN_DB))
                target = round(target / _ROUND_DB) * _ROUND_DB
                if self._threshold is None or abs(target - self._threshold) > THRESH_STEP_DB:
                    self._threshold = target
            db_part = f" at {imp_db:g} dB" if imp_db is not None else ""
            reason = f"{rate:g} impulses/s{db_part}: blanker on, threshold {self._threshold:g} dB"
            return self._settle(True, self._threshold, reason, t)

        if self._stopped_at is not None:
            reason = f"impulses stopped {t - self._stopped_at:.0f} s ago: blanker off"
        elif rate > 0:
            reason = f"{rate:g} impulses/s: below the arm rate"
        else:
            reason = "no impulses: blanker off"
        return self._settle(False, None, reason, t)

    def _settle(self, on, threshold, reason, t):
        changed = on != self._prev_on
        if changed:
            self._changed_t = t
        self._prev_on = on
        self._reason = reason
        return Decision(nb_on=on, threshold=threshold, reason=reason, changed=changed)

    # --- control port status ------------------------------------------------
    def status(self):
        """JSON-ready: what the control port shows without re-running update()."""
        since = (self._last_t - self._changed_t
                 if self._last_t is not None and self._changed_t is not None else 0.0)
        return {
            "mode": self.mode,
            "armed": bool(self._armed),
            "threshold": self._threshold if self._armed else None,
            "reason": self._reason,
            "since_s": round(float(since), 1),
        }
