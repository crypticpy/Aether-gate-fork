#
# Aether-gate — the alignment search: how many windows, held through what.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""AlignSearch: the state of one lag re-measurement, split out of
diversity_state.py so that module stays under the project's 800-line cap.

The decision this encodes (2026-09-03, an overflow on a quiet band): a
CREDIBLE re-measurement is adopted at once, whether or not its lag differs
from the one already in force -- a credible measurement is a measurement of
now. A NON-credible re-measurement changes nothing: the lock in force, if
there is one, is held through it (`_held_lag`), rather than being reset to
"not aligned" by a window that measured nothing. When ALIGN_TRIES windows
run out without a credible one, the search ends (no more windows are owed
right now) but is not abandoned: a retry is scheduled ALIGN_RETRY_S seconds
out and re-requested from `_DiversityState.ingest()` (the reader thread
already runs every block -- see `due_retry()`), and keeps retrying until a
credible window lands. A manual `request_realign()` always replaces any
pending retry and starts measuring now (`begin()` unconditionally clears it).

Pure logic: no numpy, no adapter, an injectable clock so tests do not sleep.
"""
import time

ALIGN_RETRY_S = 30.0        # how long a non-credible search waits before trying again


class AlignSearch:
    """One realign's try count, best window, held-lock state and retry
    clock. `_DiversityState` owns the `Aligner` itself and drives this class
    through `begin()` (a realign was requested), `step()` (one window
    measured), `due_retry()` (is a background retry owed), and the
    `held`/`note()`/`retry_seconds()` status accessors."""

    def __init__(self, tries, strong_peak, retry_s=ALIGN_RETRY_S, clock=time.monotonic):
        self.tries_max = int(tries)
        self.strong_peak = float(strong_peak)
        self.retry_s = float(retry_s)
        self._clock = clock
        self.try_count = 0
        self.best = None             # (lag, peak) of the best window this search
        self._why = None             # reason for the search in progress
        self._held_lag = None        # the lock in force when begin() ran, or None
        self._last_peak = 0.0        # most recent window's best-so-far peak (for the note)
        self._retry_at = None        # clock() deadline for a scheduled retry, or None
        self._retry_why = None

    # --- driven by request_realign() ---------------------------------------
    def begin(self, why, aligner):
        """A realign was requested -- the operator, a driver event, or a
        scheduled retry. Remembers the lock in force, if any, so a
        non-credible window can hold it, and cancels any pending retry: this
        search replaces it."""
        self.try_count = 0
        self.best = None
        self._why = str(why)
        self._last_peak = 0.0
        self._retry_at = None
        self._retry_why = None
        self._held_lag = int(aligner.lag) if aligner.aligned else None

    # --- driven by ingest() -------------------------------------------------
    def due_retry(self):
        """The reason to re-request, once a scheduled retry's deadline has
        passed; None otherwise. Left alone here -- begin() is what actually
        clears the schedule, once the caller calls request_realign() with it."""
        if self._retry_at is not None and self._clock() >= self._retry_at:
            return self._retry_why
        return None

    def step(self, aligner, lag, peak, min_peak):
        """One calibration window measured. Adopts it (aligner.set_lag) when
        credible; a non-credible window never touches the aligner -- the
        prior lock, if any, stays exactly as it was. When tries run out
        without a credible window, schedules a retry. Returns (ok, more,
        verdict) -- verdict is the tail of the caller's log line."""
        self.try_count += 1
        self._last_peak = peak
        improved = self.best is None or peak > self.best[1]
        if improved:
            self.best = (int(lag), float(peak))
        ok = self.best[1] >= min_peak
        if ok and improved:
            # Only a NEW best moves the aligner: set_lag restarts the delay
            # line, and a credible search that is still measuring on toward
            # the strong peak would otherwise glitch the stream every window.
            aligner.set_lag(self.best[0], self.best[1], True)
        if ok:
            self._held_lag = None
        more = self.best[1] < self.strong_peak and self.try_count < self.tries_max
        if not more and not ok:
            self._retry_at = self._clock() + self.retry_s
            self._retry_why = self._why
        if ok:
            verdict = "locked"
        elif self._held_lag is not None:
            verdict = f"NOT credible; holding lag {self._held_lag} from before"
        else:
            verdict = "NOT credible; holding lag 0"
        if more:
            verdict += f"; measuring on ({self.try_count}/{self.tries_max})"
        elif not ok:
            verdict += f"; retry in {self.retry_s:.0f} s"
        return ok, more, verdict

    # --- status() ------------------------------------------------------
    @property
    def held(self):
        """A lock is in force that predates the search in progress and has
        not yet had a credible window land -- False once one does, or when
        there was no lock to hold."""
        return self._held_lag is not None

    def retry_seconds(self):
        """Seconds until the scheduled background re-measure, or None."""
        if self._retry_at is None:
            return None
        return round(max(0.0, self._retry_at - self._clock()), 1)

    def note(self, min_peak):
        """The operator-readable line for status()'s align_note; "" when
        nothing is held and no retry is pending."""
        retry_s = self.retry_seconds()
        if self._held_lag is not None:
            base = f"held lag {self._held_lag} through {self._why}"
            if retry_s is not None:
                return (f"{base}; peak {self._last_peak:.1f}, need {min_peak:.0f}; "
                        f"next re-measure in {retry_s:.0f} s")
            return f"{base}; re-measuring (peak {self._last_peak:.1f}, need {min_peak:.0f})"
        if retry_s is not None:
            return (f"no lock yet (peak {self._last_peak:.1f}, need {min_peak:.0f}); "
                    f"next re-measure in {retry_s:.0f} s")
        return ""
