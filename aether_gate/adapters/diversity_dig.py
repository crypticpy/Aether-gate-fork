#
# Aether-gate — the "dig this out" runner: the search, on a thread, on the air.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Press the button on something weak and this spends 60 s, 3 min or 5 min
turning knobs and keeping what helped.

The strategy lives in core/digout.py and knows nothing about radios. This is
the part that touches one: it reads the settings as they stand, drives the
search on its own daemon thread, and puts everything back if the operator
says it got worse, if they cancel, or if anything throws.

It goes through the adapter's public surface and nothing else —
`diversity_status`, `filter_status`, `diversity_finder` to read;
`set_diversity` and `filter_set` to write, the same two calls the control
port already makes. It never touches the DSP thread and never blocks it.

One run at a time. When it finishes the settings are left where the search
left them and the run waits for a verdict:

    better  keep them. A focused talker's profile picks them up on its own
            (core/filter.py stores the live filter against the talker id),
            so there is nothing extra to write.
    keep    keep them, but the operator is not claiming it is better.
    worse   put every knob back where it was when the button was pressed.

The verdict is the label on the objective — the record `status()["record"]`
carries is one site-log line waiting for a home.
"""
import threading
import time

from ..core import digout

_SECONDS_ALLOWED = (60, 180, 300)
_VERDICTS = ("better", "worse", "keep")


def _dig(d, *path):
    """dict path or None, for status dicts whose sections may be missing."""
    for k in path:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def read_snapshot(div, filt):
    """Every knob the search may touch, as it stands right now.

    A section the adapter did not report is simply left out, and the search
    then never plans a trial for it.
    """
    s = {}
    post_on, post_ver = _dig(div, "post", "enabled"), _dig(div, "post", "version")
    if post_on is not None:
        s["post"] = "v2" if (post_on and post_ver == 2) else bool(post_on)
    for knob, section in (("subband", "subband"), ("mrc", "mrc")):
        v = _dig(div, section, "enabled")
        if v is not None:
            s[knob] = bool(v)
    nb_on, nb_mode = _dig(div, "nb", "enabled"), _dig(div, "nb", "auto", "mode")
    if nb_on is not None:
        s["nb"] = "auto" if nb_mode == "auto" else bool(nb_on)
    nb_db = _dig(div, "nb", "threshold_db")
    if nb_db is not None:
        s["nb_db"] = float(nb_db)
    lo, hi = _dig(filt, "low_hz"), _dig(filt, "high_hz")
    if lo is not None and hi is not None:
        s["width"] = (float(lo), float(hi))
    for knob in ("contour", "anf", "apf", "auto_eq"):
        v = _dig(filt, knob, "enabled")
        if v is not None:
            s[knob] = bool(v)
    agc = _dig(filt, "agc", "mode")
    if agc in digout.AGC_TRY:
        s["agc"] = agc
    return s


class DigRunner:
    """start / status / verdict / cancel, and the thread in between."""

    def __init__(self, adapter, clock=None, sleep=None, wall=None):
        self.a = adapter
        self._clock = clock or time.monotonic
        self._wall = wall or time.time
        self._lock = threading.Lock()
        self._cancel_ev = threading.Event()
        self._sleep = sleep or (lambda s: self._cancel_ev.wait(s))
        self._thread = None
        self._search = None
        self._snapshot = {}
        self._talker_id = None
        self._kind = None
        self._verdict = None
        self._record = None
        self._error = None
        self._cancelled = False
        self._t0_wall = None
        self._seconds = 0.0

    # ----- the operator's four buttons -----------------------------------

    def start(self, seconds, hz=None):
        seconds = int(seconds)
        if seconds not in _SECONDS_ALLOWED:
            raise ValueError(f"seconds must be one of {_SECONDS_ALLOWED}")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("a dig is already running")
            div = self.a.diversity_status()
            if not _dig(div, "available"):
                raise RuntimeError("no dual-tuner stream")
            filt = self.a.filter_status()
            self._snapshot = read_snapshot(div, filt if isinstance(filt, dict) else {})
            self._talker_id = _dig(div, "focus") or _dig(div, "talker", "id")
            self._kind = self._read_kind(hz)
            self._verdict = self._record = self._error = None
            self._cancelled = False
            self._seconds = float(seconds)
            self._t0_wall = self._wall()
            self._cancel_ev.clear()
            self._search = digout.DigSearch(seconds, kind=self._kind)
            self._search.begin(self._snapshot, self._clock())
            self._thread = threading.Thread(target=self._run, name="diversity-dig",
                                            daemon=True)
            self._thread.start()
        return self.status()

    def status(self):
        s = self._search
        if s is None:
            return {"available": True, "running": False, "phase": "idle",
                    "verdict": None, "record": None, "error": None,
                    "cancelled": False,
                    "gain_db": 0.0, "steps": [], "best": {}, "changed": {},
                    "started": None, "ends": None, "elapsed_s": None,
                    "seconds": 0.0, "objective_before": None,
                    "objective_after": None, "trials_planned": 0,
                    "trials_done": 0, "talker_id": None, "kind": None,
                    "snapshot": {}}
        out = s.report(self._clock())
        running = self._thread is not None and self._thread.is_alive()
        out["available"] = True
        out["running"] = bool(running)
        out["started"] = self._t0_wall
        out["ends"] = None if self._t0_wall is None else self._t0_wall + self._seconds
        out["remaining_s"] = max(0.0, round(self._seconds - (out["elapsed_s"] or 0.0), 1))
        out["verdict"] = self._verdict
        out["record"] = self._record
        out["error"] = self._error
        out["cancelled"] = self._cancelled
        out["talker_id"] = self._talker_id
        out["snapshot"] = dict(self._snapshot)
        return out

    def verdict(self, word):
        word = str(word or "").strip().lower()
        if word not in _VERDICTS:
            raise ValueError(f"verdict must be one of {_VERDICTS}")
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("the dig is still running")
            if self._search is None:
                raise RuntimeError("nothing to say better or worse about")
            if self._verdict is not None:
                raise RuntimeError(f"already called {self._verdict}")
            # the record is taken BEFORE the revert: it has to say what the
            # search measured, next to what the operator made of it
            self._record = self._make_record(word)
            if word == "worse":
                self._restore()
            self._verdict = word
        return self.status()

    def cancel(self):
        """Stop now and put everything back. Harmless when nothing is running."""
        self._cancel_ev.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=10.0)
        return self.status()

    # ----- the thread -----------------------------------------------------

    def _run(self):
        s = self._search
        try:
            while True:
                if self._cancel_ev.is_set():
                    s.phase = "done"
                    self._cancelled = True
                    self._restore()
                    return
                op = s.next_op(self._clock())
                if op["op"] == "done":
                    return
                if op["op"] == "set":
                    self._apply(op["target"], op["kwargs"])
                    continue
                self._sleep(op.get("settle_s") or 0.0)
                if self._cancel_ev.is_set():
                    continue                      # the top of the loop restores
                s.feed(self._measure(), self._clock())
        except Exception as e:                    # a driver that stopped answering
            self._error = f"{type(e).__name__}: {e}"
            s.phase = "done"
            self._restore()

    def _measure(self):
        return digout.objective(self.a.diversity_status(), self._kind)

    def _apply(self, target, kwargs):
        if target == "diversity":
            self.a.set_diversity(**kwargs)
        else:
            self.a.filter_set(**kwargs)

    def _restore(self):
        """Every snapshotted knob back where it was — best effort, both halves
        attempted even if one of them throws."""
        div_kw, filt_kw = {}, {}
        for knob, value in self._snapshot.items():
            kw = digout.set_kwargs(knob, value)
            (div_kw if knob in ("post", "subband", "mrc", "nb", "nb_db")
             else filt_kw).update(kw)
        for target, kw in (("diversity", div_kw), ("filter", filt_kw)):
            if not kw:
                continue
            try:
                self._apply(target, kw)
            except Exception as e:
                self._error = f"{self._error or ''} restore {target}: {e}".strip()
        if self._search is not None:
            # `changed` going empty is how the app knows the chain is back on
            # the operator's own settings; gain_db still says what was found
            self._search.current = dict(self._snapshot)

    def _read_kind(self, hz):
        """What the finder says is on this frequency — its verdict, once, at
        the start. Never its levels: they average over minutes."""
        if hz is None:
            hz = getattr(self.a, "_slice_hz", None)
        fn = getattr(self.a, "diversity_finder", None)
        if fn is None or hz is None:
            return None
        try:
            return digout.finder_kind(fn(), hz)
        except Exception:
            return None

    def _make_record(self, word):
        r = self._search.report(self._clock())
        return {"kind": "dig", "t": self._wall(), "gain_db": r["gain_db"],
                "verdict": word, "best": r["best"], "changed": r["changed"],
                "objective_before": r["objective_before"],
                "objective_after": r["objective_after"],
                "measured_best_db": r["measured_best_db"],
                "margin_db": r["margin_db"], "unsteady": r["unsteady"],
                "note": r["note"],
                "seconds": int(self._seconds), "talker_id": self._talker_id,
                "signal": self._kind, "steps": r["steps"]}
