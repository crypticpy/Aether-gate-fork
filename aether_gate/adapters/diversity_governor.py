#
# Aether-gate — B25: the governor's side of the radio (core/governor.py).
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The tick, the readings, and the writes. The POLICY is in core/governor.py
and never sees an adapter; this is the part that touches one.

Same split and the same reach as adapters/diversity_dig.py, deliberately: the
adapter's public surface and nothing else -- `diversity_status`,
`filter_status`, `diversity_finder`, `diversity_spatial`, `frontend_status`,
`diversity_dig` to read; `set_diversity`, `frontend.enabled` and
`diversity_dig(seconds=)` / `diversity_dig(cancel=True)` to write, the same
calls the control port already makes. Nothing here touches the DSP thread, and
nothing here reaches into _DiversityState (which is at its own 800-line budget)
past the public `set()` that `set_diversity` forwards to.

ONE DAEMON THREAD, started by `auto=on` and stopped by `auto=off`, ticking once
a second. It is not the read loop's: the read loop's job is to keep the stream
fed, and a status read plus a set_diversity in it would be a second of DSP work
a second. `tick()` is public and takes an explicit `now`, so a test drives the
whole runner with no thread at all.

THE SNAPSHOT is where the gate's shapes are turned into the flat dict the
policy reads, and it is the only place in B25 that knows a status key by name.
Three of its fields deserve a note:

  objective   core.digout.objective on the live diversity status, with the
              finder's verdict for the tuned frequency -- the same number, from
              the same function, the dig scores its own trials with. The
              governor's undo therefore means exactly what the dig's revert
              means.
  carriers    the finder's candidates it called "carrier", as SIGNED offsets
              from the slice centre (which is what core/squeeze.py's target is)
              and filtered to the ones actually inside the filter's passband.
              A carrier outside the passband is not audible and not the
              governor's business, however loud it is.
  floor_db    /diversity/spatial's level strip, median over the passband
              (core.governor_proxy.inband_floor_db): a level with no talker.
  dig_age_s   seconds since the last dig ended. A DURATION and not a stamp: the
              dig times itself off the wall clock and the policy off a monotonic
              one, so only the gap between the two survives the crossing.
"""
import threading
import time

from ..core import digout, governor
from ..core import governor_proxy as proxy

TICK_S = 1.0
JOIN_TIMEOUT_S = 5.0


def _safe(fn, *a, **kw):
    """A status read that never takes the tick down with it."""
    if fn is None:
        return {}
    try:
        out = fn(*a, **kw)
    except Exception:
        return {}
    return out if isinstance(out, dict) else {}


def _signed_edges(filt):
    """The passband as signed offsets from the carrier, low first. /filter
    reports the magnitudes and the sideband sign separately."""
    lo, hi = filt.get("low_hz"), filt.get("high_hz")
    if lo is None or hi is None:
        return None
    sgn = -1.0 if float(filt.get("_sign", 1) or 1) < 0 else 1.0
    a, b = sgn * float(lo), sgn * float(hi)
    return (a, b) if a <= b else (b, a)


def carriers(finder, filt, slice_hz):
    """Every candidate the finder called a carrier that is inside the passband,
    as {"hz": signed offset, "db": SNR over the floor}."""
    edges = _signed_edges(filt)
    if edges is None or slice_hz is None:
        return []
    lo, hi = edges
    out = []
    for c in (finder.get("candidates") or []):
        if c.get("kind") != "carrier":
            continue
        hz = governor._num(c.get("hz"))
        if hz is None:
            continue
        off = hz - float(slice_hz)
        if lo <= off <= hi:
            out.append({"hz": off, "db": governor._num(c.get("snr_db"), 0.0)})
    return out


class GovernorRunner:
    """auto on/off, the tick, and the writes. `status()` is what the app reads."""

    def __init__(self, adapter, clock=None):
        self.a = adapter
        self.gov = governor.Governor()
        self._clock = clock or time.monotonic
        self._stop = threading.Event()
        self._thread = None
        self._error = None

    # ----- the operator's one switch --------------------------------------

    def set_auto(self, on):
        on = bool(on)
        if on == self.gov.auto:
            return self.status()
        self.gov.auto = on
        if on:
            self._error = None
            self.start()
        else:
            self.stop()
        return self.status()

    def start(self):
        """The tick thread. Idempotent: a second call while one runs is a no-op."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="diversity-governor",
                                        daemon=True)
        self._thread.start()

    def stop(self):
        """Stop ticking and let the policy release what it holds. The SETTINGS
        are left exactly where they stand -- see core/governor.py: releasing is
        not reverting."""
        self._stop.set()
        t = self._thread
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=JOIN_TIMEOUT_S)
        self._thread = None
        ours = (self.gov.pending or {}).get("tool") == "dig"
        self.gov.auto = False
        self.gov.tick({"t": self._clock(), "available": False})
        if ours:            # a dig WE started keeps turning knobs otherwise
            _safe(getattr(self.a, "diversity_dig", None), cancel=True)

    # ----- the tick --------------------------------------------------------

    def _run(self):
        while not self._stop.wait(TICK_S):
            try:
                self.tick()
            except Exception as e:                  # a driver that stopped answering
                self._error = f"{type(e).__name__}: {e}"

    def tick(self, now=None):
        """One pass: read, decide, write. Public and thread-free on purpose."""
        now = self._clock() if now is None else float(now)
        snap = self.snapshot(now)
        for act in self.gov.tick(snap):
            self._apply(act, snap, now)
        return self.gov.status()

    def _apply(self, act, snap, now):
        try:
            self._write(act["tool"], act["params"])
        except Exception as e:
            self.gov.failed(act, f"{type(e).__name__}: {e}", now)
            return
        self.gov.applied(act, now, before=snap.get("objective"))

    def _write(self, tool, params):
        """Every write the governor can make, through the adapter's own public
        setters -- the same three calls the control port makes."""
        if tool == "guard":
            fe = getattr(self.a, "frontend", None)
            if fe is None:
                raise RuntimeError("adapter has no front-end guard")
            fe.enabled = bool(params["guard"])
        elif tool == "dig":
            self.a.diversity_dig(seconds=int(params["seconds"]))
        else:                                        # squeeze, nb, mode
            self.a.set_diversity(**params)

    # ----- the readings ----------------------------------------------------

    def snapshot(self, now=None):
        """Every number the policy reads, from the status dicts alone."""
        now = self._clock() if now is None else float(now)
        div = _safe(getattr(self.a, "diversity_status", None))
        if not div.get("available"):
            return {"t": now, "available": False}
        filt = _safe(getattr(self.a, "filter_status", None))
        fe = _safe(getattr(self.a, "frontend_status", None))
        dig = _safe(getattr(self.a, "diversity_dig", None))
        spat = _safe(getattr(self.a, "diversity_spatial", None))
        ends = governor._num(dig.get("ends"))        # wall clock: only the gap travels
        prof = div.get("noise_profile") or {}
        nb = div.get("nb") or {}
        sq = div.get("squeeze") or {}
        focus, talker = div.get("focus"), div.get("talker")
        slice_hz = getattr(self.a, "_slice_hz", None)
        return {
            "t": now, "available": True,
            "objective": digout.objective(div, self._kind(div, slice_hz)),
            "mode": div.get("mode"), "coherence": div.get("noise_coherence"),
            "focus": (focus or {}).get("id") if isinstance(focus, dict) else None,
            "talking": bool(div.get("talking")),
            "squeeze": {
                "held": bool(sq.get("held")), "tool": sq.get("tool"),
                "depth_db": sq.get("depth_db"), "target": sq.get("target"),
                "hz": sq.get("hz"),
                # core.squeeze.Squeeze.active, from the status keys: a target is
                # CONFIGURED whether or not it is currently held
                "configured": bool(sq.get("hz") is not None
                                   or (sq.get("target") == "comb"
                                       and sq.get("since") is not None)),
            },
            "nb": {"on": bool(nb.get("enabled")), "db": nb.get("threshold_db"),
                   "auto": (nb.get("auto") or {}).get("mode")},
            "impulses_per_s": prof.get("impulses_per_s") or 0.0,
            "impulse_db": prof.get("impulse_db"), "mains_hz": prof.get("mains_hz"),
            "harmonics": prof.get("harmonics") or 0, "blanked_pct": nb.get("blanked_pct"),
            "carriers": carriers(_safe(getattr(self.a, "diversity_finder", None)),
                                 filt, slice_hz),
            "frontend_available": bool(fe.get("available")), "guard": bool(fe.get("guard")),
            "headroom_db": fe.get("headroom_db"), "clips_1s": fe.get("clips_1s"),
            "slice_hz": slice_hz, "floor_db": proxy.inband_floor_db(spat),
            "talker": talker.get("id") if isinstance(talker, dict) else None,
            "dig_running": bool(dig.get("running")), "dig_note": dig.get("note"),
            "dig_gain_db": dig.get("gain_db"), "dig_verdict": dig.get("verdict"),
            "dig_unsteady": bool(dig.get("unsteady")),
            "dig_cancelled": bool(dig.get("cancelled")),
            "dig_age_s": None if ends is None else time.time() - ends,
        }

    def _kind(self, div, slice_hz):
        """What the finder says is on this frequency -- its verdict only, the
        same read adapters/diversity_dig.py makes for the same objective."""
        fn = getattr(self.a, "diversity_finder", None)
        if fn is None or slice_hz is None:
            return None
        try:
            return digout.finder_kind(fn(), slice_hz)
        except Exception:
            return None

    def status(self):
        out = self.gov.status()
        out["available"] = True
        out["error"] = self._error
        out["running"] = bool(self._thread is not None and self._thread.is_alive())
        return out
