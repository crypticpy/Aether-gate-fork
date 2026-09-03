#
# Aether-gate — the raw two-channel capture: the aligned pair, to disk.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""A diagnostic recording of the aligned raw pair, so a night's scene can be
replayed through the lab (tests, replay.py) without the antenna. Started
from the control port, collected on the reader thread, written on its own
thread: 60 s at 2 MS/s is ~1 GB of complex64."""
import os
import threading
import time


class RawCapture:
    def __init__(self):
        self._lock = threading.Lock()       # start() lands from the HTTP thread
        self._run = None
        self.last_path = None

    @property
    def active(self):
        return self._run is not None

    def start(self, directory, seconds, samp_rate, center_hz, slice_hz=None, slice_mode=None):
        """Begin collecting `seconds`; returns the path the .npz will appear at."""
        seconds = float(seconds)
        with self._lock:
            if self._run is not None:
                raise RuntimeError("a capture is already running")
            d = os.path.expanduser(directory)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, time.strftime("%Y%m%d-%H%M%S")
                                + f"_{int(center_hz)}Hz_{int(samp_rate)}sps.npz")
            self._run = {"want": int(seconds * samp_rate), "n": 0, "a": [], "b": [],
                         "path": path, "seconds": seconds,
                         "slice_hz": slice_hz, "slice_mode": slice_mode}
        return path

    def ingest(self, a, b, np, meta):
        """One aligned block pair. `meta()` is asked for the run's metadata
        (rate, centre, lag, aligned) once the last block is in."""
        with self._lock:
            c = self._run
            if c is None:
                return
            c["a"].append(a.copy()); c["b"].append(b.copy()); c["n"] += len(a)
            if c["n"] < c["want"]:
                return
            self._run = None
        m = dict(meta())
        m["seconds"] = c["seconds"]
        if c.get("slice_hz") is not None:
            m["slice_hz"] = float(c["slice_hz"])          # what the operator was listening to
            m["slice_mode"] = str(c.get("slice_mode") or "")

        def _write():
            np.savez(c["path"], a=np.concatenate(c["a"])[:c["want"]],
                     b=np.concatenate(c["b"])[:c["want"]], **m)
            self.last_path = c["path"]
            print(f"[diversity] capture written: {c['path']}", flush=True)
        threading.Thread(target=_write, name="diversity-capture", daemon=True).start()

    def status(self):
        c = self._run
        return {"active": c is not None, "path": c["path"] if c is not None else self.last_path}
