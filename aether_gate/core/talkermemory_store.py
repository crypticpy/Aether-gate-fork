#
# Aether-gate — the talker memory on disk (core/talkermemory.py's file).
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Reading and writing ~/.aether-gate/diversity-talkers.json.

Until G2 only the NAMES outlived a run (diversity-names.json): a talker heard
again after a restart got their label back, but the memory that made a recall
possible -- the steering vector, the weight it earned, which band it was earned
on -- started empty every time. What the station learned is capital (AGENTS.md),
so the whole entry is written, and the names file is read once to migrate.

The file:

    {"version": 1,
     "entries": [{"s": [re, im, ...], "m": [re, im], "hits": 3, "name": "Ann",
                  "voice": {...}|null, "band_hz": 14175000, "center_hz": 14250000.0,
                  "first_seen_wall": 1.7e9, "last_seen_wall": 1.7e9}, ...],
     "named":   [{"s": [...], "name": "Ann", "voice": {...}|null}, ...]}

IDS ARE NOT PERSISTED: they never reuse within a run and mean nothing between
them (core/talkermemory.py). Neither are the monotonic first_seen/last_seen --
a monotonic clock is uptime, so only the wall stamps cross a restart; the loader
turns them back into monotonic times against the clocks of the new run.

Every write is atomic (tmp + os.replace) and every read failure is empty, not an
exception: a memory file the operator can lose is a memory file, and a gate that
will not start because a JSON file is short one brace is not.
"""
import json
import os

import numpy as np

VERSION = 1
_ENTRY_SCALARS = ("hits", "name", "voice", "band_hz", "center_hz",
                  "first_seen_wall", "last_seen_wall")


def pack(s):
    """A complex vector as a flat list of floats, real and imaginary in turn."""
    return [float(x) for c in s for x in (c.real, c.imag)]


def unpack(v):
    """The inverse of pack, normalised. None when it is not a signature."""
    a = np.array(v, dtype=float)
    if len(a) < 4 or len(a) % 2:
        return None
    s = a[0::2] + 1j * a[1::2]
    n = float(np.linalg.norm(s))
    return None if n <= 0.0 else s / n


def _named_row(e):
    s = unpack(e.get("s"))
    if s is None or not e.get("name"):
        return None
    voice = e.get("voice")
    return {"s": s, "name": str(e["name"]),
            "voice": dict(voice) if isinstance(voice, dict) else None}


def load_names(path):
    """The old diversity-names.json: [{s, name, voice}]. Empty when there is
    nothing readable there -- this is the migration read, once."""
    raw = _read(path)
    out = []
    for e in raw if isinstance(raw, list) else []:
        row = _named_row(e) if isinstance(e, dict) else None
        if row is not None:
            out.append(row)
    return out


def load(path):
    """(entries, named) from the talkers file, or (None, None) when there is no
    readable file -- which is what tells the caller to migrate the names."""
    raw = _read(path)
    if not isinstance(raw, dict) or not isinstance(raw.get("entries"), list):
        return None, None
    entries = []
    for e in raw["entries"]:
        if not isinstance(e, dict):
            continue
        s = unpack(e.get("s"))
        m = e.get("m")
        if s is None or not isinstance(m, list) or len(m) != 2:
            continue
        row = {"s": s, "m": complex(float(m[0]), float(m[1])),
               "hits": int(e.get("hits") or 0), "name": e.get("name") or None,
               "voice": e["voice"] if isinstance(e.get("voice"), dict) else None,
               "band_hz": None if e.get("band_hz") is None else int(e["band_hz"]),
               "center_hz": _f(e.get("center_hz")),
               "first_seen_wall": _f(e.get("first_seen_wall")),
               "last_seen_wall": _f(e.get("last_seen_wall"))}
        entries.append(row)
    named = []
    for e in raw.get("named") or []:
        row = _named_row(e) if isinstance(e, dict) else None
        if row is not None:
            named.append(row)
    return entries, named


def save(path, entries, named):
    """Write the whole memory out atomically. False when the write failed."""
    doc = {"version": VERSION,
           "entries": [_entry_json(e) for e in entries],
           "named": [{"s": pack(n["s"]), "name": n["name"], "voice": n.get("voice")}
                     for n in named]}
    return _write(path, doc)


def _entry_json(e):
    out = {"s": pack(e["s"]), "m": [float(e["m"].real), float(e["m"].imag)]}
    for k in _ENTRY_SCALARS:
        out[k] = e.get(k)
    out["hits"] = int(out["hits"] or 0)
    return out


def _f(x):
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _read(path):
    if not path:
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write(path, doc):
    if not path:
        return False
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(doc, fh)
        os.replace(tmp, path)
    except (OSError, TypeError, ValueError):
        return False
    return True


def save_names(path, named):
    """The old names-only file. Still written by a memory built with a
    names_path and no store_path -- the pre-G2 constructor, and its tests."""
    return _write(path, [{"s": pack(n["s"]), "name": n["name"],
                          "voice": n.get("voice")} for n in named])
