#
# Aether-gate — two-element diversity combining for a coherent dual-tuner SDR.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Spatial signatures of recent talkers and the weight that suited each.

Split out of diversity.py, which the class used to live in directly: see
that module for the combiner and Tracker that use TalkerMemory. The class
itself has no dependency on the maths there beyond weight_to_polar, imported
lazily in status() to avoid a circular import (diversity.py imports
TalkerMemory from here).
"""
import json
import os

import numpy as np

from .focus import StationFocus

# A talker's spatial signature is recognised again when the squared cosine
# between its steering vector and the remembered one is at least this:
# 0.75 is a 60 degree phase tolerance for equal-level antennas. Live on a
# two-station QSO the fitted phase of one station scatters by 30-50
# degrees between overs (signal coherence 0.7-0.9, phase drifting ~20
# degrees in ten seconds), and 0.9 (36 degrees) filled all eight slots
# with one station. Stations in a QSO sit 100+ degrees apart.
MEMORY_MATCH = 0.75
# A busy HF net runs 10-20 check-ins; 8 forced a named regular out to make
# room for the next one-off caller every time a net went past eight voices.
# An entry is a couple of complex numbers, a handful of scalars and (once
# heard) a small voiceprint summary, so doubling the slot count costs
# nothing that matters -- recall/store is a linear scan of at most 16
# entries, done once per refit, against per-block work that is already an
# FFT or an eigendecomposition.
MEMORY_MAX = 16
# On a match the remembered signature moves towards the new one by this
# fraction, so a slowly drifting bearing is followed rather than re-added.
MEMORY_MERGE = 0.3


class TalkerMemory:
    """Spatial signatures of recent talkers and the weight that suited each.

    When someone keys up, one block is enough to tell whether they are a
    known voice (a known bearing, really); if so the weight jumps straight
    to what worked last time instead of being re-learned over a refit cycle.
    """

    def __init__(self, max_n=MEMORY_MAX, match=MEMORY_MATCH, names_path=None):
        self.max_n = int(max_n)
        self.match = float(match)
        self.entries = []                    # dicts: id, s, m, hits, first_seen, last_seen, name
        self._next_id = 1                    # ids never reuse within a run
        self.active = None                   # id of the talker whose weight is live
        self.active_since = None
        self.focus = StationFocus()          # see focus.py
        # Names outlive a run: they are keyed to the steering vector, not the
        # id, so a talker heard again after a restart gets their label back
        # as soon as their signature is stored.
        self.names_path = names_path
        self._named = self._load_names()

    def _load_names(self):
        if not self.names_path:
            return []
        try:
            with open(self.names_path) as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            return []
        out = []
        for e in raw if isinstance(raw, list) else []:
            try:
                v = np.array(e["s"], dtype=float)
                s = v[0::2] + 1j * v[1::2]
                if len(s) >= 2 and e.get("name"):
                    voice = e.get("voice")
                    out.append({"s": s / max(np.linalg.norm(s), 1e-12), "name": str(e["name"]),
                                "voice": dict(voice) if isinstance(voice, dict) else None})
            except (KeyError, TypeError, ValueError):
                continue
        return out

    def _save_names(self):
        if not self.names_path:
            return
        raw = [{"s": [float(x) for c in e["s"] for x in (c.real, c.imag)], "name": e["name"],
                "voice": e.get("voice")}
               for e in self._named]
        try:
            os.makedirs(os.path.dirname(self.names_path) or ".", exist_ok=True)
            tmp = self.names_path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(raw, fh)
            os.replace(tmp, self.names_path)
        except OSError:
            pass

    def _remember_name(self, s, name, voice=None):
        """Persist (or drop) the label for the signature s, with the voice
        it was given for, if there is one yet."""
        kept = [e for e in self._named if abs(np.vdot(e["s"], s)) ** 2 < self.match]
        if name:
            kept.append({"s": s, "name": name, "voice": voice})
        self._named = kept
        self._save_names()

    def named_voice(self, name, s=None):
        """The voice persisted with a name (at the signature s, when given:
        one name can be on file at more than one bearing), or None."""
        for e in self._named:
            if e["name"] == name and (s is None or abs(np.vdot(e["s"], s)) ** 2 >= self.match):
                return e.get("voice")
        return None

    def note_voice(self, talker_id, voice):
        """A named talker's print has grown: keep the persisted voice current,
        so the name is given by the voice next time, not the bearing alone."""
        e = self.entry(talker_id)
        if e is None or not e.get("name") or voice is None:
            return
        for n in self._named:
            if n["name"] == e["name"] and abs(np.vdot(n["s"], e["s"])) ** 2 >= self.match:
                if n.get("voice") != voice:
                    n["voice"] = dict(voice)
                    self._save_names()
                return

    def disown(self, talker_id):
        """This entry wears a name it inherited from its bearing, and the
        voice says it is somebody else: take the name off the entry (the
        persisted name keeps its bearing and voice)."""
        e = self.entry(talker_id)
        if e is None or not e.get("name"):
            return None
        name, e["name"] = e["name"], None
        return name

    def _known_name(self, s):
        best, best_c = None, self.match
        for e in self._named:
            c = abs(np.vdot(e["s"], s)) ** 2
            if c >= best_c:
                best, best_c = e["name"], c
        return best

    def _activate(self, e, now):
        if self.active != e["id"]:
            self.active_since = now
        self.active = e["id"]

    def release(self):
        """The over ended: nobody's weight is live."""
        self.active = None
        self.active_since = None

    def recall(self, s, now):
        best, best_c = None, 0.0
        for e in self.entries:
            c = abs(np.vdot(e["s"], s)) ** 2
            if c > best_c:
                best, best_c = e, c
        if best is not None and best_c >= self.match:
            best["hits"] += 1
            best["last_seen"] = now
            self._activate(best, now)
            return best["m"]
        return None

    def store(self, s, m, now):
        for e in self.entries:
            c = np.vdot(e["s"], s)
            if abs(c) ** 2 >= self.match:
                # align the new vector's global phase to the stored one
                # before blending, so the blend cannot cancel
                s_al = s * (np.conj(c) / max(abs(c), 1e-12))
                v = (1.0 - MEMORY_MERGE) * e["s"] + MEMORY_MERGE * s_al
                e["s"] = v / max(np.linalg.norm(v), 1e-12)
                e["m"], e["last_seen"] = m, now
                self._activate(e, now)
                return
        self._add(s, m, now, self._known_name(s))

    def _evict(self):
        """Which entry to drop when the memory is full, keeping named
        regulars over strangers: an unnamed hits-0 entry (nobody has
        recognised it a second time) with the oldest first_seen, so a
        one-off caller goes first; failing that, any other unnamed entry
        by oldest last_seen; and only once every entry left is named --
        the whole band has been given a name -- the named entry heard
        longest ago. The focused talker is never a candidate."""
        candidates = [e for e in self.entries if e["id"] != self.focus.talker_id]
        unnamed = [e for e in candidates if not e["name"]]
        if unnamed:
            cold = [e for e in unnamed if e["hits"] == 0]
            pool, key = (cold, "first_seen") if cold else (unnamed, "last_seen")
            return min(pool, key=lambda e: e[key])
        return min(candidates, key=lambda e: e["last_seen"])

    def _add(self, s, m, now, name):
        e = {"id": self._next_id, "s": s, "m": m, "hits": 0,
             "first_seen": now, "last_seen": now, "name": name}
        self._next_id += 1
        self.entries.append(e)
        self._activate(e, now)
        if len(self.entries) > self.max_n:
            dropped = self._evict()
            self.entries.remove(dropped)
            if dropped["id"] == self.active:
                self.release()
        return e

    def reassign(self, now, unlike):
        """The live over came from the active talker's bearing but is not
        their voice (the print said so). Move it to a remembered talker at
        the same bearing whose print it does fit -- unlike(entry) False --
        or to a new talker there. Returns the id now live, None if nobody
        was live."""
        cur = self.entry(self.active)
        if cur is None:
            return None
        same = [e for e in self.entries
                if e is not cur and abs(np.vdot(e["s"], cur["s"])) ** 2 >= self.match]
        fit = [e for e in same if not unlike(e)]
        if fit:
            e = max(fit, key=lambda e: e["last_seen"])
            e["hits"] += 1
            e["last_seen"] = now
            self._activate(e, now)
            return e["id"]
        # the name belongs to the voice, not the bearing: a stranger there is unnamed
        return self._add(cur["s"].copy(), cur["m"], now, None)["id"]

    def name(self, talker_id, name, voice=None):
        """Label an entry; '' or None clears. False when the id is unknown.
        voice: the entry's print summary, persisted beside the name."""
        for e in self.entries:
            if e["id"] == int(talker_id):
                e["name"] = (str(name).strip() or None) if name is not None else None
                self._remember_name(e["s"], e["name"], voice if e["name"] else None)
                return True
        return False

    def entry(self, talker_id):
        return next((e for e in self.entries if e["id"] == talker_id), None)

    def set_focus(self, talker_id, now):
        """Pin the combiner on a remembered talker; None releases the pin."""
        if talker_id is None:
            self.focus.clear()
            return
        if self.entry(int(talker_id)) is None:
            raise ValueError(f"unknown talker id {talker_id}")
        self.focus.pin(talker_id, now)

    def focus_entry(self):
        return self.entry(self.focus.talker_id) if self.focus.active else None

    def focus_status(self, now, nulling=False):
        return self.focus.status(self.focus_entry(), now,
                                 live=self.active == self.focus.talker_id, nulling=nulling)

    def clear(self):
        self.entries = []
        self.release()
        self.focus.clear()

    def talker(self, now):
        """{"id", "since_s"} for the live talker, or None."""
        if self.active is None:
            return None
        return {"id": int(self.active),
                "since_s": round(max(0.0, now - self.active_since), 1)}

    def status(self, now):
        from .diversity import weight_to_polar    # local: avoids a circular import
        out = []
        for e in sorted(self.entries, key=lambda e: -e["last_seen"]):
            ph, ra = weight_to_polar(e["m"])
            out.append({"id": int(e["id"]), "name": e["name"],
                        "phase_deg": round(ph, 1), "ratio_db": round(ra, 1),
                        "age_s": round(max(0.0, now - e["last_seen"]), 1),
                        "first_seen_s": round(max(0.0, now - e["first_seen"]), 1),
                        "hits": int(e["hits"])})
        return out
