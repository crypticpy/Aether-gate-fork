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

KEYED BY BAND (G2). A signature is a loop-pair phase, and that phase for a
given bearing is a function of wavelength, so an entry earned on 20 m is not
an answer on 40 m: recall() and store() only ever look at entries whose
band_hz is the band the dial is on now (set_band). Entries from other bands
are KEPT -- they are what this station learned and the operator will tune
back -- and MEMORY_MAX is a per-band budget, so a night on 40 m cannot evict
the 20 m regulars. The file, and what of an entry crosses a restart, is
core/talkermemory_store.py.
"""
import time

import numpy as np

from . import talkermemory_store as store
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
# The memory is written from the STATUS side, never from store()/recall(): those
# run on the audio thread, which must not touch a file. At most this often, and
# only when something changed -- a name is written at once, a name is rare.
PERSIST_S = 30.0


class TalkerMemory:
    """Spatial signatures of recent talkers and the weight that suited each.

    When someone keys up, one block is enough to tell whether they are a
    known voice (a known bearing, really); if so the weight jumps straight
    to what worked last time instead of being re-learned over a refit cycle.
    """

    def __init__(self, max_n=MEMORY_MAX, match=MEMORY_MATCH, names_path=None,
                 store_path=None, wall=None, mono=None):
        self.max_n = int(max_n)
        self.match = float(match)
        self.entries = []                    # dicts: id, s, m, hits, first_seen,
                                             # last_seen, name, voice, band_hz,
                                             # center_hz, first_seen_wall, last_seen_wall
        self._next_id = 1                    # ids never reuse within a run
        self.active = None                   # id of the talker whose weight is live
        self.active_since = None
        self.focus = StationFocus()          # see focus.py
        self.band_hz = None                  # the band the dial is on (core/bands.py)
        self.center_hz = None                # ...and where inside it
        self._wall = wall or time.time       # epoch: what crosses a restart
        self._mono = mono or time.monotonic  # uptime: what first_seen/last_seen are
        self._dirty = False
        self._saved_at = None
        # The WHOLE entry outlives a run, not just the name (G2). names_path is
        # the pre-G2 names file: read once to migrate, and still the file
        # written when no store_path is given.
        self.names_path = names_path
        self.store_path = store_path
        self.entries, self._named = self._load()

    def _load(self):
        """(entries, named) from the store, or the migrated names alone."""
        entries, named = store.load(self.store_path) if self.store_path else (None, None)
        if entries is None:
            named = store.load_names(self.names_path) if self.names_path else []
            self._dirty = bool(named)        # migrate: the new file owns them now
            return [], named
        now_m, now_w = self._mono(), self._wall()
        for e in entries:
            e["id"], self._next_id = self._next_id, self._next_id + 1
            for k, w in (("first_seen", "first_seen_wall"),
                         ("last_seen", "last_seen_wall")):
                stamp = e.get(w)
                e[k] = now_m - (max(0.0, now_w - stamp) if stamp is not None else 0.0)
        return entries, named

    def _save(self):
        """The whole memory, now. With no store_path only the names go out --
        the pre-G2 behaviour, which is what a memory built with names_path
        alone still gets."""
        self._saved_at, self._dirty = self._mono(), False
        if self.store_path:
            store.save(self.store_path, list(self.entries), list(self._named))
        elif self.names_path:
            store.save_names(self.names_path, list(self._named))

    def persist(self, force=False):
        """Write the memory out if something changed, at most every PERSIST_S.

        CALLED FROM THE STATUS SIDE. store()/recall() run on the audio thread
        and only ever mark the memory dirty; nothing on that thread opens a
        file. Returns True when a write was made.
        """
        if not self._dirty or not (self.store_path or self.names_path):
            return False
        if not force and self._saved_at is not None \
                and self._mono() - self._saved_at < PERSIST_S:
            return False
        self._save()
        return True

    def set_band(self, band_hz, center_hz=None):
        """The dial moved: which band recall()/store() match against now."""
        self.band_hz = None if band_hz is None else int(band_hz)
        self.center_hz = None if center_hz is None else float(center_hz)

    def _here(self, e):
        """Is this entry's band the band we are on? A match on another band is
        false by construction: the phase that made it is wavelength-dependent."""
        return e.get("band_hz") == self.band_hz

    def _touch(self, e):
        """This entry was just heard: the wall stamp and the frequency it was
        heard on, which are what survive a restart."""
        e["last_seen_wall"] = self._wall()
        if self.center_hz is not None:
            e["center_hz"] = self.center_hz
        self._dirty = True

    def _remember_name(self, s, name, voice=None):
        """Persist (or drop) the label for the signature s, with the voice
        it was given for, if there is one yet."""
        kept = [e for e in self._named if abs(np.vdot(e["s"], s)) ** 2 < self.match]
        if name:
            kept.append({"s": s, "name": name, "voice": voice})
        self._named = kept
        self._save()

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
                    n["voice"] = e["voice"] = dict(voice)
                    self._save()
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
            if not self._here(e):
                continue
            c = abs(np.vdot(e["s"], s)) ** 2
            if c > best_c:
                best, best_c = e, c
        if best is not None and best_c >= self.match:
            best["hits"] += 1
            best["last_seen"] = now
            self._touch(best)
            self._activate(best, now)
            return best["m"]
        return None

    def store(self, s, m, now):
        for e in self.entries:
            if not self._here(e):
                continue
            c = np.vdot(e["s"], s)
            if abs(c) ** 2 >= self.match:
                # align the new vector's global phase to the stored one
                # before blending, so the blend cannot cancel
                s_al = s * (np.conj(c) / max(abs(c), 1e-12))
                v = (1.0 - MEMORY_MERGE) * e["s"] + MEMORY_MERGE * s_al
                e["s"] = v / max(np.linalg.norm(v), 1e-12)
                e["m"], e["last_seen"] = m, now
                self._touch(e)
                self._activate(e, now)
                return
        self._add(s, m, now, self._known_name(s))

    def _evict(self, pool):
        """Which entry to drop when the memory is full, keeping named
        regulars over strangers: an unnamed hits-0 entry (nobody has
        recognised it a second time) with the oldest first_seen, so a
        one-off caller goes first; failing that, any other unnamed entry
        by oldest last_seen; and only once every entry left is named --
        the whole band has been given a name -- the named entry heard
        longest ago. The focused talker is never a candidate.

        `pool` is THIS BAND's entries and nothing else: MEMORY_MAX is a budget
        per band, and an evening on 40 m must not cost the operator the 20 m
        regulars they have been naming for a month."""
        candidates = [e for e in pool if e["id"] != self.focus.talker_id]
        unnamed = [e for e in candidates if not e["name"]]
        if unnamed:
            cold = [e for e in unnamed if e["hits"] == 0]
            pool, key = (cold, "first_seen") if cold else (unnamed, "last_seen")
            return min(pool, key=lambda e: e[key])
        return min(candidates, key=lambda e: e["last_seen"])

    def _add(self, s, m, now, name):
        w = self._wall()
        e = {"id": self._next_id, "s": s, "m": m, "hits": 0,
             "first_seen": now, "last_seen": now, "name": name, "voice": None,
             "band_hz": self.band_hz, "center_hz": self.center_hz,
             "first_seen_wall": w, "last_seen_wall": w}
        self._next_id += 1
        self.entries.append(e)
        self._dirty = True
        self._activate(e, now)
        here = [x for x in self.entries if self._here(x)]
        if len(here) > self.max_n:
            dropped = self._evict(here)
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
                if e is not cur and self._here(e)
                and abs(np.vdot(e["s"], cur["s"])) ** 2 >= self.match]
        fit = [e for e in same if not unlike(e)]
        if fit:
            e = max(fit, key=lambda e: e["last_seen"])
            e["hits"] += 1
            e["last_seen"] = now
            self._touch(e)
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
                e["voice"] = dict(voice) if (e["name"] and voice) else None
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
        self._dirty = True
        self._save()          # a forget the operator asked for outlives a crash

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
                        "hits": int(e["hits"]),
                        "band_hz": e.get("band_hz"), "center_hz": e.get("center_hz"),
                        "first_seen_wall": e.get("first_seen_wall"),
                        "last_seen_wall": e.get("last_seen_wall")})
        return out
