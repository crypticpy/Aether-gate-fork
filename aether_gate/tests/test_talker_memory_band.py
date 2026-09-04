#
# Aether-gate — G2: the talker memory keyed by band, and kept on disk.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""A loop pair's phase for a given bearing is a function of wavelength, so a
fingerprint earned on 20 m is not an answer on 40 m: recall and store only ever
look at this band's entries, the other bands' are kept, and MEMORY_MAX is a
budget per band. The whole entry now outlives a run, and the pre-G2 names file
is migrated once.

Each test names the mutation it catches in its own body.

Run:  .venv/bin/python -m pytest aether_gate/tests/test_talker_memory_band.py -q
"""
import json

import numpy as np

from aether_gate.core import talkermemory_store as store
from aether_gate.core.talkermemory import TalkerMemory

M20, M40 = 14_175_000, 7_150_000


def _s(theta, ratio=0.6):
    v = np.array([1.0, ratio * np.exp(1j * theta)])
    return v / np.linalg.norm(v)


def _mem(tmp_path, name="talkers.json", **kw):
    kw.setdefault("wall", _wall())
    return TalkerMemory(store_path=str(tmp_path / name), **kw)


def _wall(t0=1_700_000_000.0):
    box = {"t": t0}

    def wall():
        box["t"] += 1.0
        return box["t"]
    return wall


# --- the band gate ----------------------------------------------------------

def test_a_fingerprint_from_another_band_is_not_recalled(tmp_path):
    """Mutation: drop the band test in recall(). A talker recognised at
    14.250 would be recognised at 7.150, and steered to the wrong weight."""
    m = _mem(tmp_path)
    m.set_band(M20, 14_250_000.0)
    m.store(_s(0.4), 0.3 + 0.2j, now=0.0)
    assert m.recall(_s(0.4), now=1.0) == 0.3 + 0.2j
    m.set_band(M40, 7_150_000.0)
    assert m.recall(_s(0.4), now=2.0) is None


def test_the_same_bearing_on_a_new_band_is_a_new_entry(tmp_path):
    """Mutation: drop the band test in store(). The 20 m entry would be
    merged into and its weight overwritten by a 40 m over."""
    m = _mem(tmp_path)
    m.set_band(M20, 14_250_000.0)
    m.store(_s(0.4), 0.3 + 0.2j, now=0.0)
    m.set_band(M40, 7_150_000.0)
    m.store(_s(0.4), 0.9 + 0.0j, now=1.0)
    assert [(e["band_hz"], e["m"]) for e in m.entries] == [(M20, 0.3 + 0.2j),
                                                           (M40, 0.9 + 0.0j)]


def test_the_other_bands_entries_are_kept_and_come_back(tmp_path):
    """Mutation: clearing the memory on a band change. What the station
    learned is capital -- the operator tunes back."""
    m = _mem(tmp_path)
    m.set_band(M20, 14_250_000.0)
    m.store(_s(0.4), 0.3 + 0.2j, now=0.0)
    m.set_band(M40, 7_150_000.0)
    m.store(_s(2.5), 0.1 + 0.0j, now=1.0)
    assert len(m.entries) == 2
    m.set_band(M20, 14_260_000.0)
    assert m.recall(_s(0.4), now=2.0) == 0.3 + 0.2j


def test_memory_max_is_a_budget_per_band(tmp_path):
    """Mutation: evicting over every entry. An evening of strangers on 40 m
    would empty the operator's named 20 m regulars."""
    m = _mem(tmp_path, max_n=2)
    m.set_band(M20, 14_250_000.0)
    m.store(_s(0.0), 0j, now=0.0)
    m.store(_s(1.6), 0j, now=1.0)
    m.set_band(M40, 7_150_000.0)
    for k, th in enumerate((0.0, 1.6, 3.1)):
        m.store(_s(th), 0j, now=2.0 + k)
    assert len([e for e in m.entries if e["band_hz"] == M20]) == 2
    assert len([e for e in m.entries if e["band_hz"] == M40]) == 2


def test_a_voice_split_never_reaches_across_a_band(tmp_path):
    """Mutation: reassign() scanning every entry -- an over on 40 m would be
    handed to a 20 m talker at the same bearing."""
    m = _mem(tmp_path)
    m.set_band(M20, 14_250_000.0)
    m.store(_s(0.4), 0.3 + 0.2j, now=0.0)
    m.set_band(M40, 7_150_000.0)
    m.store(_s(0.4), 0.9 + 0.0j, now=1.0)
    new = m.reassign(now=2.0, unlike=lambda e: False)
    assert new is not None and m.entry(new)["band_hz"] == M40


def test_an_entry_carries_the_band_the_frequency_and_the_wall_clock(tmp_path):
    """Mutation: dropping any of the four keys. The app dates a talker by
    them, and a monotonic clock does not survive a restart."""
    m = _mem(tmp_path)
    m.set_band(M20, 14_250_000.0)
    m.store(_s(0.4), 0.3 + 0.2j, now=0.0)
    row = m.status(now=1.0)[0]
    assert row["band_hz"] == M20 and row["center_hz"] == 14_250_000.0
    assert row["first_seen_wall"] > 1_600_000_000.0
    assert row["last_seen_wall"] >= row["first_seen_wall"]


# --- the file ---------------------------------------------------------------

def test_store_and_recall_never_open_a_file(tmp_path):
    """Mutation: saving from store()/recall(). Those run on the AUDIO thread;
    a file write there is a click in the operator's ears."""
    path = tmp_path / "talkers.json"
    m = _mem(tmp_path)
    m.set_band(M20, 14_250_000.0)
    for k in range(4):
        m.store(_s(k * 1.5), 0.2j, now=float(k))
        m.recall(_s(0.0), now=float(k))
    assert not path.exists()
    assert m.persist() is True and path.exists()


def test_persist_is_debounced_and_only_writes_a_change(tmp_path):
    """Mutation: writing on every status read -- the file would be rewritten
    once a second for as long as the gate is up."""
    m = _mem(tmp_path)
    m.set_band(M20, 14_250_000.0)
    m.store(_s(0.4), 0.3j, now=0.0)
    assert m.persist() is True
    assert m.persist() is False                  # nothing changed
    m.store(_s(2.5), 0.1j, now=1.0)
    assert m.persist() is False                  # changed, but too soon
    assert m.persist(force=True) is True


def test_the_whole_entry_survives_a_restart(tmp_path):
    """Mutation: persisting names only (the pre-G2 file). The weight, the
    band, the hits and the times all died with the process."""
    m = _mem(tmp_path)
    m.set_band(M20, 14_250_000.0)
    m.store(_s(0.4), 0.3 + 0.2j, now=0.0)
    m.recall(_s(0.4), now=1.0)                   # one hit
    m.name(1, "Ann")
    m.set_band(M40, 7_150_000.0)
    m.store(_s(2.5), 0.9 + 0.0j, now=2.0)
    m.persist(force=True)

    again = _mem(tmp_path)
    assert [e["band_hz"] for e in again.entries] == [M20, M40]
    assert [e["name"] for e in again.entries] == ["Ann", None]
    assert again.entries[0]["hits"] == 1
    again.set_band(M20, 14_250_000.0)
    assert again.recall(_s(0.4), now=0.0) == 0.3 + 0.2j
    again.set_band(M40, 7_150_000.0)
    assert again.recall(_s(0.4), now=1.0) is None


def test_the_ages_come_back_as_ages_not_as_uptime(tmp_path):
    """Mutation: loading last_seen straight into the monotonic field. The
    memory table would say a talker heard last night was heard 4 days ago
    (whatever the host's uptime is) or 0 s ago."""
    wall = [1_700_000_000.0]
    m = TalkerMemory(store_path=str(tmp_path / "talkers.json"),
                     wall=lambda: wall[0], mono=lambda: 100.0)
    m.set_band(M20, 14_250_000.0)
    m.store(_s(0.4), 0.3j, now=0.0)
    m.persist(force=True)
    wall[0] += 3600.0                            # an hour later, a new gate
    again = TalkerMemory(store_path=str(tmp_path / "talkers.json"),
                         wall=lambda: wall[0], mono=lambda: 900_000.0)
    assert again.status(now=900_000.0)[0]["age_s"] == 3600.0


def test_the_old_names_file_is_migrated_once(tmp_path):
    """Mutation: reading only the new file. Every name the operator gave
    before G2 would be lost on the upgrade."""
    names = tmp_path / "diversity-names.json"
    names.write_text(json.dumps([{"s": store.pack(_s(0.4)), "name": "Ann",
                                  "voice": None}]))
    talkers = tmp_path / "diversity-talkers.json"
    m = TalkerMemory(names_path=str(names), store_path=str(talkers))
    assert [n["name"] for n in m._named] == ["Ann"]
    m.set_band(M20, 14_250_000.0)
    m.store(_s(0.4), 0.3j, now=0.0)              # the name comes back by bearing
    assert m.entries[0]["name"] == "Ann"
    assert m.persist(force=True) is True and talkers.exists()

    names.unlink()                               # the new file owns them now
    again = TalkerMemory(names_path=str(names), store_path=str(talkers))
    assert [n["name"] for n in again._named] == ["Ann"]
    assert [e["name"] for e in again.entries] == ["Ann"]


def test_the_names_file_alone_is_still_written_without_a_store(tmp_path):
    """Mutation: a save that needs a store_path. The pre-G2 constructor is
    still used (test_diversity.py), and it must keep its file."""
    names = tmp_path / "names.json"
    m = TalkerMemory(names_path=str(names))
    m.store(_s(0.4), 0.3j, now=0.0)
    m.name(1, "Bob")
    assert json.loads(names.read_text())[0]["name"] == "Bob"


def test_a_short_or_missing_file_is_an_empty_memory_not_a_crash(tmp_path):
    """Mutation: letting the JSON error out of __init__. A gate that will not
    start because a memory file lost a brace is worse than a lost memory."""
    bad = tmp_path / "bad.json"
    bad.write_text("{\"entries\": [{\"s\": [1.0], \"m\": null}], ")
    m = TalkerMemory(store_path=str(bad))
    assert m.entries == [] and m._named == []
    assert TalkerMemory(store_path=str(tmp_path / "none.json")).entries == []
