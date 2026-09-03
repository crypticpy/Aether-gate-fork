#
# Aether-gate — a stranger at a remembered talker's bearing becomes a talker.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The memory recalls by bearing within 50 ms, so two people from one
direction were one talker ("everybody is Ted"). The voice print is the
second key: a second into an over that is not the recalled talker's voice,
the over moves to whoever at that bearing it fits, or to a new, unnamed
talker there -- and the print learned from that over is theirs.

Run:  python -m pytest aether_gate/tests/test_talker_split.py
"""
import time

import numpy as np

from aether_gate.core.diversity import TalkerMemory
from aether_gate.adapters.diversity_state import _DiversityState
from aether_gate.tests.test_diversity_state_v2 import _FakeAdapter
from aether_gate.tests.test_voiceprint import _over

RATE = 25_000.0
BLOCK = 2000
BEARING = np.array([1.0, 0.7 * np.exp(1j * 0.4)])
BEARING = BEARING / np.linalg.norm(BEARING)


def _state():
    st = _DiversityState(_FakeAdapter(mode="USB"))
    st.aligner.set_lag(0, 20.0, True)
    st.set(mode="track", subband=False)
    rng = np.random.default_rng(1)
    pa = rng.normal(size=1024) + 1j * rng.normal(size=1024)
    st.observe(0, pa, pa)
    return st


def _speak(st, x, talking=True):
    t = st.trackers[0]
    t.talking, t.steady = talking, False
    for i in range(0, len(x), BLOCK):
        blk = x[i:i + BLOCK]
        if len(blk) < BLOCK:
            break
        st.combine_passband(0, blk, blk, 0j, 0j, RATE)


def _silence(st):
    _speak(st, np.zeros(BLOCK, dtype=np.complex128), talking=False)


def test_memory_reassign_moves_the_over_to_a_fitting_talker_or_a_new_one():
    mem = TalkerMemory()
    now = time.monotonic()
    mem.store(BEARING, 0.7 + 0j, now)
    mem.entry(1)["name"] = "Ted"
    # nobody else at that bearing fits: a new, unnamed talker there
    assert mem.reassign(now, lambda e: True) == 2
    assert mem.active == 2 and len(mem.entries) == 2
    e = mem.entry(2)
    assert e["name"] is None and np.allclose(e["s"], BEARING) and e["m"] == 0.7 + 0j
    # back on Ted by bearing, but the voice fits #2: #2 is live again, no third
    mem._activate(mem.entry(1), now)
    assert mem.reassign(now, lambda e: e["id"] != 2) == 2
    assert mem.active == 2 and len(mem.entries) == 2 and mem.entry(2)["hits"] == 1
    # a talker from elsewhere is not a candidate
    mem.release()
    assert mem.reassign(now, lambda e: False) is None


def test_two_voices_from_one_bearing_become_two_talkers_and_the_name_stays_put():
    rng = np.random.default_rng(2)
    st = _state()
    st.memory.store(BEARING, 0.7 + 0j, time.monotonic())
    st.memory.entry(1)["name"] = "Ted"
    # Ted talks for 3 s: his print is learned
    _speak(st, _over(rng, 3.2, 100.0, 2900.0, 5.0))
    _silence(st)
    ted = st.prints[0].summary(1)
    assert ted is not None and ted["overs"] == 1
    # somebody else from the same bearing, recalled as Ted: within about a
    # second the over is theirs, they are #2, unnamed, and Ted keeps his print
    st.memory._activate(st.memory.entry(1), time.monotonic())
    _speak(st, _over(rng, 2.0, 300.0, 2400.0, 3.0))
    assert st.memory.active == 2 and st.voice_splits == 1
    _silence(st)
    mem = {e["id"]: e for e in st.status()["memory"]}
    assert set(mem) == {1, 2}
    assert mem[1]["name"] == "Ted" and mem[2]["name"] is None
    assert st.prints[0].summary(1)["overs"] == 1
    other = st.prints[0].summary(2)
    assert other is not None and other["overs"] == 1 and other["high_hz"] < ted["high_hz"]
    # recalled as Ted once more, the same stranger goes straight back to #2
    st.memory._activate(st.memory.entry(1), time.monotonic())
    _speak(st, _over(rng, 2.0, 300.0, 2400.0, 3.0))
    assert st.memory.active == 2 and len(st.memory.entries) == 2 and st.voice_splits == 2
    _silence(st)
    assert st.prints[0].summary(2)["overs"] == 2
    # and Ted himself is never moved
    st.memory._activate(st.memory.entry(1), time.monotonic())
    _speak(st, _over(rng, 2.0, 100.0, 2900.0, 5.0))
    assert st.memory.active == 1 and st.voice_splits == 2
