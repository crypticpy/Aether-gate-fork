#
# Aether-gate — is this the voice we think it is? (the over's own print check)
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""One check, once per over: the running voice print against the talker the
BEARING recalled.

Split out of adapters/diversity_state.py, which was at its 800-line budget when
G1 needed room; the logic is unchanged and it still reads and writes nothing but
the state handed to it (`state.memory`, `state.prints`' VoicePrint, and the
split counter the status publishes).

Two outcomes, and they are not the same thing:

  disown    the entry has no print of its own yet and wears a name it inherited
            from its bearing, and the running voice is not the voice that name
            was given for: the name comes off the entry (the name keeps its
            bearing and its voice on file).
  split     the entry has a print and this is not it: the over moves to whoever
            at that bearing does fit, or to a new talker there.
"""
import time


def _vp():
    from ..core import voiceprint
    return voiceprint


def voice_check(state, vp, active):
    """Once per over, as soon as the running print can be judged: an over
    recalled by bearing that is not that talker's voice goes to whoever at
    that bearing it is, or to a new talker."""
    cur = vp.current()
    if cur is None:
        return
    state._voice_checked = True
    v = _vp()
    mine = vp.summary(active)
    if mine is None:
        # no print of their own yet: a name inherited from the bearing is
        # checked against the voice the name was given for
        e = state.memory.entry(active) or {}
        known = state.memory.named_voice(e["name"], e["s"]) if e.get("name") else None
        dn = vp.distance(cur, known)
        if dn is not None and dn >= v.DIFFERENT_VOICE:
            name = state.memory.disown(active)
            print(f"[diversity] #{active} has {name}'s bearing but not their voice "
                  f"(d={dn:.2f}) -> unnamed", flush=True)
        return
    d = vp.distance(cur, mine)
    if d is None or d < v.DIFFERENT_VOICE:
        return

    def unlike(e):
        s = vp.summary(e["id"])
        dd = vp.distance(cur, s)
        return dd is not None and dd >= v.DIFFERENT_VOICE
    new = state.memory.reassign(time.monotonic(), unlike)
    if new is not None:
        state.voice_splits += 1
        was = state.memory.entry(active) or {}
        print(f"[diversity] voice split: #{active}{' ' + was['name'] if was.get('name') else ''}"
              f"'s bearing but not their voice (d={d:.2f}, centroid {cur['centroid_hz']} vs "
              f"{mine['centroid_hz']} Hz, top {cur['high_hz']} vs {mine['high_hz']}, tilt "
              f"{cur['tilt_db']} vs {mine['tilt_db']} dB) -> #{new}", flush=True)
