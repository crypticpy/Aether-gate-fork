#
# Aether-gate — the noise profile's numbers into named, actionable kinds.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""What _DiversityState._noise_profile hands the control port: one row per
thing the profile found (mains hum, impulses, a periodic modulation, an ANF
tone, or -- when none of those fired -- the broadband floor), each naming
what it is, how long it was measured over, and the one control-port call
that does something about it, or why nothing can.

Split out of adapters/diversity_state.py to keep that module under the
project's per-file line budget: this is formatting over the profile's
status dict and a few scalars, not adapter state, so it stands alone.
"""


def noise_kinds(st, coh, mode, nb_on, blanked_pct, nullable_coherence, filt_status):
    """st: NoiseProfile.status(). coh: the tracker's noise coherence (or
    None). filt_status: the slice filter's status() (or None, no slice yet).
    Returns the `kinds` list -- st["kinds"] in the caller."""
    directional = coh is not None and coh >= nullable_coherence
    kinds = []

    def null_action(active_ok=True):
        if mode == "null":
            return {"label": "NULLED", "route": "/diversity/set", "query": "mode=track"}, None, True
        if directional:
            return {"label": "NULL", "route": "/diversity/set", "query": "mode=null"}, None, False
        if coh is None:
            return None, "no noise estimate yet", False
        return None, f"not directional enough to null (coherence {coh:.2f})", False

    if st["mains_hz"]:
        act, why, active = null_action()
        f2 = int(2 * st["mains_hz"])
        kinds.append({
            "kind": "mains", "label": f"Mains hum · {int(st['mains_hz'])} Hz grid",
            "detail": f"{f2} Hz comb, {st['harmonics']} harmonic{'s' if st['harmonics'] != 1 else ''}",
            "db": st["hum_db"], "window_s": st["window_s"],
            "action": act, "why": why, "active": active,
        })
    if st["impulse_db"] is not None and st["impulses_per_s"] >= 1.0:
        rec = min(30.0, max(6.0, round((st["impulse_db"] - 3.0) * 2) / 2))
        if nb_on:
            act = {"label": "UNBLANK", "route": "/diversity/set", "query": "nb=off"}
        else:
            act = {"label": "BLANK", "route": "/diversity/set", "query": f"nb=on&nb_db={rec:g}"}
        kinds.append({
            "kind": "impulse", "label": f"Impulses · {st['impulses_per_s']:g}/s",
            "detail": (f"{st['impulse_db']:g} dB over the floor"
                       + (f", blanking {blanked_pct:.1f} %" if nb_on else "")),
            "db": st["impulse_db"], "window_s": st["impulse_window_s"],
            "action": act, "why": None, "active": bool(nb_on),
        })
    for line in st["periodic"]:
        kinds.append({
            "kind": "periodic", "label": f"Periodic · {line['hz']:g} Hz",
            "detail": "a modulation rate of the noise, not a tone in the audio",
            "db": line["db"], "window_s": st["window_s"],
            "action": None, "why": "nothing to notch; ANF handles tones in the passband",
            "active": False,
        })
    if filt_status is not None:
        fs = filt_status
        have = [n["hz"] for n in fs["notches"]]
        for hz, db in zip(fs["anf"]["found_hz"], fs["anf"]["depth_db"]):
            pinned = any(abs(hz - h) <= 30 for h in have)
            kinds.append({
                "kind": "tone", "label": f"Tone · {hz:g} Hz",
                "detail": f"ANF is holding it {abs(db):g} dB down",
                "db": db, "window_s": 1.0,
                "action": (None if pinned else
                           {"label": "NOTCH", "route": "/filter/notch",
                            "query": f"add={hz:g}&width=160"}),
                "why": "already a fixed notch" if pinned else None, "active": pinned,
            })
    if not kinds:
        act, why, active = null_action()
        kinds.append({
            "kind": "floor", "label": "Broadband floor",
            "detail": ("nothing mains-locked, impulsive or periodic"
                       + (f"; coherence {coh:.2f}" if coh is not None else "")),
            "db": None, "window_s": st["window_s"],
            "action": act, "why": why, "active": active,
        })
    return kinds
