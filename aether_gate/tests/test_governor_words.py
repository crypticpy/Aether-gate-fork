#
# Aether-gate — R6-R9: the hum needs no bearing, and the words AUTO CLEAN says.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Four things the operator saw on a live band and the app could not explain:

  R6  a 60 Hz comb standing 11.8 dB over the floor at coherence 0.02, and
      AUTO CLEAN saying nothing was needed. A comb notch is a SPECTRAL tool:
      the hum's level is its gate, not a bearing. The null keeps its bearing.
  R7  "nothing on the band needs a tool right now" -- true, and it says
      nothing. What a person wants back is what was RULED OUT, on what number.
  R8  the CHAIN banner reading "holding what DIG OUT found . holding what DIG
      OUT found; nothing else...": `state_label` already carries the held list,
      so the `why` is the REMAINDER.
  R9  an event from 22:20 rendered as 11:23 -- `t` is monotonic, and nothing
      monotonic can be shown to a person. Every event gets a wall clock.

test_governor.py is at its 800-line budget; its `snap`/`gov`/`run` helpers and
its FakeAdapter are borrowed here.

Each test names the mutation it catches in its own body.

Run:  .venv/bin/python -m pytest aether_gate/tests/test_governor_words.py -q
"""
from aether_gate.adapters import diversity_governor as adapt
from aether_gate.core import governor as G

from .test_governor import WALL0, FakeAdapter, gov, run, snap

HUM = {"mains_hz": 60.0, "harmonics": 2, "hum_db": 11.8}


def _holding_the_blanker(g=None, t=0.0):
    """One site tool held, and nothing else on the band: the R8 case."""
    g = g or gov()
    s = snap(t=t, impulses_per_s=20.0, impulse_db=14.0)
    run(g, s)
    on = {"on": True, "db": 11.0, "auto": "off"}
    run(g, dict(s, t=t + 5.0, nb=on))
    g.tick(dict(s, t=t + 10.0, nb=on))
    return g


# --- R6: the comb does not need a direction ---------------------------------

def test_the_hum_threshold_is_the_profiles_own():
    """Mutation: an invented number. Every threshold in the governor is quoted
    from the module that owns it -- the harmonic count is already taken at
    LINE_MIN_DB, so the level that gates the comb has to be the same bar."""
    from aether_gate.core.noiseprofile import LINE_MIN_DB
    assert G.HUM_MIN_DB == LINE_MIN_DB


def test_an_incoherent_hum_is_squeezed_and_the_sentence_names_the_level():
    """Mutation: keeping `coh < NULLABLE_COHERENCE: return None` on the comb
    path. That is what shipped, and live it left 11.8 dB of mains hum in the
    audio all evening while AUTO CLEAN said nothing was needed."""
    act = gov().tick(snap(coherence=0.02, **HUM))[0]
    assert act["params"] == {"squeeze": "comb"} and act["kind"] == "mains"
    assert "11.8 dB over the floor" in act["why"]
    assert "coherence" not in act["why"]         # it was never the comb's gate


def test_a_hum_under_the_line_threshold_is_left_alone():
    """Mutation: dropping the coherence gate and putting nothing in its place.
    A notch is not free -- it costs the passband a hole -- so the level the
    profile itself counts a harmonic at is what has to earn it."""
    assert gov().tick(snap(coherence=0.9, mains_hz=60.0, harmonics=2,
                           hum_db=G.HUM_MIN_DB - 0.1)) == []
    act = gov().tick(snap(coherence=0.9, mains_hz=60.0, harmonics=2,
                          hum_db=G.HUM_MIN_DB))[0]
    assert act["params"] == {"squeeze": "comb"}


def test_a_null_already_deep_on_the_hum_still_stops_the_comb():
    """Mutation: taking the HUM_COVERED_DB check out with the coherence one.
    Notching a tone the null already took out is the collision the rule
    exists to avoid, and it has nothing to do with the bearing."""
    held = {"held": True, "tool": "null", "depth_db": G.HUM_COVERED_DB,
            "configured": True, "target": "signal", "hz": 1200.0}
    g = gov()
    g.holding["squeeze"] = {"tool": "squeeze", "params": {"squeeze": 1200},
                            "kind": "carrier", "why": "", "since": 0.0,
                            "delta_db": 1.0}
    assert g.tick(snap(coherence=0.02, squeeze=held, **HUM)) == []
    assert "the null already has the hum" in g.why


def test_the_null_on_the_floor_still_needs_the_loops_to_agree():
    """Mutation: dropping the coherence gate from _rule_mode as well. A null
    IS a bearing: with no direction in the floor there is nothing to steer at,
    and the combiner would be turned on a floor it cannot touch."""
    assert gov().tick(snap(mode="off", coherence=0.39)) == []
    assert gov().tick(snap(mode="off", coherence=0.4))[0]["params"] == {"mode": "null"}
    assert G.NULLABLE_COHERENCE == 0.4


def test_the_runner_carries_the_hums_level_from_the_profile():
    """Mutation: forgetting to plumb hum_db through. The policy would see None
    for ever and gate the comb on the harmonic count alone."""
    a = FakeAdapter()
    a.div["noise_profile"] = dict(a.div["noise_profile"], hum_db=11.8)
    assert adapt.GovernorRunner(a, clock=lambda: 0.0).snapshot()["hum_db"] == 11.8


# --- R7: what was ruled out -------------------------------------------------

def test_the_idle_why_is_the_rejections_in_the_chains_order():
    """Mutation: "nothing on the band needs a tool right now". True, and it
    tells the operator nothing: they cannot tell a measurement that looked
    from one that never ran."""
    g = gov()
    g.tick(snap(0.0))
    parts = g.why.split(" · ")
    assert [p.split(":")[0] for p in parts] == [
        "30 dB of ADC headroom, no clipping",
        "0 impulses/s",
        "the combiner is the operator's (track)",
        "no carrier and no mains comb over the floor",
        "no talker to dig for"]
    assert [o["tool"] for o in g.status()["ruled_out"]] == list(G.RULES)


def test_every_ruled_out_row_is_a_tool_and_a_reason():
    """Mutation: publishing the joined line only. The app elides one line; the
    list is what the CHAIN window can lay out a row at a time."""
    g = gov()
    g.tick(snap(0.0, **HUM))
    rows = g.status()["ruled_out"]
    assert all(set(r) == {"tool", "why"} for r in rows)
    assert "squeeze" not in [r["tool"] for r in rows]    # it proposed, it did not refuse
    assert g.state == "applying"


def test_a_backoff_and_the_operators_grace_are_ruled_out_in_words_too():
    """Mutation: `continue` with nothing said. These two are the reasons an
    operator is most likely to ask about: the tool they can see is needed and
    the governor will not touch it."""
    g = gov()
    g.backoff[("mains", "squeeze")] = 300.0
    g.tick(snap(0.0, **HUM))
    assert "a squeeze is backing off for 300s more" in g.why
    g2 = gov()
    g2._operator_at["squeeze"] = 0.0
    g2.tick(snap(10.0, **HUM))
    assert "a squeeze is the operator's for 50s more" in g2.why


def test_the_ruled_out_list_is_this_passs_own_and_not_a_log():
    """Mutation: appending to it for ever. The line would grow by five
    fragments a second and the app would show a paragraph."""
    g = gov()
    for t in range(4):
        g.tick(snap(float(t)))
    assert len(g.status()["ruled_out"]) == len(G.RULES)


# --- R8: holding says it once -----------------------------------------------

def test_the_why_while_holding_never_repeats_the_held_list():
    """Mutation: `why = f"holding {held}; nothing else..."`. state_label
    already says "holding the blanker", and the CHAIN banner prints both:
    "holding what DIG OUT found . holding what DIG OUT found; nothing else"."""
    g = _holding_the_blanker()
    assert g.status()["state_label"] == "holding the blanker"
    assert "holding" not in g.why
    assert "blanker already on (20/s)" in g.why


def test_with_nothing_at_all_to_report_the_words_still_say_which_case():
    """Mutation: one sentence for both. "Nothing else" is a promise that
    something IS held; without it the operator reads an idle chain."""
    g = gov()
    assert g._why_idle() == "nothing on the band needs a tool right now"
    g.holding["nb"] = {"tool": "nb", "since": 0.0}
    assert g._why_idle() == "nothing else on the band needs a tool"


# --- R9: a stamp a person can read ------------------------------------------

def test_every_event_carries_the_wall_clock_beside_the_monotonic_one():
    """Mutation: publishing `t` alone. It is time.monotonic() -- uptime -- and
    the app rendered an event from 22:20 as 11:23."""
    g = gov()
    run(g, snap(3.0, impulses_per_s=20.0, impulse_db=14.0))
    e = g.status()["events"][-1]
    assert e["t"] == 3.0 and e["wall"] == WALL0 + 3.0


def test_the_released_and_put_back_events_carry_it_too():
    """Mutation: stamping only in _event(). Every row the operator is most
    likely to be looking for -- released, put back -- would say 1970."""
    g = _holding_the_blanker()
    g.tick(snap(20.0, nb={"on": False, "db": 11.0, "auto": "off"}))
    rel = g.status()["events"][-1]
    assert rel["result"] == "released" and rel["wall"] == WALL0 + 20.0
    g.auto = False
    g.tick(snap(30.0))
    assert all(e["wall"] is not None for e in g.status()["events"])


def test_a_backoff_deadline_is_published_as_a_wall_clock_too():
    """Mutation: converting the event stamps and leaving the deadlines. The
    app shows "until" as a time of day, and it was showing the host's uptime
    plus five minutes."""
    g = gov()
    g.tick(snap(5.0))
    g.backoff[("mains", "squeeze")] = 305.0
    row = g.status()["backoff"][0]
    assert row["until"] == 305.0 and row["until_wall"] == WALL0 + 305.0


def test_the_offset_is_the_snapshots_not_the_hosts_clock():
    """Mutation: stamping time.time() when the status is READ. An event from
    an hour ago would then be dated now, which is the bug with a new coat."""
    g = gov()
    g.tick(dict(snap(1000.0), wall=WALL0))       # a host up 1000 s
    run(g, dict(snap(1000.0), wall=WALL0, impulses_per_s=20.0, impulse_db=14.0))
    assert g.status()["events"][-1]["wall"] == WALL0
    assert G.Governor()._wall_at(5.0) is None    # no snapshot: no invented time


def test_the_runner_puts_the_wall_clock_in_every_snapshot_it_makes():
    """Mutation: adding it to the healthy snapshot only. The tick that runs
    when a tuner drops out is the one that releases what was held."""
    a = FakeAdapter()
    r = adapt.GovernorRunner(a, clock=lambda: 4.0, wall=lambda: WALL0)
    assert r.snapshot()["wall"] == WALL0
    a.div["available"] = False
    assert r.snapshot() == {"t": 4.0, "wall": WALL0, "available": False}
