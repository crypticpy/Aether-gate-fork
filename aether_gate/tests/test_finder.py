#
# Aether-gate — the live spatial rows and the conversation finder, no hardware.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Synthetic frames straight into LiveSpatial and Finder: a voice-shaped
patch (syllabic modulation, both loops, fixed inter-loop phase), a steady
carrier, and band noise. The finder must rank the voice first, refuse the
carrier and the noise, say of every candidate WHAT it thinks it is, put the
dial just below the energy for USB, and carry the pair's phase there; the
spatial rows must show that phase only where the source is.

Run:  python -m pytest aether_gate/tests/test_finder.py
"""
import numpy as np
import pytest

from aether_gate.core.kinds import KINDS
from aether_gate.core.finder import (Finder, LiveSpatial, VOICE_SCORE, FAST_FRAMES,
                                     SLOW_PERIOD_S, WINDOW_STEP_POINTS)

NBINS = 2048
RATE = 125_000.0
FRAME_S = 4096 / RATE
CENTER = 14_100_000.0


def _frames(rng, n, voice=None, carrier=None, phase=1.2, ratio=0.8, snr=20.0):
    """Yield n frames of (2, NBINS) spectra in natural FFT order.

    voice: (lo_hz, hi_hz) relative to centre, energy modulated at 4 Hz with a
    hard on/off syllable pattern; carrier: hz relative to centre, steady.
    """
    f = np.fft.fftfreq(NBINS, 1.0 / RATE)
    t = 0.0
    for _ in range(n):
        Xa = (rng.normal(size=NBINS) + 1j * rng.normal(size=NBINS)) / np.sqrt(2)
        Xb = (rng.normal(size=NBINS) + 1j * rng.normal(size=NBINS)) / np.sqrt(2)
        if voice is not None:
            sel = (f >= voice[0]) & (f < voice[1])
            env = 1.0 if (t * 4.0) % 1.0 < 0.5 else 0.05          # syllables at 4 Hz
            s = (rng.normal(size=sel.sum()) + 1j * rng.normal(size=sel.sum())) * np.sqrt(snr * env / 2)
            Xa[sel] += s
            Xb[sel] += s * ratio * np.exp(1j * phase)
        if carrier is not None:
            k = int(np.argmin(np.abs(f - carrier)))
            s = np.sqrt(snr * 40) * np.exp(1j * 2 * np.pi * 0.37 * t)
            Xa[k] += s
            Xb[k] += s * ratio * np.exp(1j * phase)
        yield np.stack([Xa, Xb])
        t += FRAME_S


def _run(rng, n, **kw):
    live = LiveSpatial(NBINS, RATE)
    fd = Finder(NBINS, RATE)
    for X in _frames(rng, n, **kw):
        live.update(X, FRAME_S)
        fd.update(X, FRAME_S)
    return live, fd


def test_nothing_is_reported_before_there_is_data():
    fd = Finder(NBINS, RATE)
    live = LiveSpatial(NBINS, RATE)
    assert fd.candidates(CENTER, live) == {"available": False}
    assert live.rows(CENTER) is None


def test_voice_is_found_first_and_the_carrier_and_noise_are_not():
    rng = np.random.default_rng(7)
    live, fd = _run(rng, 400, voice=(20_000.0, 22_600.0), carrier=-30_000.0)
    out = fd.candidates(CENTER, live)
    assert out["available"] and len(out["activity"]) == out["points"] == 512
    cands = out["candidates"]
    assert cands, "the voice patch was not found"
    top = cands[0]
    assert top["score"] >= VOICE_SCORE and top["mode"] == "USB"
    # USB: the dial sits just below the energy, which starts 20 kHz up
    assert CENTER + 19_000 <= top["hz"] <= CENTER + 20_100, top
    # phone sits on whole and half kilohertz: the dial is snapped to 500 Hz
    # and the estimate it came from rides beside it
    assert top["hz"] % 500 == 0 and abs(top["hz_raw"] - top["hz"]) <= 250, top
    assert top["syllabic"] >= 0.6 and top["depth"] >= 0.4 and top["snr_db"] >= 5.0
    assert top["active_s"] >= 3 * SLOW_PERIOD_S and top["last_s"] is not None
    # and it says so: a phone-wide patch swinging at syllable rate is voice
    assert top["kind"] == "voice" and top["kind_conf"] >= 0.5
    # every row carries a verdict, and no verdict is off the list of names
    assert all(c["kind"] in KINDS and 0.0 <= c["kind_conf"] <= 1.0 for c in cands)
    # the carrier IS listed now -- a finder that only lists conversations
    # cannot answer "what is that?" -- but it is never called one, and the
    # empty band either side of the two signals is not listed at all
    carrier = [c for c in cands if CENTER - 32_000 <= c["hz"] <= CENTER - 28_000]
    assert len(carrier) == 1, cands
    assert carrier[0]["kind"] in ("carrier", "psk31", "data", "signal"), carrier
    assert len(cands) == 2, cands
    assert all(c["score"] > 0.0 for c in cands)
    # the pair's phase and level ratio are read at the candidate
    assert top["phase_deg"] == pytest.approx(np.degrees(-1.2), abs=8.0)
    assert top["ratio_db"] == pytest.approx(20 * np.log10(0.8), abs=1.5)
    assert top["coherence"] >= 0.6            # syllable gaps are noise-only time
    # equal-ish loops: the pair can earn close to the 0.8^2 MRC gain
    assert 1.5 <= top["gain_db"] <= 3.1, top


def test_activity_strip_lights_only_where_the_voice_is():
    rng = np.random.default_rng(8)
    live, fd = _run(rng, 400, voice=(20_000.0, 22_600.0))
    out = fd.candidates(CENTER, live)
    act = np.asarray(out["activity"])
    step = RATE / 512
    lo = int((RATE / 2 + 20_000) / step)
    hi = int((RATE / 2 + 22_600) / step)
    assert np.max(act[lo:hi]) >= 0.5
    assert np.max(np.concatenate([act[:lo - 20], act[hi + 20:]])) < 0.2


def test_spatial_rows_carry_the_phase_only_where_the_source_is():
    rng = np.random.default_rng(9)
    live, _ = _run(rng, FAST_FRAMES // 4, voice=(20_000.0, 22_600.0), phase=2.0, snr=40.0)
    rows = live.rows(CENTER)
    assert rows["points"] == 512 and len(rows["phase_deg"]) == 512
    assert rows["start_hz"] == pytest.approx(CENTER - RATE / 2)
    step = rows["step_hz"]
    coh = np.asarray(rows["coherence"])
    ph = np.asarray(rows["phase_deg"])
    lo = int((RATE / 2 + 20_000) / step) + 1
    hi = int((RATE / 2 + 22_600) / step) - 1
    assert np.min(coh[lo:hi]) >= 0.7
    assert np.all(np.abs(((ph[lo:hi] - np.degrees(-2.0)) + 180) % 360 - 180) <= 10)
    # plain noise is incoherent: the EMA has ~8 frames x 4 bins of averaging
    assert np.median(coh[:lo - 10]) < 0.15


def test_a_conversation_that_stopped_is_still_listed_with_its_age():
    rng = np.random.default_rng(10)
    live = LiveSpatial(NBINS, RATE)
    fd = Finder(NBINS, RATE)
    for X in _frames(rng, 300, voice=(-10_000.0, -7_400.0)):
        live.update(X, FRAME_S)
        fd.update(X, FRAME_S)
    for X in _frames(rng, 450):                     # ~15 s of silence
        live.update(X, FRAME_S)
        fd.update(X, FRAME_S)
    out = fd.candidates(CENTER, live)
    assert out["candidates"], "a 30 s recent window must keep it listed"
    top = out["candidates"][0]
    # ~15 s of silence, minus the ~4 s the 8.5 s scoring window keeps hearing it
    assert 8.0 <= top["last_s"] <= 14.0, top
    assert top["active_s"] >= 3 * SLOW_PERIOD_S
    # the row describes the conversation as it was, not the silence now
    assert top["snr_db"] >= 5.0 and top["syllabic"] >= 0.6 and top["depth"] >= 0.4, top
    # including what it was: silence is noise-shaped, the conversation was not
    assert top["kind"] == "voice" and top["kind_conf"] >= 0.5, top


# --- retune: a centre move at the same rate must not lose history -----------

def test_a_candidate_survives_a_retune_at_its_absolute_frequency_with_no_new_frames():
    rng = np.random.default_rng(30)
    live, fd = _run(rng, 400, voice=(20_000.0, 22_600.0))
    before = fd.candidates(CENTER, live)["candidates"]
    assert before, "the voice patch was not found before retuning"
    hz_before, kind_before = before[0]["hz"], before[0]["kind"]
    window_steps = 20                           # a modest, same-span retune
    delta_hz = window_steps * WINDOW_STEP_POINTS * fd.step_hz
    fd.retune(delta_hz)
    live.retune(delta_hz)
    out = fd.candidates(CENTER + delta_hz, live)          # no new frames fed
    assert out["available"]
    after = out["candidates"]
    assert after, "the candidate must still be reported without new frames"
    assert after[0]["hz"] == pytest.approx(hz_before, abs=fd.step_hz)
    assert after[0]["kind"] == kind_before


def test_activity_history_moves_with_a_retune():
    rng = np.random.default_rng(31)
    live, fd = _run(rng, 400, voice=(20_000.0, 22_600.0))
    act_before = np.asarray(fd.candidates(CENTER, live)["activity"])
    peak_before = int(np.argmax(act_before))
    assert act_before[peak_before] >= 0.5
    window_steps = 20
    delta_hz = window_steps * WINDOW_STEP_POINTS * fd.step_hz
    point_shift = round(delta_hz / fd.step_hz)
    fd.retune(delta_hz)
    live.retune(delta_hz)
    act_after = np.asarray(fd.candidates(CENTER + delta_hz, live)["activity"])
    peak_after = int(np.argmax(act_after))
    # the activity moved by exactly the retune, in bins of the points grid
    assert peak_after == peak_before - point_shift
    assert act_after[peak_after] >= 0.5


def test_a_retune_bigger_than_the_span_resets_the_finder_and_the_live_rows():
    rng = np.random.default_rng(32)
    live, fd = _run(rng, 400, voice=(20_000.0, 22_600.0))
    assert fd.candidates(CENTER, live)["available"]
    assert live.rows(CENTER) is not None
    huge = 2.0 * fd.rate_hz                     # far more than one span
    fd.retune(huge)
    live.retune(huge)
    assert fd.candidates(CENTER + huge, live) == {"available": False}
    assert live.rows(CENTER + huge) is None
    assert fd.fast_n == 0 and fd.slow_n == 0


# =========================================================================
# What the thing IS -- and, for each rule, what it would say without it
# =========================================================================
# The 2026-09-03 evening report: "it's still not 100% accurate. It's picking
# things up as voice that are clearly not." Every case below is a rule the
# classifier has to hold, checked twice: once that it holds, and once with the
# rule's own threshold pushed out of reach, so a test that would still pass
# with the rule deleted cannot sit here quietly.
#
# The mutation is always to a CONSTANT rather than to an expression, because a
# constant is the thing the rule is written in: NEVER_HZ/NEVER_FRAC put a ramp
# somewhere no measurement can reach, which is exactly "as if this term were
# not there".
from aether_gate.core import kinds                                      # noqa: E402
from aether_gate.tests.test_finder_weak import CENTER as WEAK_CENTER    # noqa: E402
from aether_gate.tests.test_finder_weak import Scene, _at               # noqa: E402
from aether_gate.tests.test_kinds import FLAP_PATH, _replay             # noqa: E402
from aether_gate.tests.test_kinds import needs_flap_capture             # noqa: E402

NEVER_HZ = (1e12, 2e12)          # a width ramp no signal can climb
NEVER_FRAC = (10.0, 11.0)        # ...and a 0..1 one, for the same purpose
ALWAYS_FRAC = (-2.0, -1.0)       # ...and its opposite, for a ramp read downwards


def _kind_at(fd, hz, margin=1500.0):
    c = _at(fd.candidates(WEAK_CENTER)["candidates"], hz, margin)
    return (c["kind"], c["kind_conf"]) if c else (None, None)


def _win(feat, **over):
    """One window's features as a dict of length-1 arrays, ready for verdict()."""
    base = {"snr_db": 8.0, "peak_db": 12.0, "bw_hz": 2000.0, "run_hz": 2400.0,
            "filled": 0.4, "depth": 0.8, "syllabic": 0.7, "occupancy": 0.5,
            "mid": 0.05, "duty": 0.5, "crest": 2.0, "floor_corr": 0.0,
            "peak_frac": 0.3, "shift_hz": 0.0}
    base.update(feat)
    base.update(over)
    out = {k: np.array([float(v)]) for k, v in base.items()}
    out["resolved"] = 1.0
    out["resolves_shift"] = 0.0
    return out


def _verdict(feat):
    code, conf = kinds.verdict(feat)
    return kinds.name(code[0]), float(conf[0])


# --- a block of digital carriers is data, not a conversation ---------------

def test_a_block_of_tones_is_data_and_only_because_it_is_on_all_the_time():
    """A 50 Hz-stepped block of tones filling a 3 kHz stretch. It swings its
    envelope and it is phone-wide, which is most of what makes a talker; what
    it is not is a man who stops to listen."""
    fd = Scene(seed=41).block(14_120_000.0, 9.0).run()
    kind, conf = _kind_at(fd, 14_121_500.0, margin=3_000.0)
    assert kind in kinds.DIGITAL, kind
    assert conf >= 0.7, conf                    # nobody is unsure about a block


# The FT8 window on 20 m as the live gate measured it on 2026-09-03, when it
# called 14074.0 "voice 0.41": phone-wide, on all the time, envelope swinging
# 0.39 with 52% of that swing at syllable rate. Synthesised blocks are flatter
# than this and would pass the rule below without exercising it.
FT8_WINDOW = {"bw_hz": 2600.0, "run_hz": 3000.0, "filled": 0.55, "depth": 0.39,
              "syllabic": 0.52, "occupancy": 1.0, "duty": 0.95, "crest": 1.5,
              "peak_frac": 0.15, "mid": 0.05}


def test_the_measured_ft8_window_is_data_and_not_the_voice_it_shipped_as():
    kind, conf = _verdict(_win(FT8_WINDOW))
    assert kind == "data" and conf >= 0.7, (kind, conf)


def test_without_the_on_all_the_time_rule_the_ft8_window_is_voice_again(monkeypatch):
    """The reported field value, reproduced: "voice", at 0.41."""
    monkeypatch.setattr(kinds, "ONTIME", NEVER_FRAC)
    kind, conf = _verdict(_win(FT8_WINDOW))
    assert kind == "voice" and conf == pytest.approx(0.41, abs=0.05), (kind, conf)


def test_a_block_wider_than_anybody_talks_is_never_a_conversation():
    """Ten kilohertz of filled band, syllables or no syllables. bw_hz cannot
    see this -- every occupied width saturates at the 2.7 kHz window -- so it
    is the contiguous RUN through the window's peak that has to say so."""
    feat = _win({}, run_hz=15_000.0)            # everything else voice-shaped
    assert _verdict(feat)[0] != "voice", _verdict(feat)
    assert _verdict(_win({}))[0] == "voice"     # ...the same window, 2.4 kHz wide


def test_without_the_run_width_rule_the_wide_block_is_called_voice(monkeypatch):
    monkeypatch.setattr(kinds, "SSB_RUN_HZ", NEVER_HZ)
    assert _verdict(_win({}, run_hz=15_000.0))[0] == "voice"


# --- a tone, keyed or held -------------------------------------------------

def test_a_steady_narrow_tone_is_a_carrier():
    fd = Scene(seed=42).carrier(14_110_000.0, 15.0).run()
    kind, conf = _kind_at(fd, 14_110_000.0, 600.0)
    assert kind == "carrier" and conf >= 0.7, (kind, conf)


def test_without_the_steady_envelope_rule_the_tone_is_not_a_carrier(monkeypatch):
    monkeypatch.setattr(kinds, "DEPTH_STEADY", ALWAYS_FRAC)   # nothing reads steady
    fd = Scene(seed=42).carrier(14_110_000.0, 15.0).run()
    assert _kind_at(fd, 14_110_000.0, 600.0)[0] != "carrier"


def test_a_keyed_narrow_tone_is_cw():
    fd = Scene(seed=43).cw(14_090_000.0, 15.0, wpm=20.0).run()
    kind, conf = _kind_at(fd, 14_090_000.0, 600.0)
    assert kind == "cw" and conf >= 0.7, (kind, conf)


def test_without_the_keying_duty_rule_the_keyed_tone_is_not_cw(monkeypatch):
    """Key-down some of the time and not all of it is what separates an
    operator from a tone left switched on."""
    monkeypatch.setattr(kinds, "DUTY_ON", NEVER_FRAC)
    fd = Scene(seed=43).cw(14_090_000.0, 15.0, wpm=20.0).run()
    assert _kind_at(fd, 14_090_000.0, 600.0)[0] != "cw"


# --- somebody talking ------------------------------------------------------

def test_a_syllabic_phone_wide_signal_is_voice():
    fd = Scene(seed=44).voice(14_170_000.0, 9.0).run()
    kind, conf = _kind_at(fd, 14_169_500.0)     # USB: the dial sits below it
    assert kind == "voice" and conf >= 0.7, (kind, conf)


def test_without_the_syllabic_rule_the_talker_is_not_voice(monkeypatch):
    monkeypatch.setattr(kinds, "SYLLABIC_VOICE", NEVER_FRAC)
    fd = Scene(seed=44).voice(14_170_000.0, 9.0).run()
    assert _kind_at(fd, 14_169_500.0)[0] != "voice"


def test_without_the_ssb_width_rule_the_talker_is_not_voice(monkeypatch):
    """Syllables alone are not speech: a keyed tone swings at a few hertz too.
    Voice needs an SSB-shaped passband as well, and this is the lower half of
    that -- at least a kilohertz and a half of it."""
    monkeypatch.setattr(kinds, "WIDE_HZ", NEVER_HZ)
    fd = Scene(seed=44).voice(14_170_000.0, 9.0).run()
    assert _kind_at(fd, 14_169_500.0)[0] != "voice"


# --- the weather is not a station ------------------------------------------

# 3.9875 MHz on the 2026-09-03 04:34 recording as the finder measured it:
# bare band, 1.9 dB, nothing standing 6 dB over the floor under it, and an
# envelope 0.78 correlated with the whole band's -- and it came back "voice".
# It is deliberately WELL present (peak_db 5.6 puts `present` at 0.87), because
# the rule under test is that presence alone is not a station.
QRN_WINDOW = {"snr_db": 1.9, "peak_db": 5.6, "floor_corr": 0.78, "bw_hz": 2000.0,
              "run_hz": 2400.0, "depth": 0.8, "syllabic": 0.7, "duty": 0.5}


def test_a_window_that_moves_with_the_whole_bands_floor_is_not_voice():
    """QRN lifts every window at once. On the 2026-09-03 80 m capture, bare
    band at 0.3-1.9 dB whose envelope tracked the band floor at 0.78-0.98 came
    back "voice 0.82" and "voice 1.00" -- the report, exactly."""
    kind, _conf = _verdict(_win(QRN_WINDOW))
    assert kind != "voice", kind
    # ...and it is not that the window is too weak to name: the same numbers
    # with the band's floor sitting still are a conversation
    assert _verdict(_win(QRN_WINDOW, floor_corr=0.0))[0] == "voice"


def test_without_the_weather_rule_the_qrn_window_is_called_voice(monkeypatch):
    monkeypatch.setattr(kinds, "FLOOR_TRACK", NEVER_FRAC)
    assert _verdict(_win(QRN_WINDOW))[0] == "voice"


# --- what "noise" may and may not be said about ----------------------------

def test_a_narrow_scored_present_line_is_never_called_noise():
    """The other half of the report: 14119.5 kHz, 70 Hz wide, 2.6 dB, listed
    at score 0.91 -- and called "noise". A line standing over the floor UNDER
    it is a signal whatever else is going on; only a window with no line in it
    can be the weather."""
    for floor_corr in (0.0, 0.5, 0.99):
        for depth, duty in ((0.02, 1.0), (0.9, 0.5)):      # held down, and keyed
            feat = _win({}, bw_hz=250.0, run_hz=250.0, peak_frac=0.99,
                        filled=0.09, syllabic=0.15, snr_db=2.6, peak_db=18.0,
                        depth=depth, duty=duty, floor_corr=floor_corr)
            kind, conf = _verdict(feat)
            assert kind != "noise", (floor_corr, depth, kind, conf)
    # ...including the row the report was actually about, where the narrow
    # verdict is only PARTLY earned. 14119.5 kHz was 2.6 dB with a peak 4 dB
    # over its own floor and a peak share of 0.6 -- narrow, certainly there,
    # and not confidently any one thing. "Nothing is here" was scored as
    # 1 - present = 0.47, which beat the 0.23 the carrier verdict earned, and
    # that is how a scored 70 Hz column came back "noise".
    half = _win({}, bw_hz=250.0, run_hz=250.0, peak_frac=0.6, filled=0.09,
                syllabic=0.15, snr_db=2.6, peak_db=4.0, depth=0.05, duty=1.0,
                floor_corr=0.0)
    kind, conf = _verdict(half)
    assert kind != "noise", (kind, conf)
    assert conf == pytest.approx(kinds.SIGNAL_MAX_CONF), (kind, conf)


def test_without_the_line_rule_the_weathered_narrow_line_is_called_noise(monkeypatch):
    monkeypatch.setattr(kinds, "PEAK_PRESENT_DB", NEVER_HZ)   # no peak reads as a line
    feat = _win({}, bw_hz=250.0, run_hz=250.0, peak_frac=0.99, filled=0.09,
                syllabic=0.15, snr_db=2.6, peak_db=18.0, depth=0.02, duty=1.0,
                floor_corr=0.99)
    assert _verdict(feat)[0] == "noise"


def test_noise_is_what_is_said_when_nothing_is_standing_over_the_floor():
    """It still has to be said, and said confidently: bare band is bare band."""
    kind, conf = _verdict(_win({}, snr_db=0.1, peak_db=0.2, depth=0.05,
                               syllabic=0.4, bw_hz=100.0, run_hz=0.0))
    assert kind == "noise" and conf >= 0.7, (kind, conf)


# --- the confidence means something ----------------------------------------

def test_a_split_verdict_is_signal_at_a_coin_tosss_confidence():
    """Two kinds that both half-fit is not a verdict. It used to need BOTH a
    low winner and a close runner-up before it would admit that; either is
    enough, and the number that ships says so."""
    tie = _win({}, depth=0.30, syllabic=0.45, bw_hz=1500.0, filled=0.5,
               duty=0.5, mid=0.5, peak_frac=0.3)
    kind, conf = _verdict(tie)
    assert kind == "signal", (kind, conf)
    assert conf == pytest.approx(kinds.SIGNAL_MAX_CONF), conf
    assert kinds.SIGNAL_MAX_CONF <= 0.4                  # a coin toss reads like one


@pytest.mark.parametrize("scene,hz,margin", [
    (lambda: Scene(seed=45).voice(14_170_000.0, 12.0), 14_169_500.0, 1500.0),
    (lambda: Scene(seed=45).cw(14_090_000.0, 15.0), 14_090_000.0, 600.0),
    (lambda: Scene(seed=45).carrier(14_110_000.0, 15.0), 14_110_000.0, 600.0),
    (lambda: Scene(seed=45).block(14_120_000.0, 9.0), 14_121_500.0, 3000.0),
])
def test_a_kind_worth_betting_on_ships_at_seven_tenths_or_better(scene, hz, margin):
    """The calibration the report asked for: rows you would bet on at 0.7 and
    up, coin tosses at 0.4 and down, and nothing in between pretending. The
    FT8-shaped block is the one that used to fail this -- its data and noise
    scores were the same expression and tied at 0.455, so it shipped at 0.23."""
    kind, conf = _kind_at(scene().run(), hz, margin)
    assert kind is not None and conf >= 0.7, (kind, conf)


# --- and the same question put to a recording ------------------------------

@needs_flap_capture
def test_on_the_recorded_80m_evening_nothing_under_two_decibels_is_a_conversation():
    """The regression, off-air. Replaying 2026-09-03 04:34 through the finder
    before this change, three of its candidates at 0.5, 1.1 and 1.9 dB -- bare
    band tracking its own floor at floor_corr 0.78-0.94, with nothing standing
    over the local floor at all -- came back "voice". Every real talker on the
    recording is 4.7 dB or better."""
    fd, center = _replay(FLAP_PATH)
    out = fd.candidates(center)
    cands = out["candidates"]
    assert len([c for c in cands if c["kind"] == "voice"]) >= 4, cands
    weak_voice = [c for c in cands if c["kind"] == "voice" and c["snr_db"] < 2.0]
    assert not weak_voice, weak_voice
    # ...and the `voice_share` strip the app paints says the same thing. The
    # score that fills it is the finder's own voice score, which had no
    # weather term at all: 3.9706 and 3.9686 MHz -- 0.2-0.3 dB of bare band
    # whose envelope is 0.97 correlated with the band floor -- were painted a
    # third of the time as somebody talking, and 3.9875 nearly always.
    # ...and the three windows the report was about, named. 3.9847, 3.9875 and
    # 3.9901 MHz are bare band on this recording -- 0.5-1.9 dB, nothing more
    # than 5.6 dB over the local floor under them, envelopes 0.78-0.94
    # correlated with the whole band's, occupied runs of 0-732 Hz -- against
    # the real talkers at 3.9230 and 3.9500, which read 10.9-14.7 dB with
    # floor_corr about zero and three-kilohertz runs. All three came back
    # "voice" at 0.82-1.00, off the one row in thirty each scored best on.
    quiet = [c for c in cands if 3_984_000.0 <= c["hz"] <= 3_991_000.0]
    assert quiet, "the QRN stretch dropped out of the list entirely"
    assert not [c for c in quiet if c["kind"] == "voice"], quiet
    vs = np.asarray(out["voice_share"])
    step = fd.rate_hz / out["points"]
    col = lambda hz: int((hz - (center - fd.rate_hz / 2)) / step)
    for hz in (3_970_600.0, 3_968_600.0):
        assert vs[col(hz)] <= 0.1, (hz, vs[col(hz)])
    assert vs[col(3_987_500.0)] <= 0.6, vs[col(3_987_500.0)]
    assert vs[col(3_950_000.0)] >= 0.8, vs[col(3_950_000.0)]   # the real talker
