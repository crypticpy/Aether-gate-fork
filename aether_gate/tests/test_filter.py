#
# Aether-gate — the receive filter (core/filter.py) and its seat in the
# soapy adapter's audio path.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Run:  python -m pytest aether_gate/tests/test_filter.py"""
import numpy as np
import pytest

from aether_gate.core.filter import (SliceFilter, Agc, blank_impulses, design_taps,
                                     ANF_WIDTH_HZ)
from aether_gate.adapters.soapy import SoapyAdapter, AUDIO_RATE

RATE = 25000.0


def _resp_db(taps, hz):
    k = np.arange(len(taps)) - (len(taps) - 1) / 2.0
    return 20 * np.log10(abs(np.sum(taps * np.exp(-2j * np.pi * hz / RATE * k))) + 1e-12)


def test_sharp_and_soft_passbands_pass_the_voice_and_stop_the_neighbour():
    sharp = design_taps(RATE, 300, 2700, "sharp")
    soft = design_taps(RATE, 300, 2700, "soft")
    for taps in (sharp, soft):
        assert abs(_resp_db(taps, 1000)) < 0.5
        assert _resp_db(taps, 2400) > -2.0
        assert _resp_db(taps, -1000) < -40             # the other sideband
    assert abs(_resp_db(sharp, 2600)) < 0.5
    assert _resp_db(sharp, 2850) < -40                 # 150 Hz past the edge
    assert _resp_db(sharp, 3500) < -60
    assert _resp_db(soft, 2800) > -30                  # soft skirts are soft
    assert _resp_db(soft, 2600) > -2.0                 # but the top 100 Hz is still there
    assert _resp_db(soft, 4000) < -40
    # LSB is the mirror
    lsb = design_taps(RATE, -2700, -300, "sharp")
    assert abs(_resp_db(lsb, -1000)) < 0.5 and _resp_db(lsb, 1000) < -40


def test_notch_contour_apf_and_tilt_are_in_the_same_taps():
    notched = design_taps(RATE, 300, 2700, "sharp", notches=[(1000, 120)])
    assert _resp_db(notched, 1000) < -30
    assert abs(_resp_db(notched, 800)) < 1.0 and abs(_resp_db(notched, 1200)) < 1.0
    contour = design_taps(RATE, 300, 2700, "sharp", contour=(1200, 6.0, 600))
    assert 5.0 < _resp_db(contour, 1200) < 7.0
    assert abs(_resp_db(contour, 2400)) < 1.0
    apf = design_taps(RATE, 300, 2700, "sharp", apf=(600, 150))
    assert abs(_resp_db(apf, 600)) < 0.5
    assert _resp_db(apf, 900) < -20
    tilt = design_taps(RATE, 300, 2700, "sharp", tilt_db=6.0)
    # a print that tilts +6 dB toward the highs is leaned back: lows up, highs down
    assert _resp_db(tilt, 550) - _resp_db(tilt, 2000) > 4.0


def _feed(sf, seconds, tones, lsb=False, noise=0.001):
    """Run seconds of a signal through the filter in 819-sample blocks;
    tones are (hz, amp). Returns the concatenated output."""
    out = []
    n = 819
    t0 = 0
    rng = np.random.default_rng(1)
    for _ in range(int(seconds * RATE) // n):
        t = (t0 + np.arange(n)) / RATE
        x = sum(a * np.exp(2j * np.pi * hz * t) for hz, a in tones)
        x = x + noise * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        out.append(sf.apply(x.astype(np.complex128), 0, lsb=lsb))
        t0 += n
    return np.concatenate(out)


def _level_db(y, hz):
    y = y[-8192:]
    t = np.arange(len(y)) / RATE
    return 20 * np.log10(abs(np.mean(y * np.exp(-2j * np.pi * hz * t))) + 1e-12)


def _voice_like(seconds, amp, seed=3):
    """Band-limited noise 300-2700 Hz: a stand-in for a voice, nothing tonal."""
    rng = np.random.default_rng(seed)
    total = int(seconds * RATE)
    w = amp * (rng.standard_normal(total + 1100) + 1j * rng.standard_normal(total + 1100))
    return np.convolve(w, design_taps(RATE, 300, 2700, "sharp"), mode="valid")[:total]


def _feed_signal(sf, sig):
    out = []
    n = 819
    for i in range(0, len(sig) - n + 1, n):
        out.append(sf.apply(sig[i:i + n], 0))
    return np.concatenate(out)


def test_auto_notch_finds_a_steady_tone_and_lets_it_go():
    sf = SliceFilter(RATE)
    sf.set(low=300, high=2700, shape="sharp", anf=True)
    voice = _voice_like(3.0, 0.02)
    t = np.arange(len(voice)) / RATE
    y = _feed_signal(sf, voice + 0.05 * np.exp(2j * np.pi * 1700 * t))
    st = sf.status()
    assert st["anf"]["found_hz"] and abs(st["anf"]["found_hz"][0] - 1700) < ANF_WIDTH_HZ
    assert st["anf"]["depth_db"][0] > 30
    # the tone (-26 dB in) has lost 30 dB or more; the voice beside it is still there
    assert _level_db(y, 1700) < -56
    assert _level_db(y, 1300) > -90
    _feed_signal(sf, _voice_like(3.0, 0.02, seed=4))
    assert sf.status()["anf"]["found_hz"] == []


def test_auto_width_follows_the_occupied_spectrum_then_the_print():
    sf = SliceFilter(RATE)
    sf.set(low=100, high=3200, auto=True)
    rng = np.random.default_rng(2)
    n = 819
    total = (int(3.0 * RATE) // n) * n
    w = 0.05 * (rng.standard_normal(total + 1100) + 1j * rng.standard_normal(total + 1100))
    voice = np.convolve(w, design_taps(RATE, 350, 2100, "sharp"), mode="valid")[:total]
    voice = voice + 0.0005 * (rng.standard_normal(total) + 1j * rng.standard_normal(total))
    for i in range(0, total, n):
        sf.apply(voice[i:i + n], 0)
    st = sf.status()
    assert st["auto"]["source"] == "spectrum"
    assert 150 <= st["auto"]["low_hz"] <= 400
    # the voice reaches 2100; the top is held at AUTO_MIN_HIGH_HZ, never closed onto it
    assert 2400 <= st["auto"]["high_hz"] <= 2450
    assert st["low_hz"] == st["auto"]["low_hz"]          # the edges in use are the auto ones
    # a print for the talker may widen the fit (its edges say the rig reaches further)
    sf.print_source = lambda: {"low_hz": 300, "high_hz": 2600, "tilt_db": -3.0}
    for _ in range(int(2.0 * RATE) // n):
        sf.apply(0.0005 * (rng.standard_normal(n) + 1j * rng.standard_normal(n)), 0)
    st = sf.status()
    assert st["auto"]["source"] == "print"
    assert 240 <= st["auto"]["low_hz"] <= 300 and 2650 <= st["auto"]["high_hz"] <= 2750
    sf.set(auto=False)
    assert sf.status()["low_hz"] == 100 and sf.status()["auto"]["low_hz"] is None


def test_auto_width_never_closes_the_bottom_onto_a_thin_channel():
    # 80 m at night, 2026-09-02: the 1 s average only cleared the floor from
    # 1.2 kHz up and AUTO put the low edge at 1233 Hz, taking the
    # fundamentals and first formant with it. The bottom never closes above
    # AUTO_MAX_LOW_HZ, as the top never closes below AUTO_MIN_HIGH_HZ.
    sf = SliceFilter(RATE)
    sf.set(low=100, high=3200, auto=True)
    rng = np.random.default_rng(5)
    n = 819
    total = (int(3.0 * RATE) // n) * n
    w = 0.05 * (rng.standard_normal(total + 1100) + 1j * rng.standard_normal(total + 1100))
    thin = np.convolve(w, design_taps(RATE, 1200, 2100, "sharp"), mode="valid")[:total]
    thin = thin + 0.0005 * (rng.standard_normal(total) + 1j * rng.standard_normal(total))
    for i in range(0, total, n):
        sf.apply(thin[i:i + n], 0)
    st = sf.status()
    assert st["auto"]["source"] == "spectrum"
    assert st["auto"]["low_hz"] <= 350, st["auto"]
    assert 2400 <= st["auto"]["high_hz"] <= 2450


def test_auto_eq_leans_against_the_prints_tilt():
    sf = SliceFilter(RATE, print_source=lambda: {"low_hz": 300, "high_hz": 2700, "tilt_db": 5.0})
    sf.set(low=300, high=2700, shape="sharp", auto_eq=True)
    _feed(sf, 1.0, [(1000, 0.01)])
    assert sf.status()["auto_eq"]["tilt_db"] == 5.0
    # +5 dB is 11 dB brighter than a normal station's -6; half of that comes off the highs
    assert sf.status()["auto_eq"]["lean_db"] == 5.5
    assert _resp_db(sf.taps, 550) - _resp_db(sf.taps, 2000) > 3.5
    # a normal station is left alone
    sf.print_source = lambda: {"low_hz": 300, "high_hz": 2700, "tilt_db": -6.0}
    _feed(sf, 1.0, [(1000, 0.01)])
    assert sf.status()["auto_eq"]["lean_db"] == 0.0


def test_agc_attacks_fast_hangs_then_decays():
    agc = Agc(target=0.25, rate_hz=1000.0)
    agc.set("med", attack_ms=5, decay_ms=200, hang_ms=100, threshold_db=0)   # 0: a plain leveller
    loud = np.full(20, 0.5)
    quiet = np.full(20, 0.05)
    for _ in range(10):
        agc.process(loud)
    g_loud = agc.gain
    assert abs(g_loud * 0.5 - 0.25) < 0.05                # settled on target within 200 ms
    gains = []
    for _ in range(20):
        agc.process(quiet)
        gains.append(agc.gain)
    assert abs(gains[3] - g_loud) < 1e-6                  # hang: 80 ms in, gain unchanged
    assert gains[-1] > g_loud * 2                          # then the decay lets it rise
    agc.set("off")
    assert agc.status()["mode"] == "off"


def test_agc_threshold_leaves_the_noise_between_words_down():
    # 20 ms chunks at 1 kS/s; speech at 0.5, the band between words 30 dB under it
    agc = Agc(target=0.25, rate_hz=1000.0)
    agc.set("med", attack_ms=5, decay_ms=100, hang_ms=0)
    assert agc.status()["threshold_db"] == 20.0
    rng = np.random.default_rng(1)
    noise = 0.5 * 10 ** (-30 / 20) * rng.standard_normal(20)
    for _ in range(20):
        agc.process(noise)                                  # the floor tracker learns the band
    for _ in range(10):
        agc.process(np.full(20, 0.5))
    g_speech = agc.gain
    out = None
    for _ in range(60):                                     # 1.2 s of silence: the decay runs out
        out = agc.process(noise)
    floor_out = float(np.sqrt(np.mean(out * out)))
    # the noise is not levelled up to the target: it stays >= 20 dB under it
    assert floor_out < 0.25 * 10 ** (-20 / 20) * 1.3
    assert agc.gain < g_speech * 10 ** (30 / 20)          # a leveller would have gone the full 30 dB
    agc.set(threshold_db=0)
    for _ in range(60):
        out = agc.process(noise)
    assert float(np.sqrt(np.mean(out * out))) > 0.15      # 0 IS the leveller: noise up to target
    with pytest.raises(ValueError):
        agc.set(threshold_db=90)


def test_blanker_removes_the_impulses_and_keeps_the_tone():
    t = np.arange(4096) / 125000.0
    x = 0.01 * np.exp(2j * np.pi * 1000 * t)
    x[500] += 1.0
    x[2000] += 0.8
    y, frac = blank_impulses(x, 12.0)
    assert 0 < frac < 0.01
    assert abs(y[500]) == 0 and abs(y[2000]) == 0
    assert abs(abs(y[100]) - 0.01) < 1e-6
    y2, frac2 = blank_impulses(0.01 * np.exp(2j * np.pi * 1000 * t), 12.0)
    assert frac2 == 0.0


def test_settings_are_validated():
    sf = SliceFilter(RATE)
    for bad in ({"shape": "medium"}, {"agc": "auto"}, {"nb_db": 50}, {"low": 1000, "high": 1020},
                {"what": 1}):
        try:
            sf.set(**bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")
    sf.set(shape="sharp")
    for _ in range(6):
        sf.notch_add(500 + _ * 200)
    try:
        sf.notch_add(2000)
    except ValueError:
        pass
    else:
        raise AssertionError("seventh notch accepted")
    sf.notch_clear(500)
    assert len(sf.spec.notches) == 5
    assert all(n["width_hz"] >= 100 for n in sf.spec.notches)
    assert sf.status()["notches"][0]["depth_db"] > 25
    sf.notch_clear()
    assert sf.spec.notches == []


# ----- through the adapter's audio path ---------------------------------------
SAMP = 125_000.0
CHUNK = 480


def _adapter():
    a = SoapyAdapter(driver="sdrplay", samp_rate=SAMP, center_hz=7_100_000.0)
    a._np = np
    a._init_demod()
    a._mode = "USB"
    return a


def _feed_iq(a, seconds, tones, block=4096):
    t0 = 0
    for _ in range(int(seconds * SAMP) // block):
        t = (t0 + np.arange(block)) / SAMP
        x = sum(amp * np.exp(2j * np.pi * hz * t) for hz, amp in tones)
        a._audio_q.append(x.astype(np.complex64))
        t0 += block


def _pull(a, n_chunks):
    out = []
    for _ in range(n_chunks):
        c = a.get_audio(CHUNK, slice_id=0)
        if c is not None:
            out.extend(c)
    return np.asarray(out, dtype=float)


def _tone_db(sig, hz):
    sig = sig[-AUDIO_RATE:]
    t = np.arange(len(sig)) / AUDIO_RATE
    return 20 * np.log10(abs(np.mean(sig * np.exp(-2j * np.pi * hz * t))) + 1e-12)


def test_the_app_edges_reach_the_audio_and_the_meter():
    a = _adapter()
    a.set_filter_edges_hz(300.0, 2700.0)
    a.filter_set(shape="sharp", agc="off")
    _feed_iq(a, 2.0, [(1000, 0.001), (3500, 0.001)])
    out = _pull(a, 2 * AUDIO_RATE // CHUNK)
    assert _tone_db(out, 1000) - _tone_db(out, 3500) > 40
    assert a._meter_band_hz() == (300.0, 2700.0)
    st = a.filter_status()
    assert st["available"] and st["width_hz"] == 2400 and st["shape"] == "sharp"
    assert st["roofing"]["analogue_hz"] == 200e3 and st["roofing"]["digital_hz"] == 25000
    assert len(st["response"]["hz"]) == len(st["response"]["db"]) == 128
    # LSB: the same audio edges, the mirrored passband
    a._mode = "LSB"
    a.set_filter_edges_hz(-2700.0, -300.0)
    assert a._meter_band_hz() == (-2700.0, -300.0)
    _feed_iq(a, 2.0, [(-1000, 0.001), (1000, 0.001)])
    out = _pull(a, 2 * AUDIO_RATE // CHUNK)
    assert _tone_db(out, 1000) > -40                      # -1 kHz IQ is 1 kHz audio on LSB
    # width-only (IC-7300 contract) keeps the centre
    a.set_filter_width_hz(1000.0)
    assert a.filter_status()["low_hz"] == 1000 and a.filter_status()["high_hz"] == 2000


def test_the_blanker_sits_ahead_of_the_filter_in_the_adapter():
    a = _adapter()
    a.set_filter_edges_hz(300.0, 2700.0)
    a.filter_set(nb=True, nb_db=12, agc="off")
    t0 = 0
    for _ in range(int(2.0 * SAMP) // 4096):
        t = (t0 + np.arange(4096)) / SAMP
        x = 0.001 * np.exp(2j * np.pi * 1000 * t)
        x[100] += 0.5                                       # one spike per block
        a._audio_q.append(x.astype(np.complex64))
        t0 += 4096
    _pull(a, 2 * AUDIO_RATE // CHUNK)
    st = a.filter_status()
    assert st["nb"]["enabled"] and 0.05 < st["nb"]["blanked_pct"] < 1.0


def test_with_a_tuner_pair_the_filters_nb_is_the_pairs_blanker():
    """The RSPduo pair is blanked once, in _DiversityState.ingest, before the
    spatial map and the trackers see it. /filter's NB must be that knob,
    not a second blanker on the slice."""
    from aether_gate.adapters.soapy import _DiversityState
    a = _adapter()
    a._div = _DiversityState(a)
    a._div.BEACONS_PATH = None
    st = a.filter_set(nb=True, nb_db=15)
    assert a._div.nb_on is True and a._div.nb_db == 15.0
    assert st["nb"]["enabled"] is True and st["nb"]["threshold_db"] == 15.0
    assert a._filt.spec.nb_on is False                    # the slice blanker stays off
    a._div.blanked_pct = 0.37
    assert a.filter_status()["nb"]["blanked_pct"] == 0.37
    a.filter_set(nb=False, shape="soft")
    assert a._div.nb_on is False and a.filter_status()["shape"] == "soft"


def test_the_status_carries_what_arrives_ahead_of_the_filter():
    """The FILTER page draws the filter over the live spectrum: a tone at
    1 kHz is the peak (0 dB) on response_db's grid, the floor sits well
    below it, and an LSB slice reads the same audio frequencies."""
    a = _adapter()
    assert a.filter_status()["spectrum"] is None                # nothing heard yet
    a.set_filter_edges_hz(300.0, 2700.0)
    a.filter_set(agc="off")
    _feed_iq(a, 1.5, [(1000.0, 0.01)])
    _pull(a, AUDIO_RATE // CHUNK)
    sp = a.filter_status()["spectrum"]
    assert sp["hz"] == a.filter_status()["response"]["hz"]
    k = int(np.argmax(sp["db"]))
    assert abs(sp["hz"][k] - 1000.0) <= 40.0 and sp["db"][k] == 0.0, (sp["hz"][k], sp["db"][k])
    assert sp["floor_db"] < -20.0 and min(sp["db"]) >= -120.0
    a._mode = "LSB"
    a._init_demod()
    a.set_filter_edges_hz(-2700.0, -300.0)
    _feed_iq(a, 1.5, [(-1000.0, 0.01)])                          # 1 kHz audio on LSB
    _pull(a, AUDIO_RATE // CHUNK)
    sp = a.filter_status()["spectrum"]
    k = int(np.argmax(sp["db"]))
    assert abs(sp["hz"][k] - 1000.0) <= 40.0, sp["hz"][k]
