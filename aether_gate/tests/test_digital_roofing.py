#
# Aether-gate — the digital roofing filter (core/roofing.py) and its seat
# ahead of the slice filter.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The stage that finally gives an FT-101MP operator the roofing menu they
asked for, because the analogue one cannot go below 200 kHz on this hardware.

Three things have to be true of it, and each is a test below: the response
is a real filter (flat where it passes, 40 dB down where it stops), it is
phase-continuous across block boundaries (a long block and the same samples
in pieces come out identical, or every block boundary is a click), and it
costs a small fraction of the real-time budget, because it runs in the audio
path on every block of both loops.

Run:  python -m pytest aether_gate/tests/test_digital_roofing.py
"""
import time

import numpy as np
import pytest

from aether_gate.adapters.soapy import AUDIO_RATE, SoapyAdapter
from aether_gate.core.roofing import (DIGITAL_ROOF_PRESETS, DigitalRoof,
                                      ROOF_MAX_TAPS, roof_ntaps, roof_taps,
                                      validate_digital_roof_hz)

RATE = 25000.0
BLOCK = 410            # a 4096-sample read at 250 kS/s, decimated by 10


def _resp_db(taps, hz):
    k = np.arange(len(taps)) - (len(taps) - 1) / 2.0
    return 20 * np.log10(abs(np.sum(taps * np.exp(-2j * np.pi * hz / RATE * k))) + 1e-15)


# ----- the menu ---------------------------------------------------------------

def test_the_presets_are_the_widths_the_radios_people_know_carry():
    assert DIGITAL_ROOF_PRESETS[0] == 25000            # off: the whole band
    assert DIGITAL_ROOF_PRESETS == sorted(DIGITAL_ROOF_PRESETS, reverse=True)
    for want in (12000, 3000, 1200, 600, 300):         # the FTdx101 menu
        assert want in DIGITAL_ROOF_PRESETS
    for want in (2800, 2700, 1800, 1000, 500, 400, 250, 200):    # the K3's
        assert want in DIGITAL_ROOF_PRESETS


def test_free_entry_runs_100_hz_to_25_khz_and_nothing_outside_it():
    assert validate_digital_roof_hz(100) == 100.0
    assert validate_digital_roof_hz("1750") == 1750.0
    assert validate_digital_roof_hz(25000) == 25000.0
    for bad in (99.9, 0, -3000, 25001, 48000):
        with pytest.raises(ValueError):
            validate_digital_roof_hz(bad)


# ----- the response -----------------------------------------------------------

def test_every_preset_passes_its_own_band_flat_and_stops_1_5x_past_the_edge():
    for hz in DIGITAL_ROOF_PRESETS:
        if 2 * hz >= RATE or 1.5 * hz > RATE / 2:
            continue                    # wider than the band, or its stopband wraps
        taps = roof_taps(RATE, hz)
        flat = max(abs(_resp_db(taps, f)) for f in np.linspace(0.0, 0.7 * hz, 40))
        stop = max(_resp_db(taps, f)
                   for f in np.linspace(1.5 * hz, RATE / 2.0, 120))
        assert flat < 0.5, (hz, flat)
        assert stop < -40.0, (hz, stop)
        assert _resp_db(taps, -hz * 0.5) == pytest.approx(_resp_db(taps, hz * 0.5), abs=1e-6)


def test_the_tap_count_is_bounded_because_this_runs_in_the_audio_path():
    assert roof_ntaps(RATE, 3000) < 64
    assert roof_ntaps(RATE, 200) == ROOF_MAX_TAPS
    assert roof_ntaps(RATE, 100) == ROOF_MAX_TAPS         # the cap, not a bigger FIR
    assert all(roof_ntaps(RATE, hz) % 2 == 1 for hz in DIGITAL_ROOF_PRESETS)


# ----- per block --------------------------------------------------------------

def _noise(n, seed=7):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex128)


def test_the_filter_state_carries_across_blocks_so_nothing_clicks():
    """One long block and the same samples fed in pieces are the same signal."""
    sig = _noise(4 * BLOCK)
    whole = DigitalRoof(RATE, 1200).apply(sig.copy())
    piecemeal = DigitalRoof(RATE, 1200)
    parts = [piecemeal.apply(sig[i:i + BLOCK].copy()) for i in range(0, len(sig), BLOCK)]
    assert np.allclose(whole, np.concatenate(parts), atol=1e-12)


def test_each_loop_of_a_pair_keeps_its_own_state():
    roof = DigitalRoof(RATE, 600)
    a, b = _noise(BLOCK, seed=1), _noise(BLOCK, seed=2)
    roof.apply(a, ch=0)
    roof.apply(b, ch=1)
    solo = DigitalRoof(RATE, 600)
    solo.apply(a.copy())
    assert np.allclose(roof.apply(a.copy(), ch=0), solo.apply(a.copy()))


def test_a_width_wider_than_the_band_is_off_and_passes_the_samples_untouched():
    for hz in (None, 15000, 25000):
        roof = DigitalRoof(RATE, hz)
        assert not roof.active and roof.status()["taps"] == 0
        sig = _noise(BLOCK)
        assert roof.apply(sig) is sig


def test_a_narrow_roof_takes_the_tone_outside_it_down_and_leaves_the_one_inside():
    roof = DigitalRoof(RATE, 1200)
    t = np.arange(8192) / RATE
    for _ in range(2):                       # let the state fill
        inside = roof.apply(np.exp(2j * np.pi * 500.0 * t))
        outside = roof.apply(np.exp(2j * np.pi * 4000.0 * t))
    assert 20 * np.log10(np.mean(abs(inside))) > -0.5
    assert 20 * np.log10(np.mean(abs(outside))) < -40.0


def test_the_worst_case_roof_costs_a_fraction_of_the_block():
    """511 taps, both loops, on the ~410-sample block the gate runs at 25 kS/s
    inside a 16.4 ms budget. The number is printed so a regression is visible
    even when the assertion still passes."""
    roof = DigitalRoof(RATE, 200)
    blocks = [_noise(BLOCK, seed=i) for i in range(50)]
    for ch in (0, 1):
        roof.apply(blocks[0].copy(), ch)     # design the taps outside the timing
    t0 = time.perf_counter()
    for blk in blocks:
        roof.apply(blk, 0)
        roof.apply(blk, 1)
    per_block_ms = (time.perf_counter() - t0) / len(blocks) * 1000.0
    print(f"\ndigital roof, 511 taps, both loops: {per_block_ms:.3f} ms per "
          f"{BLOCK}-sample block (budget 16.4 ms)")
    assert per_block_ms < 8.0


# ----- through the adapter ----------------------------------------------------
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


def test_the_roof_is_ahead_of_the_slice_filter_and_survives_its_bypass():
    """The whole point of a separate stage: "turn all the filters off" still
    leaves the operator inside a roofing bandwidth."""
    a = _adapter()
    assert a._pd_rate == 25000.0
    a.filter_set(agc="off", bypass=True, digital_roof_hz=1200.0)
    _feed_iq(a, 2.0, [(500, 0.001), (4000, 0.001)])
    out = _pull(a, 2 * AUDIO_RATE // CHUNK)
    assert _tone_db(out, 500) - _tone_db(out, 4000) > 40


def test_the_status_and_the_chain_row_report_the_width_that_is_in_force():
    a = _adapter()
    st = a.filter_set(digital_roof_hz=3000.0)
    assert st["roofing"]["digital_hz"] == 3000
    assert st["roofing"]["digital_active"] is True
    assert st["roofing"]["digital_options"] == DIGITAL_ROOF_PRESETS
    row = next(r for r in st["chain"] if r["id"] == "roof_digital")
    assert row["kind"] == "select" and row["value"] == 3000
    assert row["enabled"] and row["detail"] == "3 kHz · 37 taps"
    assert row["action"]["query"] == "digital_roof_hz="
    st = a.filter_set(digital_roof_hz=25000.0)
    row = next(r for r in st["chain"] if r["id"] == "roof_digital")
    assert not row["enabled"] and row["detail"] == "off · the full 25 kHz"
    with pytest.raises(ValueError):
        a.filter_set(digital_roof_hz=30.0)


def test_the_width_survives_a_sample_rate_change():
    """_init_demod rebuilds the chain for the new rate; the operator's roofing
    width is theirs, not the rate's."""
    a = _adapter()
    a.filter_set(digital_roof_hz=1800.0)
    a.samp_rate = 250_000.0
    a._init_demod()
    assert a._pd_rate == 25000.0
    assert a._roof.hz == 1800.0
    assert a.filter_status()["roofing"]["digital_hz"] == 1800
