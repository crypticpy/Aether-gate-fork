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
                                      ROOF_MAX_TAPS, offset_max_hz, roof_ntaps,
                                      roof_taps, validate_digital_roof_hz)

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


# ----- PEAK OFFSET -------------------------------------------------------
# The roof's centre dragged off the slice centre: a strong neighbour lands
# on the skirt instead of inside the roof, without moving anything the
# slice filter (downstream) sees. core/roofing.py's own docstring has the
# proof that the shift-down/filter/shift-back implementation is exactly a
# redesigned, shifted-centre filter -- these tests are the behaviour that
# proof is for.

def test_the_offset_shift_leaves_a_tone_at_the_slice_centre_where_it_is():
    """A frequency SHIFT, not a redesign that moves content: a tone at the
    slice's own +500 Hz stays at +500 Hz whichever way the roof is offset."""
    t = np.arange(4 * BLOCK) / RATE
    tone = np.exp(2j * np.pi * 500.0 * t)
    for off in (800.0, -800.0):
        roof = DigitalRoof(RATE, 3000.0, offset_hz=off, offset_on=True)
        out = np.concatenate([roof.apply(tone[i:i + BLOCK].copy())
                              for i in range(0, len(tone), BLOCK)])
        tail = out[-BLOCK:]
        tt = np.arange(len(tail)) / RATE
        at_500 = abs(np.mean(tail * np.exp(-2j * np.pi * 500.0 * tt)))
        assert 20 * np.log10(at_500) > -1.0, (off, at_500)


def test_offset_moves_a_neighbour_from_inside_the_roof_to_its_skirt():
    """Just outside the OFFSET roof, inside the CENTRED one: >= 40 dB down
    with the check mark on, < 3 dB down with it off -- the whole point of
    dragging the roof's centre away from a neighbour the passband edges
    alone cannot exclude."""
    hz = 2000.0
    tone_hz = hz / 2.0 + 200.0                          # 1200 Hz
    t = np.arange(8192) / RATE
    tone = np.exp(2j * np.pi * tone_hz * t)

    centred = DigitalRoof(RATE, hz, offset_hz=0.0, offset_on=False)
    for _ in range(2):
        out = centred.apply(tone)
    atten_off = -20 * np.log10(np.mean(abs(out)))
    assert atten_off < 3.0, atten_off

    offset = DigitalRoof(RATE, hz, offset_hz=-2200.0, offset_on=True)
    for _ in range(2):
        out = offset.apply(tone)
    atten_on = -20 * np.log10(np.mean(abs(out)))
    assert atten_on >= 40.0, atten_on


def test_apply_ignores_the_remembered_offset_while_the_check_mark_is_off():
    """The check mark, not just the status dict: apply() itself must not
    shift when offset_on is False, even though a nonzero offset_hz is sat
    there remembered -- the same neighbour that the offset roof would spare
    (see test_offset_moves_a_neighbour_from_inside_the_roof_to_its_skirt)
    must still be caught the way a centred roof catches it."""
    hz = 2000.0
    tone_hz = hz / 2.0 + 200.0
    t = np.arange(8192) / RATE
    tone = np.exp(2j * np.pi * tone_hz * t)

    remembered_but_off = DigitalRoof(RATE, hz, offset_hz=-2200.0, offset_on=False)
    for _ in range(2):
        out = remembered_but_off.apply(tone)
    atten = -20 * np.log10(np.mean(abs(out)))
    assert atten < 3.0, atten          # NOT spared -- the offset is not in force


def test_offset_max_hz_is_the_clamp_formula_and_zero_when_the_roof_cannot_cover_it():
    assert offset_max_hz(3000.0, 2000.0) == pytest.approx(500.0)     # (3000-2000)/2
    assert offset_max_hz(2050.0, 2050.0) == 0.0                      # exactly no room
    assert offset_max_hz(1500.0, 2000.0) == 0.0                      # too narrow: never negative
    assert offset_max_hz(None, 2000.0) == 0.0
    assert offset_max_hz(3000.0, None) == 0.0


def test_the_check_mark_remembers_the_offset_while_off():
    roof = DigitalRoof(RATE, 2000.0)
    roof.set_offset_hz(900.0)
    assert roof.offset_hz == 900.0 and not roof.offset_on
    assert roof.status(offset_max_hz=1000.0)["offset_applied_hz"] == 0
    roof.set_offset_on(True)
    assert roof.status(offset_max_hz=1000.0)["offset_applied_hz"] == 900
    roof.set_offset_on(False)                      # back off: still remembers 900
    st = roof.status(offset_max_hz=1000.0)
    assert st["offset_hz"] == 900 and st["offset_applied_hz"] == 0


def test_the_status_carries_exactly_the_four_new_roofing_keys():
    roof = DigitalRoof(RATE, 1200.0, offset_hz=300.0, offset_on=True)
    st = roof.status(offset_max_hz=450.0)
    for k in ("offset_hz", "offset_enabled", "offset_applied_hz", "offset_max_hz"):
        assert k in st, k
    assert st["offset_hz"] == 300 and st["offset_enabled"] is True
    assert st["offset_applied_hz"] == 300 and st["offset_max_hz"] == 450
    # too narrow for the passband it was asked about: held at 0 either way
    off_roof = DigitalRoof(RATE, 1200.0, offset_hz=300.0, offset_on=True)
    st2 = off_roof.status(offset_max_hz=0.0)
    assert st2["offset_hz"] == 300              # the raw value is not mutated by status()
    assert st2["offset_applied_hz"] == 300       # status() reports what apply() would do,
    # -- the "hold at 0" for a too-narrow roof is set_digital_roof_offset_hz's
    # job (it clamps what gets INTO offset_hz); status() never invents a
    # value it was not given, it only decides whether offset_hz counts as
    # applied.


# ----- through the adapter: the routes, the clamp, and the chain card ----

def test_the_route_clamps_a_requested_offset_into_range():
    a = _adapter()
    a.filter_set(low=350.0, high=2400.0)            # width 2050
    a.filter_set(digital_roof_hz=3000.0)            # max = (3000-2050)/2 = 475
    st = a.filter_set(roof_offset_hz=900.0, roof_offset=True)
    r = st["roofing"]
    assert r["offset_max_hz"] == 475
    assert r["offset_hz"] == 475 and r["offset_applied_hz"] == 475
    st = a.filter_set(roof_offset_hz=-9000.0)
    assert st["roofing"]["offset_hz"] == -475


def test_the_route_holds_the_offset_at_0_when_the_roof_is_too_narrow_for_the_passband():
    a = _adapter()
    a.filter_set(low=100.0, high=2900.0)            # width 2800 (SliceFilter's own default)
    a.filter_set(digital_roof_hz=1200.0)            # narrower than the passband
    st = a.filter_set(roof_offset_hz=5000.0, roof_offset=True)
    r = st["roofing"]
    assert r["offset_max_hz"] == 0
    assert r["offset_hz"] == 0 and r["offset_applied_hz"] == 0


def test_the_chain_cards_check_mark_and_its_detail_once_applied():
    a = _adapter()
    a.filter_set(low=350.0, high=2400.0)
    a.filter_set(digital_roof_hz=3000.0, roof_offset_hz=400.0)
    st = a.filter_set(roof_offset=False)
    row = next(r for r in st["chain"] if r["id"] == "roof_digital")
    checks = {c["key"]: c for c in row["checks"]}
    assert checks["roof_offset"]["on"] is False
    assert checks["roof_offset"]["route"] == "/filter/set"
    assert checks["roof_offset"]["query_on"] == "roof_offset=on"
    assert checks["roof_offset"]["query_off"] == "roof_offset=off"
    assert "offset" not in row["detail"]
    st = a.filter_set(roof_offset=True)
    row = next(r for r in st["chain"] if r["id"] == "roof_digital")
    assert row["checks"][0]["on"] is True
    assert "offset +400 Hz" in row["detail"]
