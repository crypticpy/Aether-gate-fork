#
# Aether-gate — soapy.py's _DiversityState invariants (no hardware).
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Three review findings against the adapter's dual-tuner state, each with a
regression test that fails on the pre-fix code:

  F2  The alignment calibration length was 0.5 s of raw IQ with no cap, so at
      2.04 MS/s it queued a 2^21-point FFT (~103 ms stall on the reader
      thread) before a lag measurement ever ran. CAL_SAMPLES_MAX caps it.
  F8  request_realign() (HTTP thread) and ingest() (reader thread) both
      mutate _cal_a/_cal_b/_cal_n with no lock, so a realign landing mid-
      accumulate could hand np.concatenate an empty list.
  F10 weight_for() used to return the configured weight even when the
      aligner had not credibly locked the two channels, combining two
      decorrelated streams for a ~3 dB SNR loss. It must hold at 0j (channel
      A alone) until aligned, while status() keeps reporting the operator's
      configured phase/ratio so the UI does not appear to reset.

Run:  python -m pytest aether_gate/tests/test_soapy_diversity_state.py
"""
import threading

import numpy as np
import pytest

from aether_gate.adapters.soapy import _DiversityState
from aether_gate.core.diversity import weight_from_polar


class _FakeAdapter:
    """Just enough of SoapyAdapter for _DiversityState to run standalone —
    no hardware, no stream, no demod chain. ingest()/request_realign()/
    weight_for()/status() only touch self.a._np and self.a.samp_rate."""

    def __init__(self, samp_rate):
        self._np = np
        self.samp_rate = float(samp_rate)
        self.center_hz = 3_900_000.0


def _noise(n, rng):
    return (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)


# --- F10: weight_for() must hold at 0j until the aligner is aligned --------

def test_weight_for_is_zero_before_any_alignment():
    """A fresh _DiversityState has never aligned; combining must not start
    just because the operator dialled in a manual weight."""
    state = _DiversityState(_FakeAdapter(2_040_000.0))
    state.set(mode="manual", phase_deg=0.0, ratio_db=0.0)     # m = 1+0j
    assert state.aligner.aligned is False
    assert state.weight_for(0) == 0j


def test_weight_for_manual_m1_still_zero_when_unaligned():
    """The exact case F10 named: manual mode with m=1 (unity) must not
    combine while the aligner has not locked — that adds a decorrelated
    copy of A, a ~3 dB SNR loss versus A alone."""
    state = _DiversityState(_FakeAdapter(2_040_000.0))
    state.mode = "manual"
    state.manual[0] = 1 + 0j
    assert state.weight_for(0) == 0j


def test_weight_for_returns_configured_weight_once_aligned():
    state = _DiversityState(_FakeAdapter(2_040_000.0))
    state.mode = "manual"
    state.manual[0] = 0.5 + 0.5j
    state.aligner.set_lag(3, peak=50.0, aligned=True)
    assert state.weight_for(0) == pytest.approx(0.5 + 0.5j)


def test_status_reports_configured_weight_but_aligned_false():
    """The UI must not see the slider snap to zero just because the aligner
    has not locked yet — but 'aligned' must stay honest."""
    state = _DiversityState(_FakeAdapter(2_040_000.0))
    state.mode = "manual"
    state.manual[0] = weight_from_polar(30.0, -2.0)
    st = state.status(0)
    assert st["aligned"] is False
    assert st["phase_deg"] == pytest.approx(30.0, abs=0.2)
    assert st["ratio_db"] == pytest.approx(-2.0, abs=0.2)
    # but the weight actually combined is 0j (channel A alone)
    assert st["weight"] == [0.0, 0.0]


def test_status_weight_matches_configured_once_aligned():
    state = _DiversityState(_FakeAdapter(2_040_000.0))
    state.mode = "manual"
    state.manual[0] = weight_from_polar(30.0, -2.0)
    state.aligner.set_lag(0, peak=50.0, aligned=True)
    st = state.status(0)
    assert st["aligned"] is True
    m = complex(*st["weight"])
    assert m == pytest.approx(weight_from_polar(30.0, -2.0), abs=1e-3)


# --- F2: the calibration length is capped, not 0.5 s of raw IQ -------------

def test_cal_samples_max_caps_the_0_5s_window_at_2_04msps():
    """At 2.04 MS/s, 0.5 s is 1,020,000 samples (a 2^21-point FFT, ~103 ms
    reader-thread stall). The cap must bring that down to CAL_SAMPLES_MAX."""
    state = _DiversityState(_FakeAdapter(2_040_000.0))
    n_cal = min(int(state.CAL_SECONDS * state.a.samp_rate), state.CAL_SAMPLES_MAX)
    assert n_cal == state.CAL_SAMPLES_MAX
    assert n_cal < int(state.CAL_SECONDS * state.a.samp_rate)


def test_ingest_completes_calibration_at_the_capped_length_not_at_0_5s():
    """Regression: on the pre-fix code this needed 1,020,000 samples (250
    blocks of 4096) before a lag measurement ever ran; feeding exactly
    CAL_SAMPLES_MAX must be enough."""
    rng = np.random.default_rng(1234)
    state = _DiversityState(_FakeAdapter(2_040_000.0))
    state.request_realign("test")
    block = 4096
    fed = 0
    while fed < state.CAL_SAMPLES_MAX:
        a, b = _noise(block, rng), _noise(block, rng)
        state.ingest(a, b)
        fed += block
    assert state.last_align["why"] == "test", (
        "calibration did not complete at the capped sample count — "
        "CAL_SECONDS*samp_rate is still being used uncapped")


def _feed_window(state, rng, lag=None, seed_b=None):
    """One calibration window of noise; with lag, B is A delayed by lag."""
    n_cal = min(int(state.CAL_SECONDS * state.a.samp_rate), state.CAL_SAMPLES_MAX)
    n = n_cal + 8192
    a = _noise(n, rng)
    if lag is None:
        b = _noise(n, rng)
    else:
        b = np.concatenate([_noise(lag, rng), a[:n - lag]])
    fed = 0
    while fed < n_cal:
        state.ingest(a[fed:fed + 4096], b[fed:fed + 4096])
        fed += 4096


def test_a_quiet_window_is_not_a_verdict_the_realign_keeps_measuring():
    """Uncorrelated windows: no lock, but the measurement goes on up to
    ALIGN_TRIES windows before it gives up; then a correlated window locks
    the right lag and, being strong, ends the search."""
    rng = np.random.default_rng(7)
    state = _DiversityState(_FakeAdapter(125_000.0))
    state.request_realign("test")
    for i in range(state.ALIGN_TRIES - 1):
        _feed_window(state, rng)
        assert state._realign == "test" and not state.aligner.aligned, i
    _feed_window(state, rng)
    assert state._realign is None and not state.aligner.aligned
    assert state.last_align["ok"] is False and state._align_try == state.ALIGN_TRIES
    # a talker keys up during the next request: locked on the first window
    state.request_realign("operator request")
    _feed_window(state, rng, lag=63)
    assert state.aligner.aligned and state.aligner.lag == 63
    assert state._realign is None, "a strong peak should end the search"
    assert state.last_align == {"lag": 63, "peak": state.aligner.peak, "ok": True,
                                "why": "operator request"}
    assert state.last_align["peak"] >= state.ALIGN_STRONG_PEAK


def test_a_credible_lag_is_adopted_at_once_and_only_replaced_by_a_better_window():
    rng = np.random.default_rng(8)
    state = _DiversityState(_FakeAdapter(125_000.0))
    state.request_realign("test")
    _feed_window(state, rng)                      # quiet: nothing yet
    assert not state.aligner.aligned
    _feed_window(state, rng, lag=40)              # strong: adopted, search over
    assert state.aligner.aligned and state.aligner.lag == 40 and state._realign is None
    assert state._align_try == 2


# --- F8: request_realign() and ingest() must not race on the accumulators --

def test_request_realign_blocks_on_the_same_lock_ingest_uses():
    """Deterministic version of the race: prove request_realign() actually
    acquires state._cal_lock, rather than relying on the two threads
    happening to interleave badly within a time budget (see the stress test
    below for that). Hold the lock from another thread and confirm
    request_realign() blocks until it is released."""
    state = _DiversityState(_FakeAdapter(240_000.0))
    lock_held = threading.Event()
    release_holder = threading.Event()

    def hold_lock():
        with state._cal_lock:
            lock_held.set()
            release_holder.wait(timeout=2)

    holder = threading.Thread(target=hold_lock, daemon=True)
    holder.start()
    assert lock_held.wait(timeout=1), "test setup failed: lock never acquired"

    done = threading.Event()

    def do_realign():
        state.request_realign("blocked-check")
        done.set()

    racer = threading.Thread(target=do_realign, daemon=True)
    racer.start()
    assert not done.wait(timeout=0.2), (
        "request_realign() completed while _cal_lock was held elsewhere — "
        "it is not using the same lock as ingest()")
    release_holder.set()
    assert done.wait(timeout=1), "request_realign() never completed after the lock was released"
    holder.join(timeout=1)
    racer.join(timeout=1)


def test_concurrent_realign_and_ingest_do_not_raise():
    """request_realign() (here: a second thread, standing in for the HTTP
    control thread) resetting _cal_a/_cal_b/_cal_n while ingest() (here: the
    main thread, standing in for the reader thread) is mid-accumulate must
    never hand np.concatenate an empty list. Without the lock this is
    flaky-but-reproducible under load; run it for a real stretch of
    wall-clock time rather than a fixed iteration count so it has a chance
    to actually interleave."""
    import time as _time

    rng = np.random.default_rng(99)
    state = _DiversityState(_FakeAdapter(240_000.0))    # small rate -> cheap FFTs, frequent completions
    errors = []
    stop = threading.Event()

    def realign_storm():
        i = 0
        while not stop.is_set():
            state.request_realign(f"race-{i}")
            i += 1

    t = threading.Thread(target=realign_storm, daemon=True)
    t.start()
    deadline = _time.monotonic() + 0.5
    try:
        while _time.monotonic() < deadline:
            a, b = _noise(256, rng), _noise(256, rng)
            try:
                state.ingest(a, b)
            except Exception as e:                      # pragma: no cover - the bug this guards
                errors.append(repr(e))
                break
    finally:
        stop.set()
        t.join(timeout=2)

    assert errors == [], f"ingest()/request_realign() raced: {errors}"


# --- the lag window is a time: the ring offset does not shrink with the rate

def test_the_lag_window_is_a_time_so_the_offset_is_found_at_250k():
    """The offset between the tuners' rings is ~33 ms whatever the rate:
    -63 samples one start at 125 kS/s, -8316 at 250 kS/s -- past the old
    +-8192-sample window, so after a span change nothing ever locked,
    REALIGN included (found on the air 2026-09-03)."""
    rng = np.random.default_rng(9)
    state = _DiversityState(_FakeAdapter(250_000.0))
    state.request_realign("stream start")
    _feed_window(state, rng, lag=8316)
    assert state.aligner.aligned and state.aligner.lag == 8316
    assert state.last_align["peak"] >= state.ALIGN_STRONG_PEAK
    assert state._realign is None


@pytest.mark.parametrize("rate,lag", [(62_500.0, 2_079), (500_000.0, 16_632),
                                      (1_000_000.0, 33_265), (2_040_000.0, 67_861)])
def test_the_offset_is_found_exactly_at_every_span(rate, lag):
    """62.5 kHz to 2 MHz of span, the same 33 ms offset each time: the
    decimated search finds it and the full-rate refine lands on the exact
    sample (the lag here is deliberately not a multiple of the decimation)."""
    rng = np.random.default_rng(int(rate) % 97)
    state = _DiversityState(_FakeAdapter(rate))
    state.request_realign("stream start")
    _feed_window(state, rng, lag=lag)
    assert state.aligner.aligned and state.aligner.lag == lag
    assert state.last_align["peak"] >= state.ALIGN_STRONG_PEAK


def test_finder_json_says_why_there_is_nothing_to_find():
    state = _DiversityState(_FakeAdapter(125_000.0))
    assert state.finder_json() == {"available": False, "reason": "not aligned"}
