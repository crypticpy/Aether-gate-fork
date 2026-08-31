#
# Aether-gate — panadapter resolution control (no hardware, no network).
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Bin width = span / bins, and both halves are operator-settable at runtime.

The bug this control was born from, 2026-08-31: an operator wanting finer bins
on an RSPdx asked for 256 kS/s — a plausible-looking number that is not a 2 MS/s
decimation. SoapySDRPlay3 does not reject it. It logs

    [WARNING] invalid sample rate. Sample rate unchanged.

and returns normally, leaving the device at 2 MS/s. The request was for FOUR
TIMES FINER bins and what landed was four times COARSER, with only a driver
warning to say so. So the rate is snapped to something the device actually
offers before it is ever handed to the driver.

Run:  python -m pytest aether_gate/tests/test_resolution.py
"""
import pytest

np = pytest.importorskip("numpy")

from aether_gate.adapters.soapy import SoapyAdapter


class _FakeDevice:
    """An SDRplay-shaped device: only 2 MS/s decimations, and it IGNORES the rest."""
    RATES = [62500.0, 125000.0, 250000.0, 500000.0, 1000000.0, 2000000.0]

    def __init__(self, rate=250000.0):
        self.rate = rate
        self.sets = []

    def listSampleRates(self, d, c):
        return list(self.RATES)

    def setSampleRate(self, d, c, hz):
        self.sets.append(hz)
        if hz in self.RATES:              # anything else: warn to stderr and no-op
            self.rate = hz

    def getSampleRate(self, d, c):
        return self.rate


def _adapter(rate=250000.0):
    a = SoapyAdapter(driver="none", samp_rate=rate)
    a._np = np
    a._sdr = _FakeDevice(rate)
    a._SOAPY_SDR_RX = 0
    return a


def _reader_tick(a):
    """Do what _read_loop does with a pending rate, without a real stream."""
    a._stop_stream = lambda: None
    a._start_stream = lambda: None
    a._verify_stream = lambda timeout_s=2.0: True
    if a._rate_to is not None:
        want = float(a._rate_to)
        if abs(want - a.samp_rate) > 1.0:
            a._apply_samp_rate(want)
        a._rate_to = None


# --- the snap ---------------------------------------------------------------

def test_an_unsupported_rate_is_snapped_not_passed_through():
    a = _adapter(2_000_000.0)
    a.set_samp_rate(256_000, wait_s=0.0)          # the 2026-08-31 request
    assert a._rate_to == 250_000.0, "256 kS/s must snap to the nearest offered rate"
    _reader_tick(a)
    assert a._sdr.sets == [250_000.0], "the driver must never see the raw request"
    assert a.samp_rate == 250_000.0


def test_a_supported_rate_is_passed_through_untouched():
    a = _adapter(2_000_000.0)
    a.set_samp_rate(500_000, wait_s=0.0)
    _reader_tick(a)
    assert a._sdr.sets == [500_000.0]
    assert a.samp_rate == 500_000.0


def test_the_rate_is_taken_from_the_device_readback_not_the_request():
    # This driver's setters lie; the whole adapter is built on never trusting one.
    a = _adapter(250_000.0)
    a._sdr.RATES = [250_000.0]                    # device will refuse everything else
    a._rate_to = 125_000.0                        # bypass the snap, as a wedged driver would
    _reader_tick(a)
    assert a.samp_rate == 250_000.0, "must report what the device says, not what we asked"


def test_no_device_means_no_rate_change():
    a = _adapter()
    a._sdr = None
    assert a.set_samp_rate(125_000, wait_s=0.0) is None
    assert a._rate_to is None


# --- the consequences of a rate change -------------------------------------

def test_the_demod_chain_is_rebuilt_for_the_new_rate():
    # samp_rate feeds the staged decimation; leaving it stale starves the audio
    # clock, which is the click-every-1.3s failure _init_demod documents.
    a = _adapter(500_000.0)
    a._init_demod()
    before = a._decim
    a.set_samp_rate(125_000, wait_s=0.0)
    _reader_tick(a)
    assert a._decim != before
    assert a._pd_rate == pytest.approx(a.samp_rate / a._decim)


def test_a_rate_change_drops_stale_audio():
    a = _adapter(500_000.0)
    a._init_demod()
    a._audio_q.append(np.zeros(64, dtype=np.complex64))
    a._iq_resid = np.zeros(8, dtype=np.complex64)
    a.set_samp_rate(125_000, wait_s=0.0)
    _reader_tick(a)
    assert len(a._audio_q) == 0, "queued IQ is at the old rate — it would click"
    assert a._iq_resid is None


def test_the_span_follows_the_rate():
    # The pan window IS the sample rate on an IQ adapter (see set_span).
    a = _adapter(500_000.0)
    a.set_samp_rate(125_000, wait_s=0.0)
    _reader_tick(a)
    assert a.current_span_hz() == 125_000.0
    assert a.capabilities.max_span_hz == 125_000.0


# --- the wire: AE has to be told the geometry changed -----------------------

class FakeConn:
    def __init__(self):
        self.out = bytearray()

    def sendall(self, b):
        self.out.extend(b)


def _radio(adapter=None):
    from aether_gate.core import Radio
    return Radio("127.0.0.1", None, adapter=adapter, port=5992, bins=4096)


def test_more_bins_narrows_the_bin_width_at_the_same_span():
    r = _radio()
    r.span_mhz = 0.25
    r.set_resolution(bins=1024)
    before = r.resolution()["bin_hz"]
    r.set_resolution(bins=2048)
    after = r.resolution()
    assert after["bins"] == 2048
    assert after["bin_hz"] == pytest.approx(before / 2.0, abs=0.001)   # bin_hz is a rounded readout


def test_a_bins_change_re_advertises_x_pixels_to_AE():
    # AE draws its frequency grid from the pan status. Change the bin count
    # without re-emitting and it paints the old grid over the new data.
    r = _radio()
    r._new_pan()
    r.conn = FakeConn()
    r.set_resolution(bins=2048)
    assert "x_pixels=2048" in r.conn.out.decode()


def test_bins_are_clamped_to_what_one_datagram_can_carry():
    # 16384 bins raised EMSGSIZE mid-send, the stream loop broke out, and the
    # panadapter stayed dark until the gate restarted (2026-08-31).
    from aether_gate.core.engine import max_pan_bins
    r = _radio()
    r.set_resolution(bins=10 ** 9)
    assert r.bins == max_pan_bins()
    r.set_resolution(bins=0)
    assert r.bins == 64


def test_the_bin_ceiling_actually_fits_a_datagram():
    from aether_gate.core.engine import max_pan_bins, udp_maxdgram, fft_packet, wf_packet
    n = max_pan_bins()
    assert len(fft_packet(1, 0, [0] * n, 0)) <= udp_maxdgram()
    assert len(wf_packet(2, 0, [0] * n, 0.0, 1.0, 0)) <= udp_maxdgram()
    # and one more bin must NOT fit, or the ceiling is leaving resolution unused
    assert len(wf_packet(2, 0, [0] * (n + 1), 0.0, 1.0, 0)) > udp_maxdgram()


def test_a_dead_stream_loop_clears_the_flag_so_it_can_restart():
    # emit_pan_status only starts a loop when streaming is False.
    r = _radio()
    r.streaming = True
    r.run = False                       # the loop's own exit condition
    r.vita_dest = ("127.0.0.1", 1)
    r.stream_loop()
    assert r.streaming is False


def test_an_adapter_without_the_seam_reports_no_rate_control():
    from aether_gate.adapters import SimAdapter
    r = _radio(SimAdapter(model="FLEX-6600"))
    res = r.resolution()
    assert res["can_set_rate"] is False
    assert res["rates"] == []
    r.set_resolution(samp_rate_hz=125_000)        # must be a no-op, not a crash
