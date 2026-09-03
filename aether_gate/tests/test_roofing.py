#
# Aether-gate — the analogue roofing filter: snap it, write it, read it back,
# and put it back after a rate change.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""setBandwidth is the one control in this chain that a pan zoom can undo.

SoapySDRPlay3's setSampleRate re-derives `bwType` from the new rate and
overwrites whatever setBandwidth last set (Settings.cpp), so the operator's
roofing choice has to be re-issued after every rate change or it reverts
silently -- and the status would go on reporting the value that is no longer
in the device. The fake driver below models exactly that: its setSampleRate
clobbers the bandwidth, and the regression test asserts the ORDER of the
calls, not just the value that survived.

Run:  python -m pytest aether_gate/tests/test_roofing.py
"""
import numpy as np
import pytest

from aether_gate.adapters.soapy import SoapyAdapter
from aether_gate.core.roofing import snap_analogue_hz

DT_BANDWIDTHS = [200000.0, 300000.0, 600000.0, 1536000.0]


class _FakeSdr:
    """Just enough of a SoapySDR.Device for the roofing path -- the same shape
    as test_device_controls._FakeSdr, plus the bandwidth calls."""

    def __init__(self, bandwidths=None, rate=250000.0):
        self.calls = []
        self.bandwidths = list(DT_BANDWIDTHS if bandwidths is None else bandwidths)
        self._bw = 200000.0
        self._rate = float(rate)

    def listBandwidths(self, direction, channel):
        self.calls.append(("listBandwidths", channel))
        return list(self.bandwidths)

    def setBandwidth(self, direction, channel, hz):
        self.calls.append(("setBandwidth", channel, float(hz)))
        self._bw = float(hz)

    def getBandwidth(self, direction, channel):
        return self._bw

    def setSampleRate(self, direction, channel, hz):
        self.calls.append(("setSampleRate", channel, float(hz)))
        self._rate = float(hz)
        self._bw = 1536000.0            # THE TRAP: the driver re-derives bwType

    def getSampleRate(self, direction, channel):
        return self._rate

    def writes(self):
        return [c for c in self.calls if c[0] in ("setBandwidth", "setSampleRate")]


def _adapter(sdr=None, rate=250000.0):
    a = SoapyAdapter(driver="sdrplay", samp_rate=rate, center_hz=3_890_000.0)
    a._np = np
    a._sdr = sdr if sdr is not None else _FakeSdr(rate=rate)
    a._SOAPY_SDR_RX = 0
    a._channels = [0]
    a._init_demod()
    a._mode = "LSB"
    return a


def _drain(a):
    """What the reader thread does with a pending bandwidth, between reads."""
    if a._roof_to is not None:
        want, a._roof_to = a._roof_to, None
        a._apply_roof_hz(want)


# ----- the snap, on its own ---------------------------------------------------

def test_a_request_snaps_down_to_a_bandwidth_the_driver_actually_offers():
    assert snap_analogue_hz(250000, DT_BANDWIDTHS) == 200000.0
    assert snap_analogue_hz(300000, DT_BANDWIDTHS) == 300000.0
    assert snap_analogue_hz(599999, DT_BANDWIDTHS) == 300000.0
    assert snap_analogue_hz(50000, DT_BANDWIDTHS) == 200000.0    # under the narrowest


def test_a_bandwidth_the_dual_tuner_list_excludes_is_refused_not_snapped():
    """5 MHz is a real RSPduo filter, but not in dual-tuner mode. Answering
    "fine, 1536 kHz" would report a filter nobody asked for."""
    with pytest.raises(ValueError):
        snap_analogue_hz(5_000_000, DT_BANDWIDTHS)
    with pytest.raises(ValueError):
        snap_analogue_hz(0, DT_BANDWIDTHS)
    with pytest.raises(ValueError):
        snap_analogue_hz(200000, [])


# ----- through the adapter ----------------------------------------------------

def test_the_options_come_from_the_driver_and_are_asked_for_once():
    a = _adapter()
    assert a.roof_options() == DT_BANDWIDTHS
    assert a.roof_options() == DT_BANDWIDTHS
    assert [c for c in a._sdr.calls if c[0] == "listBandwidths"] == [("listBandwidths", 0)]


def test_setting_the_roof_only_queues_it_for_the_reader_thread():
    """setBandwidth touches the device while readStream is running, the same
    reason retune and set_gain defer."""
    a = _adapter()
    assert a.set_roof_hz(250000) == 200000.0
    assert a._sdr.writes() == []
    _drain(a)
    assert a._sdr.writes() == [("setBandwidth", 0, 200000.0)]


def test_the_status_reports_what_the_device_read_back_not_the_request():
    a = _adapter()
    a.set_roof_hz(600000)
    _drain(a)
    a._sdr._bw = 300000.0                       # a setter that lied
    roofing = a.filter_status()["roofing"]
    assert roofing["analogue_hz"] == 600000.0   # the read-back at write time
    assert roofing["analogue_options"] == DT_BANDWIDTHS
    assert roofing["analogue_source"] == "operator"


def test_without_a_choice_the_roof_is_whatever_the_sample_rate_derived():
    roofing = _adapter().filter_status()["roofing"]
    assert roofing["analogue_hz"] == 200000.0
    assert roofing["analogue_source"] == "rate"


def test_the_operators_roofing_choice_is_re_applied_after_a_rate_change():
    """THE REGRESSION THAT MATTERS. A pan zoom goes through _apply_samp_rate,
    whose setSampleRate overwrites bwType; the order of the calls is the
    assertion, because the final value alone would also pass if the re-apply
    happened before the rate change instead of after it."""
    a = _adapter()
    a._stop_stream = lambda: None
    a._start_stream = lambda: None
    a._verify_stream = lambda: True
    a.set_roof_hz(600000)
    _drain(a)
    a._sdr.calls.clear()
    a._apply_samp_rate(500000.0)
    assert a._sdr.writes() == [("setSampleRate", 0, 500000.0),
                               ("setBandwidth", 0, 600000.0)]
    assert a.filter_status()["roofing"]["analogue_hz"] == 600000.0


def test_a_rate_change_without_a_choice_leaves_the_driver_to_derive_it():
    a = _adapter()
    a._stop_stream = lambda: None
    a._start_stream = lambda: None
    a._verify_stream = lambda: True
    a._apply_samp_rate(500000.0)
    assert [c for c in a._sdr.writes() if c[0] == "setBandwidth"] == []
    assert a.filter_status()["roofing"]["analogue_hz"] == 300000.0   # from the new rate


def test_filter_set_carries_the_roof_and_a_refusal_is_a_value_error():
    a = _adapter()
    st = a.filter_set(roof_hz=1536000.0)
    _drain(a)
    assert a._sdr.writes() == [("setBandwidth", 0, 1536000.0)]
    assert st["roofing"]["analogue_source"] == "operator"
    with pytest.raises(ValueError):
        a.filter_set(roof_hz=8_000_000.0)


def test_the_chain_row_offers_the_roof_as_a_select():
    row = next(r for r in _adapter().filter_status()["chain"] if r["id"] == "roof_rf")
    assert row["kind"] == "select" and row["options"] == DT_BANDWIDTHS
    assert row["action"] == {"label": "SET", "route": "/filter/set", "query": "roof_hz="}
    assert "the narrowest this hardware has" in row["detail"]
