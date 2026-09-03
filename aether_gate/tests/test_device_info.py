#
# Aether-gate — device_block(): which radio is this, and is diversity on (B13).
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""device_block() (aether_gate/adapters/device_info.py) is the dict the app
reads to answer three things the control panel could not: which device is
connected, whether diversity is actually running, and which tuner the audio
and pan follow on a duo that has diversity off.

No hardware here — every sdr is a small fake. The two integration tests at
the bottom instantiate the real SoapyAdapter (same pattern as
test_soapy_recovery.py's `_adapter()`) with a fake sdr wired in by hand,
since SoapyAdapter.diagnostics() is what actually carries "device_block" in
/diagnostics; everything else exercises device_block() directly.

Run:  python -m pytest aether_gate/tests/test_device_info.py
"""
import pytest

from aether_gate.adapters.device_info import device_block


class _FakeDivState:
    """Just the two attributes device_block() reads off _DiversityState,
    plus a status() stub — soapy.py's diagnostics() calls that unconditionally
    whenever self._div is not None, for its own "diversity" key."""

    def __init__(self, mode, hear):
        self.mode = mode
        self.hear = hear

    def status(self):
        return {"mode": self.mode, "source": self.hear}


class _FakeDuoSdr:
    """An RSPduo in mode=DT: getHardwareKey() names the model, getHardwareInfo()
    is empty (observed live — SoapySDRPlay3 does not populate it), so serial
    comes from --soapy-args instead."""

    def getHardwareKey(self):
        return "RSPduo"

    def getHardwareInfo(self):
        return {}


class _FakeRspdxSdr:
    def getHardwareKey(self):
        return "RSPdx"

    def getHardwareInfo(self):
        return {}


class _RaisingHardwareInfoSdr:
    """A driver quirk: getHardwareInfo() throws. getHardwareKey() still works."""

    def getHardwareKey(self):
        return "RSPduo"

    def getHardwareInfo(self):
        raise RuntimeError("driver does not implement this")


class _RaisingEverythingSdr:
    """Both calls throw — the worst case, still must not raise out of
    device_block()."""

    def getHardwareKey(self):
        raise RuntimeError("no such call")

    def getHardwareInfo(self):
        raise RuntimeError("no such call")


# --- the duo: capable, and every (mode, hear) combination -------------------

def test_duo_track_running_reports_both_tuners():
    d = device_block(_FakeDuoSdr(), "sdrplay", "serial=2405055D34,mode=DT",
                      [0, 1], _FakeDivState(mode="track", hear="combined"))
    assert d["diversity"] == {"capable": True, "running": True,
                              "mode": "track", "tuner": "both"}


def test_duo_manual_mode_counts_as_running():
    d = device_block(_FakeDuoSdr(), "sdrplay", "serial=X", [0, 1],
                      _FakeDivState(mode="manual", hear="a"))
    # mode != "off" -> running, and running always reports "both": the pair
    # is being combined regardless of which single channel `hear` names.
    assert d["diversity"]["running"] is True
    assert d["diversity"]["tuner"] == "both"


def test_duo_off_hearing_a_reports_tuner_a():
    d = device_block(_FakeDuoSdr(), "sdrplay", "serial=2405055D34,mode=DT",
                      [0, 1], _FakeDivState(mode="off", hear="a"))
    assert d["diversity"] == {"capable": True, "running": False,
                              "mode": "off", "tuner": "A"}


def test_duo_off_hearing_b_reports_tuner_b():
    d = device_block(_FakeDuoSdr(), "sdrplay", "serial=X", [0, 1],
                      _FakeDivState(mode="off", hear="b"))
    assert d["diversity"]["tuner"] == "B"


def test_duo_off_hearing_stereo_reports_both():
    d = device_block(_FakeDuoSdr(), "sdrplay", "serial=X", [0, 1],
                      _FakeDivState(mode="off", hear="stereo"))
    assert d["diversity"]["tuner"] == "both"


def test_duo_off_hearing_combined_reports_both():
    # The freshly-opened default (__init__: mode="off", hear="combined") —
    # nobody has picked a tuner yet, so "both" is the honest answer even
    # though the combiner itself is not running.
    d = device_block(_FakeDuoSdr(), "sdrplay", "serial=X", [0, 1],
                      _FakeDivState(mode="off", hear="combined"))
    assert d["diversity"]["running"] is False
    assert d["diversity"]["tuner"] == "both"


# --- single tuner: never capable, regardless of a leftover div_state -------

def test_single_tuner_rspdx_is_not_diversity_capable():
    d = device_block(_FakeRspdxSdr(), "sdrplay", "serial=2405xxxx", [0], None)
    assert d["diversity"] == {"capable": False, "running": False,
                              "mode": None, "tuner": None}
    assert d["tuners"] == 1
    assert d["model"] == "RSPdx"


def test_single_channel_with_a_div_state_present_is_still_not_capable():
    """Channel count is what decides capability, not whether a
    _DiversityState object happens to exist."""
    d = device_block(_FakeDuoSdr(), "sdrplay", "serial=X", [0],
                      _FakeDivState(mode="track", hear="combined"))
    assert d["diversity"]["capable"] is False
    assert d["diversity"]["running"] is False


def test_bare_tuner_count_is_accepted_in_place_of_a_channel_list():
    d = device_block(_FakeDuoSdr(), "sdrplay", "serial=X", 2,
                      _FakeDivState(mode="off", hear="a"))
    assert d["tuners"] == 2
    assert d["diversity"]["capable"] is True


# --- defensive: a quirky or half-open driver must never raise --------------

def test_get_hardware_info_raising_falls_back_to_soapy_args_for_serial():
    d = device_block(_RaisingHardwareInfoSdr(), "sdrplay",
                      "serial=2405055D34,mode=DT", [0, 1],
                      _FakeDivState(mode="off", hear="a"))
    assert d["serial"] == "2405055D34"
    assert d["model"] == "RSPduo"          # getHardwareKey() still answered
    assert d["hardware_key"] == "RSPduo"


def test_everything_raising_still_returns_a_dict_not_an_exception():
    d = device_block(_RaisingEverythingSdr(), "sdrplay", "serial=X", [0, 1],
                      _FakeDivState(mode="off", hear="a"))
    assert d["model"] is None
    assert d["hardware_key"] is None
    assert d["serial"] == "X"
    assert d["tuners"] == 2
    # label still falls back to the driver name when nothing else answers
    assert d["label"].startswith("sdrplay X")


def test_none_sdr_returns_empty_hardware_fields_not_an_exception():
    d = device_block(None, "rtlsdr", "", [0], None)
    assert d["model"] is None
    assert d["serial"] is None
    assert d["hardware_key"] is None
    assert d["tuners"] == 1
    assert d["diversity"]["capable"] is False


# --- the label: one honest line, no em-dash -------------------------------

def test_label_single_tuner():
    d = device_block(_FakeRspdxSdr(), "sdrplay", "serial=2405xxxx", [0], None)
    assert d["label"] == "RSPdx 2405xxxx - single tuner"


def test_label_diversity_off_tuner_a():
    d = device_block(_FakeDuoSdr(), "sdrplay", "serial=2405055D34,mode=DT",
                      [0, 1], _FakeDivState(mode="off", hear="a"))
    assert d["label"] == "RSPduo 2405055D34 - diversity off, tuner A"


def test_label_diversity_running_names_the_mode():
    d = device_block(_FakeDuoSdr(), "sdrplay", "serial=2405055D34,mode=DT",
                      [0, 1], _FakeDivState(mode="track", hear="combined"))
    assert d["label"] == "RSPduo 2405055D34 - diversity (track)"


@pytest.mark.parametrize("label", [
    "RSPdx 2405xxxx - single tuner",
    "RSPduo 2405055D34 - diversity off, tuner A",
    "RSPduo 2405055D34 - diversity (track)",
])
def test_labels_carry_no_mojibake_or_em_dash(label):
    assert "—" not in label   # em dash
    assert "Ã" not in label and "â" not in label   # mojibake tells


# --- integration: the real adapter's diagnostics() carries device_block ----

from aether_gate.adapters.soapy import SoapyAdapter  # noqa: E402


def _wired_duo_adapter():
    """A SoapyAdapter with a fake sdr wired in directly — the same shortcut
    test_soapy_recovery.py's _adapter() uses to exercise adapter methods
    without hardware or open()/_open_hw()."""
    a = SoapyAdapter(driver="sdrplay", device_args="serial=2405055D34,mode=DT")
    a._sdr = _FakeDuoSdr()
    a._SOAPY_SDR_RX = 0          # normally set in open(); diagnostics() needs it
    a._channels = [0, 1]
    a._div = _FakeDivState(mode="track", hear="combined")
    return a


def test_adapter_diagnostics_carries_device_block():
    a = _wired_duo_adapter()
    d = a.diagnostics()
    assert d["device_block"] == {
        "driver": "sdrplay", "model": "RSPduo", "serial": "2405055D34",
        "hardware_key": "RSPduo", "tuners": 2,
        "diversity": {"capable": True, "running": True, "mode": "track",
                      "tuner": "both"},
        "label": "RSPduo 2405055D34 - diversity (track)",
    }


def test_adapter_diagnostics_survives_a_hardware_info_that_raises():
    """A driver quirk in getHardwareInfo() must not take /diagnostics down —
    soapy.py's diagnostics() wraps the device_block() call in try/except as
    a backstop even though device_block() already catches this itself."""
    a = _wired_duo_adapter()
    a._sdr = _RaisingHardwareInfoSdr()
    d = a.diagnostics()
    assert d["device_block"]["serial"] == "2405055D34"
    assert d["device_block"]["model"] == "RSPduo"
