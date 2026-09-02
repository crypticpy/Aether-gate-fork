#
# Aether-gate — RSPduo diversity control-port routes (no hardware, loopback only).
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The diversity mode/weight/alignment surface the control panel drives.

Two coherent tuners means the adapter carries a mode, a phase/ratio weight,
and an alignment state that the panel has to read and set over HTTP. This
drives the real control-port handler (start_control_server) over loopback
with a fake adapter standing in for the RSPduo path, and covers:

  * an adapter with no diversity capability answers {"available": false} on
    every diversity route (never an error page, never a crash);
  * /diversity/set forwards exactly the params given, straight through to
    adapter.set_diversity;
  * a bad value (unparseable float, an out-of-set mode/source) comes back as
    {"error": "bad value: ..."}, same shape as /calibrate;
  * /diversity/align reaches adapter.diversity_realign();
  * /status carries a compact diversity dict, and None on a single-channel
    adapter.

Run:  python -m aether_gate.tests.test_diversity_routes
"""
import json
import socket
import sys
import time
import urllib.request


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class FakeDiversityAdapter:
    """Just enough of the RSPduo adapter contract for the control-port routes."""

    def __init__(self):
        self.diversity_available = True
        self.calls = []          # (method, kwargs) for assertions
        self._status = {
            "available": True, "channels": 2, "mode": "manual", "source": "combined",
            "phase_deg": 30.0, "ratio_db": -2.0, "weight": [0.9, 0.1],
            "lag_samples": 3, "aligned": True, "corr_peak": 0.92,
            "snr_db": {"a": 18.5, "b": 14.2, "out": 21.0},
            "updates": 42, "slice_id": 0,
        }

    def diversity_status(self, slice_id=None):
        self.calls.append(("status", {"slice_id": slice_id}))
        return dict(self._status)

    def set_diversity(self, mode=None, phase_deg=None, ratio_db=None, source=None, slice_id=None):
        self.calls.append(("set", {"mode": mode, "phase_deg": phase_deg,
                                    "ratio_db": ratio_db, "source": source, "slice_id": slice_id}))
        if mode is not None:
            self._status["mode"] = mode
        if phase_deg is not None:
            self._status["phase_deg"] = phase_deg
        if ratio_db is not None:
            self._status["ratio_db"] = ratio_db
        if source is not None:
            self._status["source"] = source
        return dict(self._status)

    def diversity_realign(self):
        self.calls.append(("realign", {}))
        self._status["aligned"] = False
        return dict(self._status)

    def get_audio(self, n_samples, slice_hz=None, mode=None, slice_id=None):
        self.calls.append(("get_audio", {"slice_id": slice_id}))
        return [0.0] * n_samples


class FakeSingleChannelAdapter:
    """A plain single-channel adapter — no diversity attribute at all, the way
    RTL/Kenwood/HPSDR adapters look today."""

    def get_audio(self, n_samples, slice_hz=None, mode=None):
        return [0.0] * n_samples


class FakeRaisingDiversityAdapter(FakeDiversityAdapter):
    """diversity_available is True, but the status call itself blows up —
    e.g. the adapter's numpy alignment code raising on bad hardware state."""

    def diversity_status(self, slice_id=None):
        raise RuntimeError("boom")


def _start(adapter):
    from aether_gate.core import Radio
    from aether_gate.core.engine import start_control_server
    ctl_port = _free_port()
    r = Radio("127.0.0.1", None, adapter=adapter, port=_free_port())
    start_control_server(r, ctl_port)
    time.sleep(0.3)           # let the ThreadingHTTPServer bind before we hit it
    return r, ctl_port


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2.0) as resp:
        assert resp.status == 200, resp.status
        assert resp.headers.get("Content-Type") == "application/json", resp.headers.get("Content-Type")
        return json.loads(resp.read())


# ---- unavailable adapter: every diversity route answers, never errors -----

def test_unavailable_adapter_reports_unavailable_not_an_error():
    a = FakeSingleChannelAdapter()
    _, port = _start(a)
    assert _get(port, "/diversity") == {"available": False}
    assert _get(port, "/diversity/align") == {"available": False}
    assert _get(port, "/diversity/set?mode=manual") == {"available": False}


# ---- a diversity-capable adapter's status -----------------------------------

def test_diversity_available_adapter_reports_its_status():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    s = _get(port, "/diversity")
    assert s["available"] is True
    assert s["channels"] == 2
    assert s["mode"] == "manual"
    assert a.calls[-1][0] == "status"


# ---- /diversity/set forwards exactly what was given -------------------------

def test_set_forwards_every_given_param():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    out = _get(port, "/diversity/set?mode=track&phase=45.5&ratio=-3.5&source=b&slice=1")
    method, kwargs = a.calls[-1]
    assert method == "set"
    assert kwargs == {"mode": "track", "phase_deg": 45.5, "ratio_db": -3.5,
                       "source": "b", "slice_id": 1}
    assert out["mode"] == "track" and out["source"] == "b"


def test_set_with_only_one_param_leaves_the_rest_unset():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    _get(port, "/diversity/set?ratio=5")
    _, kwargs = a.calls[-1]
    assert kwargs == {"mode": None, "phase_deg": None, "ratio_db": 5.0,
                       "source": None, "slice_id": None}


# ---- bad values come back as {"error": ...}, like /calibrate ---------------

def test_bad_phase_value_is_an_error_not_a_crash():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    out = _get(port, "/diversity/set?phase=not-a-number")
    assert "error" in out and out["error"].startswith("bad value:")
    assert not any(c[0] == "set" for c in a.calls), "adapter must not be called on a parse failure"


def test_bad_mode_value_is_an_error():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    out = _get(port, "/diversity/set?mode=bogus")
    assert "error" in out and out["error"].startswith("bad value:")


def test_bad_source_value_is_an_error():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    out = _get(port, "/diversity/set?source=c")
    assert "error" in out and out["error"].startswith("bad value:")


def test_phase_nan_is_an_error_not_a_crash():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    out = _get(port, "/diversity/set?phase=nan")
    assert "error" in out and out["error"].startswith("bad value:")
    assert not any(c[0] == "set" for c in a.calls), "adapter must not be called with a NaN weight"


def test_phase_inf_is_an_error_not_a_crash():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    out = _get(port, "/diversity/set?phase=inf")
    assert "error" in out and out["error"].startswith("bad value:")
    assert not any(c[0] == "set" for c in a.calls), "adapter must not be called with an infinite weight"


def test_ratio_nan_is_an_error_not_a_crash():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    out = _get(port, "/diversity/set?ratio=nan")
    assert "error" in out and out["error"].startswith("bad value:")
    assert not any(c[0] == "set" for c in a.calls), "adapter must not be called with a NaN weight"


# ---- /diversity/align reaches diversity_realign() ---------------------------

def test_align_calls_diversity_realign():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    out = _get(port, "/diversity/align")
    assert a.calls[-1][0] == "realign"
    assert out["aligned"] is False


# ---- /status carries the compact diversity dict, or None -------------------

def test_status_carries_compact_diversity_dict():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    s = _get(port, "/status")
    assert s["diversity"] == {
        "mode": "manual", "phase_deg": 30.0, "ratio_db": -2.0, "aligned": True,
        "snr_db": {"a": 18.5, "b": 14.2, "out": 21.0},
    }


def test_status_diversity_is_none_for_a_single_channel_adapter():
    a = FakeSingleChannelAdapter()
    _, port = _start(a)
    s = _get(port, "/status")
    assert s["diversity"] is None


def test_status_survives_diversity_status_raising():
    a = FakeRaisingDiversityAdapter()
    _, port = _start(a)
    s = _get(port, "/status")               # must still be 200 JSON, not a crash
    assert s["diversity"] is None
    assert "connected" in s and "streaming" in s and "res" in s


def main():
    tests = [
        test_unavailable_adapter_reports_unavailable_not_an_error,
        test_diversity_available_adapter_reports_its_status,
        test_set_forwards_every_given_param,
        test_set_with_only_one_param_leaves_the_rest_unset,
        test_bad_phase_value_is_an_error_not_a_crash,
        test_bad_mode_value_is_an_error,
        test_bad_source_value_is_an_error,
        test_phase_nan_is_an_error_not_a_crash,
        test_phase_inf_is_an_error_not_a_crash,
        test_ratio_nan_is_an_error_not_a_crash,
        test_align_calls_diversity_realign,
        test_status_carries_compact_diversity_dict,
        test_status_diversity_is_none_for_a_single_channel_adapter,
        test_status_survives_diversity_status_raising,
    ]
    fails = 0
    for t in tests:
        try:
            t()
            print(f"ok   {t.__name__}")
        except Exception as e:                      # noqa: BLE001 - report and continue
            fails += 1
            print(f"FAIL {t.__name__}: {e!r}")
    print("ALL PASS" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
