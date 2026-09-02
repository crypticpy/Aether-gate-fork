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
        self._capture_active = False
        self._status = {
            "available": True, "channels": 2, "mode": "manual", "source": "combined",
            "phase_deg": 30.0, "ratio_db": -2.0, "weight": [0.9, 0.1],
            "lag_samples": 3, "aligned": True, "corr_peak": 0.92,
            "snr_db": {"a": 18.5, "b": 14.2, "out": 21.0},
            "updates": 42, "slice_id": 0,
            "nb": {"enabled": False, "threshold_db": 12.0, "blanked_pct": 0.0},
            "pan": "combined",
            "sources": [
                {"lo_hz": 3512000.0, "hi_hz": 3560000.0, "phase_deg": 141.0,
                 "ratio_db": -2.1, "coherence": 0.82, "level_db": -68.0},
            ],
            "memory": [
                {"phase_deg": 12.0, "ratio_db": 1.0, "age_s": 4.0, "hits": 3},
            ],
            "rn_source": "inband", "talk_mod": 0.4,
            "capture": {"active": False, "path": None},
        }

    def diversity_status(self, slice_id=None):
        self.calls.append(("status", {"slice_id": slice_id}))
        return dict(self._status)

    def set_diversity(self, mode=None, phase_deg=None, ratio_db=None, source=None, slice_id=None,
                       nb=None, nb_db=None, pan=None, null_source=None):
        self.calls.append(("set", {"mode": mode, "phase_deg": phase_deg,
                                    "ratio_db": ratio_db, "source": source, "slice_id": slice_id,
                                    "nb": nb, "nb_db": nb_db, "pan": pan, "null_source": null_source}))
        if mode is not None:
            self._status["mode"] = mode
        if phase_deg is not None:
            self._status["phase_deg"] = phase_deg
        if ratio_db is not None:
            self._status["ratio_db"] = ratio_db
        if source is not None:
            self._status["source"] = source
        if nb is not None:
            self._status["nb"]["enabled"] = nb
        if nb_db is not None:
            self._status["nb"]["threshold_db"] = nb_db
        if pan is not None:
            self._status["pan"] = pan
        if null_source is not None:
            self._status["null_source"] = null_source
        return dict(self._status)

    def diversity_realign(self):
        self.calls.append(("realign", {}))
        self._status["aligned"] = False
        return dict(self._status)

    def diversity_map(self):
        self.calls.append(("map", {}))
        return {
            "start_hz": 3500000.0, "step_hz": 1000.0,
            "coherence": [0.1, 0.4, 0.82, 0.3],
            "level_db": [-90.0, -85.0, -68.0, -88.0],
            "sources": self._status["sources"],
        }

    def diversity_capture(self, seconds):
        self.calls.append(("capture", {"seconds": seconds}))
        if self._capture_active:
            raise RuntimeError("capture already active")
        return f"/tmp/aether-gate-capture-{seconds}s.wav"

    def diversity_memory_clear(self):
        self.calls.append(("memory_clear", {}))
        self._status["memory"] = []

    def get_audio(self, n_samples, slice_hz=None, mode=None, slice_id=None):
        self.calls.append(("get_audio", {"slice_id": slice_id}))
        return [0.0] * n_samples


class FakeDiversityAdapterCaptureBusy(FakeDiversityAdapter):
    """A diversity adapter whose capture is already running — the route must
    surface adapter.diversity_capture's RuntimeError as {"error": ...}, not a
    traceback."""

    def __init__(self):
        super().__init__()
        self._capture_active = True


class FakeDiversityAdapterNoV2(FakeDiversityAdapter):
    """diversity_available is True but the v2 methods (map/capture/memory
    clear) simply don't exist yet — the way the real adapter looks before its
    own v2 work lands. Routes must answer {"error": "not supported"}, not
    AttributeError."""

    diversity_map = None
    diversity_capture = None
    diversity_memory_clear = None


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
    out = _get(port, "/diversity/set?mode=track&phase=45.5&ratio=-3.5&source=b&slice=1"
                     "&nb=on&nb_db=18.5&pan=nulled&null_source=2")
    method, kwargs = a.calls[-1]
    assert method == "set"
    assert kwargs == {"mode": "track", "phase_deg": 45.5, "ratio_db": -3.5,
                       "source": "b", "slice_id": 1,
                       "nb": True, "nb_db": 18.5, "pan": "nulled", "null_source": 2}
    assert out["mode"] == "track" and out["source"] == "b"


def test_set_with_only_one_param_leaves_the_rest_unset():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    _get(port, "/diversity/set?ratio=5")
    _, kwargs = a.calls[-1]
    assert kwargs == {"mode": None, "phase_deg": None, "ratio_db": 5.0,
                       "source": None, "slice_id": None,
                       "nb": None, "nb_db": None, "pan": None, "null_source": None}


# ---- v2 /diversity/set params: nb, nb_db, pan, null_source ------------------

def test_nb_off_forwards_false():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    _get(port, "/diversity/set?nb=off")
    _, kwargs = a.calls[-1]
    assert kwargs["nb"] is False


def test_bad_nb_value_is_an_error():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    out = _get(port, "/diversity/set?nb=maybe")
    assert "error" in out and out["error"].startswith("bad value:")
    assert not any(c[0] == "set" for c in a.calls)


def test_nb_db_in_range_forwards():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    _get(port, "/diversity/set?nb_db=0")
    assert a.calls[-1][1]["nb_db"] == 0.0
    _get(port, "/diversity/set?nb_db=40")
    assert a.calls[-1][1]["nb_db"] == 40.0


def test_nb_db_nan_is_an_error():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    out = _get(port, "/diversity/set?nb_db=nan")
    assert "error" in out and out["error"].startswith("bad value:")
    assert not any(c[0] == "set" for c in a.calls)


def test_nb_db_out_of_range_is_an_error():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    out = _get(port, "/diversity/set?nb_db=41")
    assert "error" in out and out["error"].startswith("bad value:")
    out = _get(port, "/diversity/set?nb_db=-1")
    assert "error" in out and out["error"].startswith("bad value:")
    assert not any(c[0] == "set" for c in a.calls)


def test_pan_valid_values_forward():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    for pan in ("a", "b", "combined", "nulled"):
        _get(port, "/diversity/set?pan=" + pan)
        assert a.calls[-1][1]["pan"] == pan


def test_bad_pan_value_is_an_error():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    out = _get(port, "/diversity/set?pan=x")
    assert "error" in out and out["error"].startswith("bad value:")
    assert not any(c[0] == "set" for c in a.calls)


def test_null_source_forwards_as_int():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    _get(port, "/diversity/set?null_source=0")
    assert a.calls[-1][1]["null_source"] == 0


def test_negative_null_source_is_an_error():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    out = _get(port, "/diversity/set?null_source=-1")
    assert "error" in out and out["error"].startswith("bad value:")
    assert not any(c[0] == "set" for c in a.calls)


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


# ---- /diversity/map -----------------------------------------------------

def test_diversity_map_present():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    out = _get(port, "/diversity/map")
    assert a.calls[-1][0] == "map"
    assert out["start_hz"] == 3500000.0
    assert out["coherence"] == [0.1, 0.4, 0.82, 0.3]
    assert len(out["sources"]) == 1


def test_diversity_map_not_supported_on_v2_less_adapter():
    a = FakeDiversityAdapterNoV2()
    _, port = _start(a)
    out = _get(port, "/diversity/map")
    assert out == {"error": "not supported"}


def test_diversity_map_not_supported_on_unavailable_adapter():
    a = FakeSingleChannelAdapter()
    _, port = _start(a)
    out = _get(port, "/diversity/map")
    assert out == {"error": "not supported"}


# ---- /diversity/capture --------------------------------------------------

def test_diversity_capture_ok():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    out = _get(port, "/diversity/capture?seconds=5")
    assert out == {"ok": True, "path": "/tmp/aether-gate-capture-5s.wav"}
    assert a.calls[-1] == ("capture", {"seconds": 5})


def test_diversity_capture_default_seconds():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    _get(port, "/diversity/capture")
    assert a.calls[-1] == ("capture", {"seconds": 10})


def test_diversity_capture_bad_seconds_is_an_error():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    out = _get(port, "/diversity/capture?seconds=0")
    assert "error" in out and out["error"].startswith("bad value:")
    out = _get(port, "/diversity/capture?seconds=61")
    assert "error" in out and out["error"].startswith("bad value:")
    out = _get(port, "/diversity/capture?seconds=nope")
    assert "error" in out and out["error"].startswith("bad value:")
    assert not any(c[0] == "capture" for c in a.calls)


def test_diversity_capture_already_active_surfaces_runtime_error():
    a = FakeDiversityAdapterCaptureBusy()
    _, port = _start(a)
    out = _get(port, "/diversity/capture?seconds=5")
    assert out == {"error": "capture already active"}


def test_diversity_capture_not_supported_on_v2_less_adapter():
    a = FakeDiversityAdapterNoV2()
    _, port = _start(a)
    out = _get(port, "/diversity/capture?seconds=5")
    assert out == {"error": "not supported"}


# ---- /diversity/memory/clear ---------------------------------------------

def test_diversity_memory_clear():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    out = _get(port, "/diversity/memory/clear")
    assert out == {"ok": True}
    assert a.calls[-1][0] == "memory_clear"
    assert a._status["memory"] == []


def test_diversity_memory_clear_not_supported_on_v2_less_adapter():
    a = FakeDiversityAdapterNoV2()
    _, port = _start(a)
    out = _get(port, "/diversity/memory/clear")
    assert out == {"error": "not supported"}


# ---- /status carries the compact diversity dict, or None -------------------

def test_status_carries_compact_diversity_dict():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    s = _get(port, "/status")
    assert s["diversity"] == {
        "mode": "manual", "phase_deg": 30.0, "ratio_db": -2.0, "aligned": True,
        "snr_db": {"a": 18.5, "b": 14.2, "out": 21.0},
        "nb": False, "pan": "combined",
    }


def test_status_diversity_nb_reflects_adapter_state():
    a = FakeDiversityAdapter()
    a._status["nb"]["enabled"] = True
    _, port = _start(a)
    s = _get(port, "/status")
    assert s["diversity"]["nb"] is True
    assert s["diversity"]["pan"] == "combined"


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
        test_nb_off_forwards_false,
        test_bad_nb_value_is_an_error,
        test_nb_db_in_range_forwards,
        test_nb_db_nan_is_an_error,
        test_nb_db_out_of_range_is_an_error,
        test_pan_valid_values_forward,
        test_bad_pan_value_is_an_error,
        test_null_source_forwards_as_int,
        test_negative_null_source_is_an_error,
        test_diversity_map_present,
        test_diversity_map_not_supported_on_v2_less_adapter,
        test_diversity_map_not_supported_on_unavailable_adapter,
        test_diversity_capture_ok,
        test_diversity_capture_default_seconds,
        test_diversity_capture_bad_seconds_is_an_error,
        test_diversity_capture_already_active_surfaces_runtime_error,
        test_diversity_capture_not_supported_on_v2_less_adapter,
        test_diversity_memory_clear,
        test_diversity_memory_clear_not_supported_on_v2_less_adapter,
        test_status_carries_compact_diversity_dict,
        test_status_diversity_nb_reflects_adapter_state,
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
