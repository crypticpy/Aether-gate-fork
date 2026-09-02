#
# Aether-gate — the receive filter's control-port routes (loopback, no hardware).
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""/filter, /filter/set and /filter/notch through the real control-port
handler: flags arrive as bools, words as words, numbers as floats; an
adapter without a filter answers {"available": false}; a rejected value is
{"error": "bad value: ..."} and never a traceback page.

Run:  python -m pytest aether_gate/tests/test_filter_routes.py
"""
import json
import socket
import time
import urllib.request


def _free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class FakeFilterAdapter:
    def __init__(self):
        self.calls = []

    def filter_status(self):
        self.calls.append(("status", {}))
        return {"available": True, "low_hz": 100, "high_hz": 2900, "shape": "soft"}

    def filter_set(self, **kw):
        self.calls.append(("set", kw))
        if kw.get("shape") == "medium":
            raise ValueError("shape must be one of ['sharp', 'soft']")
        return {"available": True, **kw}

    def filter_notch(self, add_hz=None, width_hz=140.0, clear=False, clear_hz=None):
        self.calls.append(("notch", {"add_hz": add_hz, "width_hz": width_hz,
                                     "clear": clear, "clear_hz": clear_hz}))
        return {"available": True}


class NoFilterAdapter:
    pass


def _start(adapter):
    from aether_gate.core import Radio
    from aether_gate.core.engine import start_control_server
    ctl_port = _free_port()
    r = Radio("127.0.0.1", None, adapter=adapter, port=_free_port())
    start_control_server(r, ctl_port)
    time.sleep(0.3)
    return r, ctl_port


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=2.0) as resp:
        assert resp.status == 200
        return json.loads(resp.read())


def test_no_filter_adapter_answers_unavailable():
    _, port = _start(NoFilterAdapter())
    assert _get(port, "/filter") == {"available": False}
    assert _get(port, "/filter/set?low=300") == {"available": False}
    assert _get(port, "/filter/notch?add=1000") == {"available": False}


def test_set_coerces_flags_words_and_numbers():
    a = FakeFilterAdapter()
    _, port = _start(a)
    assert _get(port, "/filter")["shape"] == "soft"
    r = _get(port, "/filter/set?low=300&high=2700&shape=SHARP&anf=on&auto=0&nb=true"
                   "&agc=slow&attack_ms=5&contour_db=-3.5")
    assert r["available"]
    assert a.calls[-1] == ("set", {"low": 300.0, "high": 2700.0, "shape": "sharp", "anf": True,
                                   "auto": False, "nb": True, "agc": "slow", "attack_ms": 5.0,
                                   "contour_db": -3.5})


def test_bad_values_are_errors_not_tracebacks():
    a = FakeFilterAdapter()
    _, port = _start(a)
    assert _get(port, "/filter/set?anf=maybe")["error"].startswith("bad value")
    assert _get(port, "/filter/set?low=abc")["error"].startswith("bad value")
    assert _get(port, "/filter/set?shape=medium")["error"].startswith("bad value")
    assert _get(port, "/filter/notch?width=100")["error"].startswith("bad value")   # no add


def test_notch_add_and_clear_forward():
    a = FakeFilterAdapter()
    _, port = _start(a)
    _get(port, "/filter/notch?add=1000&width=160")
    assert a.calls[-1] == ("notch", {"add_hz": 1000.0, "width_hz": 160.0, "clear": False,
                                     "clear_hz": None})
    _get(port, "/filter/notch?clear=1000")
    assert a.calls[-1][1]["clear"] is True and a.calls[-1][1]["clear_hz"] == 1000.0
    _get(port, "/filter/notch?clear=1")
    assert a.calls[-1][1] == {"add_hz": None, "width_hz": 140.0, "clear": True, "clear_hz": None}
