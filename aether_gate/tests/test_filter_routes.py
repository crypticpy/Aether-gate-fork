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


def test_a_refused_write_is_logged_with_its_route_and_query(capsys):
    """The reply reaches one tile in the app and nothing else; without this
    line a refusal seen on screen an hour later cannot be traced to the
    write that earned it (2026-09-03: "bad value: NoneType ..." on the AUTO
    CLEAN card, route unknown). Mutation: drop the log() in _refuse."""
    a = FakeFilterAdapter()
    _, port = _start(a)
    assert _get(port, "/filter/set?low=abc")["error"].startswith("bad value")
    time.sleep(0.05)
    out = capsys.readouterr().out
    assert "/filter/set refused 'low=abc': ValueError" in out


def test_bypass_and_the_notch_flag_arrive_as_flags_not_as_floats():
    """Both are new entries in _FILTER_FLAGS; without them "bypass=on" would
    reach the adapter as float("on") and answer with a bad-value error."""
    a = FakeFilterAdapter()
    _, port = _start(a)
    assert _get(port, "/filter/set?bypass=on")["available"]
    assert a.calls[-1] == ("set", {"bypass": True})
    _get(port, "/filter/set?bypass=off&notches=0")
    assert a.calls[-1] == ("set", {"bypass": False, "notches": False})
    assert _get(port, "/filter/set?bypass=maybe")["error"].startswith("bad value")


def test_the_two_roofing_widths_pass_through_as_plain_numbers():
    """Neither needs an engine change: _filter_kwargs floats anything that is
    not a flag or a word, and the adapter is the one that validates them."""
    a = FakeFilterAdapter()
    _, port = _start(a)
    _get(port, "/filter/set?roof_hz=300000&digital_roof_hz=3000")
    assert a.calls[-1] == ("set", {"roof_hz": 300000.0, "digital_roof_hz": 3000.0})
    assert _get(port, "/filter/set?roof_hz=wide")["error"].startswith("bad value")


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


# ----- the slice exchange: edges in, edges out ---------------------------------
class FakeConn:
    def __init__(self):
        self.out = bytearray()

    def sendall(self, b):
        self.out.extend(b)


class EdgesAdapter:
    """A sim adapter that also owns a passband filter."""

    def __init__(self):
        from aether_gate.adapters import SimAdapter
        self._sim = SimAdapter(model="FLEX-6700")
        self.edges = (-2700.0, -100.0)
        self.set_calls = []

    def __getattr__(self, name):
        return getattr(self._sim, name)

    def filter_edges_hz(self):
        return self.edges

    def set_filter_edges_hz(self, low_hz, high_hz):
        self.set_calls.append((low_hz, high_hz))
        self.edges = (low_hz, high_hz)


def _radio_with(adapter):
    from aether_gate.core import Radio
    r = Radio("127.0.0.1", None, adapter=adapter, port=_free_port())
    pid = r._new_pan()
    r.slices[0] = {"freq": 7.204, "mode": "LSB", "active": True, "pan": pid}
    r.active_slice = 0
    return r


def test_slice_status_carries_the_filters_edges_and_slice_set_moves_them():
    a = EdgesAdapter()
    r = _radio_with(a)
    conn = FakeConn()
    r.emit_slice_status(conn, 0)
    text = conn.out.decode()
    assert "filter_lo=-2700 filter_hi=-100" in text, text
    r.on_line(conn, "C7|slice set 0 filter_low=-2400 filter_high=-300")
    assert a.set_calls[-1] == (-2400.0, -300.0)
    conn.out.clear()
    r.emit_slice_status(conn, 0)
    assert "filter_lo=-2400 filter_hi=-300" in conn.out.decode()


def test_an_adapter_without_a_filter_emits_no_edges():
    from aether_gate.adapters import SimAdapter
    r = _radio_with(SimAdapter(model="FLEX-6700"))
    conn = FakeConn()
    r.emit_slice_status(conn, 0)
    assert "filter_lo" not in conn.out.decode()


def test_the_flex_filt_command_moves_the_edges_and_answers():
    a = EdgesAdapter()
    r = _radio_with(a)
    conn = FakeConn()
    r.on_line(conn, "C9|filt 0 -2400 -300")
    text = conn.out.decode()
    assert "R9|0|" in text
    assert a.set_calls[-1] == (-2400.0, -300.0)
    assert "filter_lo=-2400 filter_hi=-300" in text          # status follows the write
    conn.out.clear()
    r.on_line(conn, "C10|filt 0 300 abc")
    assert "R10|50000000|" in conn.out.decode() and a.set_calls[-1] == (-2400.0, -300.0)
    conn.out.clear()
    r.on_line(conn, "C11|filt 0 2700 100")                    # low above high
    assert "R11|50000000|" in conn.out.decode()
