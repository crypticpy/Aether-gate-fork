#
# Aether-gate — the /diversity/dig control-port route (no hardware, loopback).
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The "dig this out" button, from the wire in.

The route is four things on one path — ?seconds=60|180|300 starts a run, a
bare GET is status, ?verdict=better|worse|keep labels it, ?cancel=1 stops it
— so what is tested here is the parsing and the error shapes. The search
itself is test_digout.py and the runner is test_diversity_dig.py; the fake
adapter's diversity_dig lives beside the other fakes in
test_diversity_routes.py.

Run:  python -m pytest aether_gate/tests/test_diversity_dig_routes.py
"""
from aether_gate.tests.test_diversity_routes import (
    FakeDiversityAdapter, FakeDiversityAdapterNoV2, FakeSingleChannelAdapter,
    _get, _start)


def test_dig_is_unavailable_rather_than_an_error_when_it_cannot_be_done():
    _, port = _start(FakeSingleChannelAdapter())
    assert _get(port, "/diversity/dig") == {"available": False}
    _, port2 = _start(FakeDiversityAdapterNoV2())
    assert _get(port2, "/diversity/dig") == {"available": False}


def test_bare_dig_is_status_and_starts_nothing():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    s = _get(port, "/diversity/dig")
    assert s["phase"] == "idle" and s["running"] is False
    assert a.calls[-1] == ("dig", {"seconds": None, "verdict": None,
                                    "cancel": False, "hz": None})


def test_dig_starts_for_one_of_the_three_lengths_the_button_offers():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    s = _get(port, "/diversity/dig?seconds=180")
    assert s["running"] is True and s["phase"] == "sampling"
    assert s["ends"] - s["started"] == 180
    assert a.calls[-1][1]["seconds"] == 180
    # and a second press while it runs is refused, not queued
    assert _get(port, "/diversity/dig?seconds=60")["error"] == "a dig is already running"


def test_a_dig_length_that_is_not_on_the_button_is_a_bad_value():
    _, port = _start(FakeDiversityAdapter())
    assert _get(port, "/diversity/dig?seconds=90")["error"].startswith("bad value:")
    assert _get(port, "/diversity/dig?seconds=abc")["error"].startswith("bad value:")


def test_the_operators_verdict_reaches_the_adapter_and_comes_back_as_a_record():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    _get(port, "/diversity/dig?seconds=60")
    assert _get(port, "/diversity/dig?verdict=better")["error"] == "the dig is still running"
    _get(port, "/diversity/dig?cancel=1")
    s = _get(port, "/diversity/dig?verdict=better")
    assert s["verdict"] == "better"
    assert s["record"]["kind"] == "dig" and s["record"]["gain_db"] == 2.0
    assert a.calls[-1][1]["verdict"] == "better"
    assert _get(port, "/diversity/dig?verdict=louder")["error"].startswith("bad value:")


def test_cancel_stops_the_dig():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    _get(port, "/diversity/dig?seconds=300")
    s = _get(port, "/diversity/dig?cancel=1")
    assert s["cancelled"] is True and s["running"] is False and s["phase"] == "done"
    assert a.calls[-1][1]["cancel"] is True
    # cancel=0 is not a cancel: the app can send the flag off without harm
    a2 = FakeDiversityAdapter()
    _, port2 = _start(a2)
    _get(port2, "/diversity/dig?cancel=0")
    assert a2.calls[-1][1]["cancel"] is False


def test_the_frequency_the_operator_is_on_is_forwarded():
    a = FakeDiversityAdapter()
    _, port = _start(a)
    _get(port, "/diversity/dig?seconds=60&hz=3962000")
    assert a.calls[-1][1]["hz"] == 3962000.0
    assert _get(port, "/diversity/dig?hz=nope")["error"].startswith("bad value:")
