#
# Aether-gate — runtime dossier loader tests (no hardware, no network).
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Pins the dossier loader and the IC-9700 adapter's dossier wiring:
unwrap semantics, dotted get, fail-soft on missing/broken files, the
vendored dossier actually driving the adapter's curve/bands, and the one
deliberate fail-CLOSED case (explicit empty TX whitelist = no TX anywhere,
while an ABSENT dossier falls back to the baked whitelist).

Run:  python -m aether_gate.tests.test_dossier
"""
import json
import os
import sys
import tempfile

from aether_gate import dossiers
from aether_gate.dossiers import unwrap, load, Dossier

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {name}")
    else:
        FAIL += 1
        print(f"FAIL  {name}  {detail}")


def _adapter(**kw):
    """Offline adapter construction — __init__ does no I/O (open() does)."""
    from aether_gate.adapters.icom9700 import Icom9700Adapter
    return Icom9700Adapter("192.0.2.1", "user", "pass", local_ip="192.0.2.2", **kw)


def main():
    # --- unwrap semantics -------------------------------------------------
    check("unwrap plain", unwrap(5) == 5)
    check("unwrap wrapped", unwrap({"value": 7, "$note": "x"}) == 7)
    check("unwrap strips sidecars",
          unwrap({"a": 1, "$evidence": "tbd"}) == {"a": 1})
    check("unwrap nested wrapped",
          unwrap({"k": {"value": [1, 2], "$source": "s"}}) == {"k": [1, 2]})

    # --- dotted get on a synthetic dossier --------------------------------
    d = Dossier({"schema_version": "0.1.1",
                 "identity": {"model": "T"},
                 "meters": {"fwd": {"value": {"curve": [[0, 0.0]]},
                                    "$evidence": "hw-measured"}}}, "mem")
    check("get through wrapped mid-path",
          d.get("meters.fwd.curve") == [[0, 0.0]])
    check("get missing -> default", d.get("no.such.path", 42) == 42)

    # --- fail-soft loads --------------------------------------------------
    check("load unknown model -> None", load("NO-SUCH-RADIO") is None)
    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, "ic-bad.json")
        with open(bad, "w", encoding="utf-8") as f:
            f.write("{not json")
        os.environ["AETHER_GATE_DOSSIERS"] = tmp
        try:
            check("broken JSON -> None (fail-soft)", load("IC-BAD") is None)
        finally:
            del os.environ["AETHER_GATE_DOSSIERS"]

    # --- the vendored IC-9700 dossier drives the adapter ------------------
    a = _adapter()
    check("dossier loaded", a._dossier is not None,
          "vendored dossiers/ic-9700.json missing?")
    check("po curve from dossier (8 points)",
          len(a._po_curve) == 8 and a._po_curve[0] == (0, 0.0)
          and a._po_curve[-1] == (255, 1.0), repr(a._po_curve))
    check("tx power bands loaded (70cm = 75 W per spec)",
          any(lo == 420.0 and max_w == 75.0 for lo, hi, max_w in (a._tx_power_bands or ())),
          repr(a._tx_power_bands))
    check("tuning ranges = 3 disjoint bands",
          len(a.BAND_RANGES_MHZ) == 3 and (1240.0, 1300.0) in a.BAND_RANGES_MHZ)
    check("TX whitelist from x-gate: 2m+70cm only, 23cm refused",
          a.TX_BANDS_MHZ == ((144.0, 148.0), (420.0, 450.0)), repr(a.TX_BANDS_MHZ))

    # _fwd_power_w uses the dossier band table: raw 213 = 100% of band max.
    class _CivStub:
        fwdpwr_raw = 213
        freq_hz = 435_000_000
    a._civ = _CivStub()
    check("fwd power on 70cm scales to 75 W (dossier fixes the baked 100 W)",
          a._fwd_power_w() == 75.0, repr(a._fwd_power_w()))
    _CivStub.freq_hz = 1_296_000_000
    check("fwd power on 23cm scales to 10 W", a._fwd_power_w() == 10.0)

    # --- fail-CLOSED: explicit empty whitelist ----------------------------
    with tempfile.TemporaryDirectory() as tmp:
        strict = json.load(open(os.path.join(os.path.dirname(dossiers.__file__),
                                             "..", "dossiers", "ic-9700.json"),
                                encoding="utf-8"))
        strict["x-gate"]["tx_allowed_bands"] = {"value": [],
                                                "$note": "test: no TX anywhere"}
        with open(os.path.join(tmp, "ic-9700.json"), "w", encoding="utf-8") as f:
            json.dump(strict, f)
        os.environ["AETHER_GATE_DOSSIERS"] = tmp
        try:
            a2 = _adapter()
            check("explicit empty tx_allowed_bands -> NO TX (fail-closed)",
                  a2.TX_BANDS_MHZ == (), repr(a2.TX_BANDS_MHZ))
        finally:
            del os.environ["AETHER_GATE_DOSSIERS"]

    # --- fail-SOFT: no dossier at all -> baked constants ------------------
    real_search = dossiers._search_dirs
    dossiers._search_dirs = lambda: []
    try:
        a3 = _adapter()
        check("missing dossier -> baked TX whitelist intact",
              a3.TX_BANDS_MHZ == ((144.0, 148.0), (420.0, 450.0)))
        check("missing dossier -> baked po curve",
              a3._po_curve == a3._PO_CURVE)
        check("missing dossier -> None marker", a3._dossier is None)
    finally:
        dossiers._search_dirs = real_search

    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
