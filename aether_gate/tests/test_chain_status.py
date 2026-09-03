#
# Aether-gate — the /filter `chain` array: one row per stage, in signal order.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The chain is a CONTRACT, not a convenience. The app renders whatever rows
the gate sends, including stages it has never heard of, so a row id is a
public name and a row's shape is a promise: id, name, kind in
toggle|select|value|fixed, fixed, enabled, detail, and exactly one of action
or why. The golden id list below is what makes a rename a visible failure
here rather than a blank tile in the CHAIN window.

The other half of the promise is that nothing is invented: every number in
every row is quoted from the status dict it came from, and `measured` is
present only where the gate genuinely measured a level.

Run:  python -m pytest aether_gate/tests/test_chain_status.py
"""
import numpy as np

from aether_gate.adapters.chainstatus import CHAIN_KINDS, chain_rows
from aether_gate.adapters.soapy import SoapyAdapter

GOLDEN_IDS = [
    "antenna", "traps", "lna", "ifgr", "rf_agc",        # the front end
    "roof_rf", "adc",                                   # the hardware roofing
    "align", "nb", "roof_digital",                      # full-rate gate stages
    "combiner", "subband", "squeeze", "post",           # the pair
    "slice", "passband", "auto", "shape", "notch",      # the slice FIR
    "anf", "contour", "apf", "auto_eq",
    "detect", "agc", "app",                             # out of the gate
]


def _validate(row):
    """The §0.1 contract, as a function. Returns the row so it can be chained
    into an assert."""
    assert set(row) <= {"id", "name", "kind", "fixed", "enabled", "detail",
                        "value", "options", "action", "why", "measured"}, row
    for k in ("id", "name", "kind", "detail"):
        assert isinstance(row[k], str) and row[k], (k, row)
    assert row["kind"] in CHAIN_KINDS, row
    assert isinstance(row["fixed"], bool) and isinstance(row["enabled"], bool), row
    act, why = row["action"], row["why"]
    assert (act is None) != (why is None), row
    if act is not None:
        assert set(act) == {"label", "route", "query"}, row
        assert act["route"].startswith("/"), row
        assert isinstance(act["query"], str), row
    else:
        assert isinstance(why, str) and why, row
    if row["kind"] == "fixed":
        assert row["fixed"] and act is None, row
    if row["kind"] == "select":
        assert "value" in row, row
    if "options" in row:
        assert isinstance(row["options"], list) and row["options"], row
    m = row.get("measured")
    if m is not None:
        assert set(m) == {"in_db", "out_db"}, row
        assert any(v is not None for v in m.values()), row
        assert all(v is None or isinstance(v, float) for v in m.values()), row
    return row


def _filter():
    return {
        "low_hz": 350, "high_hz": 2400, "width_hz": 2050,
        "set_low_hz": 100, "set_high_hz": 2900,
        "shape": "sharp", "taps": 1023, "transition_hz": 49, "sideband": "lsb",
        "notches": [{"hz": 1000.0, "width_hz": 140.0, "depth_db": 34.2},
                    {"hz": 1500.0, "width_hz": 140.0, "depth_db": 28.0}],
        "notches_on": True, "bypass": False,
        "anf": {"enabled": True, "found_hz": [], "depth_db": []},
        "contour": {"enabled": False, "hz": None, "db": 0.0, "width_hz": None,
                    "auto": True, "source": None},
        "apf": {"enabled": False, "hz": 600.0, "width_hz": 150.0},
        "auto": {"enabled": True, "source": "spectrum", "low_hz": 350, "high_hz": 2400},
        "auto_eq": {"enabled": True, "tilt_db": -2.5, "lean_db": 1.8},
        "nb": {"enabled": True, "threshold_db": 13.0, "blanked_pct": 0.55},
        "agc": {"mode": "med", "attack_ms": 5.0, "decay_ms": 250.0, "hang_ms": 250.0,
                "threshold_db": 20.0, "gain_db": -14.9},
        "talker": {"enabled": True, "snap": "fast", "id": 3, "remembered": [2, 3]},
        "roofing": {"analogue_hz": 200000.0,
                    "analogue_options": [200000, 300000, 600000, 1536000],
                    "analogue_source": "rate",
                    "digital_hz": 25000, "digital_options": [25000, 3000],
                    "digital_full_hz": 25000.0, "samp_rate_hz": 250000.0},
        "mode": "LSB", "available": True,
    }


def _diversity():
    return {
        "available": True, "mode": "track", "phase_deg": 140.2, "ratio_db": -7.2,
        "lag_samples": 4032, "aligned": True, "corr_peak": 40.7, "realigning": False,
        "snr_db": {"a": -0.4, "b": 3.6, "out": 5.9},
        "nb": {"enabled": True, "threshold_db": 13.0, "blanked_pct": 0.55,
               "auto": {"mode": "auto", "armed": True, "threshold": 13.0,
                        "reason": "80 impulses/s at 17.2 dB: blanker on, threshold 13 dB",
                        "since_s": 55.7}},
        "subband": {"enabled": True, "bins": 12, "extra_db": 1.4},
        "post": {"enabled": True, "floor_db": -6.0, "mean_db": -8.5},
        "squeeze": {"hz": -1200, "width_hz": 300, "held": True, "tool": "null",
                    "why": "one wavefront, coherence 0.82: the null takes it",
                    "coherence": 0.82, "depth_db": -18.4, "target": "signal",
                    "comb": None},
    }


def _device():
    return {
        "antenna": {"value": "Tuner 1 50 ohm", "options": ["Tuner 1 50 ohm"]},
        "settings": [
            {"key": "rfgain_sel", "name": "RF Gain Select",
             "options": ["0", "1", "2", "3", "4"], "value": "4"},
            {"key": "agc_setpoint", "name": "AGC Setpoint", "value": "-30"},
            {"key": "rfnotch_ctrl", "name": "RfNotch Enable", "value": "true"},
            {"key": "dabnotch_ctrl", "name": "DabNotch Enable", "value": "false"},
        ],
    }


def _frontend():
    return {"gain_db": 47.0, "gain_range": (20, 59), "agc": False}


def _rows(**kw):
    args = {"filt": _filter(), "div": _diversity(),
            "device": _device(), "frontend": _frontend()}
    args.update(kw)
    return chain_rows(args["filt"], args["div"], args["device"], args["frontend"])


def _by_id(rows):
    return {r["id"]: r for r in rows}


def test_the_rows_are_the_golden_list_in_signal_order():
    assert [r["id"] for r in _rows()] == GOLDEN_IDS


def test_every_row_validates_against_the_contract():
    for row in _rows():
        _validate(row)


def test_a_single_tuner_simply_has_no_pair_rows():
    ids = [r["id"] for r in _rows(div=None)]
    assert [i for i in GOLDEN_IDS
            if i not in ("align", "combiner", "subband", "squeeze", "post")] == ids


def test_a_gate_with_no_device_surface_still_serves_the_chain():
    rows = _rows(device=None, frontend=None)
    assert [r["id"] for r in rows][:2] == ["roof_rf", "adc"]
    for row in rows:
        _validate(row)


def test_the_front_end_rows_are_fixed_and_say_where_to_set_them():
    rows = _by_id(_rows())
    for rid in ("antenna", "traps", "lna", "ifgr", "rf_agc"):
        assert rows[rid]["kind"] == "fixed" and rows[rid]["fixed"]
        assert rows[rid]["why"] == "set on the setup page"
    assert "state 4 of 0-4" in rows["lna"]["detail"]
    assert "47 dB of 20-59" in rows["ifgr"]["detail"]
    assert rows["traps"]["detail"] == "MW/FM on · DAB off"
    assert rows["rf_agc"]["detail"] == "off · set-point -30 dBfs"


def test_the_blankers_row_carries_the_auto_states_own_reason():
    nb = _by_id(_rows())["nb"]
    assert nb["detail"] == ("13.0 dB · 0.55 % blanked · auto: 80 impulses/s at "
                            "17.2 dB: blanker on, threshold 13 dB")
    assert nb["action"] == {"label": "OFF", "route": "/filter/set", "query": "nb=off"}


def test_an_idle_auto_blanker_says_so_rather_than_going_quiet():
    div = _diversity()
    div["nb"]["auto"] = {"mode": "auto", "armed": False, "reason": None}
    assert _by_id(_rows(div=div))["nb"]["detail"].endswith("auto: idle")


def test_the_pair_rows_quote_the_diversity_status_and_never_recompute_it():
    div, rows = _diversity(), _by_id(_rows())
    assert rows["align"]["value"] == div["lag_samples"]
    assert str(div["corr_peak"]) in rows["align"]["detail"]
    assert rows["combiner"]["value"] == div["mode"]
    assert rows["subband"]["enabled"] is div["subband"]["enabled"]
    assert str(div["subband"]["bins"]) in rows["subband"]["detail"]
    assert rows["post"]["measured"]["out_db"] == div["post"]["mean_db"]
    assert rows["combiner"]["measured"] == {"in_db": div["snr_db"]["b"],
                                            "out_db": div["snr_db"]["out"]}


def test_the_squeeze_row_quotes_the_tool_and_releases_while_held():
    div = _diversity()
    row = {r["id"]: r for r in _rows(div=div)}["squeeze"]
    assert row["enabled"] is True and row["kind"] == "value"
    assert "null" in row["detail"] and div["squeeze"]["why"] in row["detail"]
    assert "-1200 Hz" in row["detail"] and row["value"] == -1200
    assert row["action"] == {"label": "RELEASE", "route": "/diversity/set",
                             "query": "squeeze=off"}
    # Off: the query is left open for the VISUAL tab to finish.
    div["squeeze"] = {"hz": None, "held": False, "target": "signal", "comb": None}
    row = {r["id"]: r for r in _rows(div=div)}["squeeze"]
    assert row["enabled"] is False and row["action"]["query"] == "squeeze="
    # Comb armed but not found yet: says so, and can still be released.
    div["squeeze"] = {"hz": None, "held": False, "target": "comb", "comb": None,
                      "reason": "no comb found"}
    row = {r["id"]: r for r in _rows(div=div)}["squeeze"]
    assert "armed" in row["detail"] and "no comb found" in row["detail"]
    assert row["action"]["query"] == "squeeze=off" and row["value"] == "comb"


def test_the_auto_clean_row_heads_the_chain_only_when_a_governor_exists():
    assert _rows()[0]["id"] == "antenna"
    gov = {"auto": True, "state": "settling",
           "holding": [{"tool": "squeeze"}, {"tool": "nb"}],
           "why": "carrier at +1200 Hz: squeezed, waiting 2 s"}
    rows = chain_rows(_filter(), _diversity(), _device(), _frontend(), gov)
    row = rows[0]
    _validate(row)
    assert row["id"] == "auto_clean" and row["enabled"] is True
    assert "settling" in row["detail"] and "holding squeeze, nb" in row["detail"]
    assert gov["why"] in row["detail"]
    assert row["action"] == {"label": "OFF", "route": "/diversity/set", "query": "auto=off"}
    off = chain_rows(_filter(), _diversity(), _device(), _frontend(), {"auto": False})[0]
    assert off["enabled"] is False and off["action"]["query"] == "auto=on"
    assert off["detail"].startswith("off")


def test_measured_appears_only_where_a_level_was_measured():
    rows = _rows()
    have = {r["id"] for r in rows if r.get("measured") is not None}
    assert have == {"combiner", "post", "notch"}
    assert _by_id(rows)["notch"]["measured"] == {"in_db": None, "out_db": -34.2}


def test_a_toggles_label_is_the_verb_and_the_query_is_the_other_state():
    rows = _by_id(_rows())
    assert rows["anf"]["enabled"] and rows["anf"]["action"]["query"] == "anf=off"
    assert not rows["apf"]["enabled"] and rows["apf"]["action"]["query"] == "apf=on"
    assert rows["slice"]["action"] == {"label": "BYPASS", "route": "/filter/set",
                                       "query": "bypass=on"}


def test_a_bypassed_slice_filter_says_that_nothing_below_it_is_in_circuit():
    filt = _filter()
    filt["bypass"] = True
    row = _by_id(_rows(filt=filt))["slice"]
    assert not row["enabled"] and row["action"]["query"] == "bypass=off"
    assert "nothing below" in row["detail"]


def test_the_contour_row_switches_the_auto_fit_while_the_auto_fit_is_what_runs():
    rows = _by_id(_rows())
    assert rows["contour"]["action"]["query"] == "auto_contour=off"
    filt = _filter()
    filt["contour"] = {"enabled": False, "hz": 1200.0, "db": 0.0, "width_hz": 600.0,
                       "auto": False, "source": "manual"}
    assert _by_id(_rows(filt=filt))["contour"]["action"]["query"] == "contour=on"


def test_the_roofing_rows_offer_the_drivers_own_options():
    rows = _by_id(_rows())
    assert rows["roof_rf"]["options"] == [200000, 300000, 600000, 1536000]
    assert rows["roof_rf"]["action"]["query"] == "roof_hz="
    assert "the narrowest this hardware has" in rows["roof_rf"]["detail"]
    assert rows["roof_digital"]["action"]["query"] == "digital_roof_hz="
    assert rows["roof_digital"]["detail"] == "off · the full 25 kHz"
    assert not rows["roof_digital"]["enabled"]


def test_a_driver_that_will_not_list_its_bandwidths_says_why_instead():
    filt = _filter()
    filt["roofing"].pop("analogue_options")
    filt["roofing"].pop("digital_options")
    rows = _by_id(_rows(filt=filt))
    assert rows["roof_rf"]["kind"] == "value" and rows["roof_rf"]["action"] is None
    assert "does not offer its IF bandwidths" in rows["roof_rf"]["why"]
    assert _validate(rows["roof_digital"])["action"] is None


def test_the_detector_row_names_the_mode_the_gate_is_actually_running():
    filt = _filter()
    filt["mode"] = "NFM"
    assert _by_id(_rows(filt=filt))["detect"]["detail"] == "FM discriminator"
    assert _by_id(_rows())["detect"]["detail"] == "LSB product detector"


def test_the_adapter_serves_the_chain_from_filter_status():
    a = SoapyAdapter(driver="sdrplay", samp_rate=250_000.0, center_hz=3_890_000.0)
    a._np = np
    a._init_demod()
    a._mode = "LSB"
    st = a.filter_status()
    ids = [r["id"] for r in st["chain"]]
    assert ids == [i for i in GOLDEN_IDS
                   if i not in ("antenna", "traps", "lna", "align",
                                "combiner", "subband", "squeeze", "post")]
    for row in st["chain"]:
        _validate(row)
    assert st["roofing"]["samp_rate_hz"] == 250_000.0
    assert st["roofing"]["digital_full_hz"] == 25_000.0
