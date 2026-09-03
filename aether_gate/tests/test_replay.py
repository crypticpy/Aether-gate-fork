#
# Aether-gate — the replay lab on a synthetic capture, no hardware.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""A two-second synthetic capture (a talker in the USB passband on both
loops, a coherent noise source, white floor) through every configuration:
four WAVs of the same length at the same rate, a summary, and the combined
configurations at least as clean as the better loop.

Run:  python -m pytest aether_gate/tests/test_replay.py
"""
import json
import os
import wave

import numpy as np
import pytest

from aether_gate import replay

RATE = 125_000.0


def _capture(path, seconds=3.0):
    rng = np.random.default_rng(21)
    n = int(seconds * RATE)
    t = np.arange(n) / RATE
    f = np.fft.fftfreq(n, 1.0 / RATE)

    def band(lo, hi, p):
        X = np.zeros(n, dtype=np.complex128)
        sel = (f >= lo) & (f < hi)
        X[sel] = rng.normal(size=sel.sum()) + 1j * rng.normal(size=sel.sum())
        x = np.fft.ifft(X)
        return x * np.sqrt(p / max(1e-30, np.mean(np.abs(x) ** 2)))
    voice = band(300.0, 2700.0, 3.0) * (((t * 3.0) % 1.0) < 0.6)
    qrm = band(-20_000.0, 20_000.0, 4.0)
    a = voice + qrm + (rng.normal(size=n) + 1j * rng.normal(size=n)) * 0.5
    b = voice * 0.9 * np.exp(-0.6j) + qrm * np.exp(2.1j) \
        + (rng.normal(size=n) + 1j * rng.normal(size=n)) * 0.5
    np.savez(path, a=a.astype(np.complex64), b=b.astype(np.complex64), rate_hz=RATE,
             center_hz=14_200_000.0, lag_samples=0, aligned=True, seconds=seconds)


def test_replay_writes_comparable_wavs_and_a_summary(tmp_path, capsys):
    cap = str(tmp_path / "cap.npz")
    _capture(cap)
    out = str(tmp_path / "out")
    assert replay.main([cap, "--out", out, "--mode", "USB"]) == 0
    with open(os.path.join(out, "summary.json")) as fh:
        s = json.load(fh)
    assert set(s["results"]) == set(replay.CONFIGS)
    lengths = set()
    for c in replay.CONFIGS:
        with wave.open(os.path.join(out, f"{c}.wav")) as w:
            assert w.getnchannels() == 1 and w.getsampwidth() == 2
            assert w.getframerate() == 25_000
            lengths.add(w.getnframes())
            assert w.getnframes() >= 2.5 * 25_000
    assert len(lengths) == 1, lengths
    r = s["results"]
    # the combined outputs are cleaner than either loop alone: a louder
    # loud-over-quiet ratio (the over against the pauses)
    best_single = max(r["a"]["loud_over_quiet_db"], r["b"]["loud_over_quiet_db"])
    assert r["wideband"]["loud_over_quiet_db"] >= best_single - 0.5, r
    assert r["subband"]["loud_over_quiet_db"] >= best_single - 0.5, r
    assert r["subband"]["tracker"]["subband"]["enabled"] is True
    assert r["wideband"]["tracker"]["subband"]["enabled"] is False
    assert "wrote a.wav" in capsys.readouterr().out


def test_replay_filtered_config_runs_the_slice_filter_and_agc(tmp_path):
    cap = str(tmp_path / "cap.npz")
    _capture(cap)
    out = str(tmp_path / "out")
    assert replay.main([cap, "--out", out, "--mode", "USB", "--configs", "a,filtered",
                        "--filter", "low=250,high=2600,shape=soft,agc=slow,threshold_db=12"]) == 0
    with open(os.path.join(out, "summary.json")) as fh:
        r = json.load(fh)["results"]
    assert set(r) == {"a", "filtered"}
    f = r["filtered"]["filter"]
    # the settings reached core.filter and are the ones in force
    assert (f["set_low_hz"], f["set_high_hz"], f["shape"]) == (250.0, 2600.0, "soft")
    assert f["agc"]["mode"] == "slow" and f["agc"]["threshold_db"] == 12.0
    assert f["agc"]["gain_db"] is not None                  # the AGC ran
    assert "filter" not in r["a"]
    # the AGC put the audio near its target (0.25 -> -12 dB) whatever the
    # raw passband level was, and the threshold kept the pauses down
    assert -20.0 < r["filtered"]["rms_db"] < -6.0, r
    assert r["filtered"]["loud_over_quiet_db"] >= r["a"]["loud_over_quiet_db"], r
    assert r["filtered"]["tracker"]["subband"]["enabled"] is True
    lengths = set()
    for c in ("a", "filtered"):
        with wave.open(os.path.join(out, f"{c}.wav")) as w:
            lengths.add(w.getnframes())
            frames = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        assert np.max(np.abs(frames)) > 20_000, c          # each normalised on its own
    assert len(lengths) == 1, lengths


def test_replay_rejects_a_malformed_filter_setting(tmp_path):
    cap = str(tmp_path / "cap.npz")
    _capture(cap, seconds=1.0)
    with pytest.raises(SystemExit):
        replay.main([cap, "--out", str(tmp_path / "o"), "--configs", "a", "--filter", "low"])
