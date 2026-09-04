#
# Aether-gate — the weak, the narrow and the digital: what the finder missed.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The 2026-09-03 report, put to the finder as a scene it has to get right.

20 m, mid-afternoon, the pair on two loops. The spatial waterfall showed a
dense data block at 14070-14082, keyed columns at 14085, 14090, 14093, 14100
and 14110, a strong talker near 14170 and a weak one at 14178 that the
operator was copying by ear -- and /diversity/finder returned three
candidates, all called "voice", two of which were the FT8 and FT4 windows,
and not the one being listened to. The activity strip peaked at 0.09.

Everything here is synthesised into the same span the gate was on
(14068.75-14193.75 kHz), so the frequencies in the assertions are the
frequencies in the report.

Run:  python -m pytest aether_gate/tests/test_finder_weak.py
"""
import math

import numpy as np
import pytest

from aether_gate.core import finder_bands, finder_floor, kinds
from aether_gate.core.finder import Finder, WINDOW_STEP_POINTS
from aether_gate.core.finder_report import CANDIDATE_MAX, VOICE_SCORE

NBINS = 2048
RATE = 125_000.0
CHUNK = 4096
FRAME_S = CHUNK / RATE                 # ~30 frames a second, as the reader runs
CENTER = 14_131_250.0                  # the span the operator was on, exactly
FRAMES = 340                           # ~11 s: the fast ring holds 8.5 s of it
BIN_HZ = RATE / NBINS
WIN_BINS = 44                          # bins in one 2.7 kHz finder window
SYLLABLE_HZ = 4.0
TILT_DB = 2.5                          # measured across the field span that day


def _tilt(f):
    """Noise power per bin, tilted across the span as the band was."""
    return 10.0 ** (TILT_DB * (f - f.min()) / (f.max() - f.min()) / 10.0)


def _amp(snr_db, bins):
    """Per-bin signal amplitude for `snr_db` of WINDOW SNR against unit noise,
    where the signal covers `bins` of the window's WIN_BINS."""
    return math.sqrt((10.0 ** (snr_db / 10.0) - 1.0) * WIN_BINS / max(bins, 1))


class Scene:
    """A band, built one frame at a time, in the frequency domain."""

    def __init__(self, seed=3):
        self.rng = np.random.default_rng(seed)
        self.f = np.fft.fftfreq(NBINS, 1.0 / RATE) + CENTER
        self.floor = _tilt(self.f)
        self.parts = []

    def _lvl(self, sel):
        """Amplitude scale for the noise where the signal IS: an SNR asked for
        here is an SNR over the LOCAL floor, and the span is tilted."""
        return math.sqrt(float(np.mean(self.floor[sel])))

    def voice(self, hz, snr_db, width_hz=2400.0, duty=0.5):
        sel = (self.f >= hz) & (self.f < hz + width_hz)
        self.parts.append(("voice", sel, self._lvl(sel)
                           * _amp(snr_db, int(sel.sum())) / math.sqrt(duty), duty))
        return self

    def block(self, hz, snr_db, width_hz=3000.0, tone_hz=50.0):
        """A sub-band full of constant-envelope tones: FT8/FT4 as the map sees
        it -- filled edge to edge, on all the time, no syllables in it."""
        tones = [hz + i * tone_hz for i in range(int(width_hz / tone_hz))]
        sel = np.zeros(NBINS, dtype=bool)
        for t in tones:
            sel[int(np.argmin(np.abs(self.f - t)))] = True
        self.parts.append(("block", sel,
                           self._lvl(sel) * _amp(snr_db, int(sel.sum())), 1.0))
        return self

    def cw(self, hz, snr_db, wpm=20.0):
        sel = np.zeros(NBINS, dtype=bool)
        sel[int(np.argmin(np.abs(self.f - hz)))] = True
        self.parts.append(("cw", sel, self._lvl(sel) * _amp(snr_db, 1)
                           / math.sqrt(0.5), wpm / 12.0 * 5.0))
        return self

    def carrier(self, hz, snr_db):
        sel = np.zeros(NBINS, dtype=bool)
        sel[int(np.argmin(np.abs(self.f - hz)))] = True
        self.parts.append(("carrier", sel, self._lvl(sel) * _amp(snr_db, 1), 1.0))
        return self

    def rtty(self, hz, snr_db, shift_hz=170.0):
        sel = np.zeros(NBINS, dtype=bool)
        for t in (hz, hz + shift_hz):
            sel[int(np.argmin(np.abs(self.f - t)))] = True
        self.parts.append(("rtty", sel, self._lvl(sel) * _amp(snr_db, 2), 1.0))
        return self

    def _noise(self):
        n = (self.rng.normal(size=NBINS) + 1j * self.rng.normal(size=NBINS))
        return n * np.sqrt(self.floor / 2.0)

    def frames(self, n=FRAMES):
        t = 0.0
        for i in range(n):
            Xa, Xb = self._noise(), self._noise()
            for kind, sel, amp, rate in self.parts:
                if kind == "voice":
                    on = 1.0 if (t * SYLLABLE_HZ) % 1.0 < rate else 0.0
                elif kind == "cw":
                    on = 1.0 if (t * rate) % 1.0 < 0.5 else 0.0
                elif kind == "rtty":
                    on = 1.0
                else:
                    on = 1.0
                if on <= 0.0:
                    continue
                k = int(sel.sum())
                s = ((self.rng.normal(size=k) + 1j * self.rng.normal(size=k))
                     / math.sqrt(2) * amp if kind in ("voice", "block")
                     else np.full(k, amp) * np.exp(2j * np.pi * 0.37 * t))
                Xa[sel] += s
                Xb[sel] += s * np.exp(1j * 0.7)
            yield np.stack([Xa, Xb])
            t += FRAME_S

    def run(self, points=None, n=FRAMES):
        fd = Finder(NBINS, RATE, points=points)
        for X in self.frames(n):
            fd.update(X, FRAME_S)
        return fd


def _at(cands, hz, margin=1500.0):
    """The candidate nearest hz within margin, or None."""
    near = [c for c in cands if abs(c["hz"] - hz) <= margin]
    return min(near, key=lambda c: abs(c["hz"] - hz)) if near else None


# --- the local floor -------------------------------------------------------

def test_the_floor_is_local_so_a_tilt_and_a_dense_block_do_not_hide_a_talker():
    """The span the operator was on ran 2.5 dB downhill and had a solid block
    of digital modes across a fifth of it. One median for the whole span puts
    the floor above the quiet stretch a weak talker is sitting in."""
    points = 512
    step = RATE / points
    f = np.arange(points) * step
    floor = 10.0 ** (TILT_DB * f / f.max() / 10.0)
    band = np.copy(floor)
    for lo in (60, 100, 140, 180, 220):                 # five 5 kHz digital blocks
        band[lo:lo + 20] *= 30.0
    band[20:31] *= 2.0                                  # ...and one weak talker
    P = np.repeat(band[None, :], 64, axis=0)
    fl = finder_floor.local_floor(P, step)
    # the floor follows the tilt...
    assert fl[-1] / fl[0] == pytest.approx(10.0 ** (TILT_DB / 10.0), rel=0.15)
    # ...is not lifted by a block sitting in it...
    assert fl[70] == pytest.approx(floor[70], rel=0.2)
    # ...and leaves the weak talker the 3 dB it really has over the floor beside
    # it, where one median for the whole span -- dragged up the tilt by the
    # blocks, which is what the band did on the day -- reads under 1.5 dB
    local = 10.0 * np.log10(band[25] / fl[25])
    globalish = 10.0 * np.log10(band[25] / np.median(band))
    assert local == pytest.approx(3.0, abs=0.5), local
    assert globalish < 1.5, globalish


def test_presence_is_the_share_of_the_ring_a_point_stood_over_its_own_floor():
    rng = np.random.default_rng(5)
    points, n = 512, 256
    P = rng.gamma(2.0, 0.5, size=(n, points))           # bare band, two loops
    P[: n // 4, 200] *= 30.0                            # a signal, a quarter of it
    fl = finder_floor.local_floor(P, RATE / points)
    pres = finder_floor.presence(P, fl, RATE / points, 0.033)
    assert pres[200] == pytest.approx(0.25, abs=0.1), pres[200]
    assert np.median(pres) == 0.0
    assert float(np.mean(pres > 0.05)) < 0.05           # noise does not trip it


# --- the weak talker -------------------------------------------------------

@pytest.mark.parametrize("snr_db", (3.0, 6.0, 12.0))
def test_a_weak_talker_over_a_local_floor_is_found(snr_db):
    """Three decibels in a phone passband with syllables in it is copyable by
    ear, so it is a candidate. On the day, 4.0 dB scored 0.496 against a gate
    of 0.5 and everything under it was invisible."""
    fd = Scene(seed=11).voice(14_178_000.0, snr_db).block(14_074_000.0, 9.0).run()
    out = fd.candidates(CENTER)
    c = _at(out["candidates"], 14_178_000.0)
    assert c is not None, out["candidates"]
    assert c["score"] >= VOICE_SCORE, c
    assert c["kind"] == "voice" and c["syllabic"] >= 0.4, c
    assert c["snr_db"] == pytest.approx(snr_db, abs=2.5), c
    assert c["active_s"] >= 2.0, c


def test_the_dense_block_does_not_swallow_the_talker_beside_it():
    """Both of them, from one pass: the block IS a detection and so is the
    talker 100 kHz up the band."""
    fd = Scene(seed=12).voice(14_178_000.0, 4.0).block(14_074_000.0, 9.0).run()
    cands = fd.candidates(CENTER)["candidates"]
    assert _at(cands, 14_178_000.0) is not None, cands
    assert _at(cands, 14_075_000.0, margin=3_000.0) is not None, cands


# --- what the thing IS -----------------------------------------------------

def test_the_ft8_window_is_never_called_voice():
    """The bug, exactly: 14074.0 came back "voice 0.41" and 14080.5 "voice
    0.65". They are the FT8 and FT4 windows on 20 m."""
    fd = Scene(seed=13).block(14_074_000.0, 9.0).block(14_080_000.0, 6.0).run()
    cands = fd.candidates(CENTER)["candidates"]
    ft8 = _at(cands, 14_075_500.0, margin=3_000.0)
    ft4 = _at(cands, 14_081_500.0, margin=3_000.0)
    assert ft8 is not None and ft4 is not None, cands
    assert ft8["kind"] == "ft8" and ft8["hz"] == 14_074_000.0, ft8
    assert ft4["kind"] == "ft4" and ft4["hz"] == 14_080_000.0, ft4
    assert ft8["mode"] == "USB", ft8
    for c in cands:
        if finder_bands.sub_band(c["hz"] - 1500.0, c["hz"] + 1500.0):
            assert c["kind"] != "voice", c


def test_a_block_of_tones_away_from_any_allocation_is_data_not_voice():
    """Without the band plan to lean on, the structure alone still has to say
    "not a conversation": filled edge to edge, on all the time, no syllables."""
    fd = Scene(seed=14).block(14_120_000.0, 9.0).run()
    c = _at(fd.candidates(CENTER)["candidates"], 14_121_500.0, margin=3_000.0)
    assert c is not None
    assert c["kind"] in kinds.DIGITAL, c


def test_a_keyed_column_is_a_candidate_and_is_called_cw():
    """No CW column was EVER a candidate: a 200 Hz tone lifts its 2.7 kHz
    window by 2.8 dB however loud it is, and the list was gated on that."""
    fd = Scene(seed=15).cw(14_090_000.0, 15.0).cw(14_093_000.0, 12.0).run()
    cands = fd.candidates(CENTER)["candidates"]
    a, b = _at(cands, 14_090_000.0, 600.0), _at(cands, 14_093_000.0, 600.0)
    assert a is not None and b is not None, cands
    assert a["kind"] == "cw" and b["kind"] == "cw", (a, b)
    # ...and the dial goes ON a keyed tone, not beside its passband
    assert abs(a["hz_raw"] - 14_090_000.0) <= 300.0, a
    assert a["occupied_hz"] <= kinds.NARROW_HZ[1], a


def test_a_steady_tone_is_a_carrier():
    fd = Scene(seed=16).carrier(14_110_000.0, 15.0).run()
    c = _at(fd.candidates(CENTER)["candidates"], 14_110_000.0, 600.0)
    assert c is not None and c["kind"] == "carrier", c


def test_an_rtty_pair_is_named_where_the_map_can_resolve_the_shift():
    """170 Hz is under a map point at every normal span, so RTTY is honestly
    "data" there; give the finder the resolution and it says which."""
    fd = Scene(seed=17).rtty(14_100_000.0, 12.0).run(points=NBINS)
    c = _at(fd.candidates(CENTER)["candidates"], 14_100_100.0, 600.0)
    assert c is not None, "the pair was not found at all"
    assert c["kind"] == "rtty", c
    coarse = Scene(seed=17).rtty(14_100_000.0, 12.0).run()
    c2 = _at(coarse.candidates(CENTER)["candidates"], 14_100_100.0, 900.0)
    assert c2 is not None and c2["kind"] != "voice", c2


def test_what_cannot_be_named_is_kept_as_a_signal():
    """A candidate the classifier cannot name is not dropped and is not given
    the least-bad of the other names."""
    feat = {"snr_db": np.array([8.0]), "peak_db": np.array([8.0]),
            "bw_hz": np.array([1500.0]), "filled": np.array([0.5]),
            "depth": np.array([0.3]), "syllabic": np.array([0.42]),
            "occupancy": np.array([0.5]), "mid": np.array([0.5]),
            "duty": np.array([0.5]), "crest": np.array([2.0]),
            "floor_corr": np.array([0.0]), "peak_frac": np.array([0.3]),
            "resolved": 1.0, "shift_hz": np.array([0.0]), "resolves_shift": 0.0}
    code, conf = kinds.verdict(feat)
    assert kinds.name(code[0]) == "signal", (kinds.name(code[0]), conf[0])
    # ...and it is not a bet. "signal" says something is certainly there and
    # nothing named it, so the confidence in the KIND is a coin toss whatever
    # the confidence in the presence: it used to ship the presence itself, and
    # a row reading "signal 1.00" claims certainty about an admission.
    assert 0.0 < conf[0] <= kinds.SIGNAL_MAX_CONF, conf[0]


# --- the list, and the strips ----------------------------------------------

def test_the_list_is_sorted_and_capped():
    scene = Scene(seed=18)
    for i in range(60):                                 # a band full of columns
        scene.cw(14_072_000.0 + i * 2_000.0, 10.0 + (i % 5))
    cands = scene.run().candidates(CENTER)["candidates"]
    assert len(cands) == CANDIDATE_MAX, len(cands)
    scores = [c["score"] for c in cands]
    assert scores == sorted(scores, reverse=True), scores


def test_the_tuned_column_is_always_listed_even_when_it_scored_nothing():
    fd = Scene(seed=19).voice(14_170_000.0, 12.0).run()
    out = fd.candidates(CENTER, tuned_hz=14_178_000.0)
    tuned = [c for c in out["candidates"] if c["tuned"]]
    assert len(tuned) == 1, out["candidates"]
    assert abs(tuned[0]["hz"] - 14_178_000.0) <= 3_000.0, tuned
    assert all(c["tuned"] is False for c in out["candidates"] if c is not tuned[0])


def test_the_tuned_flag_lands_on_the_candidate_the_operator_is_listening_to():
    fd = Scene(seed=20).voice(14_178_000.0, 9.0).run()
    out = fd.candidates(CENTER, tuned_hz=14_178_500.0)
    tuned = [c for c in out["candidates"] if c["tuned"]]
    assert len(tuned) == 1 and tuned[0]["kind"] == "voice", out["candidates"]
    assert len([c for c in out["candidates"]
                if abs(c["hz"] - 14_178_000.0) <= 1_500.0]) == 1


def test_the_strips_report_every_kind_and_carry_their_own_maximum():
    """`activity` is anything at all now, `voice_share` is what it used to be,
    and `activity_max` is there so the app can stretch the strip: 186 columns
    of the field payload were non-zero and the largest was 0.092."""
    fd = Scene(seed=21).voice(14_170_000.0, 9.0).cw(14_110_000.0, 15.0).run()
    out = fd.candidates(CENTER)
    act = np.asarray(out["activity"])
    vs = np.asarray(out["voice_share"])
    assert len(act) == len(vs) == out["points"]
    assert out["activity_max"] == pytest.approx(float(act.max()), abs=1e-3)
    assert out["activity_max"] >= 0.5, out["activity_max"]
    step = RATE / out["points"]
    lo = CENTER - RATE / 2

    def col(hz):
        return int((hz - lo) / step)

    assert act[col(14_110_000.0)] >= 0.5, "the keyed column is not in the strip"
    assert vs[col(14_110_000.0)] <= 0.2, "a keyed column is not somebody talking"
    assert vs[col(14_171_000.0)] >= 0.3, "the talker is not in the voice strip"
    assert np.median(act) == 0.0                        # bare band stays dark


def test_the_payload_keeps_every_key_the_app_reads():
    fd = Scene(seed=22).voice(14_170_000.0, 12.0).run()
    out = fd.candidates(CENTER, tuned_hz=14_170_500.0)
    for k in ("available", "span_hz", "history_s", "points", "activity",
              "activity_max", "voice_share", "candidates"):
        assert k in out, k
    for c in out["candidates"]:
        for k in ("hz", "hz_raw", "mode", "width_hz", "score", "kind", "kind_conf",
                  "snr_db", "syllabic", "depth", "active_s", "last_s", "gain_db",
                  "occupied_hz", "tuned"):
            assert k in c, (k, c)
        assert c["kind"] in kinds.KINDS
