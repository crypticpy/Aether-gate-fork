#
# Aether-gate — a print of each remembered talker: their voice and their rig.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The talker memory tells stations apart by where they arrive from. Two
stations from the same direction are one talker to it, and a station heard
again after moving the antennas is a stranger. A print of the audio itself
is the second opinion: it is free, it survives a change of geometry, and it
reads as a sentence an operator can check by ear.

Per over, from the combined passband while the tracker says someone is
talking:

  * the audio spectrum in 100 Hz bands to 3.2 kHz. Its lower and upper
    edges (20 dB below the peak band) are the TRANSMITTER: rigs differ in
    their TX filter (2.4 vs 2.9 kHz, a 300 Hz roll-off vs ESSB to 80 Hz)
    and the same operator keeps the same one. The centroid and the tilt
    (1.5–2.5 kHz over 300–800 Hz, in dB) are mostly the VOICE.
  * the syllabic rate: the peak of the envelope's spectrum between 2 and
    8 Hz. How fast someone talks.
  * how long their overs run.

Overs shorter than MIN_OVER_S teach nothing; a print is a slow EMA over
overs, so one shout does not rewrite it.
"""
import math

import numpy as np

HOP = 1024               # analysis hop, passband samples (~41 ms at 25 kHz)
BAND_HZ = 100.0
BANDS = 32               # 0 .. 3.2 kHz
EDGE_DB = 20.0
ENV_MAX_S = 20.0         # envelope kept per over for the syllabic rate
SYL_LO, SYL_HI = 2.0, 8.0
MIN_OVER_S = 1.5
MERGE = 0.3              # a new over's weight in the talker's print
# THE SECOND OPINION, USED. The memory recalls a talker by bearing within
# 50 ms; two people from one bearing are one talker to it. From VOICE_CHECK_S
# into an over the running print is compared with the recalled talker's,
# and a distance of DIFFERENT_VOICE or more (centroid 250 Hz, top edge
# 400 Hz, tilt 4 dB each count as one) says this is somebody else at that
# bearing. An over that ends unlike the print it was headed for does not
# teach it either.
VOICE_CHECK_S = 1.0
DIFFERENT_VOICE = 0.9


class _Over:
    __slots__ = ("bands", "n", "env", "seconds", "talker")

    def __init__(self):
        self.bands = np.zeros(BANDS)
        self.n = 0
        self.env = []
        self.seconds = 0.0
        self.talker = None


class VoicePrint:
    def __init__(self, rate_hz, hop=HOP):
        self.rate_hz = float(rate_hz)
        self.hop = int(hop)
        self.win = np.hanning(self.hop)
        f = np.fft.rfftfreq(self.hop, 1.0 / self.rate_hz)
        self._band = np.minimum((f / BAND_HZ).astype(int), BANDS)      # BANDS = the bin above 3.2 kHz
        self._buf = np.zeros(0)
        self._cur = None
        self.prints = {}            # talker id -> {bands, syllabic, over_s, overs}
        self.env_hz = self.rate_hz / self.hop

    # --- feeding ----------------------------------------------------------------
    def feed(self, y, talking, talker_id):
        """One passband block (complex, one-sided) with the tracker's VAD and
        the memory's live talker id (None until recalled or stored)."""
        audio = np.real(np.asarray(y))
        if talking:
            if self._cur is None:
                self._cur = _Over()
            if talker_id is not None:
                self._cur.talker = talker_id
            self._buf = np.concatenate([self._buf, audio])
            while len(self._buf) >= self.hop:
                self._hop(self._buf[:self.hop])
                self._buf = self._buf[self.hop:]
        elif self._cur is not None:
            self._finish()

    def _hop(self, x):
        cur = self._cur
        X = np.abs(np.fft.rfft(x * self.win)) ** 2
        cur.bands += np.bincount(self._band, weights=X, minlength=BANDS + 1)[:BANDS]
        cur.n += 1
        if len(cur.env) < int(ENV_MAX_S * self.env_hz):
            cur.env.append(math.sqrt(float(np.mean(x * x))))
        cur.seconds += self.hop / self.rate_hz

    def _finish(self):
        cur, self._cur, self._buf = self._cur, None, np.zeros(0)
        if cur.talker is None or cur.seconds < MIN_OVER_S or cur.n == 0:
            return
        bands = cur.bands / cur.n
        syl = self._syllabic(np.asarray(cur.env))
        p = self.prints.get(cur.talker)
        if p is None:
            self.prints[cur.talker] = {"bands": bands, "syllabic": syl,
                                       "over_s": cur.seconds, "overs": 1}
            return
        d = self.distance(self._summarise(bands, syl, cur.seconds, 0),
                          self._summarise(p["bands"], p["syllabic"], p["over_s"], p["overs"]))
        if d is not None and d >= DIFFERENT_VOICE:
            return                                # somebody else's over: not this print's
        p["bands"] = (1 - MERGE) * p["bands"] + MERGE * bands
        if syl is not None:
            p["syllabic"] = syl if p["syllabic"] is None else (1 - MERGE) * p["syllabic"] + MERGE * syl
        p["over_s"] = (1 - MERGE) * p["over_s"] + MERGE * cur.seconds
        p["overs"] += 1

    def _syllabic(self, env):
        if len(env) < int(2.0 * self.env_hz):
            return None
        e = env - env.mean()
        n = 1 << max(9, int(math.ceil(math.log2(len(e) * 4))))
        S = np.abs(np.fft.rfft(e * np.hanning(len(e)), n)) ** 2
        f = np.fft.rfftfreq(n, 1.0 / self.env_hz)
        sel = (f >= SYL_LO) & (f <= SYL_HI)
        if not sel.any() or S[sel].max() <= 0:
            return None
        return float(f[sel][np.argmax(S[sel])])

    # --- reading ----------------------------------------------------------------
    def summary(self, talker_id):
        p = self.prints.get(talker_id)
        if p is None:
            return None
        return self._summarise(p["bands"], p["syllabic"], p["over_s"], p["overs"])

    def current(self):
        """The running over so far, once it is long enough to be judged
        (VOICE_CHECK_S); None otherwise."""
        cur = self._cur
        if cur is None or cur.n == 0 or cur.seconds < VOICE_CHECK_S:
            return None
        return self._summarise(cur.bands / cur.n, self._syllabic(np.asarray(cur.env)),
                               cur.seconds, 0)

    @staticmethod
    def distance(a, b):
        """How unlike two summaries are; DIFFERENT_VOICE and up is another
        person (or another rig). None when either is missing."""
        if a is None or b is None:
            return None
        d = ((a["centroid_hz"] - b["centroid_hz"]) / 250.0) ** 2 \
            + ((a["high_hz"] - b["high_hz"]) / 400.0) ** 2
        if a["tilt_db"] is not None and b["tilt_db"] is not None:
            d += ((a["tilt_db"] - b["tilt_db"]) / 4.0) ** 2
        return math.sqrt(d)

    @staticmethod
    def _summarise(b, syllabic, over_s, overs):
        centres = (np.arange(BANDS) + 0.5) * BAND_HZ
        total = float(b.sum())
        if total <= 0:
            return None
        peak = float(b.max())
        loud = np.nonzero(b >= peak * 10 ** (-EDGE_DB / 10))[0]
        lo_hz, hi_hz = loud[0] * BAND_HZ, (loud[-1] + 1) * BAND_HZ
        p_lo = float(b[3:8].sum())            # 300 .. 800 Hz
        p_hi = float(b[15:25].sum())          # 1.5 .. 2.5 kHz
        tilt = 10 * math.log10(p_hi / p_lo) if p_lo > 0 and p_hi > 0 else None
        return {"centroid_hz": round(float((centres * b).sum() / total)),
                "low_hz": round(lo_hz), "high_hz": round(hi_hz),
                "tilt_db": None if tilt is None else round(tilt, 1),
                "syllabic_hz": None if syllabic is None else round(syllabic, 1),
                "over_s": round(over_s, 1), "overs": int(overs)}

    def forget(self, keep_ids=None):
        """Drop every print, or every print whose talker is not in keep_ids."""
        if keep_ids is None:
            self.prints = {}
        else:
            self.prints = {k: v for k, v in self.prints.items() if k in keep_ids}
