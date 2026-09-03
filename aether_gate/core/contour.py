#
# Aether-gate — the auto contour: one bell that takes the microphone out.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The filter's CONTOUR is a bell (hz, dB, width). This fits one to a
talker from their voice print's 100 Hz band profile, so the operator does
not have to find the boom or the honk by ear for every station.
"""
import numpy as np

# THE AUTO CONTOUR. A talker's 100 Hz band profile (their print, learned
# from 1.5 s of them live) is read against a long-term average speech
# spectrum with the tilt taken out (the tilt is auto EQ's job). What is left
# is the microphone and the speech processor: a proximity boom, a presence
# peak, a scooped or honking midrange. The largest smooth deviation becomes
# one bell that leans against it -- part of it, never all, and never more
# than AUTO_CONTOUR_MAX_DB -- so formants stay the voice they are.
AUTO_CONTOUR_LOW_HZ = 300.0
AUTO_CONTOUR_HIGH_HZ = 2500.0
AUTO_CONTOUR_MAX_DB = 6.0
AUTO_CONTOUR_STRENGTH = 0.6
AUTO_CONTOUR_MIN_DB = 1.5            # smaller deviations are the voice, not the microphone
AUTO_CONTOUR_MIN_WIDTH_HZ = 200.0
AUTO_CONTOUR_MAX_WIDTH_HZ = 1200.0
PROFILE_BAND_HZ = 100.0
PROFILE_BANDS = 32
PROFILE_HZ = (np.arange(PROFILE_BANDS) + 0.5) * PROFILE_BAND_HZ
# long-term average speech spectrum, both sexes (after Byrne et al. 1994),
# dB re its peak, on the profile's grid
SPEECH_DB = np.interp(PROFILE_HZ,
                      (100, 200, 300, 400, 500, 600, 800, 1000, 1250, 1600, 2000, 2500, 3200),
                      (-8, -4, -1, 0, 0, -1, -4, -6, -8, -10, -12, -14, -16))


def fit_contour(profile_db):
    """One bell that takes the microphone out of a talker's band profile
    (dB per 100 Hz band, PROFILE_BANDS of them): (hz, db, width_hz), or None
    when the profile is speech-shaped enough to leave alone."""
    p = np.asarray(profile_db, dtype=float)
    if p.shape != (PROFILE_BANDS,) or not np.all(np.isfinite(p)):
        return None
    sel = (PROFILE_HZ >= AUTO_CONTOUR_LOW_HZ) & (PROFILE_HZ <= AUTO_CONTOUR_HIGH_HZ)
    hz = PROFILE_HZ[sel]
    r = p[sel] - SPEECH_DB[sel]
    x = np.log2(hz)                                  # the tilt, in dB per octave, comes out
    a = np.vstack([x, np.ones_like(x)]).T
    line = a @ np.linalg.lstsq(a, r, rcond=None)[0]
    keep = np.abs(r - line) <= AUTO_CONTOUR_MIN_DB   # ...fitted past the bump, not through it
    if keep.sum() >= 3:
        line = a @ np.linalg.lstsq(a[keep], r[keep], rcond=None)[0]
    r = r - line
    r = np.convolve(np.pad(r, 1, mode="edge"), np.ones(3) / 3.0, mode="valid")
    i = int(np.argmax(np.abs(r)))
    d = float(r[i])
    if abs(d) < AUTO_CONTOUR_MIN_DB:
        return None
    # the lobe: the run around the peak still on the same side, past half depth
    i0 = i1 = i
    while i0 > 0 and r[i0 - 1] * d > 0 and abs(r[i0 - 1]) >= abs(d) / 2:
        i0 -= 1
    while i1 < len(r) - 1 and r[i1 + 1] * d > 0 and abs(r[i1 + 1]) >= abs(d) / 2:
        i1 += 1
    width = (i1 - i0 + 1) * PROFILE_BAND_HZ
    width = max(AUTO_CONTOUR_MIN_WIDTH_HZ, min(AUTO_CONTOUR_MAX_WIDTH_HZ, width))
    db = max(-AUTO_CONTOUR_MAX_DB, min(AUTO_CONTOUR_MAX_DB, -AUTO_CONTOUR_STRENGTH * d))
    return float(hz[i]), round(db, 1), float(width)
