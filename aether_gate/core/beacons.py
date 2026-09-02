#
# Aether-gate — the NCDXF/IARU beacons as a calibration source for the pair.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Eighteen beacons round the world share five frequencies on a three-minute
cycle: each sends, in a ten-second slot, its callsign at 100 W and then four
one-second dashes at 100, 10, 1 and 0.1 W. The schedule is GPS-locked to
UTC, so at any instant we know who SHOULD be on 14.100 (and 18.110, 21.150,
24.930, 28.200): a labelled, timed, four-level test signal from eighteen
known directions, every three minutes, for free.

For a two-loop pair that is a calibration lab. Per beacon and band:

  * SNR on each loop (in a 500 Hz bandwidth, the CW convention) and the
    pair's phase difference and coherence at the beacon's frequency (the
    phase from a known direction is what a geometry solve wants);
  * which of the four power steps was still heard: the band's real reach in
    dB, not a guess from the noise floor;
  * how much the second loop would add (the MRC bound from the two SNRs).

Detection is narrowband: the aligned pair is mixed to the beacon frequency
(phase-continuous), boxcar-decimated x16 and spectrum-analysed in ~20 ms
chunks; the beacon is the strongest bin within +-100 Hz of nominal over the
slot, the floor is the median of the other bins. A slot is scored when it
ends. Nothing is done unless a beacon frequency is inside the span.
"""
import math

import numpy as np

SLOT_S = 10.0
SLOTS = 18
BANDS_HZ = (14_100_000.0, 18_110_000.0, 21_150_000.0, 24_930_000.0, 28_200_000.0)
# in slot order on 14.100 (4U1UN at 00:00:00 UTC); on the next band one slot later
BEACONS = (
    ("4U1UN", "United Nations, New York"), ("VE8AT", "Eureka, Nunavut"),
    ("W6WX", "Mt Umunhum, California"), ("KH6RS", "Maui, Hawaii"),
    ("ZL6B", "Masterton, New Zealand"), ("VK6RBP", "Rolystone, Australia"),
    ("JA2IGY", "Mt Asama, Japan"), ("RR9O", "Novosibirsk, Russia"),
    ("VR2B", "Hong Kong"), ("4S7B", "Colombo, Sri Lanka"),
    ("ZS6DN", "Pretoria, South Africa"), ("5Z4B", "Kikuyu, Kenya"),
    ("4X6TU", "Tel Aviv, Israel"), ("OH2B", "Lohja, Finland"),
    ("CS3B", "Sao Jorge, Madeira"), ("LU4AA", "Buenos Aires, Argentina"),
    ("OA4B", "Lima, Peru"), ("YV5B", "Caracas, Venezuela"),
)
STEPS_W = (100.0, 10.0, 1.0, 0.1)
DECIM = 16
CHUNK_S = 0.02
SEARCH_HZ = 100.0
HEARD_DB = 6.0             # a step counts as heard this far over the bin's floor
REF_BW_HZ = 500.0          # SNRs are reported in this bandwidth, the CW convention
EDGE_MARGIN_HZ = 5_000.0


def slot_at(t_utc):
    return int(t_utc // SLOT_S) % SLOTS


def on_air(t_utc, band_hz):
    """(callsign, location, seconds left in the slot) for band_hz at t_utc."""
    j = BANDS_HZ.index(band_hz)
    k = (slot_at(t_utc) - j) % SLOTS
    call, loc = BEACONS[k]
    return call, loc, SLOT_S - (t_utc % SLOT_S)


class BeaconWatch:
    def __init__(self, rate_hz):
        self.rate_hz = float(rate_hz)
        self.chunk = max(8, int(round(CHUNK_S * self.rate_hz / DECIM)))
        self.band_hz = None
        self._offset_hz = 0.0
        self._phase = 0.0                  # NCO phase, continuous across blocks
        self._fifo_a = np.zeros(0, dtype=np.complex128)
        self._fifo_b = np.zeros(0, dtype=np.complex128)
        self._slot = None                  # (slot index, band) being collected
        self._spec = []                    # per chunk: (Paa, Pbb, Pab) over the bins
        self.results = {}                  # (band_hz, call) -> dict
        self.last = None

    # --- driving ---------------------------------------------------------------
    def update(self, a, b, center_hz, t_utc):
        band = self._band_in_span(center_hz)
        if band is None:
            if self._slot is not None:
                self._slot, self._spec = None, []
            self.band_hz = None
            return
        if band != self.band_hz:
            self.band_hz = band
            self._offset_hz = band - center_hz
            self._phase = 0.0
            self._fifo_a = self._fifo_a[:0]; self._fifo_b = self._fifo_b[:0]
            self._slot, self._spec = None, []
        slot = slot_at(t_utc)
        if self._slot is not None and self._slot[0] != slot:
            self._score(self._slot, t_utc)
            self._slot, self._spec = None, []
        if self._slot is None:
            self._slot = (slot, band, t_utc)
        n = min(len(a), len(b))
        k = np.arange(n)
        w = -2.0 * math.pi * self._offset_hz / self.rate_hz
        rot = np.exp(1j * (self._phase + w * k))
        self._phase = (self._phase + w * n) % (2.0 * math.pi)
        m = n // DECIM
        xa = (np.asarray(a[:m * DECIM], dtype=np.complex128) * rot[:m * DECIM]).reshape(m, DECIM).mean(axis=1)
        xb = (np.asarray(b[:m * DECIM], dtype=np.complex128) * rot[:m * DECIM]).reshape(m, DECIM).mean(axis=1)
        self._fifo_a = np.concatenate([self._fifo_a, xa])
        self._fifo_b = np.concatenate([self._fifo_b, xb])
        c = self.chunk
        while len(self._fifo_a) >= c:
            A = np.fft.fft(self._fifo_a[:c]); B = np.fft.fft(self._fifo_b[:c])
            self._fifo_a = self._fifo_a[c:]; self._fifo_b = self._fifo_b[c:]
            self._spec.append((np.abs(A) ** 2, np.abs(B) ** 2, A * np.conj(B)))

    def _band_in_span(self, center_hz):
        half = self.rate_hz / 2.0 - EDGE_MARGIN_HZ
        for band in BANDS_HZ:
            if abs(band - center_hz) <= half:
                return band
        return None

    # --- scoring a finished slot ---------------------------------------------
    def _score(self, slot, t_now):
        idx, band, t0 = slot
        if len(self._spec) < int(0.8 * SLOT_S / CHUNK_S):
            return                                          # a partial slot: retune, start
        Paa = np.stack([s[0] for s in self._spec]); Pbb = np.stack([s[1] for s in self._spec])
        Pab = np.stack([s[2] for s in self._spec])
        c = Paa.shape[1]
        f = np.fft.fftfreq(c, DECIM / self.rate_hz)
        bin_hz = self.rate_hz / DECIM / c
        ref = 10.0 * math.log10(REF_BW_HZ / bin_hz)         # bin SNR -> SNR in REF_BW_HZ
        search = np.abs(f) <= SEARCH_HZ
        both = Paa + Pbb
        mean_spec = np.mean(both, axis=0)
        k = int(np.argmax(np.where(search, mean_spec, -np.inf)))
        others = ~search
        floor_a = float(np.median(Paa[:, others])); floor_b = float(np.median(Pbb[:, others]))
        pa = Paa[:, k]; pb = Pbb[:, k]; p = pa + pb
        floor = floor_a + floor_b
        # 1 s moving average, the 100 W dash is the strongest second
        per_s = int(round(1.0 / CHUNK_S))
        ma = np.convolve(p, np.ones(per_s) / per_s, mode="valid")
        i0 = int(np.argmax(ma))
        peak = float(ma[i0])
        bin_snr = 10.0 * math.log10(max(peak - floor, 1e-30) / max(floor, 1e-30))
        snr_db = bin_snr - ref
        heard = bin_snr >= HEARD_DB
        # the four dashes: 1 s each from the strongest second on; a step is
        # heard when its second stands HEARD_DB over the floor
        steps = []
        for s in range(len(STEPS_W)):
            j = i0 + s * per_s
            if j >= len(ma):
                break
            steps.append(10.0 * math.log10(max(float(ma[j]) - floor, 1e-30) / max(floor, 1e-30)))
        steps_heard = 0
        for d in steps:
            if d >= HEARD_DB:
                steps_heard += 1
            else:
                break
        strong = p >= floor + 0.5 * (peak - floor)
        if heard and strong.any():
            xab = complex(np.sum(Pab[strong, k]))
            saa = float(np.sum(pa[strong])); sbb = float(np.sum(pb[strong]))
            phase = math.degrees(math.atan2(xab.imag, xab.real))
            coh = min(1.0, abs(xab) ** 2 / max(saa * sbb, 1e-30))
            sa = max(float(np.mean(pa[strong])) - floor_a, 0.0)
            sb = max(float(np.mean(pb[strong])) - floor_b, 0.0)
            snr_a = 10.0 * math.log10(max(sa, 1e-30) / max(floor_a, 1e-30)) - ref
            snr_b = 10.0 * math.log10(max(sb, 1e-30) / max(floor_b, 1e-30)) - ref
            r = min(sa / sb, sb / sa) if sa > 0 and sb > 0 else 0.0
            gain = 10.0 * math.log10(1.0 + r)
        else:
            phase = coh = None
            snr_a = snr_b = gain = None
        call, loc = BEACONS[(idx - BANDS_HZ.index(band)) % SLOTS]
        res = {
            "call": call, "location": loc, "band_hz": band, "at": float(t0 - (t0 % SLOT_S)),
            "heard": bool(heard), "snr_db": round(snr_db, 1),
            "offset_hz": round(float(f[k]), 1),
            "snr_a": None if snr_a is None else round(snr_a, 1),
            "snr_b": None if snr_b is None else round(snr_b, 1),
            "phase_deg": None if phase is None else round(phase, 1),
            "coherence": None if coh is None else round(coh, 2),
            "gain_db": None if gain is None else round(gain, 1),
            "steps_db": [round(d - ref, 1) for d in steps],
            "steps_heard": int(steps_heard),
            "lowest_w": STEPS_W[steps_heard - 1] if steps_heard else None,
        }
        self.results[(band, call)] = res
        self.last = res

    def status(self, t_utc):
        now = None
        if self.band_hz is not None:
            call, loc, left = on_air(t_utc, self.band_hz)
            now = {"call": call, "location": loc, "seconds_left": round(left, 1)}
        rows = sorted(self.results.values(), key=lambda r: (r["band_hz"], -r["at"]))
        return {"available": True, "band_hz": self.band_hz, "slot": slot_at(t_utc),
                "now": now, "results": rows, "last": self.last}
