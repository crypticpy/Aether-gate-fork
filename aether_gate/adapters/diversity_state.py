#
# Aether-gate — the adapter's side of dual-tuner diversity.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""_DiversityState: everything the Soapy adapter keeps for an RSPduo whose two
tuners flow on one stream. core/diversity and core/spatial hold the maths;
this module wires them to the adapter's three threads:

  READER   ingest(a, b): alignment measurement, the noise blanker, the raw
           two-channel capture, the per-bin spatial map, and the block the
           panadapter and meters see (A, B, the slice's beam, or the map's
           per-bin nulls).
  AUDIO    observe(sid, xa, xb): the slice's tracker sees the in-band and
           guard-band covariances of the mixed blocks and hands back the
           (previous, new) weight pair the demod ramps between.
  CONTROL  status()/set()/map()/capture()/memory_clear().

Every shared value is a Python scalar, a small dict swapped whole, or an
object replaced atomically, so no lock is needed for a reader to see a
consistent weight; the two places two threads mutate the same list
(calibration accumulation, capture accumulation) take a lock.

Weights are PER SLICE: the beam is arithmetic on the same two streams, so
the slice on a net can be steered at whoever is talking while a second slice
keeps its own weight. The panadapter and S-meter follow the slice the audio
is on (active_slice).
"""
import os
import threading
import time

SSB_PASS_HZ = 3000.0        # mirrors soapy.SSB_PASS_HZ / FM_PASS_HZ: the bands
FM_PASS_HZ = 8000.0         # the demodulator actually passes (see _init_demod)


def _dv():
    """core.diversity, imported on first use: it needs numpy, this module must not."""
    from ..core import diversity
    return diversity


def _sp():
    from ..core import spatial
    return spatial


def _is_fm(mode):
    return (mode or "").upper() in ("FM", "NFM", "DFM")


class _DiversityState:
    CAL_SECONDS = 0.5          # of raw IQ cross-correlated to find the lag
    CAL_SAMPLES_MAX = 1 << 17  # 131072 — caps the FFT at 2.04 MS/s to ~64 ms of
                                # reader-thread stall instead of the 103 ms a
                                # full 0.5 s (2^21-point FFT) costs, which was
                                # long enough to overflow the driver and trigger
                                # another realign under its own stall
    MODES = ("off", "manual", "null", "track")
    SOURCES = ("combined", "a", "b")             # the v1 `source` vocabulary
    PANS = ("combined", "a", "b", "nulled")      # what the panadapter shows
    MAP_BINS = 2048            # spatial-map resolution (61 Hz at 125 kS/s)
    GUARD_GAP_HZ = 300.0       # between the passband edge and its guard band
    NB_DEFAULT_DB = 12.0
    CAPTURE_DIR = "~/aether-gate-captures"

    def __init__(self, adapter):
        self.a = adapter
        self.aligner = _dv().Aligner()
        self.mode = "off"
        self.pan = "combined"
        self.manual = {}                    # slice_id -> complex weight m
        self.trackers = {}
        self.passband = {}                  # sid -> PassbandPhase                  # slice_id -> Tracker (rebuilt on a rate change)
        self.last_m = {}                    # slice_id -> weight the last block ended on
        self.memory = _dv().TalkerMemory()  # shared by every slice's tracker
        self.active_slice = 0
        self._cal_a, self._cal_b, self._cal_n = [], [], 0
        # Guards _cal_a/_cal_b/_cal_n: request_realign() can land from the
        # HTTP thread (diversity_realign) while ingest() is mid-accumulate on
        # the reader thread, and an empty list handed to np.concatenate raises.
        self._cal_lock = threading.Lock()
        self._realign = None                # why a measurement is owed, or None
        self.last_align = {"lag": 0, "peak": 0.0, "ok": False, "why": None}
        # noise blanker
        self.nb_on = False
        self.nb_db = self.NB_DEFAULT_DB
        self.blanked_pct = 0.0
        # spatial map: rebuilt whenever the hardware centre or rate moves,
        # since its bins are absolute frequencies
        self.map = None
        self._map_key = None
        self._win = {}                      # length -> Hann window
        # raw two-channel capture (see capture())
        self._cap_lock = threading.Lock()
        self._capture = None
        self.last_capture = None

    # --- the `source` alias -------------------------------------------------
    @property
    def source(self):
        """v1 name for the panadapter selection; 'nulled' reads as combined."""
        return self.pan if self.pan in self.SOURCES else "combined"

    # --- reader thread -------------------------------------------------
    def request_realign(self, why):
        with self._cal_lock:
            self._cal_a, self._cal_b, self._cal_n = [], [], 0
        self._realign = str(why)

    def _configured_weight(self, sid):
        """What the operator has dialled in for slice sid, regardless of
        whether the aligner currently trusts the two channels enough to
        combine them. Used by status() so the UI shows what is set even
        while weight_for() is holding at 0j."""
        if self.mode == "off":
            return 0j
        if self.mode == "manual":
            return self.manual.get(sid, 1 + 0j)
        t = self.trackers.get(sid)
        return t.m if t is not None else 0j

    def weight_for(self, sid):
        """The complex weight ACTUALLY used to combine A and B for slice sid.

        0j (channel A alone) whenever the aligner is not aligned, in every
        mode — including manual m=1. Combining two streams the aligner has
        not credibly locked adds a decorrelated copy of the same signal,
        which costs ~3 dB SNR rather than gaining anything (found in review,
        F10).
        """
        if not self.aligner.aligned:
            return 0j
        return self._configured_weight(sid)

    def _window(self, n):
        w = self._win.get(n)
        if w is None:
            self._win = {n: self.a._np.hanning(n)}     # one length at a time is plenty
            w = self._win[n]
        return w

    def ingest(self, a, b):
        """One raw block pair -> (block for the pan/meters, pair for the demod)."""
        np = self.a._np
        if self._realign is not None:
            n_cal = min(int(self.CAL_SECONDS * self.a.samp_rate), self.CAL_SAMPLES_MAX)
            snapshot = None
            with self._cal_lock:
                self._cal_a.append(a); self._cal_b.append(b); self._cal_n += len(a)
                if self._cal_n >= n_cal:
                    snapshot = (list(self._cal_a), list(self._cal_b), self._realign)
                    self._cal_a, self._cal_b, self._cal_n = [], [], 0
                    self._realign = None
            if snapshot is not None:
                cal_a, cal_b, why = snapshot
                A = np.concatenate(cal_a); B = np.concatenate(cal_b)
                lag, peak, ok = self.aligner.calibrate(A, B, min(8192, len(A) // 4))
                self.last_align = {"lag": int(lag), "peak": float(peak), "ok": bool(ok), "why": why}
                print(f"[diversity] alignment ({why}): lag {lag:+d} samples, correlation "
                      f"peak {peak:.1f}x the floor — "
                      f"{'locked' if ok else 'NOT credible; holding lag 0'}", flush=True)
        a, b = self.aligner.apply(a, b)
        if self.nb_on:
            a, b, frac = _dv().blank_impulses(a, b, self.nb_db)
            self.blanked_pct = 0.9 * self.blanked_pct + 0.1 * 100.0 * frac
        elif self.blanked_pct:
            self.blanked_pct = 0.0
        if self._capture is not None:
            self._capture_ingest(a, b)
        if self.aligner.aligned:
            self._map_update(a, b)
        if self.pan == "a":
            pan = a
        elif self.pan == "b":
            pan = b
        elif self.pan == "nulled" and self.map is not None and self.aligner.aligned:
            pan = self._nulled(a, b)
        else:
            pan = _dv().combine(a, b, self.weight_for(self.active_slice))
        return pan, (a, b)

    def _map_update(self, a, b):
        np = self.a._np
        n = self.MAP_BINS
        if len(a) < n or len(b) < n:
            return
        key = (float(self.a.center_hz), float(self.a.samp_rate))
        if self.map is None or self._map_key != key:
            self.map = _sp().SpatialMap(n, self.a.samp_rate)
            self._map_key = key
        w = self._window(n)
        X = np.fft.fft(np.stack([a[:n], b[:n]]) * w, axis=1)
        self.map.update(X, len(a) / self.a.samp_rate)

    def _nulled(self, a, b):
        """The pan block with every coherent bin steered to its own null and
        the rest on the slice's weight — the noise map applied wholesale."""
        np = self.a._np
        n = min(len(a), len(b))
        m = self.map.null_weights(fallback=self.weight_for(self.active_slice))
        if n != self.MAP_BINS:
            # the block and the map differ in length: index the map's bins
            # by frequency (both are natural FFT order, so go via fftshift)
            ms = np.fft.fftshift(m)
            idx = (np.arange(n) * self.MAP_BINS) // n
            m = np.fft.ifftshift(ms[idx])
        Xa = np.fft.fft(a[:n]); Xb = np.fft.fft(b[:n])
        Y = (Xa + m * Xb) / np.sqrt(1.0 + np.abs(m) ** 2)
        return np.fft.ifft(Y).astype(a.dtype)

    # --- audio thread ----------------------------------------------------
    def _bands(self, n, rate, mode):
        """(in-band index, guard index) for a length-n spectrum of the MIXED
        block: the slice sits at DC, so the passband is what _init_demod
        builds around it. The guard bands sit GUARD_GAP_HZ beyond each edge
        and are as wide as the passband."""
        np = self.a._np
        f = np.fft.fftfreq(n, 1.0 / rate)
        if _is_fm(mode):
            lo, hi = -FM_PASS_HZ, FM_PASS_HZ
        elif (mode or "").upper().startswith("LSB"):
            lo, hi = -SSB_PASS_HZ, 0.0
        else:
            lo, hi = 0.0, SSB_PASS_HZ
        bw, gap = hi - lo, self.GUARD_GAP_HZ
        idx_in = (f >= lo) & (f < hi)
        idx_g = ((f >= hi + gap) & (f < hi + gap + bw)) | ((f < lo - gap) & (f >= lo - gap - bw))
        return idx_in, idx_g

    def observe(self, sid, xa, xb):
        """Both channels of one MIXED block (slice at DC) for slice sid.
        Returns (m_from, m_to): the weight the previous block ended on and
        the one this block should end on, for combine_ramp."""
        np = self.a._np
        rate = float(self.a.samp_rate)
        t = self.trackers.get(sid)
        if t is None:
            t = self.trackers[sid] = _dv().Tracker(rate, memory=self.memory, t0=time.monotonic())
        n = min(len(xa), len(xb))
        if n >= 64:
            w = self._window(n)
            X = np.fft.fft(np.stack([xa[:n], xb[:n]]) * w, axis=1)
            idx_in, idx_g = self._bands(n, rate, getattr(self.a, "_mode", "USB"))
            rc = _sp().region_covariance
            t.update(rc(X, idx_in), rc(X, idx_g, trim=True), n, self.mode)
            pb = self.passband.get(sid)
            if pb is None:
                from ..core.passband import PassbandPhase
                pb = self.passband[sid] = PassbandPhase(rate)
            f = np.fft.fftfreq(n, 1.0 / rate)
            pb.update(X[0][idx_in], X[1][idx_in], f[idx_in], n, t.talking)
        m1 = self.weight_for(sid)
        m0 = self.last_m.get(sid, m1)
        self.last_m[sid] = m1
        return m0, m1

    # --- capture ---------------------------------------------------------
    def capture(self, seconds):
        """Start recording `seconds` of the aligned raw pair; returns the path
        the .npz will appear at once the reader has collected it."""
        seconds = float(seconds)
        with self._cap_lock:
            if self._capture is not None:
                raise RuntimeError("a capture is already running")
            d = os.path.expanduser(self.CAPTURE_DIR)
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, time.strftime("%Y%m%d-%H%M%S")
                                + f"_{int(self.a.center_hz)}Hz_{int(self.a.samp_rate)}sps.npz")
            self._capture = {"want": int(seconds * self.a.samp_rate), "n": 0,
                             "a": [], "b": [], "path": path, "seconds": seconds}
        return path

    def _capture_ingest(self, a, b):
        with self._cap_lock:
            c = self._capture
            if c is None:
                return
            c["a"].append(a.copy()); c["b"].append(b.copy()); c["n"] += len(a)
            if c["n"] < c["want"]:
                return
            self._capture = None
        np = self.a._np
        meta = {"rate_hz": float(self.a.samp_rate), "center_hz": float(self.a.center_hz),
                "lag_samples": int(self.aligner.lag), "aligned": bool(self.aligner.aligned),
                "seconds": c["seconds"]}

        def _write():
            np.savez(c["path"], a=np.concatenate(c["a"])[:c["want"]],
                     b=np.concatenate(c["b"])[:c["want"]], **meta)
            self.last_capture = c["path"]
            print(f"[diversity] capture written: {c['path']}", flush=True)
        # 60 s at 2 MS/s is ~1 GB of complex64: not on the reader thread
        threading.Thread(target=_write, name="diversity-capture", daemon=True).start()

    def memory_clear(self):
        self.memory.clear()

    # --- control port ----------------------------------------------------
    def _sources(self):
        if self.map is None or self.map.R is None:
            return []
        return self.map.sources(float(self.a.center_hz))

    def map_json(self):
        if self.map is None or self.map.R is None:
            return {"available": False, "coherence": [], "sources": []}
        out = self.map.map(float(self.a.center_hz))
        out["available"] = True
        return out

    def status(self, sid=None):
        sid = self.active_slice if sid is None else int(sid)
        m = self.weight_for(sid)                     # what is ACTUALLY combined
        # phase/ratio report the operator's CONFIGURED weight, not weight_for's
        # 0j-while-unaligned — the slider must not appear to snap to zero just
        # because the aligner has not locked yet.
        ph, ra = _dv().weight_to_polar(self._configured_weight(sid))
        t = self.trackers.get(sid)
        cap = self._capture
        return {
            "available": True, "channels": 2,
            "mode": self.mode, "source": self.source, "pan": self.pan,
            "phase_deg": round(ph, 1), "ratio_db": round(ra, 1),
            "weight": [round(m.real, 4), round(m.imag, 4)],
            "lag_samples": int(self.aligner.lag), "aligned": bool(self.aligner.aligned),
            "corr_peak": round(float(self.aligner.peak), 1),
            "realigning": self._realign is not None,
            "snr_db": t.snr_db() if t is not None else {"a": None, "b": None, "out": None},
            "talking": bool(t.talking) if t is not None else False,
            "talk_mod": (round(t.talk_mod, 2) if t is not None and t.talk_mod is not None
                         else None),
            "rn_source": t.rn_source if t is not None else None,
            "steady_qrm": bool(t.steady) if t is not None else False,
            "passband": (self.passband[sid].status() if sid in self.passband else None),
            # how directional the noise is (0 = isotropic, nothing to null)
            "noise_coherence": (round(_dv()._coherence(t.Rn), 2)
                                if t is not None and t.Rn is not None else None),
            "updates": int(t.updates) if t is not None else 0,
            "nb": {"enabled": self.nb_on, "threshold_db": self.nb_db,
                   "blanked_pct": round(self.blanked_pct, 2)},
            "sources": self._sources(),
            "memory": self.memory.status(time.monotonic()),
            "capture": {"active": cap is not None,
                        "path": cap["path"] if cap is not None else self.last_capture},
            "slice_id": sid,
        }

    def set(self, mode=None, phase_deg=None, ratio_db=None, source=None, sid=None,
            nb=None, nb_db=None, pan=None, null_source=None):
        sid = self.active_slice if sid is None else int(sid)
        if mode is not None:
            mode = str(mode).lower()
            if mode not in self.MODES:
                raise ValueError(f"mode must be one of {self.MODES}")
            if mode == "manual" and self.mode in ("null", "track"):
                # start the sliders where the automatic fit left off
                t = self.trackers.get(sid)
                if t is not None and t.m != 0:
                    self.manual[sid] = t.m
            self.mode = mode
        if phase_deg is not None or ratio_db is not None:
            ph, ra = _dv().weight_to_polar(self.manual.get(sid, 1 + 0j))
            if phase_deg is not None:
                ph = float(phase_deg)
            if ratio_db is not None:
                ra = float(ratio_db)
            self.manual[sid] = _dv().weight_from_polar(ph, ra)
        if source is not None:
            source = str(source).lower()
            if source not in self.SOURCES:
                raise ValueError(f"source must be one of {self.SOURCES}")
            self.pan = source
        if pan is not None:
            pan = str(pan).lower()
            if pan not in self.PANS:
                raise ValueError(f"pan must be one of {self.PANS}")
            self.pan = pan
        if nb is not None:
            self.nb_on = bool(nb)
        if nb_db is not None:
            nb_db = float(nb_db)
            if not (0.0 <= nb_db <= 40.0):
                raise ValueError("nb_db must be 0..40")
            self.nb_db = nb_db
        if null_source is not None:
            srcs = self._sources()
            i = int(null_source)
            if not (0 <= i < len(srcs)):
                raise ValueError(f"no such source: {i} (have {len(srcs)})")
            # the source's per-bin null, applied as the slice's whole-band
            # weight: manual mode with the sliders parked on it
            self.manual[sid] = _dv().weight_from_polar(srcs[i]["phase_deg"], srcs[i]["ratio_db"])
            self.mode = "manual"
        return self.status(sid)
