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


def _nba():
    from ..core import nbarm
    return nbarm


def _fd():
    from ..core import finder
    return finder


def _sb():
    from ..core import subband
    return subband


def _npf():
    from ..core import noiseprofile
    return noiseprofile


def _npk():
    from . import noise_kinds
    return noise_kinds


def _vp():
    from ..core import voiceprint
    return voiceprint


def _pf():
    from ..core import postfilter
    return postfilter


def _voice_key(summary):
    """What of a print summary is persisted with a name: enough for
    VoicePrint.distance, nothing that grows."""
    if summary is None:
        return None
    return {k: summary.get(k) for k in ("centroid_hz", "high_hz", "tilt_db", "overs")}


def _bc():
    from ..core import beacons
    return beacons


def _balance():
    from ..core import balance
    return balance


def _is_fm(mode):
    return (mode or "").upper() in ("FM", "NFM", "DFM")


class _DiversityState:
    CAL_SECONDS = 0.5          # of raw IQ cross-correlated to find the lag
    # A quiet half second of band noise from two loops a few metres apart
    # peaks at only 6-10x the floor (measured 2026-09-03: 34x and 49x with a
    # talker, 9x and 10x without, on the same capture), so one window is not
    # a verdict. Keep measuring, up to ALIGN_TRIES windows, and adopt the
    # best; a peak past ALIGN_STRONG_PEAK ends the search early.
    ALIGN_TRIES = 8
    ALIGN_STRONG_PEAK = 30.0
    # How the lag is searched -- over a time, at every span -- is core/alignsearch.py.
    CAL_SAMPLES_MAX = 1_000_000  # 0.49 s at 2.04 MS/s: the window is a TIME (the
                                # offset is ~33 ms); alignsearch bounds the FFT
    MODES = ("off", "manual", "null", "track")
    SOURCES = ("combined", "a", "b", "stereo")   # HEAR: what reaches the audio
    PANS = ("combined", "a", "b", "nulled")      # what the panadapter shows
    MAP_BINS = 2048            # spatial-map resolution (61 Hz at 125 kS/s)
    GUARD_GAP_HZ = 300.0       # between the passband edge and its guard band
    NB_DEFAULT_DB = 12.0
    CAPTURE_DIR = "~/aether-gate-captures"
    NAMES_PATH = "~/.aether-gate/diversity-names.json"   # talker labels, by signature
    BEACONS_PATH = "~/.aether-gate/beacons.json"          # beacon samples + station grid
    NULLABLE_COHERENCE = 0.4     # below this the noise has no direction to null

    def __init__(self, adapter):
        self.a = adapter
        self.aligner = _dv().Aligner()
        self.mode = "off"
        self.pan = "combined"
        self.hear = "combined"              # the audio: combined, a, b, or stereo (A left, B right)
        self.manual = {}                    # slice_id -> complex weight m
        self.trackers = {}
        self.passband = {}                  # sid -> PassbandPhase                  # slice_id -> Tracker (rebuilt on a rate change)
        self.last_m = {}                    # slice_id -> weight the last block ended on
        self.memory = _dv().TalkerMemory(   # shared by every slice's tracker
            names_path=os.path.expanduser(self.NAMES_PATH))
        self.active_slice = 0
        self._cal_a, self._cal_b, self._cal_n = [], [], 0
        # Guards _cal_a/_cal_b/_cal_n: request_realign() can land from the
        # HTTP thread (diversity_realign) while ingest() is mid-accumulate on
        # the reader thread, and an empty list handed to np.concatenate raises.
        self._cal_lock = threading.Lock()
        self._realign = None                # why a measurement is owed, or None
        self._align_try = 0                 # windows measured for the realign in progress
        self._align_best = None             # (lag, peak) of the best window so far
        self.last_align = {"lag": 0, "peak": 0.0, "ok": False, "why": None}
        # noise blanker
        self.nb_on = False
        self.nbarm = _nba().NbArm()      # the profile arms it until the operator says on/off
        self.balance = _balance().LoopBalance()   # G7: a sick loop, said out loud
        self.nb_db = self.NB_DEFAULT_DB
        self.blanked_pct = 0.0
        # per-bin refinement of the tracker's weight in the demod passband
        self.subband_on = True
        self.subbands = {}                  # sid -> SubbandCombiner
        self.post_on = True                 # the coherence post-filter (core.postfilter)
        self.post_floor_db = None           # None = the module's default
        self.prints = {}                    # sid -> VoicePrint (per talker voice/rig prints)
        self._voice_checked = False         # the running over has been judged against its print
        self.voice_splits = 0               # overs moved off a recalled talker by their voice
        self.profile = None                 # what kind of noise this is (core.noiseprofile)
        self.beacons = None                 # the NCDXF beacons on the pair (core.beacons)
        # spatial map: rebuilt whenever the hardware centre or rate moves,
        # since its bins are absolute frequencies
        self.map = None
        self._map_key = None
        self.live = None                    # LiveSpatial: the span right now
        self.finder = None                  # Finder: where people are talking
        self._win = {}                      # length -> Hann window
        # raw two-channel capture (see capture())
        self._cap_lock = threading.Lock()
        self._capture = None
        self.last_capture = None

    # --- the `source` alias -------------------------------------------------
    @property
    def source(self):
        """The `source` key: what the operator hears. Until 2026-09-02 this
        was an alias of `pan` and HEAR in the window changed the panadapter,
        not the audio."""
        return self.hear

    # --- reader thread -------------------------------------------------
    def request_realign(self, why):
        with self._cal_lock:
            self._cal_a, self._cal_b, self._cal_n = [], [], 0
        self._align_try, self._align_best = 0, None
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

    def _align_window(self, A, B, why):
        """One calibration window measured: adopt the best lag so far when
        it is credible, and keep measuring while it is not yet strong."""
        dv = _dv()
        from ..core.alignsearch import measure_lag      # numpy: first use only
        lag, peak = measure_lag(A, B, self.a.samp_rate)
        self._align_try += 1
        best = self._align_best
        if best is None or peak > best[1]:
            best = self._align_best = (int(lag), float(peak))
        ok = best[1] >= dv.ALIGN_MIN_PEAK
        if best == (int(lag), float(peak)) or self._align_try == 1:
            # first window, or a better one: the aligner follows it (an
            # unchanged best is left alone -- set_lag restarts the delay line)
            self.aligner.set_lag(best[0] if ok else 0, best[1], ok)
        more = best[1] < self.ALIGN_STRONG_PEAK and self._align_try < self.ALIGN_TRIES
        self.last_align = {"lag": best[0] if ok else 0, "peak": best[1], "ok": ok, "why": why}
        if more:
            self._realign = why             # keep collecting
        verdict = ("locked" if ok else "NOT credible; holding lag 0") + (
            f"; measuring on ({self._align_try}/{self.ALIGN_TRIES})" if more else "")
        print(f"[diversity] alignment ({why}): lag {lag:+d} samples, correlation "
              f"peak {peak:.1f}x the floor — {verdict}", flush=True)

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
                self._align_window(A, B, why)
        a, b = self.aligner.apply(a, b)
        if self.aligner.aligned:
            # before the blanker: the profile must see the impulses it counts
            if self.profile is None or self.profile.rate_hz != float(self.a.samp_rate):
                self.profile = _npf().NoiseProfile(self.a.samp_rate)
            self.profile.update(a, b)
            d = self.nbarm.update(self.profile.status(), time.time())
            self.nb_on = d.nb_on
            if d.threshold is not None:
                self.nb_db = d.threshold
            if self.beacons is None or self.beacons.rate_hz != float(self.a.samp_rate):
                self.beacons = self._beacon_watch()
            self.beacons.update(a, b, float(self.a.center_hz), time.time())
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
        center = float(self.a.center_hz)
        rate = float(self.a.samp_rate)
        key = (center, rate)
        if self.map is None or self._map_key[1] != rate:
            # no map yet, or the rate moved: every bin's meaning changed,
            # nothing to slide
            self.map = _sp().SpatialMap(n, rate)
            self.live = _fd().LiveSpatial(n, rate)
            self.finder = _fd().Finder(n, rate)
        elif self._map_key[0] != center:
            # same span, different centre: the bins are still THIS wide, just
            # sitting somewhere else -- slide the history rather than lose it
            delta = center - self._map_key[0]
            self.map.retune(delta)
            self.live.retune(delta)
            self.finder.retune(delta)
        self._map_key = key
        w = self._window(n)
        X = np.fft.fft(np.stack([a[:n], b[:n]]) * w, axis=1)
        frame_s = len(a) / self.a.samp_rate
        self.map.update(X, frame_s)
        self.live.update(X, frame_s)
        self.finder.update(X, frame_s)

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
    @staticmethod
    def _pass_edges(mode):
        """The demodulated passband relative to the slice, in Hz."""
        if _is_fm(mode):
            return -FM_PASS_HZ, FM_PASS_HZ
        if (mode or "").upper().startswith("LSB"):
            return -SSB_PASS_HZ, 0.0
        return 0.0, SSB_PASS_HZ

    def _bands(self, n, rate, mode):
        """(in-band index, guard index) for a length-n spectrum of the MIXED
        block: the slice sits at DC, so the passband is what _init_demod
        builds around it. The guard bands sit GUARD_GAP_HZ beyond each edge
        and are as wide as the passband."""
        np = self.a._np
        f = np.fft.fftfreq(n, 1.0 / rate)
        lo, hi = self._pass_edges(mode)
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
            if t.Rn is not None:
                self.balance.update(t.Rn, time.monotonic())
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
                             "a": [], "b": [], "path": path, "seconds": seconds,
                             "slice_hz": getattr(self.a, "_slice_hz", None),
                             "slice_mode": getattr(self.a, "_mode", None)}
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
        if c.get("slice_hz") is not None:
            meta["slice_hz"] = float(c["slice_hz"])          # what the operator was listening to
            meta["slice_mode"] = str(c.get("slice_mode") or "")

        def _write():
            np.savez(c["path"], a=np.concatenate(c["a"])[:c["want"]],
                     b=np.concatenate(c["b"])[:c["want"]], **meta)
            self.last_capture = c["path"]
            print(f"[diversity] capture written: {c['path']}", flush=True)
        # 60 s at 2 MS/s is ~1 GB of complex64: not on the reader thread
        threading.Thread(target=_write, name="diversity-capture", daemon=True).start()

    def memory_clear(self):
        self.memory.clear()
        for vp in self.prints.values():
            vp.forget()

    def memory_name(self, talker_id, name):
        vp = self.prints.get(self.active_slice)
        voice = _voice_key(vp.summary(int(talker_id))) if vp is not None else None
        if not self.memory.name(talker_id, name, voice):
            raise ValueError(f"unknown talker id {talker_id}")

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
        out["passband_hz"] = self._passband_hz()
        return out

    def _passband_hz(self):
        """Where the receiver's passband sits inside the span, so a strip can
        mark it: absolute Hz, or None when the adapter has no slice yet."""
        slice_hz = getattr(self.a, "_slice_hz", None)
        if slice_hz is None:
            return None
        lo, hi = self._pass_edges(getattr(self.a, "_mode", "USB"))
        return [float(slice_hz) + lo, float(slice_hz) + hi]

    def spatial_json(self):
        rows = self.live.rows(float(self.a.center_hz)) if self.live is not None else None
        if rows is None:
            return {"available": False}
        rows["available"] = True
        rows["passband_hz"] = self._passband_hz()
        return rows

    def _beacon_watch(self):
        return _bc().BeaconWatch(self.a.samp_rate, store_path=self.BEACONS_PATH)

    def beacons_json(self):
        if self.beacons is None:
            return {"available": False}
        return self.beacons.status(time.time())

    def finder_json(self):
        # Fed only while aligned (_map_update): say which it is, not just "no".
        if not self.aligner.aligned:
            return {"available": False, "reason": "not aligned"}
        if self.finder is None:
            return {"available": False, "reason": "no frames yet"}
        return self.finder.candidates(float(self.a.center_hz), self.live)

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
            "idle_null": bool(t.idle_null) if t is not None else False,
            "passband": (self.passband[sid].status() if sid in self.passband else None),
            # how directional the noise is (0 = isotropic, nothing to null)
            "noise_coherence": (round(_dv()._coherence(t.Rn), 2)
                                if t is not None and t.Rn is not None else None),
            "updates": int(t.updates) if t is not None else 0,
            "nb": {"enabled": self.nb_on, "threshold_db": self.nb_db,
                   "blanked_pct": round(self.blanked_pct, 2),
                   "auto": self.nbarm.status()},
            "subband": {"enabled": self.subband_on,
                        **(self.subbands[sid].status() if sid in self.subbands
                           else {"bins": 0, "extra_db": 0.0})},
            "post": self._post_status(sid),
            "sources": self._sources(),
            "noise_profile": self._noise_profile(t),
            "memory": self._memory_status(sid),
            "talker": self.memory.talker(time.monotonic()),
            "voice_splits": self.voice_splits,
            "loops": self.balance.status(time.monotonic()),
            "focus": self.memory.focus_status(time.monotonic(),
                                              nulling=bool(t.interferer) if t is not None else False),
            "capture": {"active": cap is not None,
                        "path": cap["path"] if cap is not None else self.last_capture},
            "slice_id": sid,
        }

    def _noise_profile(self, t):
        """The profile's numbers plus `kinds`: one row per thing found, each
        naming what it is, how long it was measured over, and the one
        control-port call that does something about it (or why nothing can).
        The kinds themselves are built in noise_kinds.py (see there)."""
        if self.profile is None:
            return None
        st = self.profile.status()
        coh = (float(_dv()._coherence(t.Rn))
               if t is not None and t.Rn is not None else None)
        filt = getattr(self.a, "_filt", None)
        st["kinds"] = _npk().noise_kinds(
            st, coh, self.mode, self.nb_on, self.blanked_pct,
            self.NULLABLE_COHERENCE, filt.status() if filt is not None else None)
        return st

    def _memory_status(self, sid):
        """The memory's entries with each talker's voice/rig print attached."""
        mem = self.memory.status(time.monotonic())
        vp = self.prints.get(sid)
        if vp is not None:
            vp.forget(keep_ids={e["id"] for e in mem})
        for e in mem:
            e["voice"] = vp.summary(e["id"]) if vp is not None else None
            if e.get("name") and e["voice"] is not None:
                self.memory.note_voice(e["id"], _voice_key(e["voice"]))
        return mem

    def _print_feed(self, sid, y, pa, rate_hz):
        """Teach the slice's VoicePrint from what was just combined (loop A
        when the monitor is on)."""
        vp = self.prints.get(sid)
        if vp is None or vp.rate_hz != float(rate_hz):
            vp = self.prints[sid] = _vp().VoicePrint(rate_hz)
        t = self.trackers.get(sid)
        talking = t is not None and bool(t.talking) and not bool(t.steady)
        active = self.memory.active if talking else None
        vp.feed(y if getattr(y, "ndim", 1) == 1 else pa, talking, active)
        if not talking:
            self._voice_checked = False
        elif active is not None and not self._voice_checked:
            self._voice_check(vp, active)

    def _voice_check(self, vp, active):
        """Once per over, as soon as the running print can be judged: an
        over recalled by bearing that is not that talker's voice goes to
        whoever at that bearing it is, or to a new talker."""
        cur = vp.current()
        if cur is None:
            return
        self._voice_checked = True
        v = _vp()
        mine = vp.summary(active)
        if mine is None:
            # no print of their own yet: a name inherited from the bearing is
            # checked against the voice the name was given for
            e = self.memory.entry(active) or {}
            known = self.memory.named_voice(e["name"], e["s"]) if e.get("name") else None
            dn = vp.distance(cur, known)
            if dn is not None and dn >= v.DIFFERENT_VOICE:
                name = self.memory.disown(active)
                print(f"[diversity] #{active} has {name}'s bearing but not their voice "
                      f"(d={dn:.2f}) -> unnamed", flush=True)
            return
        d = vp.distance(cur, mine)
        if d is None or d < v.DIFFERENT_VOICE:
            return

        def unlike(e):
            s = vp.summary(e["id"])
            dd = vp.distance(cur, s)
            return dd is not None and dd >= v.DIFFERENT_VOICE
        new = self.memory.reassign(time.monotonic(), unlike)
        if new is not None:
            self.voice_splits += 1
            was = self.memory.entry(active) or {}
            print(f"[diversity] voice split: #{active}{' ' + was['name'] if was.get('name') else ''}"
                  f"'s bearing but not their voice (d={d:.2f}, centroid {cur['centroid_hz']} vs "
                  f"{mine['centroid_hz']} Hz, top {cur['high_hz']} vs {mine['high_hz']}, tilt "
                  f"{cur['tilt_db']} vs {mine['tilt_db']} dB) -> #{new}", flush=True)

    def _monitor(self, pa, pb):
        if self.hear == "a":
            return pa
        if self.hear == "b":
            return pb
        n = min(len(pa), len(pb))
        return self.a._np.stack([pa[:n], pb[:n]], axis=1)

    def _post_status(self, sid):
        sb = self.subbands.get(sid)
        pf = sb.post if sb is not None else None
        if pf is None:
            return {"enabled": self.post_on and self.subband_on,
                    "floor_db": _pf().FLOOR_DB if self.post_floor_db is None else self.post_floor_db,
                    "mean_db": 0.0}
        return {"enabled": True, **pf.status()}

    def _talker_profile(self, sid):
        """The live talker's print bands, for the post-filter's floor."""
        vp = self.prints.get(sid)
        if vp is None or self.memory.active is None:
            return None
        s = vp.summary(self.memory.active)
        return None if s is None else s.get("bands_db")

    def combine_passband(self, sid, pa, pb, m0, m1, rate_hz):
        y = self._combine(sid, pa, pb, m0, m1, rate_hz)
        self._print_feed(sid, y, pa, rate_hz)
        return y

    def _combine(self, sid, pa, pb, m0, m1, rate_hz):
        """The demod passband pair of slice sid -> one combined block.

        In null/track with the sub-band refinement on, every bin gets its own
        weight (core.subband): the tracker's wideband weight, refined to a
        per-bin null wherever the learned noise has a direction, with the
        talker's steering vector held distortionless. Otherwise, and whenever
        the pair is not aligned, the wideband combiner with its click-free
        ramp from m0 to m1.

        HEAR a / b hands that loop's passband straight through; stereo hands
        both as an (n, 2) array, A left and B right, for the operator with
        two speakers to hear the loops as a soundstage. The tracker keeps
        learning from observe() either way, so a comparison costs nothing."""
        if self.hear != "combined":
            return self._monitor(pa, pb)
        t = self.trackers.get(sid)
        if (not self.subband_on or self.mode not in ("null", "track")
                or not self.aligner.aligned or t is None):
            return _dv().combine_ramp(pa, pb, m0, m1)
        np = self.a._np
        sb = self.subbands.get(sid)
        if sb is None or sb.rate_hz != float(rate_hz):
            sb = self.subbands[sid] = _sb().SubbandCombiner(rate_hz)
        sb.set_post(self.post_on, self.post_floor_db)
        s = None
        if t.Rs is not None and t.Rn is not None:
            S = t.Rs - t.Rn
            if float(np.real(np.trace(S))) > 0:
                s = _dv().steering_of(S)
        if s is None:
            s = np.array([1.0, np.conj(m1)], dtype=np.complex128)    # the weight's own beam
        # a steady carrier is "talking" to the VAD but noise to the listener:
        # the tracker learns it into Rn_in, and so does the per-bin model
        talking = bool(t.talking) and not bool(t.steady)
        return sb.process(pa, pb, m1, s, talking,
                          self._talker_profile(sid) if self.post_on else None)

    def set(self, mode=None, phase_deg=None, ratio_db=None, source=None, sid=None,
            nb=None, nb_db=None, pan=None, null_source=None, focus=None, subband=None,
            grid=None, post=None, post_floor_db=None):
        sid = self.active_slice if sid is None else int(sid)
        if post is not None:
            self.post_on = bool(post)
        if post_floor_db is not None:
            self.post_floor_db = max(-20.0, min(0.0, float(post_floor_db)))
        if grid is not None:
            if self.beacons is None:
                self.beacons = self._beacon_watch()
            self.beacons.set_station(grid)           # ValueError on a bad locator
        if subband is not None:
            self.subband_on = bool(subband)
            if not self.subband_on:
                self.subbands.clear()       # relearn from scratch when it comes back
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
            self.hear = source
        if pan is not None:
            pan = str(pan).lower()
            if pan not in self.PANS:
                raise ValueError(f"pan must be one of {self.PANS}")
            self.pan = pan
        if nb is not None:
            # True/False are the old callers; "auto" hands it to the profile
            mode = nb if isinstance(nb, str) else ("on" if nb else "off")
            self.nbarm.set_mode(mode)           # ValueError on junk
            if mode != "auto":
                self.nb_on = mode == "on"
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
        if focus is not None:
            # '' releases; an id pins that talker (ValueError when unknown)
            fid = int(focus) if str(focus).strip() not in ("", "off", "none") else None
            self.memory.set_focus(fid, time.monotonic())
            e = self.memory.focus_entry()
            for t in self.trackers.values():
                # pre-steer at the pinned station while nobody is talking, so
                # the first syllable of their next over is already on beam
                if e is not None and not t.talking and t.m != e["m"]:
                    t.m = e["m"]
                    t.updates += 1
        return self.status(sid)
