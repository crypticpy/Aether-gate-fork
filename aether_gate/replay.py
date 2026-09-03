#
# Aether-gate — the replay lab: a captured pair through the combiner, offline.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Take a /diversity/capture (.npz: the aligned raw pair plus its metadata)
and run it through the same tracker and combiner the gate runs live, once
per configuration, writing one WAV per configuration so they can be heard
side by side and one summary so they can be compared in numbers:

  a          loop A alone
  b          loop B alone
  wideband   the tracker's weight (mode track), one weight for the passband
  subband    the same, refined per bin (core.subband)
  filtered   subband through the slice filter and the AGC (core.filter):
             what the operator hears, so a filter or AGC change can be
             judged on the same over as the combiner

Every configuration hears the same seconds of the same over at the same
gain, so a difference is the algorithm's and nothing else's. A change to
the DSP is judged here on last night's capture before it is judged on the
air. The one exception is `filtered`: its AGC sets its own level, so that
file is normalised on its own and compared for shape, not loudness.

Usage:
    python -m aether_gate.replay CAPTURE.npz [--slice-hz HZ] [--mode USB|LSB]
                                             [--seconds N] [--out DIR]
                                             [--configs a,b,wideband,subband,filtered]
                                             [--filter low=300,high=2700,threshold_db=20,...]
"""
import argparse
import json
import os
import sys
import time
import wave

import numpy as np

from .adapters.diversity_state import _DiversityState
from .core.diversity import combine_ramp
from .core.filter import SliceFilter

BLOCK = 4096
PD_RATE_TARGET = 25_000.0
CONFIGS = ("a", "b", "wideband", "subband", "filtered")


class _Adapter:
    """What _DiversityState needs of an adapter, and nothing else."""
    def __init__(self, rate_hz, center_hz, mode):
        self._np = np
        self.samp_rate = float(rate_hz)
        self.center_hz = float(center_hz)
        self._mode = mode


class _Fir:
    """Overlap-save FIR with decimation by M, comb phase kept across blocks."""
    def __init__(self, taps, M=1):
        self.taps = np.asarray(taps, dtype=np.complex128)
        self.M = int(M)
        self.state = np.zeros(len(taps) - 1, dtype=np.complex128)
        self.pos = 0

    def __call__(self, x):
        x = np.concatenate([self.state, np.asarray(x, dtype=np.complex128)])
        n_out = len(x) - len(self.taps) + 1
        if n_out <= 0:
            self.state = x
            return np.zeros(0, dtype=np.complex128)
        y = np.convolve(x, self.taps, mode="valid")
        keep = (np.arange(n_out) + self.pos) % self.M == 0
        self.pos += n_out
        self.state = x[-(len(self.taps) - 1):]
        return y[keep]


def _lowpass(ntaps, cutoff_norm):
    k = np.arange(ntaps) - (ntaps - 1) / 2.0
    h = np.sinc(2 * cutoff_norm * k) * np.hamming(ntaps)
    return h / h.sum()


def _ssb_taps(pd_rate, mode):
    """The gate's own one-sided SSB filter: 63 taps, 300..2700 Hz-ish."""
    ntaps = 63
    k = np.arange(ntaps) - (ntaps - 1) / 2.0
    lp = _lowpass(ntaps, 1500.0 / pd_rate)
    usb = lp * np.exp(2j * np.pi * (1500.0 / pd_rate) * k)
    return usb if mode == "USB" else np.conj(usb)


class _Chain:
    """Mixer + decimator + SSB filter for one channel, block by block."""
    def __init__(self, rate_hz, offset_hz, mode):
        self.rate = float(rate_hz)
        self.M = max(1, int(round(self.rate / PD_RATE_TARGET)))
        self.pd_rate = self.rate / self.M
        self.w = -2.0 * np.pi * offset_hz / self.rate
        self.phase = 0.0
        self.dec = _Fir(_lowpass(127, 0.4 / self.M), self.M)
        self.ssb = _Fir(_ssb_taps(self.pd_rate, mode))

    def mix(self, x):
        n = len(x)
        rot = np.exp(1j * (self.phase + self.w * np.arange(n)))
        self.phase = (self.phase + self.w * n) % (2.0 * np.pi)
        return np.asarray(x, dtype=np.complex128) * rot

    def passband(self, xm):
        return self.ssb(self.dec(xm))


def _parse_filter(text):
    """'low=300,high=2700,shape=soft' -> kwargs for SliceFilter.set (numbers as floats)."""
    kw = {}
    for item in (text or "").split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"--filter wants KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        k = k.strip(); v = v.strip()
        try:
            kw[k] = float(v)
        except ValueError:
            kw[k] = v
    return kw


def run(cap, config, slice_hz, mode, seconds=None, progress=None, filter_kw=None):
    """One configuration over the capture -> (audio at pd_rate, pd_rate, overs, status)."""
    rate = float(cap["rate_hz"]); center = float(cap["center_hz"])
    a_all = cap["a"]; b_all = cap["b"]
    if seconds is not None:
        n = int(seconds * rate)
        a_all = a_all[:n]; b_all = b_all[:n]
    st = _DiversityState(_Adapter(rate, center, mode))
    st.aligner.set_lag(0, 99.0, True)            # a capture is the aligned pair
    st.set(mode="track", subband=(config in ("subband", "filtered")))
    st.active_slice = 0
    ca = _Chain(rate, slice_hz - center, mode)
    cb = _Chain(rate, slice_hz - center, mode)
    cb.w = ca.w
    filt = None
    if config == "filtered":
        # the live chain: the operator's filter on each loop's decimated IQ
        # (channel 0 feeds its spectrum), the combiner, then one AGC on the
        # audio — soapy._passband and get_audio, at pd_rate instead of 24 k
        filt = SliceFilter(ca.pd_rate)
        filt.agc.rate_hz = ca.pd_rate
        filt.set(**(filter_kw or {}))
    lsb = mode == "LSB"
    out = []
    overs = []
    talking = False
    t_block = BLOCK / rate
    nblocks = (len(a_all) - BLOCK) // BLOCK + 1
    for i in range(nblocks):
        a = a_all[i * BLOCK:(i + 1) * BLOCK]; b = b_all[i * BLOCK:(i + 1) * BLOCK]
        st.ingest(a, b)                          # the map, the profile, the blanker
        xa = ca.mix(a); xb = cb.mix(b)
        m0, m1 = st.observe(0, xa, xb)
        if filt is not None:
            pa = filt.apply(ca.dec(xa), 0, lsb=lsb); pb = filt.apply(cb.dec(xb), 1, lsb=lsb)
        else:
            pa = ca.passband(xa); pb = cb.passband(xb)
        if config == "a":
            y = pa
        elif config == "b":
            y = pb
        elif config == "wideband":
            y = combine_ramp(pa, pb, m0, m1)
        else:
            y = st.combine_passband(0, pa, pb, m0, m1, ca.pd_rate)
        y = np.real(y)
        if filt is not None:
            y = filt.agc.process(y)
        out.append(y)
        t = st.trackers.get(0)
        now = bool(t.talking and not t.steady) if t is not None else False
        if now and not talking:
            overs.append([round(i * t_block, 2), None])
        elif talking and not now and overs:
            overs[-1][1] = round(i * t_block, 2)
        talking = now
        if progress and i % 200 == 0:
            progress(i, nblocks)
    if overs and overs[-1][1] is None:
        overs[-1][1] = round(nblocks * t_block, 2)
    audio = np.concatenate(out) if out else np.zeros(0)
    status = st.status()
    if filt is not None:
        status["filter"] = filt.status()
    return audio, ca.pd_rate, overs, status


def _frame_powers(x, rate, frame_s=0.1):
    n = int(frame_s * rate)
    m = len(x) // n
    if m < 2:
        return np.array([np.mean(x ** 2)])
    return np.mean(x[:m * n].reshape(m, n) ** 2, axis=1)


def _write_wav(path, x, rate, scale):
    pcm = np.clip(x * scale, -1.0, 1.0)
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(round(rate)))
        w.writeframes((pcm * 32767.0).astype("<i2").tobytes())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("capture")
    ap.add_argument("--slice-hz", type=float, default=None,
                    help="dial frequency (default: the capture's centre)")
    ap.add_argument("--mode", default="USB", choices=("USB", "LSB"))
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--out", default=None, help="output directory (default: beside the capture)")
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--filter", default="",
                    help="settings for the filtered configuration, as /filter/set takes them: "
                         "low=300,high=2700,shape=soft,auto=1,auto_eq=1,agc=med,threshold_db=20,...")
    args = ap.parse_args(argv)
    try:
        filter_kw = _parse_filter(args.filter)
    except ValueError as e:
        sys.exit(str(e))
    cap = np.load(args.capture)
    slice_hz = args.slice_hz if args.slice_hz is not None else float(cap["center_hz"])
    out_dir = args.out or os.path.splitext(args.capture)[0] + "-replay"
    os.makedirs(out_dir, exist_ok=True)
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        if c not in CONFIGS:
            sys.exit(f"unknown config {c!r}; pick from {', '.join(CONFIGS)}")
    print(f"capture {os.path.basename(args.capture)}: {float(cap['seconds']):.1f} s at "
          f"{float(cap['rate_hz']) / 1e3:.0f} kS/s, centre {float(cap['center_hz']) / 1e6:.6f} MHz, "
          f"lag {int(cap['lag_samples'])} ({'aligned' if bool(cap['aligned']) else 'NOT aligned'})")
    print(f"dial {slice_hz / 1e6:.6f} MHz {args.mode}; writing to {out_dir}")
    results = {}
    audios = {}
    for c in configs:
        t0 = time.time()
        tick = (lambda i, n: print(f"  {c}: {i}/{n}", end="\r", flush=True)) if sys.stdout.isatty() else None
        audio, pd_rate, overs, status = run(cap, c, slice_hz, args.mode, args.seconds,
                                            progress=tick, filter_kw=filter_kw)
        fp = _frame_powers(audio, pd_rate)
        loud = float(np.percentile(fp, 90)); quiet = float(np.percentile(fp, 10))
        results[c] = {
            "seconds": round(len(audio) / pd_rate, 2), "rate_hz": pd_rate,
            "rms_db": round(10 * np.log10(max(np.mean(audio ** 2), 1e-30)), 1),
            "loud_over_quiet_db": round(10 * np.log10(max(loud, 1e-30) / max(quiet, 1e-30)), 1),
            "overs": overs,
            "tracker": {k: status.get(k) for k in ("snr_db", "weight", "phase_deg", "ratio_db",
                                                    "noise_coherence", "subband", "noise_profile")},
            "took_s": round(time.time() - t0, 1),
        }
        if "filter" in status:
            results[c]["filter"] = status["filter"]
        audios[c] = (audio, pd_rate)
        print(f"  {c}: {results[c]['seconds']} s, loud/quiet {results[c]['loud_over_quiet_db']} dB, "
              f"{len(overs)} overs, {results[c]['took_s']} s        ")
    # the per-bin path holds up to half a frame at the end (it emits late,
    # it does not shift): pad every file to the longest so sample k is the
    # same instant in each
    longest = max(len(a) for a, _ in audios.values())
    audios = {c: (np.concatenate([a, np.zeros(longest - len(a))]), r) for c, (a, r) in audios.items()}
    # one gain for all, so the files compare by ear — except the filtered
    # one, whose AGC has already chosen its level; it is normalised alone
    raw = [a for c, (a, _) in audios.items() if c != "filtered"]
    peak = max((float(np.max(np.abs(a))) for a in raw), default=0.0) or 1.0
    scale = 0.9 / peak
    for c, (audio, pd_rate) in audios.items():
        s = scale if c != "filtered" else 0.9 / (float(np.max(np.abs(audio))) or 1.0)
        _write_wav(os.path.join(out_dir, f"{c}.wav"), audio, pd_rate, s)
    summary = {"capture": os.path.abspath(args.capture), "slice_hz": slice_hz, "mode": args.mode,
               "results": results}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {', '.join(c + '.wav' for c in audios)} and summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
