#
# Aether-gate -- the site log: what this site heard, written down.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Every three minutes the beacon watch measures eighteen known directions
and every second the noise profile passes a verdict on the interference --
and until now both were display-only, thrown away the moment the window
repainted. A site is not learned in one evening. The array's geometry, the
band's real reach, the hour the switch-mode supply next door comes on: all
of that is a MONTH of these measurements, not one.

So: one line of JSON per event, appended, never rewritten. JSONL because an
append is the one file operation that survives a kill -9 with everything
before it intact (a crash mid-write costs the last line, which `read` then
skips as unparseable), and because a month of it is still grep-able,
awk-able and a few hundred kilobytes.

Two kinds:

  "beacon"  one scored slot: who, which band, the bearing and distance the
            locator gives, both loops' SNR and noise floor, the pair's
            COMPLEX inter-loop ratio at the beacon bin, the coherence, how
            far down the four power steps it was still heard, and the MRC
            bound.
  "noise"   the profile's verdict: the mains comb and its depth, the
            impulse rate and excess, the strongest non-mains lines, and --
            if the caller has it -- how coherent the noise is across the
            pair.

The ratio is kept as [re, im] and not as a phase on purpose. A phase is all
a two-element pair needs today; a four-element array needs the amplitude
too, because the vector of per-element ratios IS the steering vector it
will be calibrated against. Written now, the same log is that calibration
later. For the same reason the log has no notion of "loop A" and "loop B"
beyond the field names: an N-element version adds `ratio_2`, `ratio_3`,
... and every line already written still reads.

Noise events are rate-limited: the profile speaks once a second and a day
of that is 86400 lines that all say the same thing. One line a minute
unless the VERDICT changed -- a different mains frequency, a harmonic count
that moved, hum or impulses a bucket louder, a new line in the band. The
buckets are coarse so a number breathing over a boundary does not write a
line a second, and a five-second floor catches the case where it does.
"""
import json
import math
import os
import time
from datetime import datetime, timezone

DEFAULT_PATH = "~/.aether-gate/site-log.jsonl"
NOISE_PERIOD_S = 60.0        # a repeat of the same verdict waits this long
NOISE_MIN_S = 5.0            # ... and even a changed verdict waits this long
HUM_STEP_DB = 3.0            # verdict buckets: a level moves when it moves this far
COH_STEP = 0.2
LINE_STEP_HZ = 10.0


def _iso(ts):
    """UTC, offset-explicit ('+00:00', not 'Z'): datetime.fromisoformat reads
    it back on every Python that can run this."""
    return datetime.fromtimestamp(float(ts), timezone.utc).isoformat()


def _epoch(v):
    """Seconds since the epoch from a float, a datetime, or one of our own
    timestamps. None when it is not a time at all."""
    if v is None:
        return None
    if isinstance(v, datetime):
        d = v if v.tzinfo is not None else v.replace(tzinfo=timezone.utc)
        return d.timestamp()
    if isinstance(v, (int, float)):
        return float(v)
    try:
        s = str(v)
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        d = datetime.fromisoformat(s)
        if d.tzinfo is None:
            d = d.replace(tzinfo=timezone.utc)
        return d.timestamp()
    except ValueError:
        return None


def _pair(z):
    """A complex ratio as [re, im], from a complex, a 2-sequence, or None."""
    if z is None:
        return None
    if isinstance(z, complex):
        re, im = z.real, z.imag
    else:
        try:
            re, im = float(z[0]), float(z[1])
        except (TypeError, ValueError, IndexError, KeyError):
            return None
    if not (math.isfinite(re) and math.isfinite(im)):
        return None
    return [round(float(re), 5), round(float(im), 5)]


def _num(x, nd=1):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if math.isfinite(f) else None


def _bucket(x, step):
    return None if x is None else int(round(float(x) / step))


def _rate_bucket(r):
    """Impulse rates span 0.1/s (a thermostat) to 300/s (PLT): half-decade
    buckets, so 'a bit more often' is not a new verdict but 10x is."""
    if r is None:
        return None
    return int(round(2.0 * math.log10(max(float(r), 0.1))))


class SiteLog:
    """Append-only JSONL. Never raises: a write that fails is said once on
    stdout and remembered in `error`, because the caller is the DSP thread."""

    def __init__(self, path=DEFAULT_PATH, noise_period_s=NOISE_PERIOD_S,
                 clock=time.time):
        self.path = os.path.expanduser(str(path))
        self.noise_period_s = float(noise_period_s)
        self._clock = clock
        self._noise_at = None
        self._noise_verdict = None
        self.written = 0
        self.skipped = 0             # noise verdicts the rate limit swallowed
        self.error = None            # the first write failure, kept for status

    # --- writing ------------------------------------------------------------
    def beacon(self, *, band_hz, callsign, bearing_deg=None, distance_km=None,
               snr_a_db=None, snr_b_db=None, floor_a_db=None, floor_b_db=None,
               ratio=None, coherence=None, steps_heard=None, lowest_w=None,
               mrc_gain_db=None):
        """One scored beacon slot. Returns the line written, or None."""
        return self._append({
            "t": _iso(self._clock()), "kind": "beacon",
            "band_hz": _num(band_hz, 0), "callsign": str(callsign),
            "bearing_deg": _num(bearing_deg, 1), "distance_km": _num(distance_km, 0),
            "snr_a_db": _num(snr_a_db), "snr_b_db": _num(snr_b_db),
            "floor_a_db": _num(floor_a_db), "floor_b_db": _num(floor_b_db),
            "ratio": _pair(ratio), "coherence": _num(coherence, 3),
            "steps_heard": None if steps_heard is None else int(steps_heard),
            "lowest_w": _num(lowest_w, 2), "mrc_gain_db": _num(mrc_gain_db),
        })

    def beacon_result(self, res):
        """The same, from a core.beacons result dict (what BeaconWatch.last
        is). Slots nobody was heard in are still worth a line: 'not heard on
        20 m at 0300 towards New Zealand' is propagation too."""
        if not res:
            return None
        return self.beacon(
            band_hz=res.get("band_hz"), callsign=res.get("call"),
            bearing_deg=res.get("bearing_deg"), distance_km=res.get("distance_km"),
            snr_a_db=res.get("snr_a"), snr_b_db=res.get("snr_b"),
            floor_a_db=res.get("floor_a_db"), floor_b_db=res.get("floor_b_db"),
            ratio=res.get("ratio"), coherence=res.get("coherence"),
            steps_heard=res.get("steps_heard"), lowest_w=res.get("lowest_w"),
            mrc_gain_db=res.get("gain_db"))

    def noise(self, *, samp_rate, center_hz, mains_hz=None, hum_db=None,
              harmonics=None, impulses_per_s=None, impulse_db=None, lines=None,
              noise_coherence=None, force=False):
        """The profile's verdict, rate-limited. Returns the line written, or
        None when this second said nothing the last line did not."""
        rec = {
            "t": _iso(self._clock()), "kind": "noise",
            "samp_rate": _num(samp_rate, 0), "center_hz": _num(center_hz, 0),
            "mains_hz": _num(mains_hz, 0), "hum_db": _num(hum_db),
            "harmonics": None if harmonics is None else int(harmonics),
            "impulses_per_s": _num(impulses_per_s), "impulse_db": _num(impulse_db),
            "lines": list(lines) if lines else [],
            "noise_coherence": _num(noise_coherence, 3),
        }
        now = float(self._clock())
        verdict = _verdict(rec)
        if not force and self._noise_at is not None:
            since = now - self._noise_at
            same = verdict == self._noise_verdict
            if since < NOISE_MIN_S or (same and since < self.noise_period_s):
                self.skipped += 1
                return None
        self._noise_at, self._noise_verdict = now, verdict
        return self._append(rec)

    def noise_status(self, status, samp_rate, center_hz, noise_coherence=None,
                     force=False):
        """The same, straight from core.noiseprofile.status()."""
        if not status:
            return None
        return self.noise(
            samp_rate=samp_rate, center_hz=center_hz,
            mains_hz=status.get("mains_hz"), hum_db=status.get("hum_db"),
            harmonics=status.get("harmonics"),
            impulses_per_s=status.get("impulses_per_s"),
            impulse_db=status.get("impulse_db"), lines=status.get("periodic"),
            noise_coherence=noise_coherence, force=force)

    def _append(self, rec):
        try:
            line = json.dumps(rec, separators=(",", ":"))
            d = os.path.dirname(self.path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except (OSError, TypeError, ValueError) as exc:
            if self.error is None:
                self.error = f"{type(exc).__name__}: {exc}"
                print(f"[sitelog] cannot write {self.path}: {exc}", flush=True)
            return None
        self.written += 1
        return rec

    # --- reading ------------------------------------------------------------
    def read(self, kind=None, since=None):
        """Yield the lines back, oldest first. A missing file yields nothing;
        a line a crash truncated is skipped, not raised."""
        cut = _epoch(since)
        try:
            f = open(self.path, "r", encoding="utf-8")
        except OSError:
            return
        with f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(rec, dict):
                    continue
                if kind is not None and rec.get("kind") != kind:
                    continue
                if cut is not None:
                    ts = _epoch(rec.get("t"))
                    if ts is None or ts < cut:
                        continue
                yield rec

    def status(self):
        return {"path": self.path, "written": int(self.written),
                "skipped": int(self.skipped), "error": self.error}


def _verdict(rec):
    """What makes this second a DIFFERENT verdict from the last one. Coarse
    on purpose: levels in 3 dB steps, rates in half decades, lines to 10 Hz,
    the centre to the nearest MHz (a band change is a new verdict; nudging
    the VFO inside one is not)."""
    lines = rec.get("lines") or []
    hz = []
    for ln in lines:
        try:
            hz.append(int(round(float(ln["hz"]) / LINE_STEP_HZ)))
        except (TypeError, ValueError, KeyError, IndexError):
            continue
    return (rec.get("mains_hz"), rec.get("harmonics"),
            _bucket(rec.get("hum_db"), HUM_STEP_DB),
            _rate_bucket(rec.get("impulses_per_s")),
            _bucket(rec.get("impulse_db"), HUM_STEP_DB),
            _bucket(rec.get("noise_coherence"), COH_STEP),
            tuple(sorted(hz)),
            _bucket(rec.get("center_hz"), 1e6), rec.get("samp_rate"))
