#
# Aether-gate — the roofing filters: the analogue IF one, and a digital one.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The stage an FT-101MP operator means when they say "roofing filter", in
the two places this receiver actually has one.

THE ANALOGUE ROOF is the RSP's IF filter, `bwType` in SoapySDRPlay3. Its
legal widths are 200, 300, 600 and 1536 kHz (the 5-8 MHz entries disappear
in the RSPduo's dual-tuner mode), and 200 kHz is the narrowest the hardware
owns in any mode -- two orders of magnitude wider than the 3 kHz roofing
filter in the radio the question came from. Explicit control over it can
only ever make it WIDER, and that is worth saying on the row rather than
implying otherwise. What lives here is the snap: the driver silently rounds
a requested bandwidth (Settings.cpp getBwEnumForRate), so the gate snaps
explicitly, against the driver's OWN listBandwidths, and reports what it
snapped to -- the same rule _request_rate follows for sample rates.

THE DIGITAL ROOF is the one that can be narrow, and the one an operator
recognises from a menu: a short windowed-sinc FIR at the post-decimation
rate, ahead of the slice filter, with the widths the radios people know
carry (12 k / 6 k / 3 k / 2.7 k / 1.8 k / 1.2 k / 600 / 500 / 300 / 250 Hz
and the rest). It is deliberately NOT the slice filter: it sits in front of
it, so bypassing the slice FIR still leaves the operator hearing a roofing
bandwidth rather than the whole 25 kHz.

PEAK OFFSET is the digital roof's centre dragged off the slice centre, the
Icom "VC-Tune" / Kenwood "Digi-Sel" move: park a strong neighbour on the
roof's skirt instead of inside it, which the passband edges alone cannot do
once the neighbour is inside THEIR span. Implemented as a frequency shift,
not a redesign: the block is shifted down by the offset, run through the
existing (unshifted, real-tapped) lowpass, then shifted back up, with the
same index-matched phase both ways. That is exactly equivalent to designing
the taps with a shifted centre (h[k]*exp(j*2*pi*offset*k/rate) commutes
through the convolution the same way -- see test_the_offset_shift_is_exactly_
a_centre_shifted_filter) but far cheaper: two O(N) complex multiplies per
block against taps that stay real, rather than doubling every multiply in
the O(N*taps) convolution by making the taps complex. The shift is a
continuous NCO (phase carried in `_shift_phase`, per channel, across
blocks) so a preset or offset change clicks once at the change and nothing
else does.

numpy only -- no scipy on the gate hosts (see core/filter.py).
"""
import numpy as np


def snap_analogue_hz(want, options):
    """The bandwidth the driver would actually take for `want`, snapped
    against ITS OWN list rather than a table copied out of its source.

    Rounds DOWN to the widest option that is not wider than the request,
    which is what getBwEnumForRate does (a 250 kHz request becomes the
    200 kHz filter). A request under the narrowest option becomes the
    narrowest. A request WIDER than anything offered is refused rather than
    quietly snapped down: in dual-tuner mode the 5-8 MHz filters are not on
    the list at all, and answering "fine, 1536 kHz" to a request for 5 MHz
    would report a filter the operator did not ask for.
    """
    opts = sorted(float(o) for o in options or ())
    if not opts:
        raise ValueError("this driver offers no IF bandwidths")
    hz = float(want)
    if hz <= 0:
        raise ValueError(f"roof_hz must be positive, not {want!r}")
    if hz > opts[-1]:
        raise ValueError(f"roof_hz {hz:.0f} is wider than this device offers "
                         f"({opts[-1]:.0f} Hz)")
    return max([o for o in opts if o <= hz], default=opts[0])


# ----- the digital roof -------------------------------------------------------

# The menu, grouped in the design notes by the radios these widths come from:
# 25 k is "off" (the whole post-decimation band), then the FTdx101 / K3 /
# PT-8000A / IC-7851 / FTdx10 roofing menus merged and sorted.
DIGITAL_ROOF_PRESETS = [25000, 15000, 12000, 6000, 3000, 2800, 2700, 2400,
                        1800, 1200, 1000, 600, 500, 400, 300, 250, 200]
DIGITAL_ROOF_MIN_HZ = 100.0
DIGITAL_ROOF_MAX_HZ = 25000.0

ROOF_MIN_TAPS = 31
ROOF_MAX_TAPS = 511        # 511 taps at 250 Hz still puts 1.5x the edge 50 dB down
_ROOF_TRANSITION = 1.3 * 3.3    # Hamming's transition width, times a safety factor


def validate_digital_roof_hz(hz):
    """100 Hz to 25 kHz, free entry. Outside that it is a ValueError, which
    the control port turns into {"error": "bad value: ..."}."""
    v = float(hz)
    if not (DIGITAL_ROOF_MIN_HZ <= v <= DIGITAL_ROOF_MAX_HZ):
        raise ValueError(f"digital_roof_hz must be "
                         f"{DIGITAL_ROOF_MIN_HZ:.0f}..{DIGITAL_ROOF_MAX_HZ:.0f} Hz, "
                         f"not {hz!r}")
    return v


def roof_ntaps(rate_hz, hz):
    """How long the FIR has to be for this width, bounded. Odd, so the delay
    is a whole number of samples."""
    n = int(np.ceil(_ROOF_TRANSITION * float(rate_hz) / float(hz)))
    n = max(ROOF_MIN_TAPS, min(ROOF_MAX_TAPS, n))
    return n + 1 if n % 2 == 0 else n


def roof_taps(rate_hz, hz):
    """A windowed-sinc lowpass with its -6 dB point at `hz`, in taps.

    `hz` is measured FROM THE SLICE CENTRE, like a passband edge, so a 3 kHz
    roof passes +/-3 kHz around the carrier and a 2.4 kHz SSB passband still
    fits inside it whole -- which is what an operator means when they say
    their 3 kHz roofing filter passes their 2.4 kHz filter.

    THE TAP COUNT IS BOUNDED, because this runs in the real-time path. The
    length is set from the transition the width needs (Hamming: about
    3.3 * rate / N) and capped at ROOF_MAX_TAPS, which is where the narrowest
    preset lands: 511 taps at 25 kS/s. The gate's block at that rate is ~410
    samples (a 4096-sample read at 250 kS/s decimated by 10) inside a 16.4 ms
    budget; the worst case -- 511 taps, both loops of a tuner pair -- measured
    0.11 ms of that (test_digital_roofing.py, which prints the number so a
    regression shows even while the assertion still passes).
    """
    n = roof_ntaps(rate_hz, hz)
    idx = np.arange(n) - (n - 1) / 2.0
    h = np.sinc(2.0 * (float(hz) / float(rate_hz)) * idx) * np.hamming(n)
    return (h / h.sum()).astype(np.float64)


def offset_max_hz(roof_hz, passband_width_hz):
    """The widest PEAK OFFSET may move the roof's centre and still cover the
    whole slice passband: |offset| <= (roof_hz - passband_width) / 2, where
    roof_hz is the roof's own radius (its `hz`, "3 kHz roof passes +/-3 kHz")
    and passband_width_hz is the slice filter's own high_hz - low_hz. 0 --
    never negative -- when the roof itself is too narrow for the passband it
    is meant to carry, or either number is unknown; the caller holds the
    offset at 0 in that case rather than clamping into a negative range.
    """
    if roof_hz is None or passband_width_hz is None:
        return 0.0
    return max(0.0, (float(roof_hz) - float(passband_width_hz)) / 2.0)


class DigitalRoof:
    """The digital roofing filter, one per audio chain, state per channel.

    Redesigns lazily, on the thread that calls apply() -- the control port
    sets `hz` and marks it dirty, the reader thread builds the taps, exactly
    as SliceFilter does. Phase-continuous across blocks: the overlap state is
    kept per channel, so a long block and the same samples fed in pieces come
    out identical.
    """

    def __init__(self, rate_hz, hz=None, offset_hz=0.0, offset_on=False):
        self.rate_hz = float(rate_hz)
        self.hz = None if hz is None else validate_digital_roof_hz(hz)
        self.taps = None
        self.state = {}
        self.dirty = True
        self.offset_hz = float(offset_hz)         # PEAK OFFSET, signed, from the slice centre
        self.offset_on = bool(offset_on)           # the check mark; remembered while off
        self._shift_phase = {}                     # per-channel NCO phase, cycles, carried across blocks

    @property
    def active(self):
        """False when the chosen width is wider than the band the decimation
        delivers -- 15 k and 25 k at a 25 kS/s post-decimation rate. Nothing
        to filter, so nothing is filtered, and the status says so."""
        return self.hz is not None and 2.0 * self.hz < self.rate_hz

    def set(self, hz):
        """hz, or None / anything at least half the rate for 'off'."""
        self.hz = None if hz is None else validate_digital_roof_hz(hz)
        self.taps = None
        self.state = {}
        self.dirty = True
        self._shift_phase = {}
        return self.hz

    def set_offset_hz(self, hz):
        """PEAK OFFSET's Hz. Clamping against the current passband is the
        caller's job (soapy.py -- it alone knows the slice filter's edges);
        this just stores what it is given. Resets the NCO phase and the
        roof's overlap state so a change does not blend the old shift into
        the new one -- one click at the change, same as a width change."""
        self.offset_hz = float(hz)
        self.state = {}
        self._shift_phase = {}
        return self.offset_hz

    def set_offset_on(self, on):
        """The check mark. Off leaves `offset_hz` exactly as it was --
        remembered, not applied -- so flipping it back on does not lose the
        operator's setting."""
        self.offset_on = bool(on)
        self.state = {}
        self._shift_phase = {}
        return self.offset_on

    def apply(self, sig, ch=0):
        if not self.active:
            return sig
        if self.dirty or self.taps is None:
            self.taps = roof_taps(self.rate_hz, self.hz)
            self.state = {}
            self.dirty = False
        sig = np.asarray(sig, dtype=np.complex128)
        shifting = self.offset_on and self.offset_hz != 0.0
        if shifting:
            # Shift down by the offset, filter with the existing (real-tap,
            # centred-at-0) lowpass, shift back up -- with the SAME
            # index-matched phase both times. For an FIR h and a per-sample
            # phasor m[n] = exp(-j*2*pi*offset*n/rate), m[n-k] = m[n] *
            # exp(j*2*pi*offset*k/rate), so filtering (sig*m) with h and then
            # multiplying by conj(m) is exactly filtering sig with
            # h[k]*exp(j*2*pi*offset*k/rate) -- the shifted-centre taps this
            # module deliberately does NOT build, because that would double
            # every multiply the O(N*taps) convolution below does.
            step = self.offset_hz / self.rate_hz            # cycles per sample
            phase0 = self._shift_phase.get(ch, 0.0)
            ph = phase0 + step * np.arange(len(sig))
            sig = sig * np.exp(-2j * np.pi * ph)
            self._shift_phase[ch] = (phase0 + step * len(sig)) % 1.0
        n = len(self.taps)
        st = self.state.get(ch)
        if st is None:
            st = np.zeros(n - 1, dtype=np.complex128)
        x = np.concatenate([st, sig])
        y = np.convolve(x, self.taps, mode="valid")
        self.state[ch] = x[len(x) - (n - 1):]
        if shifting:
            y = y * np.exp(2j * np.pi * ph)
        return y

    def status(self, offset_max_hz=0.0):
        """`hz` is what the operator chose (or the full band when they have
        chosen nothing), never a width that is silently different.
        `offset_max_hz` is the caller's -- see core/roofing.offset_max_hz --
        because only it knows the current slice passband."""
        applied = (round(self.offset_hz)
                   if self.offset_on and self.active and self.offset_hz != 0.0
                   else 0)
        return {"hz": round(self.hz) if self.hz is not None else round(self.rate_hz),
                "full_hz": float(self.rate_hz),
                "taps": roof_ntaps(self.rate_hz, self.hz) if self.active else 0,
                "active": self.active,
                "options": list(DIGITAL_ROOF_PRESETS),
                "offset_hz": round(self.offset_hz),
                "offset_enabled": self.offset_on,
                "offset_applied_hz": applied,
                "offset_max_hz": round(float(offset_max_hz))}
