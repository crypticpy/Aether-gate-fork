# NBFM demodulation in the SoapySDR adapter.
#
# Every mode that was not LSB fell through to the USB taps, so asking for FM got
# an SSB product detector. That is why 2 m AX.25 never decoded through an SDR
# gate while the audio still sounded clean to the ear (found live 2026-08-07 on
# an RSP1a: clean-sounding audio, zero packet decodes, no control in AE made any
# difference because every mode landed on the same code path).
#
# The tests below feed a SYNTHESISED FM signal with a KNOWN modulating tone and
# assert the tone comes back out. A steady carrier proves nothing about a
# demodulator — constant amplitude is constant by design — so every case here
# modulates, and the sideband/tone-ratio cases are the ones that would fail if
# the FM path silently reverted to SSB.

import math

import pytest

np = pytest.importorskip("numpy")

from aether_gate.adapters.soapy import SoapyAdapter, AUDIO_RATE


def _adapter(samp_rate=240_000.0):
    """An adapter with the demod chain built, but no hardware opened."""
    a = SoapyAdapter(driver="none", samp_rate=samp_rate, center_hz=145_070_000.0)
    a._np = np
    a._init_demod()
    return a


def _fm_iq(n, fs, tone_hz, dev_hz, amp=1.0):
    """Complex baseband FM: a carrier at 0 Hz whose frequency swings +/-dev_hz
    at tone_hz. This is what the NCO hands the demodulator after mixing.

    PHASE IS THE INTEGRAL OF FREQUENCY, so integrating 2*pi*dev*cos(2*pi*f*t)
    gives (dev/f)*sin(...) radians — the modulation index beta = dev/tone, NOT
    2*pi*dev/tone. Getting that wrong puts 6.28x too much phase in: beta=15.7 rad
    wraps through np.angle and the "demodulator" appears to output the third
    harmonic. That was a bug in this test, not in the adapter, and it is worth
    naming because the failure looks exactly like a broken discriminator.
    """
    t = np.arange(n) / fs
    phase = (dev_hz / tone_hz) * np.sin(2.0 * np.pi * tone_hz * t)
    return (amp * np.exp(1j * phase)).astype(np.complex128)


def _dominant_hz(x, fs):
    """Frequency of the largest spectral peak, ignoring DC."""
    w = np.hanning(len(x))
    sp = np.abs(np.fft.rfft(x * w))
    sp[: max(1, int(len(sp) * 0.002))] = 0.0        # kill DC/very-low bins
    return float(np.argmax(sp)) * fs / len(x)


def _tone_amp(x, fs, f0):
    """Amplitude at f0 via a direct Goertzel-ish projection."""
    t = np.arange(len(x)) / fs
    return 2.0 * abs(np.mean(x * np.exp(-2j * np.pi * f0 * t)))


@pytest.mark.parametrize("mode", ["FM", "NFM", "DFM", "FM-N"])
def test_fm_modes_take_the_discriminator(mode):
    """All the FM spellings AE can send must route to the FM path.

    DFM in particular is what AE actually sent on 2 m (data-mode FM); if only
    a literal "FM" were handled, packet would still fail in the field.
    """
    a = _adapter()
    assert a._is_fm_mode(mode) is True


@pytest.mark.parametrize("mode", ["USB", "LSB", "DIGU", "DIGL", "CW", "AM", ""])
def test_non_fm_modes_stay_on_ssb(mode):
    a = _adapter()
    assert a._is_fm_mode(mode) is False


def test_demodulates_a_1200hz_tone():
    """The AX.25 mark tone must come out of the discriminator at 1200 Hz."""
    a = _adapter()
    a._mode = "FM"
    fs = a.samp_rate
    iq = _fm_iq(int(fs * 0.25), fs, tone_hz=1200.0, dev_hz=3000.0)
    out = a._demod_block(iq)
    assert len(out) > 0
    got = _dominant_hz(out, a._pd_rate)
    assert abs(got - 1200.0) < 40.0, f"expected ~1200 Hz, got {got:.0f} Hz"


def test_demodulates_a_2200hz_tone():
    """...and the space tone at 2200 Hz. Both must survive the channel filter;
    an SSB-width filter would attenuate 2200 relative to 1200 and skew the
    slicer downstream."""
    a = _adapter()
    a._mode = "FM"
    fs = a.samp_rate
    iq = _fm_iq(int(fs * 0.25), fs, tone_hz=2200.0, dev_hz=3000.0)
    out = a._demod_block(iq)
    got = _dominant_hz(out, a._pd_rate)
    assert abs(got - 2200.0) < 40.0, f"expected ~2200 Hz, got {got:.0f} Hz"


def test_bell202_tones_come_back_at_similar_amplitude():
    """THE AX.25 ASSERTION. AFSK slices on the relative level of the 1200 and
    2200 Hz tones, so the demodulator must not favour one over the other.

    A discriminator is flat with modulating frequency, so equal deviation ->
    equal amplitude. The SSB path was not: sliding a 3 kHz-wide one-sided
    bandpass over the pair attenuates 2200 far more than 1200, which is what
    stopped packets decoding even though voice sounded fine.
    """
    a = _adapter()
    a._mode = "FM"
    fs = a.samp_rate
    amps = {}
    for f in (1200.0, 2200.0):
        a._fm_prev = np.complex128(0)
        a._fm_dc = None
        a._fm_state = np.zeros(len(a._fm_taps) - 1, dtype=np.complex128)
        out = a._demod_block(_fm_iq(int(fs * 0.25), fs, tone_hz=f, dev_hz=3000.0))
        amps[f] = _tone_amp(out, a._pd_rate, f)
    ratio = amps[2200.0] / amps[1200.0]
    assert 0.7 < ratio < 1.4, (
        f"tone imbalance {ratio:.2f} (1200={amps[1200.0]:.4f} 2200={amps[2200.0]:.4f}) "
        "— a flat discriminator should treat both nearly equally")


def test_output_is_independent_of_rf_amplitude():
    """FM carries information in deviation, not amplitude. A 10x stronger
    signal must demodulate to essentially the same audio — this is what makes
    the AGC unnecessary (and harmful) on this path."""
    a = _adapter()
    a._mode = "FM"
    fs = a.samp_rate
    outs = []
    for amp in (0.1, 1.0):
        a._fm_prev = np.complex128(0)
        a._fm_dc = None
        a._fm_state = np.zeros(len(a._fm_taps) - 1, dtype=np.complex128)
        outs.append(a._demod_block(_fm_iq(int(fs * 0.25), fs, 1200.0, 3000.0, amp=amp)))
    r = _tone_amp(outs[1], a._pd_rate, 1200.0) / _tone_amp(outs[0], a._pd_rate, 1200.0)
    assert 0.8 < r < 1.25, f"amplitude dependence {r:.2f}x — discriminator should be flat"


def test_deviation_scales_the_output():
    """Louder modulation = more deviation = bigger audio. Guards against a
    discriminator that saturates or normalises the swing away."""
    a = _adapter()
    a._mode = "FM"
    fs = a.samp_rate
    levels = []
    for dev in (1500.0, 3000.0):
        a._fm_prev = np.complex128(0)
        a._fm_dc = None
        a._fm_state = np.zeros(len(a._fm_taps) - 1, dtype=np.complex128)
        out = a._demod_block(_fm_iq(int(fs * 0.25), fs, 1200.0, dev))
        levels.append(_tone_amp(out, a._pd_rate, 1200.0))
    assert levels[1] > levels[0] * 1.6, (
        f"doubling deviation only changed output {levels[1]/levels[0]:.2f}x")


def test_block_boundaries_do_not_glitch():
    """Feeding one long block and the same signal in chunks must agree.

    The discriminator differences consecutive samples, so a reset between
    blocks injects a phase step — an audible tick at the block rate and a bit
    error mid-packet. _fm_prev carries that state.
    """
    fs = 240_000.0
    sig = _fm_iq(int(fs * 0.2), fs, 1200.0, 3000.0)

    whole = _adapter(fs)
    whole._mode = "FM"
    ref = whole._demod_block(sig)

    chunked = _adapter(fs)
    chunked._mode = "FM"
    parts = [chunked._demod_block(sig[i:i + 4096]) for i in range(0, len(sig), 4096)]
    got = np.concatenate([p for p in parts if len(p)])

    n = min(len(ref), len(got))
    # Compare the recovered TONE, not sample-by-sample: the staged decimators
    # carry their own comb phase, so chunking legitimately shifts the output.
    assert abs(_dominant_hz(ref[:n], whole._pd_rate)
               - _dominant_hz(got[:n], chunked._pd_rate)) < 40.0
    # and no huge discontinuity spikes from a reset discriminator
    assert float(np.max(np.abs(np.diff(got[:n])))) < 5.0 * float(np.std(got[:n])) + 1.0


def test_fm_does_not_use_the_ssb_taps():
    """Direct regression on the original defect: the FM path must not be the
    SSB path. Demodulating the same FM signal as FM and as USB must differ."""
    fs = 240_000.0
    sig = _fm_iq(int(fs * 0.25), fs, 1200.0, 3000.0)

    afm = _adapter(fs); afm._mode = "FM"
    assb = _adapter(fs); assb._mode = "USB"
    out_fm = afm._demod_block(sig)
    out_ssb = assb._demod_block(sig)

    n = min(len(out_fm), len(out_ssb))
    # normalise both, then require they are not near-identical
    def _n(x):
        s = float(np.std(x)) or 1.0
        return x / s
    corr = float(np.corrcoef(_n(out_fm[:n]), _n(out_ssb[:n]))[0, 1])
    assert abs(corr) < 0.9, f"FM and USB outputs correlate {corr:.3f} — FM is still on the SSB path"
