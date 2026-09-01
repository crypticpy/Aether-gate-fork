"""The panadapter and the S-meter must agree, and neither may follow RF gain.

Regression cover for the 2026-08-31 finding. The two paths had grown separate
calibrations: core.fft.iq_to_dbm applied NO gain correction, so raising the RF
gain 20 dB relabelled the whole dBm axis 20 dB louder while the signal at the
antenna had not moved, and SoapyAdapter.read_meters applied its own. Measured on
identical white noise they agreed to 3.8 dB at 12 dB of gain and disagreed by
16.2 dB at 32 dB — which is what put a quiet 80 m band at S9 on the meter.
"""
import numpy as np
import pytest

from aether_gate.core.fft import iq_to_dbm, dbm_offset_for
from aether_gate.adapters.soapy import SoapyAdapter, SSB_PASS_HZ

FS = 250_000.0
CENTER = 3_875_000.0
BINS = 4096
BIN_HZ = FS / BINS
# The S-meter integrates the demodulator passband; the pan reports one bin.
BANDWIDTH_RATIO_DB = 10.0 * np.log10(SSB_PASS_HZ / BIN_HZ)


def _noise(sigma=1.35e-3, n=8192, seed=3):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, sigma, n) + 1j * rng.normal(0, sigma, n)).astype(np.complex128)


def _pan_floor(iq, gain_db, trim=0.0):
    bins = iq_to_dbm(iq[:BINS], BINS, -200.0, 20.0, dbm_offset_for(gain_db, trim))
    return float(np.median(bins))


def _meter(iq, gain_db, trim=0.0):
    a = SoapyAdapter(driver="none", samp_rate=FS, center_hz=CENTER, gain_db=gain_db)
    a._np = np
    a._init_demod()
    a._mode = "LSB"
    a._slice_hz = CENTER
    a.dbm_trim = trim
    a._latest = iq
    return a.read_meters().s_meter_dbm


def _at_gain(gain_db, trim=0.0):
    """The same signal at the antenna, seen through `gain_db` of front end."""
    iq = _noise() * (10 ** ((gain_db - 12.0) / 20.0))
    return _pan_floor(iq, gain_db, trim), _meter(iq, gain_db, trim)


@pytest.mark.parametrize("gain", [12.0, 22.0, 32.0, 45.0])
def test_neither_scale_follows_the_rf_gain(gain):
    """Turning the front end up must not relabel the dBm axis.

    THE bug: the pan moved 1:1 with gain, so the operator's own gain setting
    read as signal strength.
    """
    ref_pan, ref_meter = _at_gain(12.0)
    pan, meter = _at_gain(gain)
    assert pan == pytest.approx(ref_pan, abs=0.5), (
        f"pan floor moved {pan - ref_pan:+.1f} dB going from 12 to {gain:.0f} dB of gain")
    assert meter == pytest.approx(ref_meter, abs=0.5), (
        f"S-meter moved {meter - ref_meter:+.1f} dB going from 12 to {gain:.0f} dB of gain")


@pytest.mark.parametrize("gain", [12.0, 32.0])
def test_pan_and_meter_agree_on_the_same_noise(gain):
    """Corrected for measurement bandwidth, the two must report the same thing.

    They look at identical samples; a disagreement is a calibration split, which
    is exactly what dbm_offset_for exists to make impossible.
    """
    pan, meter = _at_gain(gain)
    assert meter == pytest.approx(pan + BANDWIDTH_RATIO_DB, abs=1.5), (
        f"pan says {pan:.1f} dBm/bin (= {pan + BANDWIDTH_RATIO_DB:.1f} over "
        f"{SSB_PASS_HZ:.0f} Hz), meter says {meter:.1f} dBm")


def test_trim_moves_both_scales_together():
    """Operator calibration must not re-open the split it was added to close."""
    base_pan, base_meter = _at_gain(12.0)
    trim_pan, trim_meter = _at_gain(12.0, trim=-12.0)
    assert trim_pan == pytest.approx(base_pan - 12.0, abs=0.2)
    assert trim_meter == pytest.approx(base_meter - 12.0, abs=0.2)


def test_full_scale_carrier_reads_zero_dbfs():
    """With calibration backed out, a full-scale carrier is the 0 dBFS anchor.

    Guards the window coherent-gain division: without it the pan sat 6 dB low
    and every absolute reading inherited the error.
    """
    n = BINS
    carrier = np.exp(2j * np.pi * 10_000.0 * np.arange(n) / FS).astype(np.complex128)
    peak = max(iq_to_dbm(carrier, BINS, -200.0, 20.0))
    assert peak == pytest.approx(0.0, abs=0.5), f"full-scale carrier read {peak:.1f} dBFS"
