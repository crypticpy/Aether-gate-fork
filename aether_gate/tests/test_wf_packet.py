# Pin the waterfall tile frequency encoding to FlexLib "VitaFrequency" (Hz * 2^20).
#
# AE >= #4412 (VitaTileFrequency.h) decodes FrameLowFreq/BinBandwidth as
# raw / (2^20 * 1e6) MHz UNCONDITIONALLY — no plain-Hz fallback. When the gate
# sent plain Hz, every tile mapped to ~13 Hz and the waterfall rendered black
# while the panadapter stayed correct (found live on the Pi appliance,
# 2026-07-31). This test decodes exactly as AE does and asserts the tile lands
# on the pan, so a regression to plain Hz (or a double-scaling) fails loudly.

import struct

from aether_gate.core.engine import wf_packet

TILE_SUB = ">qqIHHIIHH"          # FrameLowFreq, BinBandwidth, dur, W, H, timecode, auto_black, W, 0
VITA_FREQ_TO_MHZ = 1048576.0 * 1e6   # AE's kVitaFrequencyToMhz


def _decode_like_ae(pkt, n_bins):
    sub_len = struct.calcsize(TILE_SUB)
    sub = pkt[-(sub_len + 2 * n_bins):-(2 * n_bins)]
    low_raw, binbw_raw = struct.unpack(">qq", sub[:16])
    return low_raw / VITA_FREQ_TO_MHZ, binbw_raw / VITA_FREQ_TO_MHZ


def test_wf_tile_frequency_is_vita_hz_times_2pow20():
    low_hz, binbw_hz, bins = 13_926_700.0, 244.140625, 32
    pkt = wf_packet(0x42000000, 0, [0] * bins, low_hz, binbw_hz, timecode=1)
    low_mhz, binbw_mhz = _decode_like_ae(pkt, bins)

    # AE must land the tile at the pan frequency, not 2^20 below it.
    assert abs(low_mhz - low_hz / 1e6) < 1e-6, \
        f"tile decodes to {low_mhz} MHz — plain-Hz regression (AE #4412 has no fallback)"
    assert abs(binbw_mhz * 1e6 - binbw_hz) < 1e-3

    # The whole tile must span the pan width, not collapse near DC.
    high_mhz = low_mhz + binbw_mhz * bins
    assert high_mhz > low_mhz > 13.0
