# Aether-gate — where the second tuner's ring sits against the first.
"""
The offset between the two tuners' rings is a TIME: the second
activateStream lands some tens of milliseconds after the first (33 ms
measured 2026-09-03: -63 samples one start at 125 kS/s, -8316 the next at
250 kS/s, which would be ~68 000 at 2.04 MS/s). So the search window has
to be a time as well. It used to be +-8192 samples: 65 ms at 125 kS/s,
found every time; 33 ms at 250 kS/s, 124 samples short of the offset, so
after a span change every window read as noise -- "NOT credible; holding
lag 0", REALIGN included -- and the finder went dark and the spatial
waterfall lost its colours (found on the air 2026-09-03).

The span runs 62.5 kHz to 2 MHz and the pair has to lock at every one of
them, on the reader thread, without the stall a 2^21-point FFT costs
(103 ms, enough to overflow the driver and trigger another realign under
its own stall). Above SEARCH_RATE the search therefore runs on a
boxcar-decimated copy (D consecutive samples averaged: a cheap low-pass
that keeps the central rate/D of the band, never less than a 250 kHz
span's worth), and the coarse lag is refined at full rate on a
REFINE_SAMPLES segment with B pre-shifted by it. The refine is always run
and is what gets reported: its peak is measured on the full band and is
the credibility verdict, so a coarse pass that came out weak (a quiet
window keeps the loops' noise only partly coherent) is not the answer.
"""
from .diversity import find_lag

MAX_LAG_S = 0.1              # the offset is ~33 ms; three times that either way
SEARCH_RATE = 250_000.0      # the coarse pass runs at or under this rate
REFINE_SAMPLES = 1 << 17     # the full-rate segment the verdict is measured on


def measure_lag(A, B, rate, max_lag_s=MAX_LAG_S, search_rate=SEARCH_RATE,
                refine_samples=REFINE_SAMPLES):
    """(lag, peak): the lag of B behind A -- a[i] ~ b[i + lag] -- searched
    over +-max_lag_s at any rate, with the peak in units of the floor."""
    rate = float(rate)
    want = int(max_lag_s * rate)
    D = -(-int(rate) // int(search_rate))                  # ceil
    if D <= 1:
        return find_lag(A, B, min(len(A) // 4, want))
    m = min(len(A), len(B)) // D
    Ad = A[:m * D].reshape(m, D).mean(axis=1)
    Bd = B[:m * D].reshape(m, D).mean(axis=1)
    coarse, _weak = find_lag(Ad, Bd, min(m // 4, want // D))
    shift = coarse * D
    n = min(refine_samples, min(len(A), len(B)) - abs(shift))
    if shift >= 0:
        a_seg, b_seg = A[:n], B[shift:shift + n]
    else:
        a_seg, b_seg = A[-shift:-shift + n], B[:n]
    residual, peak = find_lag(a_seg, b_seg, max(256, 4 * D))
    return shift + residual, peak
