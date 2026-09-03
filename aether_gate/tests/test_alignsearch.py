# The lag search at every span: a time, not a sample count, and exact after
# the decimated coarse pass. The state-level cases (test_soapy_diversity_state)
# drive the same thing through ingest(); these pin the function itself.
import numpy as np
import pytest

from aether_gate.core import alignsearch
from aether_gate.core.diversity import ALIGN_MIN_PEAK


def _pair(n, lag, rng):
    a = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    if lag >= 0:                                    # B lags A: a[i] ~ b[i + lag]
        b = np.concatenate([(rng.normal(size=lag) + 1j * rng.normal(size=lag)).astype(np.complex64), a[:n - lag]])
    else:
        b = np.concatenate([a[-lag:], (rng.normal(size=-lag) + 1j * rng.normal(size=-lag)).astype(np.complex64)])
    return a, b


@pytest.mark.parametrize("rate,lag", [(62_500.0, -2_079), (250_000.0, 8_316),
                                      (1_000_000.0, -33_265), (2_040_000.0, 67_861),
                                      (2_040_000.0, -67_861)])
def test_the_offset_is_found_to_the_sample_either_way_round(rate, lag):
    rng = np.random.default_rng(int(abs(lag)) % 101)
    n = min(int(0.5 * rate), 1_000_000)
    a, b = _pair(n, lag, rng)
    found, peak = alignsearch.measure_lag(a, b, rate)
    assert found == lag
    assert peak >= ALIGN_MIN_PEAK


def test_a_weak_coarse_pass_still_refines_and_a_wrong_one_is_not_credible():
    """The refine is the verdict: two unrelated channels come back with a
    peak under the bar whatever the coarse pass guessed."""
    rng = np.random.default_rng(5)
    a, _ = _pair(1_000_000, 0, rng)
    b, _ = _pair(1_000_000, 0, rng)
    _lag, peak = alignsearch.measure_lag(a, b, 2_040_000.0)
    assert peak < ALIGN_MIN_PEAK
