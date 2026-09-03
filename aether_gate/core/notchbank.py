#
# Aether-gate — SQUEEZE's notch bank: deep regardless of the slice shape.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""core/filter.py's own notches (the operator's IF NOTCH table and the ANF)
are folded into the slice FIR's design -- one convolution, whatever else the
FIR is doing. That is fine for them: they are a few tens of hertz on top of
a passband the operator already shaped. SQUEEZE's own notch, though, must
be deep NO MATTER what shape the operator picked for the passband itself --
the "soft" 255-tap window's main lobe is ~200 Hz wide at 25 kS/s, and a
comb tooth folded into that design as just another dip in the same window
only reaches a few dB before the window's own skirt swamps it. Only the
"sharp" 1023-tap Kaiser window was ever narrow enough to cut it, and the
operator should not have to give up their chosen shape to make SQUEEZE work.

So this is separate machinery: a bank of first-order complex IIR notch
sections, one per target frequency, applied to the complex baseband IQ the
slice filter itself carries (SliceFilter.apply's `sig` -- see its own
docstring: "filters one channel's decimated IQ"). FIRST order, not the
textbook real second-order biquad: a real notch needs a conjugate zero/pole
PAIR (at +-f0) to keep a real-valued output symmetric; a complex baseband
signal has no such symmetry to keep, and a target's signed frequency is not
its own mirror (the mirror is a DIFFERENT, unrelated part of the spectrum,
e.g. the far edge of the other sideband's image) -- one zero, one pole, at
the target's own signed frequency, is the correct complex-domain equivalent
and costs half what a real second-order section would:

    H_k(z) = (1 - zc_k/z) / (1 - pc_k/z),   zc_k = e^{jθ},  pc_k = r e^{jθ}

θ = 2π f0/fs puts a zero exactly on the unit circle at f0 -- the notch is
exact, not approximate, limited only by floating point. r < 1 sets how far
the paired pole sits inside the circle, which is what controls the notch's
width: the closer r is to 1, the narrower the notch and the flatter the
response away from it. See `_coeffs` for the r <-> width_hz relationship (a
first-order filter's own 3 dB half-bandwidth).

DESIGN is a CASCADE (series): the bank's response is every H_k multiplied
together, one factor per target, near-unity everywhere but that target's
own notch -- `response_db` evaluates exactly that, analytically, no
simulation involved.

RUNTIME is the algebraic equivalent in PARALLEL (partial-fraction) form,
not a literal series of K stages. A cascade of K first-order sections is
one Kth-order IIR filter; run as K stages in series that is a K-iteration
Python loop, each iteration many numpy calls over the whole block -- fine
at one section, ruinous at 24 run every audio block. Every proper rational
H(z) with distinct poles decomposes into a constant plus K independent
first-order terms fed the SAME input (not each other's output):

    H(z) = D + sum_k A_k / (1 - pc_k/z),   y[n] = D*x[n] + sum_k y_k[n]
    y_k[n] = pc_k*y_k[n-1] + A_k*x[n]

D and every A_k are residues of the product-form H(z) (`_partial_fraction`),
computed ONCE per `set_targets` call (O(K^2), trivial), never per block.
Because every y_k now shares the SAME input x, the whole bank's per-block
work is a handful of 2D (section x sample) numpy calls -- independent of K
-- via the same closed-form cumsum trick a single first-order section uses
(`_parallel_block`), rather than K times as many small calls. The two forms
are mathematically identical; `tests/test_filter.py`'s notch-bank tests
check the fast (parallel) runtime path against depths designed on the slow
(cascade) product directly, and cross-check it against a brute-force
per-sample reference for a small bank.

STATE. Only y_k[n-1] per section is carried across blocks (no x_prev: the
parallel form's recursion needs none) -- kept per channel (`_state`),
since SliceFilter.apply() runs both channels of a pair through the SAME
bank instance with the same coefficients but independent signals.
"""
import math

import numpy as np

MAX_SECTIONS = 24                 # a comb past this many in-band teeth is not realistic
R_MIN = 0.5                       # this loose a pole is already most of the passband
R_MAX = 0.9995                    # this tight is a handful of hertz; width_hz -> 0 clamps here
_SAFE_LOG_RANGE = 250.0 * math.log(10.0)   # keep pc**(-chunk) well inside float64's ~1e308


def _coeffs(hz, width_hz, rate_hz):
    """(zc, pc) for one section: a zero on the unit circle at hz, a pole at
    the same angle, radius r set from width_hz -- the first-order filter's
    own 3 dB bandwidth is (1-r)*rate_hz/pi, so r = 1 - pi*width_hz/rate_hz,
    clamped to a sane pole radius either way."""
    theta = 2.0 * math.pi * float(hz) / float(rate_hz)
    r = 1.0 - math.pi * max(float(width_hz), 1.0) / float(rate_hz)
    r = min(R_MAX, max(R_MIN, r))
    zc = complex(math.cos(theta), math.sin(theta))
    return zc, r * zc


def _partial_fraction(zc, pc):
    """D, A[k] such that prod_j (1-zc_j*z^-1)/(1-pc_j*z^-1) == D +
    sum_k A_k/(1-pc_k*z^-1) -- the residues of the product-form cascade,
    assuming distinct poles (true of any bank a real target set builds:
    two sections sharing a pole would mean two targets at the same
    frequency AND width, which callers dedupe before this is ever reached).
    D is the ratio of the two polynomials' leading coefficients; each A_k
    is the standard residue at z=1/pc_k, both closed-form in the K
    coefficients -- O(K^2), and only ever done on a retarget."""
    k = len(pc)
    if k == 0:
        return 1.0 + 0j, np.zeros(0, dtype=np.complex128)
    d = complex(np.prod(zc) / np.prod(pc))
    num = np.prod(1.0 - zc[None, :] / pc[:, None], axis=1)          # (K,): prod_j(1-zc_j/pc_k)
    den_terms = 1.0 - pc[None, :] / pc[:, None]                      # (K, K)
    np.fill_diagonal(den_terms, 1.0)                                 # drop the j==k factor
    den = np.prod(den_terms, axis=1)                                 # (K,): prod_{j!=k}(1-pc_j/pc_k)
    return d, num / den


def _powers(pc_col, m):
    """pc_col**[0, 1, ..., m-1], shape (K, m) -- via cumprod, not `**` with
    an array exponent: numpy's complex power falls back to a slow generic
    path for that (measured ~5x _powers's own cost at K=24, m=819), where a
    cumulative product of the same K numbers is a fast, native reduction."""
    k_cols = pc_col.shape[0]
    arr = np.empty((k_cols, m), dtype=np.complex128)
    arr[:, 0] = 1.0
    if m > 1:
        arr[:, 1:] = pc_col
    return np.cumprod(arr, axis=1)


def _pow_and_coef(pc, a, m):
    """(pc**[0..m), a/pc**[0..m)), both shape (K, m) -- the second is what
    `_parallel_block` actually multiplies the block's samples by every
    call; neither depends on the block's own samples, only on the bank's
    coefficients and the block length, so both belong in the same
    length-keyed cache (see NotchBank._coef_of) rather than dividing by
    pc**n freshly inside every apply()."""
    pc_pow = _powers(pc[:, None], m)
    return pc_pow, a[:, None] / pc_pow


def _parallel_block(x, pc, y_prev, coef_of):
    """The whole bank's response to one block, vectorized over every
    section at once: y_k[n] = pc_k*y_k[n-1] + a_k*x[n] for every section in
    parallel (shape (K, len(x))), closed form same as a single first-order
    section's (see core/notchbank.py's own module docstring) -- chunked to
    whatever length keeps pc**-n inside float64's range for the SHORTEST-
    lived pole in the bank (the others decay no slower). `coef_of(m)` is the
    bank's own cache of (pc**[0..m), a/pc**[0..m)) (see NotchBank._coef_of):
    a receive chain calls apply() with the SAME block length every time, so
    after the first call this is a dict lookup, and the a_k/pc_k**n divide
    -- constant for a given (bank, m), never depends on the block's own
    samples -- is paid once per retarget instead of once per block."""
    n = len(x)
    r_min = float(np.min(np.abs(pc))) if len(pc) else 1.0
    chunk = n if r_min >= 1.0 else max(64, min(n, int(_SAFE_LOG_RANGE / max(-math.log(r_min), 1e-12))))
    out = np.zeros(n, dtype=np.complex128)
    yp = y_prev.copy()
    pc_col = pc[:, None]
    for i0 in range(0, n, chunk):
        xs = x[i0:i0 + chunk]
        m = len(xs)
        pc_pow, coef = coef_of(m)                    # (K, m) each, cached across calls
        u = coef * xs[None, :]
        s = np.cumsum(u, axis=1)
        ys = pc_pow * (pc_col * yp[:, None] + s)      # (K, m)
        out[i0:i0 + m] = np.sum(ys, axis=0)
        yp = ys[:, -1]
    return out, yp


class NotchBank:
    """A bank of up to MAX_SECTIONS complex notch sections on one slice's
    complex baseband, DESIGNED as a cascade (see response_db) and RUN as the
    algebraically equivalent parallel form (see _parallel_block) -- one
    instance shared by both channels of a pair, state kept per channel."""

    def __init__(self, rate_hz):
        self.rate_hz = float(rate_hz)
        self.targets = []                          # [(hz, width_hz)], as last set
        self.recomputes = 0                         # coefficient rebuilds, for the cache test
        self._zc = self._pc = np.zeros(0, dtype=np.complex128)
        self._d = 1.0 + 0j
        self._a = np.zeros(0, dtype=np.complex128)
        self._state = {}                            # ch -> y_prev, shape (K,)
        self._pow_cache = {}                        # block length -> (pc**[0..len), a/pc**[0..len))

    @property
    def n_sections(self):
        return len(self._zc)

    def set_targets(self, targets):
        """targets: [(hz, width_hz), ...] in the same signed-hertz frame as
        core.filter.design_taps' own `notches` -- capped at MAX_SECTIONS (the
        earliest given win). A no-op, same shape as SliceFilter's own
        set_squeeze_notches, when the (rounded) set has not actually
        changed: this is called every block the notch tool holds, and a
        coefficient rebuild -- let alone the state reset it would force --
        is not free."""
        new = [(round(float(hz), 1), round(max(float(w), 1.0), 1))
              for hz, w in list(targets)[:MAX_SECTIONS]]
        if new == self.targets:
            return
        self.targets = new
        self.recomputes += 1
        if new:
            zc, pc = zip(*(_coeffs(hz, w, self.rate_hz) for hz, w in new))
            self._zc = np.array(zc, dtype=np.complex128)
            self._pc = np.array(pc, dtype=np.complex128)
        else:
            self._zc = np.zeros(0, dtype=np.complex128)
            self._pc = np.zeros(0, dtype=np.complex128)
        self._d, self._a = _partial_fraction(self._zc, self._pc)
        self._pow_cache = {}                        # keyed on self._pc, now a new array
        self._state = {}                            # section count changed: a stale splice
                                                     # would sound worse than the click of a reset

    def _coef_of(self, m):
        """(pc**[0..m), a/pc**[0..m)), cached per block length: a receive
        chain's own block size does not change block to block, so after the
        first call this is a dict lookup, not a cumulative product AND a
        divide over (K, m) again."""
        pair = self._pow_cache.get(m)
        if pair is None:
            pair = _pow_and_coef(self._pc, self._a, m)
            self._pow_cache[m] = pair
        return pair

    def apply(self, sig, ch=0):
        """Filter one channel's block of complex IQ through the whole bank,
        state carried under `ch` across calls. A pass-through, at the cost
        of nothing, with no targets set."""
        k = self.n_sections
        if k == 0 or len(sig) == 0:
            return sig
        y_prev = self._state.get(ch, np.zeros(k, dtype=np.complex128))
        x = np.asarray(sig, dtype=np.complex128)
        par, new_yp = _parallel_block(x, self._pc, y_prev, self._coef_of)
        self._state[ch] = new_yp
        return self._d * x + par

    def response_db(self, hz):
        """The bank's own DESIGNED response, dB, at one signed frequency or
        an array of them (matching core.filter.response_at's signature for
        one value) -- the analytic |H(e^{jw})| of every section's CASCADE
        factor multiplied together (summed in dB): not a simulation, and
        independent of the parallel runtime form apply() actually uses."""
        scalar = np.isscalar(hz)
        f = np.atleast_1d(np.asarray(hz, dtype=np.float64))
        if self.n_sections == 0:
            out = np.zeros_like(f)
            return float(out[0]) if scalar else out
        w = np.exp(-2j * np.pi * f[:, None] / self.rate_hz)          # (M, 1)
        h = (1.0 - self._zc[None, :] * w) / (1.0 - self._pc[None, :] * w)   # (M, K)
        mag = np.maximum(np.prod(np.abs(h), axis=1), 1e-12)
        out = 20.0 * np.log10(mag)
        return float(out[0]) if scalar else out
