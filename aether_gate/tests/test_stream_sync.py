#
# Aether-gate — the two tuners' FIFOs, squared up at the driver (no hardware).
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""The dual-tuner pair used to come up 66 driver packets out of step.

_FakeSDRPlay below is not a stub: it is SoapySDRPlay3's Streaming.cpp
written out in Python — a per-channel FIFO of 65-packet buffers, a `reset`
flag set by activateStream and consumed LAZILY inside the next read of THAT
channel, an `overflowEvent` that does the same and returns SOAPY_SDR_OVERFLOW,
and readStream serving one closed buffer at a time. Nothing here knows the
number 4158; the skew falls out of those rules, exactly as it does on the
air, because the reader reads A and only then B.

The air is a single noise record, and each test says how far apart the two
antennas hear it. -63 is the figure the gate used to settle on after a
fault; it is ONE driver packet, not geometry, and with the FIFOs squared up
the real pair reads 0. The tests keep it as an injected residual to prove
the fix preserves a real offset rather than flattening everything to zero.

Run:  python -m pytest aether_gate/tests/test_stream_sync.py
"""
import numpy as np
import pytest

from aether_gate.adapters import stream_sync
from aether_gate.core import alignsearch

RATE = 125_000.0
PACKET = 63                     # 1008 IF samples decimated by 16
BUFFER_LIMIT = 65_536 // 16     # a fill buffer closes at this many samples
PACKETS_PER_BUFFER = 65         # ...so a CLOSED buffer holds 65 packets
BUFFER = PACKET * PACKETS_PER_BUFFER            # 4095
PHYSICAL_LAG = -63              # B hears the air one packet early


class _Ret:
    """readStream's return value: SoapySDR hands back an object with .ret."""

    def __init__(self, ret):
        self.ret = ret


class _Stream:
    def __init__(self, ch):
        self.ch = ch


class _Fifo:
    def __init__(self):
        self.closed = []        # [(global start index, sample count)]
        self.fill = None        # (start, count) still being written
        self.cur = None         # (start, count) part-served to readStream
        self.reset = False
        self.overflow = False
        self.registered = False

    def flush(self):
        self.closed, self.fill, self.cur = [], None, None


class _FakeSDRPlay:
    """SoapySDRPlay3's two FIFOs and their lazy flush, on a sample clock."""

    def __init__(self, air, packet=PACKET, buffer_samples=BUFFER_LIMIT):
        self.air = air                          # air[ch] -> the noise record
        self.packet = packet
        self.buffer_samples = buffer_samples
        self.now = 0                            # global index of the next packet
        self.fifos = [_Fifo(), _Fifo()]
        self.reads = [0, 0]

    # --- the callback thread ------------------------------------------
    def _deliver(self):
        """One sdrplay_api packet into every REGISTERED channel's FIFO."""
        for f in self.fifos:
            if not f.registered:
                continue                        # _streams[ch] == 0: dropped
            if f.fill is None:
                f.fill = (self.now, self.packet)
            elif f.fill[1] + self.packet >= self.buffer_samples:
                f.closed.append(f.fill)         # close it, start the new one
                f.fill = (self.now, self.packet)
            else:
                f.fill = (f.fill[0], f.fill[1] + self.packet)
        self.now += self.packet

    # --- the Soapy API the adapter uses --------------------------------
    def activateStream(self, stream):
        f = self.fifos[stream.ch]
        f.registered = True
        f.reset = True                          # consumed at the next read

    def readStream(self, stream, buffs, numElems, timeoutUs=0):
        ch = stream.ch
        f = self.fifos[ch]
        self.reads[ch] += 1
        if f.reset or f.overflow:
            was_overflow = f.overflow and not f.reset
            f.flush()
            f.reset = f.overflow = False
            if was_overflow:
                return _Ret(-4)                 # SOAPY_SDR_OVERFLOW
        if f.cur is None:
            if not f.closed and timeoutUs > 0:
                for _ in range(4 * PACKETS_PER_BUFFER):
                    self._deliver()
                    if f.closed:
                        break
            if not f.closed:
                return _Ret(-1)                 # SOAPY_SDR_TIMEOUT
            f.cur = f.closed.pop(0)
        start, left = f.cur
        k = min(int(numElems), left)
        buffs[0][:k] = self.air[ch][start:start + k]
        f.cur = None if k == left else (start + k, left - k)
        return _Ret(k)


class _FakeAdapter:
    """Just enough of SoapyAdapter for stream_sync to drive the fake driver."""

    def __init__(self, dev):
        self._np = np
        self._sdr = dev
        self._streams = [_Stream(0), _Stream(1)]
        self.samp_rate = RATE


def _air(n=600_000, seed=7, physical=PHYSICAL_LAG):
    rng = np.random.default_rng(seed)
    sig = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    # B hears the same air `physical` samples early: with a[n] ~ b[n + lag]
    # and lag < 0, B's index for an event is the lower one.
    return [sig, np.roll(sig, physical)]


def _bench(physical=PHYSICAL_LAG):
    dev = _FakeSDRPlay(_air(physical=physical))
    ad = _FakeAdapter(dev)
    for st in ad._streams:
        dev.activateStream(st)
    return dev, ad


def _read_pairs(ad, n_want, chunk=4096):
    """The read loop's own shape: a block off A, then exactly as many off B."""
    A, B = [], []
    got = 0
    buf = np.empty(chunk, dtype=np.complex64)
    while got < n_want:
        k = stream_sync._ret(ad._sdr.readStream(ad._streams[0], [buf], chunk,
                                                timeoutUs=200_000))
        if k <= 0:
            raise AssertionError(f"channel A read returned {k}")
        A.append(buf[:k].copy())
        out = np.empty(k, dtype=np.complex64)
        have = 0
        while have < k:                                     # _read_exact_b
            j = stream_sync._ret(ad._sdr.readStream(ad._streams[1], [buf],
                                                    k - have, timeoutUs=200_000))
            if j <= 0:
                raise AssertionError(f"channel B read returned {j}")
            out[have:have + j] = buf[:j]
            have += j
        B.append(out)
        got += k
    return np.concatenate(A), np.concatenate(B)


def _lag(ad, n=65_536):
    A, B = _read_pairs(ad, n)
    lag, peak = alignsearch.measure_lag(A, B, RATE)
    return lag, peak


# --- the bug, reproduced from the driver's own rules -----------------------

def test_the_pair_comes_up_66_packets_out_of_step_without_priming():
    """Read A then B the way the reader does and the lazy per-channel flush
    puts a whole driver buffer plus a packet between them: 66 * 63 = 4158
    samples on top of the physical -63, which is the -4221 the gate logs at
    stream start. Nothing in this test names 4158; the driver's rules do."""
    _dev, ad = _bench()
    lag, peak = _lag(ad)
    assert peak >= 10.0
    assert lag == PHYSICAL_LAG - 66 * PACKET == -4221


# --- the fix ---------------------------------------------------------------

def test_priming_leaves_only_the_physical_residual():
    """prime_rings consumes both pending flushes microseconds apart, so the
    next packet lands in both FIFOs and the correlator sees -63, not -4221."""
    _dev, ad = _bench()
    report = stream_sync.prime_rings(ad, "stream start")
    assert report is not None and report["passes"] >= 1
    lag, peak = _lag(ad)
    assert peak >= 10.0
    assert lag == PHYSICAL_LAG


def test_priming_a_pair_with_no_air_delay_reads_zero():
    """The real hardware case. Two loops at one site are a small fraction of
    a sample apart, so once the FIFO skew is gone the honest answer is 0 —
    and the -63 the gate used to settle on was never geometry, it was one
    driver packet of whichever channel the reader was behind on."""
    _dev, ad = _bench(physical=0)
    stream_sync.prime_rings(ad, "stream start")
    assert _lag(ad)[0] == 0


def _rephase_by_an_overflow(dev, ad, packets):
    """Overflow channel A after `packets` packets, which re-bases A's buffer
    boundaries and leaves B's where they were: the two FIFOs are then that
    many packets out of phase, which is what an overflow does on the air."""
    for _ in range(packets):
        dev._deliver()
    dev.fifos[0].overflow = True
    buf = np.empty(4096, dtype=np.complex64)
    assert stream_sync._ret(dev.readStream(ad._streams[0], [buf], 4096,
                                           timeoutUs=200_000)) == -4


def test_an_overflow_mid_run_is_folded_out_and_the_residual_holds(capsys):
    """An overflow re-phases one channel; priming clears the stranded backlog
    and the realign's fold takes the whole packets off the stream that is
    ahead. The residual the correlator reports never moves."""
    dev, ad = _bench(physical=-10)
    stream_sync.prime_rings(ad, "stream start")
    assert _lag(ad)[0] == -10
    _rephase_by_an_overflow(dev, ad, 100)
    stream_sync.prime_rings(ad, "overflow on channel A")
    capsys.readouterr()
    lag, peak = stream_sync.measured_lag(*_read_pairs(ad, 65_536), ad)
    assert peak >= 10.0
    assert lag == -10
    assert "ring alias, folding" in capsys.readouterr().out
    assert _lag(ad)[0] == -10               # ...and it stays there afterwards


def test_without_the_prime_the_overflow_strands_a_backlog_on_channel_b():
    """The mutation check for the overflow edit in soapy.py: drop the
    prime_rings call and B's stranded backlog is served to the pairing read,
    so the two channels are further apart than the phase alone."""
    dev, ad = _bench(physical=-10)
    stream_sync.prime_rings(ad, "stream start")
    _lag(ad)
    _rephase_by_an_overflow(dev, ad, 100)
    stranded = sum(n for _s, n in dev.fifos[1].closed)
    assert stranded > 0                      # B kept everything A threw away
    lag, _peak = _lag(ad)                    # no prime_rings here
    assert lag != -10
    assert (lag + 10) % PACKET == 0          # ...and it is a whole-packet skew


def test_priming_keeps_draining_until_a_whole_pass_moves_nothing():
    """The mutation check for the drain loop: one pass is not enough when a
    channel has a backlog, because draining it takes several reads and lets
    more arrive. Both FIFOs must end EMPTY, not merely shorter."""
    dev, ad = _bench()
    stream_sync.prime_rings(ad, "stream start")
    for _ in range(5 * PACKETS_PER_BUFFER):     # a real backlog on both
        dev._deliver()
    report = stream_sync.prime_rings(ad, "backlog")
    assert report["dropped"][0] > BUFFER and report["dropped"][1] > BUFFER
    for f in dev.fifos:
        assert f.closed == [] and f.cur is None


def test_prime_is_a_no_op_when_there_is_no_pair():
    ad = _FakeAdapter(_FakeSDRPlay(_air(n=8192)))
    ad._streams = ad._streams[:1]
    assert stream_sync.prime_rings(ad, "single tuner") is None


def test_prime_survives_a_driver_that_raises():
    class _Angry:
        def readStream(self, *a, **k):
            raise RuntimeError("Device has been removed. Stopping.")

    ad = _FakeAdapter(_Angry())
    assert stream_sync.prime_rings(ad, "stream start") is None


# --- resync_rings: the sign, and the defensive fold ------------------------

@pytest.mark.parametrize("shift,ch", [(-4158, 0), (4158, 1)])
def test_resync_drops_off_whichever_tuner_is_ahead(shift, ch):
    """a[n] ~ b[n + lag]: a negative lag means A's FIFO started earlier and
    is holding the extra samples, so A is the one that gives them up."""
    dev, ad = _bench()
    stream_sync.prime_rings(ad, "stream start")
    for _ in range(3 * PACKETS_PER_BUFFER):
        dev._deliver()
    before = list(dev.reads)
    got = stream_sync.resync_rings(ad, shift, "test")
    assert got == abs(shift)
    assert dev.reads[ch] > before[ch]
    assert dev.reads[1 - ch] == before[1 - ch]


def test_resync_of_zero_touches_nothing():
    dev, ad = _bench()
    before = list(dev.reads)
    assert stream_sync.resync_rings(ad, 0, "test") == 0
    assert dev.reads == before


def test_measured_lag_folds_a_whole_packet_skew_and_squares_the_rings(capsys):
    """The net under the reader: if the pair skews anyway, the packets come
    off the stream that is ahead instead of living in the delay line. The
    physical residual here is -10, so the fold has something to preserve."""
    _dev, ad = _bench(physical=-10)
    lag, peak = stream_sync.measured_lag(*_read_pairs(ad, 65_536), ad)
    assert peak >= 10.0
    assert lag == -10                           # -4168 folded to its residual
    out = capsys.readouterr().out
    assert "ring alias, folding" in out
    assert "-66 driver FIFO packets" in out
    assert stream_sync._ret(ad._sdr.readStream(ad._streams[0], [np.empty(8, np.complex64)],
                                               8, timeoutUs=200_000)) > 0


def test_the_fold_cannot_tell_a_one_packet_residual_from_the_skew(capsys):
    """Honest about the limit, and about why the fold is only the net. A
    physical -63 IS one driver packet, so -4221 folds to 0 and the residual
    is lost -- which is exactly why the fix is prime_rings in the reader
    (test_priming_leaves_only_the_physical_residual), where the skew is
    removed before it can be confused with anything."""
    _dev, ad = _bench()
    lag, _peak = stream_sync.measured_lag(*_read_pairs(ad, 65_536), ad)
    assert lag == 0
    assert "-67 driver FIFO packets" in capsys.readouterr().out


def test_measured_lag_leaves_an_ordinary_measurement_alone(capsys):
    """Requirement 4: the ordinary case must come back byte-for-byte what
    measure_lag returned, and must not print anything."""
    _dev, ad = _bench()
    stream_sync.prime_rings(ad, "stream start")
    A, B = _read_pairs(ad, 65_536)
    capsys.readouterr()
    assert stream_sync.measured_lag(A, B, ad) == alignsearch.measure_lag(A, B, RATE)
    assert capsys.readouterr().out == ""


def test_a_residual_too_far_off_a_packet_boundary_is_not_folded(capsys):
    """The mutation check for RING_FOLD_TOL. A lag a quarter of a packet or
    more away from a whole number of them is not a FIFO skew we can account
    for, so it is reported exactly as measured and nothing is dropped."""
    _dev, ad = _bench(physical=-20)             # 0.32 of a packet
    A, B = _read_pairs(ad, 65_536)
    before = list(ad._sdr.reads)
    lag, _peak = stream_sync.measured_lag(A, B, ad)
    assert lag == -20 - 66 * PACKET
    assert capsys.readouterr().out == ""
    assert ad._sdr.reads == before          # nothing said, nothing dropped


def test_an_incredible_window_never_talks_the_reader_into_dropping_samples(capsys):
    """measured_lag drops samples off a live stream, so it must not act on a
    peak the aligner would refuse. Two unrelated channels at a huge apparent
    lag come back untouched."""
    dev, ad = _bench()
    rng = np.random.default_rng(3)
    n = 65_536
    A = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    B = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
    before = list(dev.reads)
    lag, peak = stream_sync.measured_lag(A, B, ad)
    assert peak < 10.0
    assert (lag, peak) == alignsearch.measure_lag(A, B, RATE)
    assert dev.reads == before
    assert capsys.readouterr().out == ""


def test_a_skew_that_will_not_come_off_the_stream_is_carried_not_folded(capsys):
    """Mutation check on the resync gate. The fold is only earned when the
    samples actually left the FIFO: an adapter with no streams to drop from
    (the state-level tests use one) must get the measured lag back, because
    the two channels really are that far apart until something moves them."""
    class _NoStreams:
        _np = np
        _streams = []
        samp_rate = RATE

    _dev, ad = _bench()
    A, B = _read_pairs(ad, 65_536)
    capsys.readouterr()
    assert stream_sync.measured_lag(A, B, _NoStreams()) == alignsearch.measure_lag(A, B, RATE)
    out = capsys.readouterr().out
    assert "ring alias:" in out and "carrying the measured lag" in out
    assert "folding" not in out
