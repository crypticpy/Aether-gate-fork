#
# Aether-gate — the two tuners' FIFOs, squared up at the driver.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Why the two tuners come up 33 ms apart, and how to stop them.

SoapySDRPlay3 gives every channel its own FIFO and refuses a two-channel
stream, so dual-tuner mode is TWO streams (see soapy.py `_start_stream`).
Both are fed from the one `sdrplay_api_Init()` the first activateStream
performs, and the API hands the driver a packet of 1008 samples at the
RSPduo's fixed 2 MS/s dual-tuner IF — 504 us of air, whatever output rate
the operator asked for. The driver decimates each packet by D and appends
it to that channel's fill buffer; a buffer closes at 65536/D samples, so
one buffer is exactly 65 packets = 32.76 ms (Streaming.cpp rx_callback,
`bufferLength / decimationFactor` with bufferLength = 65536 * 2 * 2).

THE SKEW. activateStream sets `reset = true` on ITS stream and nothing
else; the flag is consumed LAZILY, inside the next acquireReadBuffer for
that channel, which drains that channel's FIFO right then
(Streaming.cpp:505). The reader reads A, and only once A has returned a
whole buffer does it read B — so B's FIFO is flushed one full buffer
LATER than A's, and B's data therefore starts ~66 packets after A's:

    lag = -66 packets = -4158 samples at 125 kS/s   (-33.264 ms)
                        -2079            at  62.5 kS/s
                        -8316            at 250 kS/s
                       -16632            at 500 kS/s

which is every stream-start figure in three nights of gate logs, to the
sample, at four different rates. -4221 and -4284 are 67 and 68 packets
(a packet or two of jitter across the drain). It is NOT a ring alias and
it is NOT a delay between the antennas: two loops a few metres apart are
a small fraction of ONE sample at 125 kS/s. An overflow does the same
thing with the same lazy flush — `overflowEvent` drains that channel and
returns SOAPY_SDR_OVERFLOW — so the sign of the skew afterwards says
which tuner was flushed: -63 (one packet, A still holding a backlog)
after a channel B failure, +4032 (64 packets) after an overflow on A.

THE FIX, and it is not the correlator's job. `prime_rings` consumes the
pending flush on BOTH channels back to back with timeoutUs=0, then keeps
draining until a whole pass moves nothing. At stream start that is exact:
neither FIFO has been written yet, both flushes land inside the same 504 us
packet, and from there the two run in lockstep — the correlator sees the
residual, under a packet, and the aligner's delay line stops holding 33 ms
of channel B.

WHAT THE DRAIN CANNOT DO. A drain only takes CLOSED buffers; the partially
filled tail stays, and its start was fixed by that channel's last flush. So
after an overflow has re-phased ONE channel mid-run, priming empties the
stranded backlog but leaves the two buffer phases up to 64 packets apart
(seen live as +4032). `measured_lag` is the net for exactly that: a lag
that is a whole number of driver packets and far too big to be geometry is
folded to its residual and the packets are taken off the stream that is
ahead, rather than accepted and carried in the delay line forever.
"""
IF_RATE_HZ = 2_000_000.0        # the RSPduo's dual-tuner IF rate, fixed
PACKET_IF_SAMPLES = 1008        # one sdrplay_api stream callback at that rate
PACKET_S = PACKET_IF_SAMPLES / IF_RATE_HZ                   # 504 us

DRAIN_BUF = 1 << 14             # scratch: two buffers' worth at any decimation
DRAIN_READS_MAX = 64            # a FIFO is 8 buffers; this is a stop, not a budget
DRAIN_PASSES = 8                # passes over both channels before giving up
RESYNC_TIMEOUT_US = 200_000     # same patience as the reader's own readStream


def _ret(sr):
    """The sample count out of a readStream return, whatever shape it is."""
    return sr.ret if hasattr(sr, "ret") else (sr[0] if isinstance(sr, tuple) else 0)


def _streams_of(adapter):
    return list(getattr(adapter, "_streams", None) or ())


def _drain(adapter, stream, buf):
    """Empty one stream's FIFO; returns the samples discarded.

    timeoutUs=0 is the whole point. acquireReadBuffer consumes the pending
    reset/overflow flush first and then, with count == 0, waits zero
    microseconds and returns TIMEOUT — so a pass over both channels costs
    microseconds and the two flushes land inside the same 504 us packet.
    """
    got = 0
    for _ in range(DRAIN_READS_MAX):
        k = _ret(adapter._sdr.readStream(stream, [buf], len(buf), timeoutUs=0))
        if k <= 0:
            break
        got += k
    return got


def prime_rings(adapter, why):
    """Flush both tuners' FIFOs together so the pair starts on one packet.

    Call it once the streams are activated, and again whenever the driver
    has flushed one channel behind our back (an overflow, or a channel B
    read that failed). Returns the report dict, or None if there is no pair
    to square up or the driver raised.
    """
    streams = _streams_of(adapter)
    if len(streams) < 2:
        return None
    buf = adapter._np.empty(DRAIN_BUF, dtype=adapter._np.complex64)
    dropped = [0] * len(streams)
    passes = 0
    try:
        for passes in range(1, DRAIN_PASSES + 1):
            moved = 0
            for i, stream in enumerate(streams):
                n = _drain(adapter, stream, buf)
                dropped[i] += n
                moved += n
            if moved == 0:
                break
    except Exception as e:
        print(f"[soapy] ring sync ({why}) failed: {e!r} — the tuners' FIFOs are "
              f"still whatever the driver left them", flush=True)
        return None
    print(f"[soapy] ring sync ({why}): both tuners' FIFOs flushed together — "
          f"dropped {dropped[0]} from A, {dropped[1]} from B in {passes} pass(es); "
          f"the pair now starts on the same 504 us packet", flush=True)
    return {"why": why, "dropped": dropped, "passes": passes}


def resync_rings(adapter, shift, why):
    """Take `shift` samples off whichever tuner is ahead; returns how many.

    Sign follows the aligner's: a[n] ~ b[n + lag], so a NEGATIVE lag means
    A's FIFO started earlier and is holding |lag| extra samples at the
    front — drop them off A. A positive lag says the same of B.
    """
    streams = _streams_of(adapter)
    shift = int(shift)
    if len(streams) < 2 or shift == 0:
        return 0
    ch = 0 if shift < 0 else 1
    want = abs(shift)
    buf = adapter._np.empty(min(want, DRAIN_BUF), dtype=adapter._np.complex64)
    got = 0
    try:
        while got < want:
            k = _ret(adapter._sdr.readStream(streams[ch], [buf],
                                             min(want - got, len(buf)),
                                             timeoutUs=RESYNC_TIMEOUT_US))
            if k <= 0:
                break
            got += k
    except Exception as e:
        print(f"[soapy] ring resync ({why}) raised: {e!r} — {got} of {want} "
              f"samples dropped off channel {'AB'[ch]}", flush=True)
        return got
    print(f"[soapy] ring resync ({why}): dropped {got} of {want} samples off "
          f"channel {'AB'[ch]} — the two FIFOs now hold the same instant",
          flush=True)
    return got


def measured_lag(A, B, adapter):
    """measure_lag(), with a whole-packet FIFO skew taken out at the driver.

    The correlator's answer is honest either way — the two FIFOs really are
    that far apart when it says so — but carrying 33 ms of channel B in the
    aligner's delay line for the rest of the session is not the handling
    that offset deserves. A lag that is a whole number of driver packets and
    far too big to be antenna geometry is a FIFO skew: fold it to its
    residual and take the packets out of the stream that is ahead.

    Note the one thing arithmetic cannot do: a residual that is ITSELF a
    whole packet is indistinguishable from one more packet of skew, so a
    physical -63 at 125 kS/s folds to 0. That is why this is the net and
    prime_rings is the fix — the prime removes the skew before it can ever
    be confused with the residual.
    """
    from ..core.alignsearch import measure_lag                  # numpy: first use only
    from ..core.diversity import ALIGN_MIN_PEAK, fold_ring_skew
    rate = float(adapter.samp_rate)
    lag, peak = measure_lag(A, B, rate)
    # An incredible window is not evidence of anything, and this path DROPS
    # SAMPLES: never let a noise peak talk the reader into a resync.
    folded, packets = fold_ring_skew(lag, rate) if peak >= ALIGN_MIN_PEAK else (lag, 0)
    if not packets:
        return lag, peak
    shift = lag - folded
    ms = abs(packets) * PACKET_S * 1e3
    # Fold only what actually came off the stream. If the samples could not be
    # dropped there is nothing to fold: the two FIFOs really are that far
    # apart and the delay line has to go on holding it, which is the handling
    # this had before and is still the safe one.
    dropped = resync_rings(adapter, shift, f"{packets:+d} packet skew")
    if dropped != abs(shift):
        print(f"[diversity] ring alias: lag {lag:+d} is {packets:+d} driver FIFO "
              f"packets ({ms:.1f} ms), but only {dropped} of {abs(shift)} samples "
              f"came off the stream — carrying the measured lag instead",
              flush=True)
        return lag + (dropped if shift < 0 else -dropped), peak
    print(f"[diversity] ring alias, folding: lag {lag:+d} is {packets:+d} driver "
          f"FIFO packets ({ms:.1f} ms) — a skew between the two tuners' FIFOs, "
          f"not a delay. Folded to {folded:+d}; the packets came off the stream "
          f"that was ahead.", flush=True)
    return folded, peak
