"""Loop balance: notice when one antenna has quietly gone sick.

Most diversity disappointment is one loop's preamp, feedline or connector,
not the combiner: with one channel several dB down the max-SNR fit simply
leans on the other and the operator sees "no gain" for days. The noise
floors of the two channels sit within a couple of dB of each other on a
healthy pair (same band, same loops), so a floor gap that holds for
minutes is worth saying out loud.
"""
import math

WARN_DB = 6.0        # floor gap that counts as an imbalance
HOLD_S = 600.0       # ...once it has held this long
TC_S = 30.0          # smoothing on the gap: a fade is not a fault


class LoopBalance:
    def __init__(self, warn_db=WARN_DB, hold_s=HOLD_S, tc_s=TC_S):
        self.warn_db = float(warn_db)
        self.hold_s = float(hold_s)
        self.tc_s = float(tc_s)
        self.gap_db = None           # smoothed B minus A noise floor, dB
        self._t = None
        self._since = None           # when the gap first exceeded warn_db

    def update(self, Rn, now):
        """Rn: the 2x2 noise covariance; now: seconds."""
        pa, pb = float(Rn[0, 0].real), float(Rn[1, 1].real)
        if pa <= 0 or pb <= 0:
            return
        gap = 10.0 * math.log10(pb / pa)
        if self.gap_db is None or self._t is None:
            self.gap_db = gap
        else:
            al = 1.0 - math.exp(-max(0.0, now - self._t) / self.tc_s)
            self.gap_db += al * (gap - self.gap_db)
        self._t = now
        if abs(self.gap_db) >= self.warn_db:
            if self._since is None:
                self._since = now
        else:
            self._since = None

    def status(self, now):
        if self.gap_db is None:
            return {"b_minus_a_db": None, "warning": None}
        held = (now - self._since) if self._since is not None else 0.0
        warning = None
        if self._since is not None and held >= self.hold_s:
            side = "B" if self.gap_db < 0 else "A"
            warning = (f"{side} is {abs(self.gap_db):.0f} dB down for {held / 60:.0f} min: "
                       f"check that loop, its preamp and feedline")
        return {"b_minus_a_db": round(self.gap_db, 1), "warning": warning}
