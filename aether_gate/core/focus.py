"""Station focus: hold the combiner on one remembered talker.

With two antennas there is one degree of freedom, so the combiner can
either steer at the wanted station or null one interferer, and the
tracker's default -- steer at whoever is talking -- is the wrong choice
when the operator is digging one DX station out from under a pile-up.
A focus pins that station: its over gets the remembered weight, and any
other over is treated as an interferer and nulled from its own steering
vector, so the receiver stays deaf to the pile-up between the wanted
station's transmissions instead of following each caller in turn.
"""
import numpy as np

WEIGHT_MAX_ABS = 10.0        # mirrors diversity.WEIGHT_MAX_ABS; kept import-free


def null_of(s):
    """The multiplier m for which y = a + m b cancels a signal with unit
    steering vector s = [s_a, s_b]: s_a + m s_b = 0."""
    sa, sb = complex(s[0]), complex(s[1])
    if abs(sb) < abs(sa) / WEIGHT_MAX_ABS:
        # the signal is (almost) only on A: the deepest null allowed is
        # B alone, at the phase that still opposes what little reaches B
        ang = np.angle(-sa * np.conj(sb)) if abs(sb) > 0 else 0.0
        return complex(WEIGHT_MAX_ABS * np.exp(1j * ang))
    m = -sa / sb
    if abs(m) > WEIGHT_MAX_ABS:
        m = m / abs(m) * WEIGHT_MAX_ABS
    return complex(m)


class StationFocus:
    """Which memory entry is pinned, and what has happened since."""

    def __init__(self):
        self.talker_id = None
        self.since = None
        self.overs = 0               # focus overs heard since the pin
        self.nulled = 0              # interferer overs nulled since the pin
        self.best_db = None          # best output SNR during a focus over

    @property
    def active(self):
        return self.talker_id is not None

    def pin(self, talker_id, now):
        self.talker_id = int(talker_id)
        self.since = now
        self.overs = self.nulled = 0
        self.best_db = None

    def clear(self):
        self.__init__()

    def note_snr(self, out_db):
        if out_db is not None and (self.best_db is None or out_db > self.best_db):
            self.best_db = out_db

    def status(self, entry, now, live, nulling):
        if not self.active:
            return None
        return {"id": self.talker_id,
                "name": entry["name"] if entry is not None else None,
                "since_s": round(max(0.0, now - self.since), 1),
                "live": bool(live), "nulling": bool(nulling),
                "overs": int(self.overs), "nulled": int(self.nulled),
                "best_db": self.best_db}
