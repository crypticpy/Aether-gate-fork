#
# Aether-gate - runtime radio-dossier loader.
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Load a vendored radio dossier (dossiers/<model>.json) at runtime.

The dossier format's canonical home is shack-experiments/radio-dossiers/
(schema, validator, library); the gate ships a PINNED vendored copy under
<repo-root>/dossiers/ and reads it here. This is the "served metadata"
demonstration: adapter facts (power curves, band tables, TX policy) come from
evidence-tagged data, not baked constants.

FAIL-SOFT BY DESIGN: a missing or unreadable dossier returns None and the
adapter falls back to its baked constants — the gate must never fail to start
(or, worse, change TX policy) because a data file moved. The ONE exception is
deliberate: an x-gate.tx_allowed_bands key that is PRESENT but empty means
"no TX anywhere" and is honoured fail-closed (an absent dossier is a fallback;
an explicit empty whitelist is an instruction).

Values may be plain JSON or wrapped as {"value": X, "$evidence": ..., ...};
unwrap() normalises. Keys starting "$" are sidecar metadata and are dropped
from data views (get()/section()) but preserved in .raw.
"""
import json
import os

_SIDECAR_PREFIX = "$"


def unwrap(node):
    """Normalise a dossier node: {"value": X, "$...": ...} -> unwrap(X);
    dicts lose their $-sidecar keys; lists unwrap element-wise."""
    if isinstance(node, dict):
        if "value" in node:
            return unwrap(node["value"])
        return {k: unwrap(v) for k, v in node.items()
                if not k.startswith(_SIDECAR_PREFIX)}
    if isinstance(node, list):
        return [unwrap(v) for v in node]
    return node


class Dossier:
    def __init__(self, raw, path):
        self.raw = raw
        self.path = path
        self.schema_version = raw.get("schema_version", "?")
        self.model = unwrap(raw.get("identity", {})).get("model", "?")

    def get(self, dotted, default=None):
        """Unwrapped value at a dotted path, e.g.
        get('meters.forward_power.curve_raw_to_fraction')."""
        node = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict):
                return default
            if part not in node:
                # allow the wrapped form {"value": {...}} mid-path
                if "value" in node and isinstance(node["value"], dict):
                    node = node["value"]
                    if part not in node:
                        return default
                else:
                    return default
            node = node[part]
        return unwrap(node)

    def has(self, dotted):
        _MISSING = object()
        return self.get(dotted, _MISSING) is not _MISSING


def _search_dirs():
    """Candidate dossier directories: <repo-root>/dossiers (source checkout AND
    the Pi's plain-files deploy, where aether_gate/ and dossiers/ are siblings),
    then CWD/dossiers, then $AETHER_GATE_DOSSIERS."""
    here = os.path.dirname(os.path.abspath(__file__))     # .../aether_gate
    dirs = [os.path.join(os.path.dirname(here), "dossiers"),
            os.path.join(os.getcwd(), "dossiers")]
    env = os.environ.get("AETHER_GATE_DOSSIERS")
    if env:
        dirs.insert(0, env)
    return dirs


def load(model):
    """Load the dossier for a model name ('IC-9700' -> ic-9700.json).
    Returns a Dossier, or None (fail-soft) with a one-line notice."""
    fname = model.strip().lower().replace(" ", "-").replace("_", "-") + ".json"
    for d in _search_dirs():
        path = os.path.join(d, fname)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, ValueError) as e:
            print(f"[dossier] {path} unreadable ({e}) - using baked constants",
                  flush=True)
            return None
        if "schema_version" not in raw:
            print(f"[dossier] {path} has no schema_version - using baked constants",
                  flush=True)
            return None
        return Dossier(raw, path)
    return None
