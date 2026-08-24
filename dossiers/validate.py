#!/usr/bin/env python3
#
# Aether-gate - radio dossier structural validator (schema v0.1).
# Copyright (C) 2026 Nigel Fenton (G0JKN). GPL-3.0-or-later.
#
"""Validate a radio dossier against the v0.1 structural rules.

Checks (see SCHEMA.md):
  * valid JSON, schema_version present
  * required sections present
  * every $evidence value is from the vocabulary
  * every non-extension top-level section RESOLVES an evidence tag
    (its own, inherited to it, or on every leaf claim inside it)
  * sidecar keys limited to the known set ($evidence/$note/$source/$date,
    plus $<field>_note conveniences)
  * extension sections must be named x-*

Usage: python dossiers/validate.py dossiers/ic-9700.json [more.json ...]
Exit 0 = all files pass.
"""
import json
import sys

EVIDENCE = {"hw-measured", "guide-verified", "cross-referenced", "tbd"}
SIDECAR = {"$evidence", "$note", "$source", "$date"}
REQUIRED = {"identity", "capabilities"}
KNOWN_CORE = {"identity", "transport", "capabilities", "modes", "meters",
              "scope", "audio", "quirks"}


def is_sidecar(key: str) -> bool:
    # $evidence / $note / ... plus the "$tuningRanges_note" convenience form.
    return key.startswith("$")


def check_sidecars(obj, path, errors):
    if not isinstance(obj, dict):
        return
    for k, v in obj.items():
        if is_sidecar(k):
            base = k if k in SIDECAR else None
            if base is None and not (k.endswith("_note") or k.endswith("_source")):
                errors.append(f"{path}: unknown sidecar key {k!r}")
            if k == "$evidence" and v not in EVIDENCE:
                errors.append(f"{path}: bad $evidence {v!r} (want one of {sorted(EVIDENCE)})")
        else:
            check_sidecars(v, f"{path}.{k}", errors)
        if isinstance(v, list):
            for i, item in enumerate(v):
                check_sidecars(item, f"{path}.{k}[{i}]", errors)


def claims_without_evidence(obj, inherited, path, errors):
    """Every dict that carries a 'value' or is a leaf-bearing section must see
    an evidence tag from itself or an ancestor."""
    if not isinstance(obj, dict):
        return
    ev = obj.get("$evidence", inherited)
    has_claims = any(not is_sidecar(k) for k in obj)
    if has_claims and ev is None:
        # Only flag at objects that directly carry non-dict claims — pure
        # containers are fine as long as their children resolve.
        direct = [k for k, v in obj.items()
                  if not is_sidecar(k) and not isinstance(v, dict)]
        if direct:
            errors.append(f"{path}: claims {direct} carry no $evidence "
                          f"(own or inherited)")
    for k, v in obj.items():
        if is_sidecar(k):
            continue
        if isinstance(v, dict):
            claims_without_evidence(v, ev, f"{path}.{k}", errors)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    claims_without_evidence(item, ev, f"{path}.{k}[{i}]", errors)


def validate(fname) -> list:
    errors = []
    try:
        with open(fname, encoding="utf-8") as f:
            d = json.load(f)
    except Exception as e:  # noqa: BLE001 - report, don't crash the batch
        return [f"{fname}: unreadable JSON: {e}"]

    if "schema_version" not in d:
        errors.append("missing schema_version")

    top = {k for k in d if not is_sidecar(k) and k != "schema_version"}
    for miss in REQUIRED - top:
        errors.append(f"missing required section {miss!r}")
    for k in top:
        if k not in KNOWN_CORE and not k.startswith("x-"):
            errors.append(f"unknown top-level section {k!r} "
                          f"(core sections: {sorted(KNOWN_CORE)}; extensions must be x-*)")

    check_sidecars(d, fname, errors)
    for k in sorted(top & KNOWN_CORE):
        claims_without_evidence(d[k], d.get("$evidence"), f"{fname}.{k}", errors)
    return errors


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    bad = 0
    for fname in argv[1:]:
        errs = validate(fname)
        if errs:
            bad += 1
            print(f"FAIL {fname}")
            for e in errs:
                print(f"  - {e}")
        else:
            print(f"OK   {fname}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
