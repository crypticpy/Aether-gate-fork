# Radio dossiers — schema draft v0.1 (2026-08-23)

A **dossier** is one JSON file describing one radio model: what it truly is,
with an **evidence tag on every claim**. Dossiers separate *radio truth* from
*consumer presentation* (the gate's Flex costume, AE's UI choices) — a radio
graduating from gate-bridged to native-AE-backend takes its dossier with it.

This is the private proof-of-concept for the AetherSDR "radio dossiers" idea
(see AE's `RadioCapabilities` / Icom `IcomModelProfile` facet system, aetherd
RFC §4.1/§5.5). Vocabulary deliberately mirrors AE's so the schema can be
proposed upstream without translation.

## Design rules

1. **Truth only in the core.** The core sections describe the radio as built.
   Anything a consumer decides (which Flex model to impersonate, which bands an
   operator allows TX on) lives in a namespaced extension (`x-gate`, `x-ae`, …)
   — mirroring AE's "typed core + namespaced extensions" capability design.
   Policy is never allowed to masquerade as hardware fact.

2. **Every claim carries evidence.** The `$evidence` key applies to the object
   it appears in and **inherits downward** until overridden. A single value can
   override by wrapping: `{"value": X, "$evidence": "...", "$note": "..."}`.
   Evidence levels (ordered, strongest first):

   | tag | meaning |
   |---|---|
   | `hw-measured` | measured on real hardware; `$source` says when/how |
   | `guide-verified` | confirmed against the model's own official reference (CI-V Reference Guide etc.); `$source` names the document |
   | `cross-referenced` | taken from a credible secondary source (another model's guide, community knowledge, vendor marketing) — **not licence to stream/key** |
   | `tbd` | unknown. Generators MUST emit nothing for `tbd` fields — an absent control is a better answer than one that lies |

3. **Notes carry the "why".** `$note` preserves the reasoning that stops the
   next reader from "fixing" a deliberate value (AE precedent:
   `hasManualNotch=false` *because a TNF is a different instrument*). A refuted
   hypothesis is recorded as a note on the true value, not deleted — e.g. the
   IC-9700 "scope firmware cap" that wire-diffing disproved.

4. **Sidecar keys** (allowed in any object): `$evidence`, `$note`, `$source`,
   `$date`. Keys beginning `x-` are namespaced consumer extensions and follow
   the same evidence rules internally.

5. **Field names mirror AE** where an AE name exists (`maxSlices`,
   `canTransmit`, `txPowerBands`, `tuningMinHz`…). Family-specific facts live
   under `family_specific` (for Icom: CI-V address, wire mode bytes, scope
   command geometry — the `IcomModel`/`IcomModelProfile` vocabulary).

## Sections

```
identity          model / manufacturer / family / aliases
transport         how you reach it (lan|usb|serial), protocol quirks that are
                  RADIO behaviour (session limits, remembered ports)
capabilities      the AE RadioCapabilities-shaped core: receivers, vfos,
                  canTransmit, txPowerBands (per-band watts), tuner, bands
modes             neutral mode list, wire encodings, fold table for consumer
                  vocabularies that are supersets (Flex DIGU/DFM…), and the
                  facts that shape folding ("no data-mode byte")
meters            calibration: power curves (raw→fraction), level semantics
scope             waveform output geometry + keepalive behaviour
audio             audio paths the radio offers, formats, rates
quirks            reproducible behaviours that aren't capabilities (stalls,
                  recovery requirements) — each with evidence + note
x-gate            Aether-gate presentation & policy: advertised Flex model,
                  operator TX whitelist, XVTR mapping. NOT radio truth.
```

## Versioning

`schema_version` (semver) at the top of every dossier. This draft: `0.1.0`.
Breaking field renames bump minor while <1.0.

## Validation

`python dossiers/validate.py dossiers/ic-9700.json` — structural checks:
required sections, evidence-tag vocabulary, every top-level section resolves an
inherited evidence tag, `x-` prefix rule, no unknown sidecar keys.

## Open questions (deliberately unresolved in v0.1)

- Per-band capability variance (the IC-705 preamp collapsing above 50 MHz —
  AE handles it by publishing the HF ladder and letting the radio refuse;
  a dossier probably wants `bands[].overrides` eventually).
- Whether `modes.wire` belongs in core or `family_specific` (it is wire truth,
  but its byte values are family vocabulary). v0.1 keeps it under
  `family_specific` next to the other CI-V facts.
- How a dossier states firmware-version dependence (`$firmware` sidecar?).
