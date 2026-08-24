# dossiers/ — vendored radio dossiers (pinned consumer copy)

Canonical home: `nigelfenton/shack-experiments/radio-dossiers/` (schema,
validator, full library, producers). **Edit dossiers THERE, then re-vendor
here** — this copy exists so a deployed gate never breaks when the canonical
schema moves (consumers pin; see the canonical README's rules of the road).

Vendored from: shack-experiments `ef448f0` (schema 0.1.0), 2026-08-23.

Files here are exactly what the gate's runtime dossier loader (planned) will
read. Validate before vendoring:
`python ../shack-experiments/radio-dossiers/tools/validate.py dossiers/*.json`
