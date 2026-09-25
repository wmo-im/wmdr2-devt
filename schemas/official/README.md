# Pinned official WMDR2 schema snapshot

`wmdr2-devt` references the official `wmo-im/wmdr2` schema directly for every
compatible part of the development model.

Pinned upstream revision:

- repository: `wmo-im/wmdr2`
- commit: `d30f13c3be6395466a778360c4a4ef59625be6af`
- file: `schemas/wmdr2-bundled.json`
- Git blob SHA-1: `649bdea87f7a641bd009a4ee118f2b5a77ee0234`

The development schemas use the immutable raw-GitHub URL at that exact commit,
not `main`.  `official/wmdr2-bundled.json` is an exact, unmodified local copy
used so tests and local validation do not depend on network access.

Refresh/check it with:

```bash
python schemas/sync_official_wmdr2_schema.py
# or, if you have a local wmo-im/wmdr2 clone at the pinned commit:
python schemas/sync_official_wmdr2_schema.py --source ~/Public/git/wmdr2/schemas/wmdr2-bundled.json
python schemas/sync_official_wmdr2_schema.py --check
```

Do not edit the vendored file.  To move to a later official WMDR2 revision,
change the pinned commit and expected Git blob SHA in the sync script and in the
development schema references, then rerun the full test suite.


The official schema is used as a source of compatible property and class constraints. The development schema does **not** inherit the complete official record with `allOf`, because that would also import official cardinalities and the currently incompatible embedded-Instrument constraint. Existing wmdr2-devt cardinalities are therefore preserved explicitly while compatible official definitions are referenced directly.

## Deliberate wmdr2-devt differences

The local schemas isolate differences instead of copying the official schemas:

- `Configuration.instrument` is a record-local reference to
  `properties.instruments[].id`; the pinned official schema still embeds an
  Instrument object.
- `Observation.observedGeometry` is mandatory in `wmdr2-devt`.
- `Observation` additionally carries the rich-model `observedFeature`,
  observing/reporting procedures and related metadata.
- `Instrument` extends the official Instrument with optional
  `observingMethods` and `verticalRange`; serial numbers remain on
  Configuration.
- `wmdr2-devt` retains reusable schedules, contact assignments, environment and
  related extension structures.

Everything else that has a compatible official definition is referenced from
the pinned official bundle.

### WCMP time dependency

The official modular WMDR2 schemas delegate `time` to WCMP. At the pinned WCMP revision, the `resolution` regular expression is over-escaped in the WCMP YAML itself, and the generated WMDR2 bundle inherits the same problem. Consequently, valid values such as `P1D` and `PT1H` are rejected by the upstream `resolution` constraint. wmdr2-devt therefore reuses the pinned WCMP `date`, `timestamp`, and `interval` sub-schemas directly, while isolating only `resolution` behind a corrected local ISO-8601 duration constraint. The WCMP snapshot remains pinned at `wmo-im/wcmp2@f05037aa8d8bf5911a44a511b7b99a0be009c9ab` (Git blob `73ac9326613473e2534b48c7b33ab6eb1f1e50b3`).
