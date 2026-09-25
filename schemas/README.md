# WMDR2 development schemas

The development schema is modular and reuses the official WMDR2 schema rather
than maintaining a parallel copy.

```text
official wmo-im/wmdr2 (pinned snapshot)
        │
        ├── record/Facility properties ──────────────┐
        ├── Observation controlled properties ──────┤
        ├── Configuration properties ───────────────┤
        └── Instrument base ─────────────────────────┤
                                                    ▼
wmdr2-devt local modules
  wmdr2-common.schema.json
  wmdr2-instrument.schema.json
  wmdr2-configuration.schema.json
  wmdr2-observation.schema.json
  wmdr2-facility-properties.schema.json
  wmdr2-record-feature.schema.json
```

The official schema is used as a source of compatible property and class
constraints. The development schema does **not** inherit the complete official
record with `allOf`, because that would also import official cardinalities and
the currently incompatible embedded-Instrument constraint. Existing
`wmdr2-devt` cardinalities are therefore preserved explicitly while compatible
official definitions are referenced directly.

The authoritative entry point for a full development record remains:

`schemas/wmdr2-record-feature.schema.json`

Official definitions are referenced using an immutable raw-GitHub URL pinned
to:

`wmo-im/wmdr2@987f5896c45e30c9c5f2c7bcf22cd9142a7adbf2`

For deterministic offline/local validation, synchronize the exact verified
snapshot:

```bash
python schemas/sync_official_wmdr2_schema.py
```

Or, if a local `wmo-im/wmdr2` clone is already at the pinned commit:

```bash
python schemas/sync_official_wmdr2_schema.py \
  --source ~/Public/git/wmdr2/schemas/wmdr2-bundled.json
```

The synchronization script verifies the exact Git blob before writing:

`schemas/official/wmdr2-bundled.json`

That local file is only a verified cache of the pinned upstream artifact; it
must not be edited.

## Scope of official-schema reuse

Compatible official definitions are reused directly for:

- the GeoJSON/OGC Records root where compatible;
- Facility properties already defined by official WMDR2;
- Observation controlled properties and programme affiliations;
- Configuration properties;
- Instrument as the base reusable entity;
- OGC/WCMP temporal structures carried by the corrected official bundled
  schema.

Local development modules add or refine semantics that are not yet represented
the same way upstream, including:

- the record-local reusable Instrument registry and
  `Configuration.instrument` reference;
- the richer ObservedFeature representation;
- observing and reporting procedures;
- reusable context-neutral schedules;
- contextual contact assignments;
- environment metadata;
- temporal geometry;
- additional development-model cardinality constraints.

## Deliberate Instrument divergence

The current official WMDR2 schema embeds an Instrument object directly in
`Configuration.instrument`.

`wmdr2-devt` instead models Instrument as a reusable entity stored in
`properties.instruments[]`, with `Configuration.instrument` containing the
context-local Instrument identifier.

This divergence is intentional and isolated so that the surrounding official
Configuration constraints can still be reused.

## Time dependency

The official WMDR2 bundled schema now contains the corrected WCMP
`time.resolution` pattern. `wmdr2-devt` therefore reuses the temporal
definitions from the pinned official WMDR2 bundle directly.

The earlier temporary workaround that separately pinned WCMP and replaced the
`resolution` constraint locally is no longer required.

## Validation

Tests use `tests/schema_registry.py` to register:

1. the local `wmdr2-devt` schema modules; and
2. the verified pinned official `wmdr2-bundled.json`.

This allows normal validation and the test suite to run without network access
after synchronization.

The schema-reuse tests also verify that compatible local schema elements
actually reference the pinned official definitions rather than silently
duplicating them.
