# WMDR2 GC-DAR profile

GC-DAR means **Global Catalogue Discovery, Access and Retrieval**.

It is a derived profile of a WMDR2 v0.2.5 full record. It is not intended as an
editing source of truth. Nodes should edit and publish the full WMDR2 record; the
Global Catalogue should derive GC-DAR records for discovery and portal retrieval.

## Design intent

GC-DAR is a **current-state discovery projection**. The full WMDR2 record remains
where historical detail, complete deployment history, reporting definitions and
editing semantics live.

Current/latest record-level search facets are emitted directly under `properties`.
GC-DAR deliberately avoids a separate `summary` wrapper because the whole profile
is already a discovery/access/retrieval projection. These top-level facets should
remain shallow and should not contain unrelated bags of values whose combinations
cannot be reconstructed.

Program affiliations are not represented as aligned arrays, because positional
coupling is fragile. They are represented as objects, for example
`{"program": "GAWregional", "reportingStatus": "operational"}`.

The linked combinations belong in `observationSeries`. In GC-DAR, each
observation-series summary resolves the current/latest reporting, deployment,
instrument and observing-method information directly on the observation-series
object. This avoids the ambiguous pattern where `observedProperty`,
`observedGeometry`, `observedFeatureDomain`, `observingMethod`, `deployment`,
`uom`, `internationalExchange` and `temporalAggregate` are all independent arrays.

The converter therefore:

- emits `properties.territory` as a scalar latest value, not as both `territory[]`
  and `latestTerritory`;
- emits `properties.programAffiliation` as an array of explicit objects with
  `program`, `reportingStatus` and, when available, `date`;
- emits current-state discovery facets such as `observedProperty`,
  `observedGeometry`, `observedFeatureDomain`, `instrument` and
  `observationSeriesCount` directly under `properties`;
- removes reporting-definition indirection from GC-DAR and resolves reporting
  fields onto the relevant `observationSeries` element;
- resolves deployment-to-instrument references onto the relevant
  `observationSeries` element;
- keeps top-level `deployments` and `instruments` only for the current/latest
  observation series included in GC-DAR;
- uses `provenance`, not `ancestry`, for source derivation metadata.

Recommended workflow:

```text
XML -> WMDR10 -> WMDR2 full -> WMDR2 catalogue
                           \-> WMDR2 GC-DAR
```

Recommended repository placement:

```text
wmdr2-devt/
  converters/
    convert_wmdr10_xml_to_wmdr10_json.py
    convert_wmdr10_json_to_wmdr2_full.py
    convert_wmdr2_full_to_catalogue.py
    convert_wmdr2_json_to_wmdr2_gc_dar.py
  schemas/
    wmdr2-common.schema.json
    wmdr2-record-feature.schema.json
    wmdr2-facility-sets.schema.json
    wmdr2-gc-dar-record.schema.json
  examples/
    wmdr2_full/
    wmdr2_catalogue/
    wmdr2_gc_dar/
  tests/
    test_convert_wmdr10_xml_to_wmdr10_json.py
    test_convert_wmdr10_json_to_wmdr2_full.py
    test_convert_wmdr2_full_to_catalogue.py
    test_convert_wmdr2_json_to_wmdr2_gc_dar.py
    test_wmdr2_schemas.py
    test_wmdr2_gc_dar_schema.py
```

For a minimally disruptive first commit, keep the new converter at repository
root, matching the current converter scripts, and add only:

```text
convert_wmdr2_json_to_wmdr2_gc_dar.py
schemas/wmdr2-gc-dar-record.schema.json
tests/test_convert_wmdr2_json_to_wmdr2_gc_dar.py
tests/test_wmdr2_gc_dar_schema.py
examples/wmdr2_full_minimal_example.json
examples/wmdr2_gc_dar_example.json
```

Suggested config section:

```yaml
convert_wmdr2_json_to_wmdr2_gc_dar:
  source: results/wmdr2_json_examples
  target: results/wmdr2_json_examples/gc_dar
  pattern: "*.json"
  recursive: true
  full_record_href_template: "https://example.org/wmdr2/full/{plain_id}.json"
  dar_record_href_template: "https://example.org/wmdr2/gc-dar/{plain_id}.json"
```

Run:

```bash
python convert_wmdr2_json_to_wmdr2_gc_dar.py \
  --source results/wmdr2_json_examples \
  --target results/wmdr2_json_examples/gc_dar

pytest -q tests/test_convert_wmdr2_json_to_wmdr2_gc_dar.py tests/test_wmdr2_gc_dar_schema.py
```
