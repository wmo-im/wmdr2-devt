# WMDR2 development model v0.3.2

This repository contains the current development version of the simplified WMDR2 JSON representation, converter utilities, JSON Schemas, generated examples, and tests.

The format represents WIGOS station metadata in an OGC Records / GeoJSON-oriented structure. The conversion path is intended to preserve information available in WMDR 1.0 source records, use stable controlled-concept identifiers, and avoid inventing metadata that is not present in the source.

The current model version described here is WMDR2 v0.3.2. The schema has been tightened to make the intended core contract explicit while retaining the existing conceptual model.

## Design principles

The current model follows these principles.

1. A WMDR2 station record is a GeoJSON `Feature` whose root `id` is the primary WIGOS Station Identifier (WSI).
2. Facility names and identifiers are normalized to one primary value plus explicit additional values.
3. Temporal history, where recorded, uses a `time` object with an interval. A history-capable semantic object does not automatically require time; class-specific cardinalities determine whether the temporal anchor is mandatory. Older source-specific temporal field names are not part of the public model.
4. Source equipment and configuration history is represented directly in `observingConfigurations[]`; no nested deployment/location wrapper is emitted.
5. Reusable contacts, instruments, and schedules are registries in the facility record and are referenced from the places where they are used.
6. Controlled values use absolute concept URIs. Existing canonical WMO identifiers such as `http://codes.wmo.int/...` are preserved; the converter does not contract them to notations or rewrite `http://` identifiers to `https://`.
7. Mandatory controlled properties may use `{"nilReason": "..."}` where the model explicitly allows a nil reason. Optional controlled properties are omitted when unknown rather than populated with a nil reason.
8. The converter preserves recorded information and must not fabricate validity dates, phone country codes, instrument serial numbers, observing methods, source of observation, data policy, programme affiliations, operating status, or other missing metadata.
9. Source-derived records that cannot satisfy the tightened schema without invented information are handled explicitly by the end-to-end test policy; converter regressions remain hard failures.

## Controlled values and code-list URIs

Reviewed controlled properties are represented by absolute concept URIs, for example:

```json
{
  "facilityType": "http://codes.wmo.int/wmdr/FacilityType/landFixed",
  "observedProperty": "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006",
  "observedGeometry": "http://codes.wmo.int/wmdr/Geometry/point",
  "observingMethod": "http://codes.wmo.int/wmdr/ObservingMethod/266",
  "sourceOfObservation": "http://codes.wmo.int/wmdr/SourceOfObservation/automaticReading"
}
```

The schema accepts absolute HTTP(S) URIs. This does **not** imply that identifiers should be changed from `http://` to `https://`: URI identity is preserved from the source/registry.

For a mandatory controlled property where the information is explicitly unknown, the schema may allow a nil reason:

```json
{
  "observingMethod": {
    "nilReason": "unknown"
  }
}
```

For an optional controlled property, unknown means omission. For example, `operatingStatus` is optional and non-nillable:

```json
{
  "time": {"interval": ["2020-01-01", ".."]},
  "observingMethod": "http://codes.wmo.int/wmdr/ObservingMethod/266",
  "sourceOfObservation": "http://codes.wmo.int/wmdr/SourceOfObservation/automaticReading"
}
```

The converter therefore omits an absent or explicitly unknown optional `operatingStatus`; it does not manufacture `"unknown"`.

## Record shape

A WMDR2 facility record has this top-level shape:

```json
{
  "id": "0-20008-0-THE",
  "conformsTo": [
    "http://wigos.wmo.int/spec/wmdr/2/conf/core"
  ],
  "type": "Feature",
  "geometry": {
    "type": "Point",
    "coordinates": [22.957, 40.631, 60.0]
  },
  "temporalGeometry": {
    "type": "MovingPoint",
    "coordinates": [[22.957, 40.631, 60.0]],
    "dates": ["1982-03-13"],
    "methods": [
      ["http://codes.wmo.int/wmdr/GeopositioningMethod/gps"]
    ]
  },
  "time": {
    "interval": ["1982-03-13", ".."],
    "resolution": "P1D"
  },
  "properties": {
    "type": "facility",
    "title": "Thessaloniki",
    "observationSeries": []
  },
  "links": []
}
```

### Root members

| Member | Meaning |
| --- | --- |
| `id` | Primary WIGOS Station Identifier. It is a bare WSI, for example `0-20008-0-THE`, not a prefixed identifier. |
| `conformsTo` | Conformance classes. A core record contains `http://wigos.wmo.int/spec/wmdr/2/conf/core`. |
| `type` | Always `Feature`. |
| `geometry` | Latest or representative GeoJSON point geometry. Coordinates are GeoJSON order: longitude, latitude, optional elevation. |
| `temporalGeometry` | Optional movement/location-history extension using aligned `coordinates`, `dates`, and optional controlled-URI `methods` arrays. |
| `time` | Facility temporal extent, i.e. establishment/closure of the facility. |
| `properties` | Facility metadata and related WMDR metadata blocks. |
| `links` | OGC-style links about the record. |

## Facility properties

`properties.type` is always `facility`. The facility object carries the primary description of the station and registries for reusable objects.

```json
{
  "type": "facility",
  "title": "Flüela permafrost",
  "additionalTitles": ["Flüelapass"],
  "additionalIds": ["0-756-1-387493"],
  "facilityType": "http://codes.wmo.int/wmdr/FacilityType/landFixed",
  "wmoRegion": "http://codes.wmo.int/wmdr/WMORegion/6",
  "description": "Example station description.",
  "contacts": [],
  "contactAssignments": [],
  "instruments": [],
  "observationSeries": [],
  "schedules": []
}
```

`facilityType` is mandatory and controlled. `wmoRegion` is optional and controlled.

Facility operating status is **not stored as a duplicate facility-level property**. Where an overall facility status is needed, it is derived from programme-specific affiliation/reporting status. For example, a facility may still be operational overall while observations for one programme have stopped.

### Facility names and identifiers

The converter applies deterministic primary/additional rules.

| Source concept | WMDR2 output |
| --- | --- |
| First recorded facility identifier | root `id` |
| Further recorded WSI values | `properties.additionalIds[]` |
| First recorded facility name | `properties.title` |
| Further recorded facility names | `properties.additionalTitles[]` |

`additionalIds[]` contains only values matching the WSI pattern:

```text
^(0|1|2|3)-([1-9]\d*)-([0-9]+)-([A-Za-z0-9._-]+)$
```

This avoids hiding alternate official station identifiers while keeping the root feature identifier single-valued.

## Time model

Temporal metadata is represented with OGC-style `time` objects:

```json
{
  "time": {
    "interval": ["2020-01-01", ".."],
    "resolution": "P1D"
  }
}
```

The interval is a two-element array. Each endpoint is either a date-like value (`YYYY`, `YYYY-MM`, `YYYY-MM-DD`) or `..` for an open/unknown end. `time.resolution`, where present, is an ISO 8601 duration such as `P1D`, `PT1H`, or `PT10M`.

The same structure is used wherever temporal history is recorded. `ObservingConfiguration`, `ObservingProcedure`, and `OfficialStatus` entries require time. Facility `environment`, `territory`, and `programAffiliations` occurrences may be untimed when the source records the semantic value but no validity period; the converter preserves that value rather than inventing or discarding information.

When a class requires a time anchor and the source does not provide one, the converter does not invent one.

### Observation-series temporal extent

`ObservationSeries` does **not** carry an independent `time` property.

Its temporal extent is derived from `ObservingConfiguration.time`:

- each observing-configuration interval contributes to the observation-series temporal extent;
- an interval is excluded only when its `operatingStatus`, when present, explicitly denotes that observations were not collected;
- absence of `operatingStatus` means that no status assertion was made and does **not** exclude the interval;
- detailed gaps and status history remain in `observingConfigurations[]`.

A catalogue/discovery projection may reduce these intervals to a simple envelope (earliest start to latest/open end), but that projection can hide gaps and is therefore not the authoritative history.

The classification of operating-status concepts into collecting/non-collecting states is a semantic rule and is better enforced in application/semantic validation than by JSON Schema alone.

## Spatial model

The root `geometry` is the current or representative facility position. The optional root `temporalGeometry` records location history:

```json
{
  "type": "MovingPoint",
  "coordinates": [
    [7.0, 46.0, 100.0],
    [7.1, 46.1, 101.0]
  ],
  "dates": ["2000-01-01", "2020-01-01"],
  "methods": [
    [],
    ["http://codes.wmo.int/wmdr/GeopositioningMethod/gps"]
  ]
}
```

`coordinates`, `dates`, and `methods` are aligned by array index. Empty method arrays are allowed when the source does not record the positioning method.

## Contacts

Reusable contacts are stored in `properties.contacts[]` using the OGC Records Contact model. Contact roles in WMDR are contextual, so they are represented separately through `contactAssignments[]` at the facility or observation-series level.

```json
{
  "contacts": [
    {
      "identifier": "contact:met-service-example",
      "organization": "Example Meteorological Service",
      "emails": [
        {"value": "ops@example.org"}
      ],
      "phones": [
        {"value": "+41123456789"}
      ],
      "links": [
        {
          "rel": "about",
          "href": "https://example.org",
          "type": "text/html"
        }
      ]
    }
  ],
  "contactAssignments": [
    {
      "contact": "contact:met-service-example",
      "roles": ["owner"]
    }
  ]
}
```

Phone values follow the strict OGC Contact schema used in this repository and must be E.164-style when emitted. The converter may normalize clearly international numbers, for example `00...` to `+...`, but it must not infer a country code for a local-only value.

Some historical OSCAR/Surface records contain an HTTP(S) URL in an `electronicMailAddress` slot. The converter preserves such a value as an OGC Contact `links[]` entry instead of emitting it as an invalid email address. This is a lossless correction of the source container, not invented metadata.

## Environment, territory, and programme affiliations

Facility-level environmental and administrative metadata are arrays of semantic occurrences. Each occurrence must contain semantic payload. A `time` interval is included when recorded; it is not required merely because the concept is history-capable.

```json
{
  "environment": [
    {
      "time": {"interval": ["2020-01-01", ".."]},
      "climateZone": "http://codes.wmo.int/wmdr/ClimateZone/equatorialSavannahDrySummer",
      "surfaceCover": {
        "value": "http://codes.wmo.int/wmdr/SurfaceCoverGlob2009/mosaicForest",
        "scheme": "http://codes.wmo.int/wmdr/SurfaceCoverClassification/globCover2009"
      },
      "surfaceRoughness": "http://codes.wmo.int/wmdr/SurfaceRoughness/rough",
      "topographyBathymetry": {
        "localTopography": "http://codes.wmo.int/wmdr/LocalTopography/slope",
        "relativeElevation": "http://codes.wmo.int/wmdr/RelativeElevation/middle",
        "topographicContext": "http://codes.wmo.int/wmdr/TopographicContext/rises",
        "altitudeOrDepth": "http://codes.wmo.int/wmdr/AltitudeOrDepth/veryHighAltitude"
      }
    }
  ],
  "programAffiliations": [
    {
      "time": {"interval": ["2020-01-01", ".."]},
      "program": "http://codes.wmo.int/wmdr/ProgramAffiliation/GAW",
      "programSpecificFacilityId": "GAW-TEST",
      "programSpecificFacilityTitle": "Example GAW station",
      "reportingStatus": "http://codes.wmo.int/wmdr/ReportingStatus/operational"
    }
  ]
}
```

Programme affiliations at facility level require the programme itself. A `time` interval is optional so that a valid WMDR1 affiliation without a recorded validity period is preserved. `programSpecificFacilityId` and `programSpecificFacilityTitle` are ordinary programme-specific strings, not controlled values.

Observation-series programme memberships are represented by controlled programme concept URIs in `programAffiliations[]`.

## Instruments

`properties.instruments[]` is a reusable instrument-type registry, not a list of individual physical instances. It may contain manufacturer, model, observing-method metadata, and vertical range where known.

```json
{
  "instruments": [
    {
      "id": "instrument:thermo--49i",
      "manufacturer": "Thermo",
      "model": "49i",
      "observingMethods": [
        "http://codes.wmo.int/wmdr/ObservingMethod/266"
      ],
      "verticalRange": {
        "min": 0.0,
        "max": 30.0
      }
    }
  ]
}
```

Serial numbers are not part of the instrument catalogue because they identify individual physical items rather than catalogue entries. A serial number is optional instance metadata on an `observingConfiguration`.

## Observation series

An observation series describes observations of one property or a closely related property/feature/geometry combination at the facility.

```json
{
  "id": "observationSeries:0-20008-0-THE--12006",
  "title": "Air temperature",
  "observedProperty": "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006",
  "observedFeature": {
    "domain": "http://codes.wmo.int/wmdr/Domain/atmosphere",
    "featureName": "air"
  },
  "observedGeometry": "http://codes.wmo.int/wmdr/Geometry/point",
  "applicationAreas": [
    "http://codes.wmo.int/wmdr/ApplicationArea/nowcasting"
  ],
  "programAffiliations": [
    "http://codes.wmo.int/wmdr/ProgramAffiliation/GBON"
  ],
  "observingConfigurations": [],
  "observingProcedures": [],
  "reportingProcedures": [],
  "officialStatus": [],
  "contactAssignments": []
}
```

The core requires `id`, `observedProperty`, `observedGeometry`, `observedFeature`, and `programAffiliations`.

`observedFeature.domain` is mandatory and controlled. `domainFeature` is optional and, where used, is intended to be a controlled URI. `featureName` is optional free text.

There is no independent `ObservationSeries.time`; see the temporal derivation rule above.

The XML-derived WMDR1 source may contain singular `applicationArea` values. The WMDR2 converter collects them into the plural `applicationAreas[]` list.

## Observing configurations

`observingConfigurations[]` is the time-bound history of how and where an observation series is made.

```json
{
  "time": {"interval": ["2020-01-01", ".."]},
  "observingMethod": "http://codes.wmo.int/wmdr/ObservingMethod/266",
  "operatingStatus": "http://codes.wmo.int/wmdr/InstrumentOperatingStatus/operational",
  "sourceOfObservation": "http://codes.wmo.int/wmdr/SourceOfObservation/automaticReading",
  "instrument": "instrument:thermo--49i",
  "serialNumber": "SN-001",
  "exposure": "http://codes.wmo.int/wmdr/Exposure/good",
  "geometry": {
    "type": "Point",
    "coordinates": [7.0, 46.0, 2.0]
  },
  "referenceSurface": "http://codes.wmo.int/wmdr/ReferenceSurfaceType/localGround",
  "verticalDistanceFromReferenceSurface": {
    "value": 2.0,
    "uom": "m"
  }
}
```

An observing configuration requires:

- `time`;
- `observingMethod`;
- `sourceOfObservation`.

`observingMethod` and `sourceOfObservation` are mandatory controlled properties and may use an allowed `nilReason` when explicitly unknown.

`operatingStatus` is optional and non-nillable. If the source has no status, or explicitly records a nil/unknown optional status, the converter omits the property. When present, it is a single controlled concept URI.

`serialNumber` is optional and belongs to the observing configuration, not the instrument catalogue.

If `verticalDistanceFromReferenceSurface` is present, `referenceSurface` is required. The converter does not guess a missing reference surface.

If the source carries a temporal operating-status history, the converter creates separate observing-configuration entries with the applicable `time.interval` and scalar `operatingStatus`; it does not emit an array-valued status inside one configuration.

## Observing procedures

`observingProcedures[]` contains time-bound procedure history and references reusable schedules through `observingSchedules[]`.

```json
{
  "time": {"interval": ["2020-01-01", ".."]},
  "strategy": "continuous",
  "observingSchedules": ["schedule_001"]
}
```

The validity of the procedure is carried by `time.interval`; the schedule describes a reusable temporal pattern.

## Reporting procedures

`reportingProcedures[]` contains reporting metadata for an observation series. Reporting procedures are not time-bound objects in this model.

```json
{
  "internationalExchange": true,
  "dataPolicy": "http://codes.wmo.int/wmdr/DataPolicy/noLimitation",
  "temporalReportingInterval": "PT1H",
  "temporalAggregate": "PT10M",
  "reportingSchedules": ["schedule_002"],
  "numberOfObservationsInReportingInterval": 6,
  "timeliness": "PT30M",
  "uom": "http://codes.wmo.int/wmdr/unit/K",
  "contactAssignments": []
}
```

`dataPolicy` is mandatory and controlled.

`internationalExchange` is explicit. When it is `true`, the tightened schema requires:

- `temporalReportingInterval`;
- `reportingSchedules`.

`temporalAggregate` is optional, including for international exchange. An hourly reporting interval does not imply that the reported value is an hourly aggregate.

`temporalReportingInterval` and `temporalAggregate` are ISO 8601 durations and remain properties of the `ReportingProcedure`.

They must **not** be moved to a reusable schedule or interpreted as `wmo.int:aggregationInterval`. A schedule aggregation interval is emitted only when an explicit aggregation interval is present in the source schedule semantics.

## Schedules

Reusable schedules are stored once in `properties.schedules[]` and referenced from observing and reporting procedures.

```json
{
  "uid": "schedule_001",
  "@type": "Event",
  "start": "0001-01-01T06:00:00",
  "duration": "PT12H",
  "timeZone": "UTC",
  "wmo.int:samplingFrequency": "PT10M",
  "wmo.int:aggregationInterval": "PT10M",
  "wmo.int:diurnalBaseTime": "06:00:00"
}
```

The schedule fields are intentionally JSCalendar-like, with WMO extension members for sampling, explicit aggregation, and diurnal base time. A reusable schedule requires `uid` and `start`, plus at least one meaningful schedule semantic: `duration`, a non-empty `recurrenceRules` entry, `wmo.int:samplingFrequency`, or `wmo.int:aggregationInterval`. `recurrenceOverrides` and `wmo.int:diurnalBaseTime` are modifiers and do not make an identifier-only schedule meaningful.

`duration` is reserved for a within-day coverage window. When a source gives a daily window, the converter anchors `start` to the dummy date `0001-01-01T<time>`. Real-world validity remains on the relevant time-bound WMDR object.

A single schedule may be referenced by both observing and reporting procedures when the normalized pattern is truly the same. Distinct patterns must have distinct `uid` values. No schedule-type discriminator is needed because the referencing property provides the context.

## Official status

`officialStatus[]` is a time-bound observation-series history.

```json
{
  "time": {"interval": ["2020-01-01", ".."]},
  "officialStatus": "primary"
}
```

Where a source uses a boolean official-status value, the current mapping is:

| Source value | WMDR2 value |
| --- | --- |
| `true` | `primary` |
| `false` | `additional` |
| absent | no `officialStatus` entry |

This part of the model still uses the transitional `codeValue` definition until its controlled-value contract is reviewed.

## Catalogues and derived views

The main facility record may be transformed into catalogue-oriented views. In a catalogue view, reusable contacts and instruments can be externalized to separate catalogue files while the facility record keeps lightweight references.

Typical outputs are:

```text
results/wmdr2_json_examples/
results/wmdr2_json_examples/catalogues/contacts.json
results/wmdr2_json_examples/catalogues/instruments.json
```

Instrument catalogue entries remain type-level entries. Instance-level information such as serial number is not introduced into the catalogue.

A catalogue projection may also derive simplified search fields, such as an observation-series temporal envelope, from the authoritative WMDR2 structures. Derived catalogue values should not be written back as duplicate authoritative metadata.

## Converter workflow

The main converter is:

```text
convert_wmdr10_json_to_wmdr2_json.py
```

It converts intermediate WMDR1/WMDR10 JSON into current WMDR2 facility records.

### Recommended conversion chain

For end-to-end regeneration:

```text
WMDR1 XML
  -> convert_wmdr10_xml_to_wmdr10_json.py
  -> WMDR1 JSON
  -> convert_wmdr10_json_to_wmdr2_json.py
  -> WMDR2 JSON
```

The WMDR1 stage is treated as source/intermediate metadata. Tightening in this repository is applied at the WMDR1-to-WMDR2 boundary; the converter preserves full controlled-concept URIs supplied by the WMDR1 representation.

### Configuration

From the repository root:

```bash
python convert_wmdr10_json_to_wmdr2_json.py
```

With no arguments, the converter discovers `config.yaml` or `config.yml`, reads the `convert_wmdr10_json_to_wmdr2_json` section, and uses the configured source and target paths.

A minimal configuration is:

```yaml
convert_wmdr10_json_to_wmdr2_json:
  source: resources/wmdr10_json_examples
  target: results/wmdr2_json_examples
  pattern: "*.json"
  recursive: true
```

The converter also accepts command-line paths:

```bash
python convert_wmdr10_json_to_wmdr2_json.py --source resources/wmdr10_json_examples --target results/wmdr2_json_examples
```

`--source` is an alias for `--input`, and `--target` is an alias for `--output`.

## Schemas

The primary validation schema is:

```text
schemas/wmdr2-record-feature.schema.json
```

It validates the full WMDR2 facility record as a GeoJSON Feature and uses JSON Schema draft 2020-12.

The tightened schema distinguishes reviewed controlled properties from transitional/unreviewed code-like properties. The generic `codeValue` definition remains intentionally broad only for properties whose final controlled-value contract has not yet been reviewed; it should not be interpreted as the target representation for reviewed WMDR core properties.

The schema describes current public WMDR2 output only. Earlier WMDR2 draft aliases and structures are not part of the supported converter contract merely for backwards compatibility.

## Test policy

The canonical test set verifies:

- converter helpers and record conversion;
- URI preservation for controlled values;
- CLI behaviour;
- tightened schema constraints;
- temporal geometry alignment;
- WMDR1-to-WMDR2 mapping contracts;
- XML-to-WMDR2 end-to-end conversion;
- rejection of obsolete public-model keys;
- PR-22 compatibility checks.

Recommended checks:

```bash
python -m py_compile convert_wmdr10_json_to_wmdr2_json.py
pytest -q
```

The complete suite must pass before committing; focused semantic-wrapper cardinality tests are part of the canonical test set.

### End-to-end source deficiencies

The XML examples are real/legacy source records, not a curated set of fully conformant tightened-WMDR2 fixtures.

The end-to-end test therefore allows only narrowly reviewed **source-deficiency signatures** where the missing information cannot be supplied without invention. These currently cover cases such as:

- missing `ObservingConfiguration.sourceOfObservation`;
- missing `ReportingProcedure.dataPolicy`;
- missing required time on a time-bound source-derived object;
- vertical distance without a recorded reference surface;
- phone values that cannot be safely normalized to E.164.

The converter does not fill these gaps with guessed defaults merely to make a source record validate.

The allow-list is semantic and narrow: any other validation error remains a hard failure. This means a known incomplete source record can still expose a new converter or schema regression.

### PR-22 schema compatibility check

The optional PR-22 compatibility validator checks generated WMDR2 examples against the current `wmo-im/wmdr2` PR-22 schema. It is a transition check, not the native `wmdr2-devt` validator. Temporary adaptations are applied in memory and do not modify the generated examples on disk.

## Current non-goals

The current development model deliberately does not try to solve missing metadata by inference. In particular, it does not:

- derive country codes for local phone numbers;
- construct validity intervals where no time anchor is recorded;
- invent source of observation, data policy, operating status, observing method, programme affiliation, exposure, or reference surface;
- create instrument catalogue entries for individual serial-numbered physical items;
- infer aggregation from reporting frequency;
- force observing and reporting procedures to share schedules when their temporal patterns differ;
- preserve compatibility with obsolete WMDR2 draft shapes at the expense of the current schema contract.

These constraints keep conversion faithful to the source and keep schema validation meaningful.
