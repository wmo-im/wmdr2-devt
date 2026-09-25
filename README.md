# WMDR2 development model v0.3.3

This repository contains the current development version of the simplified WMDR2 JSON representation, converter utilities, JSON Schemas, generated examples, and tests.

The model represents WIGOS station metadata in an OGC Records / GeoJSON-oriented structure. The conversion path is designed to preserve information available in WMDR 1.0 source records, use stable controlled-concept identifiers, reuse the official `wmo-im/wmdr2` schemas wherever possible, and avoid inventing metadata that is not present in the source.

The current conceptual model is WMDR2 **v0.3.3**. The JSON schema follows the v0.3.3 model while deliberately isolating a small number of development-model extensions or divergences from the current official WMDR2 schema.

## Design principles

1. A WMDR2 station record is a GeoJSON `Feature` whose root `id` is the primary WIGOS Station Identifier (WSI).
2. The conceptual UML model and JSON serialization are related but are not identical. A singular UML role with multiplicity may serialize as a plural JSON array, for example `Instrument.observingMethod [0..*]` -> `observingMethods[]`.
3. Controlled values are represented as OGC API Records Concept objects with the complete URI in `id`.
4. The converter preserves recorded metadata and must not fabricate missing validity dates, programme affiliations, observing methods, source of observation, data policy, operating status, instrument serial numbers, reference surfaces, or other missing information.
5. Reusable Contacts, Instruments, and Schedules are serialized once in record-local registries and referenced from the semantic objects where they are used. Registry placement is a serialization mechanism, not an additional Facility ownership relationship in the conceptual model.
6. Configuration history is represented directly in `configurations[]`; there is no nested deployment/location wrapper in the WMDR2 output.
7. Observation temporal extent is derived from Configuration history rather than duplicated as an independent Observation `time`.
8. Observing and reporting cadence are properties of their respective procedures. A reusable Schedule describes a calendar pattern and does not itself carry observing-, reporting-, or aggregation-specific WMDR semantics.
9. Aggregation is a processing/provenance concept. It is not inferred from reporting frequency and is not represented as a Schedule property.
10. Earlier WMDR2 development aliases are not retained in the public output merely for backwards compatibility.

## Official WMDR2 schema reuse

The development schemas reuse the official `wmo-im/wmdr2` schema wherever the semantics are compatible.

The current pinned official revision is:

```text
wmo-im/wmdr2
987f5896c45e30c9c5f2c7bcf22cd9142a7adbf2
```

The primary local schema modules are:

```text
schemas/wmdr2-record-feature.schema.json
schemas/wmdr2-facility-properties.schema.json
schemas/wmdr2-observation.schema.json
schemas/wmdr2-configuration.schema.json
schemas/wmdr2-instrument.schema.json
schemas/wmdr2-common.schema.json
```

Compatible official properties and class definitions are referenced directly from the pinned official bundle. Local schemas contain only development-model extensions, explicit tightenings, or intentional divergences.

The main current divergence is:

- official WMDR2 embeds an Instrument object in `Configuration.instrument`;
- `wmdr2-devt` uses a record-local Instrument registry and serializes `Configuration.instrument` as a reference to `properties.instruments[].id`.

Other development-model additions include the richer ObservedFeature, observing/reporting procedure structures, record-local reusable schedules and contacts, environment metadata, and temporal geometry.

The development schema does **not** inherit the complete official record wholesale with `allOf`, because doing so would also import official cardinalities and the currently incompatible embedded-Instrument constraint. Compatible official definitions are reused selectively while the development-model contract remains explicit.

For deterministic local validation, synchronize the exact pinned official bundle:

```bash
python schemas/sync_official_wmdr2_schema.py
```

If a local clone of `wmo-im/wmdr2` is already at the pinned revision:

```bash
python schemas/sync_official_wmdr2_schema.py \
  --source ~/Public/git/wmdr2/schemas/wmdr2-bundled.json
```

The synchronization script verifies the exact Git blob before writing `schemas/official/wmdr2-bundled.json`.

The earlier temporary WCMP-specific workaround is no longer needed. The pinned official WMDR2 bundle contains the corrected `time.resolution` pattern.

## Controlled values and code-list URIs

Reviewed controlled properties are represented as Concept objects:

```json
{
  "facilityType": {
    "id": "http://codes.wmo.int/wmdr/FacilityType/landFixed"
  },
  "observedProperty": {
    "id": "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006"
  },
  "observedGeometry": {
    "id": "http://codes.wmo.int/wmdr/Geometry/point"
  },
  "observingMethod": {
    "id": "http://codes.wmo.int/wmdr/ObservingMethod/266"
  }
}
```

The converter retains complete absolute HTTP(S) identifiers. It does not contract controlled values to local notations.

One deliberate canonicalization is `wmoRegion`: historical `http://codes.wmo.int/wmdr/WMORegion/...` source values are normalized to the canonical HTTPS identifier required by the development schema:

```json
{
  "wmoRegion": {
    "id": "https://codes.wmo.int/wmdr/WMORegion/6"
  }
}
```

Where an official property is `Concept-or-null`, explicit unknown/nil source information may be represented as JSON `null`. The old development representation:

```json
{"nilReason": "unknown"}
```

is not part of the current WMDR2 output contract.

Optional controlled properties are omitted when the source contains no usable assertion unless the schema explicitly permits `null`.

## Record shape

A WMDR2 Facility record has this overall shape:

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
  "time": {
    "interval": ["1982-03-13", ".."],
    "resolution": "P1D"
  },
  "temporalGeometry": {
    "type": "MovingPoint",
    "coordinates": [[22.957, 40.631, 60.0]],
    "dates": ["1982-03-13"],
    "methods": [
      [
        {
          "id": "http://codes.wmo.int/wmdr/GeopositioningMethod/gps"
        }
      ]
    ]
  },
  "properties": {
    "type": "facility",
    "title": "Thessaloniki",
    "facilityType": {
      "id": "http://codes.wmo.int/wmdr/FacilityType/landFixed"
    },
    "observations": []
  },
  "links": []
}
```

### Root members

| Member | Meaning |
| --- | --- |
| `id` | Primary WIGOS Station Identifier. It is a bare WSI, for example `0-20008-0-THE`, not a prefixed identifier. |
| `conformsTo` | Conformance classes. A core record contains `http://wigos.wmo.int/spec/wmdr/2/conf/core`. |
| `type` | GeoJSON Feature type; always `Feature`. |
| `geometry` | Latest/current GeoJSON geometry. Coordinates are longitude, latitude, optional elevation. The property itself is required by the development record schema but may be `null` when the source has no usable position. |
| `time` | Facility temporal extent, i.e. establishment/closure of the Facility. |
| `temporalGeometry` | Optional location-history extension using aligned `coordinates`, `dates`, and optional positioning `methods`. |
| `properties` | Facility metadata plus embedded record-local registries and Observation objects. |
| `links` | Optional OGC-style links about the record. |

## Facility properties

`properties.type` is always `facility`.

The development schema currently requires:

- `type`;
- `title`;
- `facilityType`.

Additional Facility metadata may include `description`, `created`, `updated`, `contacts`, `contactAssignments`, `externalIds`, `additionalIds`, `additionalTitles`, `wmoRegion`, `territories`, `environment`, `instruments`, `schedules`, `observations`, `keywords`, and `facilitySets`.

The schema deliberately remains source-friendly at the outer Facility level so that conversion of incomplete legacy records does not require invented metadata. Inner Observation and Configuration structures are tightened independently.

### Facility names and identifiers

The converter applies deterministic primary/additional rules.

| Source concept | WMDR2 output |
| --- | --- |
| First recorded WSI | root `id` |
| Further recorded WSI values | `properties.additionalIds[]` |
| First recorded Facility name | `properties.title` |
| Further recorded Facility names | `properties.additionalTitles[]` |

`additionalIds[]` is reserved for additional WSI values.

Programme-specific Facility identifiers from WMDR1 are currently preserved through generic OGC Records `externalIds[]`:

```json
{
  "externalIds": [
    {
      "scheme": "GAW",
      "value": "PAY"
    }
  ]
}
```

Likewise, a WMDR1 `programSpecificFacilityTitle` is preserved in `additionalTitles[]`.

These are **converter mapping conventions for legacy WMDR1 information**. They do not redefine or constrain the generic OGC Records semantics of `externalIds`.

## Time model

OGC-style temporal metadata uses `time` objects:

```json
{
  "time": {
    "interval": ["2020-01-01", ".."],
    "resolution": "P1D"
  }
}
```

A period uses a two-element `interval`; an endpoint may be an ISO date-like value or `..` for an open/unknown endpoint. `resolution`, where present, is an ISO 8601 duration such as `P1D`, `PT1H`, or `PT10M`.

The conceptual model distinguishes temporal extents from arrays of individual dates:

- `Facility.time`, `Environment.time`, `Configuration.time`, and `ObservingProcedure.time` are OGC temporal objects;
- Territory and ProgramAffiliation temporal validity is serialized by the official WMDR2 structures as `dates`;
- `TemporalGeometry.date [1..*]` in UML serializes as aligned JSON `dates[]`.

The converter never fabricates a missing temporal anchor.

### Observation temporal extent

`Observation` does **not** carry an independent `time` property.

Its temporal extent is derived from `Configuration.time`:

- each Configuration interval contributes to the Observation temporal extent;
- an interval is excluded only when an explicitly recorded operating status denotes that observations were not collected;
- absence of `operatingStatus` means that no status assertion was made and does not by itself exclude the interval;
- gaps and detailed status history remain in `configurations[]`.

A catalogue/discovery projection may reduce these intervals to a simple envelope, but that projection is derived metadata and is not the authoritative history.

## Spatial model and temporal geometry

The root `geometry` is the current or representative Facility position. `temporalGeometry` records location history:

```json
{
  "type": "MovingPoint",
  "coordinates": [
    [7.0, 46.0, 100.0],
    [7.1, 46.1, 101.0]
  ],
  "dates": [
    "2000-01-01",
    "2020-01-01"
  ],
  "methods": [
    [],
    [
      {
        "id": "http://codes.wmo.int/wmdr/GeopositioningMethod/gps"
      }
    ]
  ]
}
```

`coordinates`, `dates`, and `methods`, when present, are aligned by array index. Empty method arrays are allowed when the source does not record the positioning method.

## Contacts

Contacts are reusable objects stored in `properties.contacts[]` using the OGC Records Contact model.

In the conceptual UML, WMDR entities associate directly with Contact. The role of a Contact is contextual rather than intrinsic to the reusable Contact itself. JSON therefore serializes the relationship through `contactAssignments[]`:

```json
{
  "contacts": [
    {
      "identifier": "contact:met-service-example",
      "organization": "Example Meteorological Service",
      "emails": [
        {"value": "ops@example.org"}
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

The same Contact may therefore be reused with different roles in different contexts.

Contact values referenced through `contactAssignments` require a usable `identifier`.

Phone values follow the OGC Contact schema. The converter may normalize clearly international forms such as `00...` to `+...`, but it does not invent a country code for a local-only number.

Historical source values that contain an HTTP(S) URL in an email slot are preserved as Contact `links[]` rather than emitted as invalid email addresses.

## Territory and programme affiliations

Facility Territory occurrences follow the official WMDR2 structure:

```json
{
  "territories": [
    {
      "territory": {
        "id": "http://codes.wmo.int/wmdr/TerritoryName/CHE"
      },
      "dates": ["2020-01-01", ".."]
    }
  ]
}
```

The `dates` member is emitted only when the WMDR1 source actually carries temporal information.

Facility-level programme affiliation is **not independently serialized** as a second authoritative programme-membership structure. Programme membership and reporting status belong to Observation.

Observation programme affiliations are structured objects:

```json
{
  "programAffiliations": [
    {
      "programAffiliation": {
        "id": "http://codes.wmo.int/wmdr/ProgramAffiliation/GAW"
      },
      "reportingStatus": {
        "id": "http://codes.wmo.int/wmdr/ReportingStatus/operational"
      },
      "dates": ["2020-01-01", ".."]
    }
  ]
}
```

`programAffiliation` is required for every occurrence. `reportingStatus` and `dates` are optional. Dates are preservation-only: the converter carries them when they exist in the source but never fabricates them.

Programme-specific Facility identifiers/titles from WMDR1 are preserved through the Facility mapping described above.

## Instruments

`properties.instruments[]` is a reusable Instrument catalogue. It describes a logical instrument/model/capability rather than an individual physical instance.

```json
{
  "instruments": [
    {
      "id": "vaisala-hmp155",
      "manufacturer": "Vaisala",
      "model": "HMP155",
      "observingMethods": [
        {
          "id": "http://codes.wmo.int/wmdr/ObservingMethod/266"
        }
      ]
    }
  ]
}
```

Instrument identifiers are **context-local**. A redundant type prefix such as `instrument:` is not required.

The UML property remains singular with multiplicity:

```text
Instrument.observingMethod [0..*]
```

while JSON serializes the collection as `observingMethods[]`.

Physical serial numbers are not catalogue metadata. They are optional instance metadata on Configuration.

The current development model intentionally uses a simple semantic association from Configuration to Instrument. JSON serializes that relationship as a record-local Instrument ID reference.

## Observations

A Facility contains `observations[]`.

An Observation describes observations of one property/feature/geometry combination and its programme affiliations and Configuration history.

```json
{
  "id": "12006-point",
  "title": "Air temperature",
  "observedProperty": {
    "id": "http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006"
  },
  "observedGeometry": {
    "id": "http://codes.wmo.int/wmdr/Geometry/point"
  },
  "observedFeature": {
    "domain": {
      "id": "http://codes.wmo.int/wmdr/Domain/atmosphere"
    },
    "featureName": "air"
  },
  "applicationAreas": [
    {
      "id": "http://codes.wmo.int/wmdr/ApplicationArea/nowcasting"
    }
  ],
  "programAffiliations": [
    {
      "programAffiliation": {
        "id": "http://codes.wmo.int/wmdr/ProgramAffiliation/GBON"
      }
    }
  ],
  "configurations": [
    {
      "id": "cfg-1",
      "time": {
        "interval": ["2020-01-01", ".."]
      }
    }
  ]
}
```

The current development schema requires:

- `id`;
- `observedProperty`;
- `observedGeometry`;
- `observedFeature`;
- at least one `programAffiliations` occurrence;
- at least one `configurations` occurrence.

`observedGeometry` is deliberately mandatory in `wmdr2-devt`.

There is no independent Observation `time`.

The conceptual UML may use singular role names with multiplicity, for example `applicationArea [1..*]`, while JSON serializes `applicationAreas[]`.

## Configurations

`configurations[]` is the time-bound history of how and where an Observation is made.

```json
{
  "id": "cfg-1",
  "time": {
    "interval": ["2020-01-01", ".."]
  },
  "observingMethod": {
    "id": "http://codes.wmo.int/wmdr/ObservingMethod/266"
  },
  "operatingStatus": {
    "id": "http://codes.wmo.int/wmdr/InstrumentOperatingStatus/operational"
  },
  "sourceOfObservation": {
    "id": "http://codes.wmo.int/wmdr/SourceOfObservation/automaticReading"
  },
  "instrument": "vaisala-hmp155",
  "instrumentSerialNumber": "SN-001",
  "verticalDistance": {
    "distances": [2.0],
    "unit": {
      "id": "http://codes.wmo.int/wmdr/unit/m"
    },
    "referenceSurface": {
      "id": "http://codes.wmo.int/wmdr/ReferenceSurfaceType/localGround"
    }
  }
}
```

The JSON schema requires `id` and `time`.

Controlled Configuration properties follow the corresponding official WMDR2 definitions. Where an official property permits Concept-or-null, explicit JSON `null` is valid. The converter does not create fake controlled-value URIs to satisfy a schema.

`instrumentSerialNumber` belongs to Configuration, not Instrument.

`verticalDistance` uses the official shape:

- `distances [1..*]`;
- `unit`;
- `referenceSurface`.

`unit` and `referenceSurface` are required members of the object and may be `null` where the official Concept-or-null definition permits it.

## Observing procedures

`observingProcedures[]` describes how observations are acquired. An ObservingProcedure is time-bound.

Two complementary mechanisms describe when observation occurs:

- `temporalObservingInterval [0..1]` for a simple regular cadence;
- `observingSchedules [0..*]` for a reusable calendar pattern.

At least one of these must be specified. Both may be present.

```json
{
  "time": {
    "interval": ["2020-01-01", ".."]
  },
  "strategy": {
    "id": "http://codes.wmo.int/wmdr/SamplingStrategy/continuous"
  },
  "temporalObservingInterval": "PT10M",
  "spatialObservingResolution": {
    "value": [10.0],
    "uom": "m"
  },
  "observingSchedules": ["schedule_daylight"]
}
```

`temporalObservingInterval` is an ISO 8601 duration.

`spatialObservingResolution` is quantitative:

```json
{
  "value": [10.0, 20.0],
  "uom": "m"
}
```

One value denotes a distance; two values may describe grid/pixel dimensions.

Older WMDR1 source names such as `temporalSamplingInterval` are mapped to the current `temporalObservingInterval` output property.

## Reporting procedures

`reportingProcedures[]` describes reporting and exchange metadata. ReportingProcedure is **not** time-bound in the current model.

It uses the same timing pattern as ObservingProcedure:

- `temporalReportingInterval [0..1]` for a simple regular cadence;
- `reportingSchedules [0..*]` for a reusable calendar pattern.

At least one must be specified; both may be present.

```json
{
  "internationalExchange": true,
  "dataPolicy": {
    "id": "http://codes.wmo.int/wmdr/DataPolicy/noLimitation"
  },
  "temporalReportingInterval": "PT1H",
  "spatialReportingResolution": {
    "value": [10.0],
    "uom": "km"
  },
  "diurnalBaseTime": "06:00:00",
  "numberOfObservationsInReportingInterval": 6,
  "timeliness": "PT30M",
  "uom": {
    "id": "http://codes.wmo.int/wmdr/unit/K"
  }
}
```

`internationalExchange` and `dataPolicy` are required by the development schema.

`temporalReportingInterval` is an ISO 8601 duration.

`diurnalBaseTime`, when present, is a normalized `HH:MM:SS` clock value and remains on ReportingProcedure rather than on Schedule.

`spatialReportingResolution` has the same quantitative structure as `spatialObservingResolution`.

A reporting interval describes **how often values are reported or exchanged**. It does not imply an aggregation interval.

## Reusable schedules

`properties.schedules[]` contains reusable, context-neutral JSCalendar-like Schedule objects.

```json
{
  "uid": "schedule_daylight",
  "@type": "Event",
  "start": "0001-01-01T06:00:00",
  "duration": "PT12H",
  "recurrenceRules": [
    {
      "frequency": "daily"
    }
  ],
  "timeZone": "UTC"
}
```

The development schema requires `uid` and `start`.

Schedule may additionally use `duration`, `recurrenceRules`, `recurrenceOverrides`, and `timeZone`.

Observing- and reporting-specific WMDR semantics do **not** belong to Schedule. The following earlier extensions are rejected by the current schema:

```text
wmo.int:samplingFrequency
wmo.int:aggregationInterval
wmo.int:diurnalBaseTime
```

Observation cadence remains on `ObservingProcedure.temporalObservingInterval`; reporting cadence and diurnal base time remain on ReportingProcedure.

The same Schedule may be referenced by both an ObservingProcedure and a ReportingProcedure when the actual calendar pattern is the same. Distinct patterns use distinct Schedule IDs.

## Aggregation and provenance

An aggregation interval is a processing parameter, not a reporting or scheduling parameter.

For example:

```text
temporalObservingInterval = PT10M
aggregationInterval       = PT1H
temporalReportingInterval = PT3H
```

represent three different concepts:

- observations are acquired every 10 minutes;
- hourly results may be derived from those observations;
- those results may be reported every 3 hours.

The current core WMDR2 output does not yet define the processing/provenance representation for `aggregationInterval`. The intended direction is an OGC/W3C PROV-O-based provenance model in which aggregation parameters belong to the processing Activity that generated a result.

The converter therefore does not reinterpret `temporalReportingInterval` as aggregation, and it does not place aggregation semantics on Schedule.

## Official status

OfficialStatus is still a transitional part of the development model and should not be treated as a settled controlled-value contract in this README.

Its detailed representation should be reviewed separately from the current schema/official-alignment work. The converter must not fabricate an official status when none is present in the source.

## Catalogues and derived views

The main Facility record contains record-local reusable Contacts and Instruments.

The converter can optionally produce catalogue-oriented outputs in which reusable objects are externalized while records keep lightweight references. Instrument catalogue entries remain logical/type-level entries; serial-numbered physical instances are not promoted to Instrument catalogue objects.

Derived catalogue/search values, such as an Observation temporal envelope, are projections. They must not be written back as duplicate authoritative metadata.

## Converter workflow

The main semantic converter is:

```text
convert_wmdr10_json_to_wmdr2_json.py
```

It consumes the information-preserving WMDR1/WMDR10 JSON representation and maps it to the current WMDR2 model.

### Recommended conversion chain

```text
WMDR1 XML
  -> convert_wmdr10_xml_to_wmdr10_json.py
  -> WMDR1 JSON
  -> convert_wmdr10_json_to_wmdr2_json.py
  -> WMDR2 JSON
```

The first stage is a faithful representation conversion. Semantic/model transformation is performed at the WMDR1-JSON -> WMDR2-JSON boundary.

The XML-to-WMDR1 converter should therefore not be changed merely to make WMDR2 validation pass.

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

The converter also accepts explicit paths:

```bash
python convert_wmdr10_json_to_wmdr2_json.py \
  --source resources/wmdr10_json_examples \
  --target results/wmdr2_json_examples
```

`--source` is an alias for `--input`; `--target` is an alias for `--output`.

## Schemas

The primary development-record validation schema is:

```text
schemas/wmdr2-record-feature.schema.json
```

It uses JSON Schema draft 2020-12.

The schema is modular. `tests/schema_registry.py` registers both the local modules and the exact pinned official WMDR2 bundle so that normal test/validation runs do not depend on network access.

The official-schema-reuse tests verify that compatible Facility, Observation, Configuration, Instrument, root GeoJSON, and temporal definitions really resolve to the pinned upstream schema rather than to duplicated local copies.

## Test policy

The canonical test suite covers:

- converter helpers and record conversion;
- official-schema reuse;
- current v0.3.3 naming and structure;
- controlled Concept objects and URI preservation;
- CLI behaviour;
- tightened schema constraints;
- observing/reporting timing invariants;
- context-neutral reusable Schedule semantics;
- temporal geometry alignment;
- WMDR1-to-WMDR2 mapping contracts;
- XML-to-WMDR2 end-to-end conversion;
- rejection of obsolete public-model keys.

Recommended checks:

```bash
python -m py_compile convert_wmdr10_json_to_wmdr2_json.py
pytest
```

The complete suite must pass before committing.

### End-to-end source deficiencies

The XML examples are real/legacy source records, not curated fully conformant WMDR2 fixtures.

The end-to-end tests therefore allow only narrowly reviewed source-deficiency signatures where the missing information cannot be supplied without invention. The current allow-list covers:

- Configuration without a source validity `time`;
- ObservingProcedure without a source validity `time`;
- ReportingProcedure without a usable `dataPolicy`;
- Observation without a recorded `observedGeometry`;
- Observation without a recorded programme affiliation;
- Contact phone numbers that cannot be safely normalized to the required format.

Any other validation error remains a hard failure.

## Current non-goals

The development model and converter deliberately do not:

- invent missing source metadata merely to make a record validate;
- derive country codes for ambiguous local phone numbers;
- fabricate Configuration or procedure validity intervals;
- fabricate programme affiliation, data policy, operating status, observing method, source of observation, exposure, or reference surface;
- store physical serial numbers in the reusable Instrument catalogue;
- duplicate Observation temporal extent when it can be derived from Configuration history;
- infer aggregation from observing or reporting frequency;
- encode observing/reporting semantics inside reusable Schedule;
- force observing and reporting procedures to share a Schedule when their calendar patterns differ;
- retain obsolete WMDR2 development aliases in current output solely for backwards compatibility.

These constraints keep migration faithful to the source, make validation meaningful, and keep the development model aligned as closely as possible with the official WMDR2 schema while preserving the extensions needed to test the richer WIGOS metadata model.
