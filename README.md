# WMDR2 development

This repository contains experimental tooling for transforming legacy WMDR 1.0 XML records into a simplified WMDR10 JSON representation and then into a draft WMDR2 JSON representation.

The current WMDR2 output is a facility-centric full record encoded as a GeoJSON-like `Feature` with WMDR2-specific content in `properties`. Output files use the `.json` extension.

## References

- [OGC API - Records - Part 1: Core](https://docs.ogc.org/is/20-004r1/20-004r1.html)
- [WMO-No. 1192 WIGOS Metadata Standard](https://library.wmo.int/documents/wmo-1192)
- [WMDR 1.0 schemas](https://schemas.wmo.int/wmdr/1.0/)
- [WMDR2 draft UML aligned to OMS](https://wmo-im.github.io/wmdr2-devt/oms/html/)

## Current workflow

The current conversion workflow has two active stages.

### 1. Convert WMDR 1.0 XML to WMDR10 JSON

```bash
python convert_wmdr10_xml_to_wmdr10_json.py
```

or explicitly:

```bash
python convert_wmdr10_xml_to_wmdr10_json.py \
  --config config.yaml \
  --source resources/wmdr10_xml_examples \
  --target resources/wmdr10_json_examples
```

This stage simplifies the XML representation while preserving relevant WMDR content. XML/GML bookkeeping identifiers are stripped from container and descriptive objects, but source identifiers are preserved for referenceable WMDR entities such as deployments, contacts, equipment, or instruments when present.

### 2. Convert WMDR10 JSON to WMDR2 JSON

```bash
python convert_wmdr10_json_to_wmdr2_json.py
```

or explicitly:

```bash
python convert_wmdr10_json_to_wmdr2_json.py \
  --config config.yaml \
  --source resources/wmdr10_json_examples \
  --target results/wmdr2_json_examples
```

This stage writes one `.json` WMDR2 full record per facility.

## WMDR2 full-record structure

Each WMDR2 output file is a facility-centric JSON `Feature`.

```json
{
  "type": "Feature",
  "id": "facility:0-20000-0-06725",
  "geometry": {
    "type": "Point",
    "coordinates": [7.8232, 46.4204, 1540]
  },
  "temporalGeometry": {
    "coordinates": [
      [7.823197, 46.420453, 1538],
      [7.8232, 46.4204, 1540]
    ],
    "dates": ["2000-08-17", "2024-01-17"]
  },
  "time": {
    "interval": ["2000-08-17", "2025-05-28"]
  },
  "conformsTo": [
    "http://www.opengis.net/spec/ogcapi-records-1/1.0/conf/record-core",
    "https://schemas.wmo.int/wmdr/2.0/core/full-record"
  ],
  "properties": {
    "type": "facility",
    "title": "Blatten",
    "created": "2025-07-29T00:00:00Z",
    "updated": "2025-07-29T00:00:00Z",
    "facilitySets": ["facilitySet:gaw"],
    "keywords": ["0-20000-0-06725", "Blatten"],
    "observations": [],
    "deployments": [],
    "instruments": []
  }
}
```

### Root members

- `type`: always `Feature`.
- `id`: facility identifier, usually based on the WIGOS station identifier.
- `geometry`: current GeoJSON point geometry, derived from the most recent known coordinates.
- `temporalGeometry`: optional WMDR2 coordinate history. It remains the only temporal object that uses aligned `coordinates` and `dates` arrays.
- `time`: facility lifecycle interval. This uses date resolution only. Unknown bounds are represented with `..`.
- `conformsTo`: declares OGC Records core and the WMDR2 full-record profile.
- `properties`: contains the facility, observation, deployment, instrument, schedule, and facility-set references.

`externalIds` is not emitted, because it only repeats the feature `id`.

## Temporal-history convention

`temporalGeometry` is special and remains an aligned-array object:

```json
"temporalGeometry": {
  "coordinates": [
    [7.823197, 46.420453, 1538],
    [7.8232, 46.4204, 1540]
  ],
  "dates": ["2000-08-17", "2024-01-17"]
}
```

All other `temporal*` histories use arrays of dated objects. This avoids parallel-array alignment errors and keeps each historical assertion self-contained.

```json
"temporalProgramAffiliation": [
  {
    "programAffiliation": "GOSGeneral",
    "reportingStatus": "operational",
    "programSpecificFacilityId": "GOS-06725",
    "programSpecificFacilityTitle": "Blatten GOS facility",
    "date": "2000-08-17"
  },
  {
    "programAffiliation": "GBON",
    "reportingStatus": "operational",
    "date": "2022-09-08"
  }
]
```

Examples of the same convention include:

```json
"temporalTerritory": [
  {"territory": "CHE", "date": "2000-08-17"}
]
```

```json
"deployments": [
  {
    "id": "deployment:abc123",
    "temporalObservingSchedule": [
      {"observingSchedule": "schedule_daily_12", "date": "2025-01-01"}
    ]
  }
]
```

## Environment

Environmental histories are grouped under `properties.environment`. `temporalTopographyBathymetry` is not emitted. Its former sub-elements are promoted to first-level environment temporal histories.

```json
"environment": {
  "temporalClimateZone": [
    {"climateZone": "Cfb", "date": "1980-01-01"}
  ],
  "temporalSurfaceCover": [
    {"surfaceCover": "urbanBuiltup", "date": "1981-01-01"}
  ],
  "temporalPopulationDensities": [
    {"populationDensity": [100.0, 200.0], "date": "1990-01-01"}
  ],
  "temporalSurfaceRoughness": [
    {"surfaceRoughness": "rough", "date": "1991-01-01"}
  ],
  "temporalLocalTopography": [
    {"localTopography": "flat", "date": "1970-01-01"}
  ],
  "temporalRelativeElevation": [
    {"relativeElevation": "hilltop", "date": "1970-01-01"}
  ],
  "temporalTopographicContext": [
    {"topographicContext": "valley", "date": "1970-01-01"}
  ],
  "temporalAltitudeOrDepth": [
    {"altitudeOrDepth": 1540, "date": "1970-01-01"}
  ]
}
```

## Facility sets

A facility record references facility sets with `facilitySets`:

```json
"facilitySets": ["facilitySet:gaw"]
```

Facility-set catalogue entries are validated separately by `schemas/wmdr2-facility-sets.schema.json`:

```json
{
  "facilitySets": [
    {
      "id": "facilitySet:gaw",
      "title": "GAW",
      "description": "Global Atmosphere Watch facilities."
    }
  ]
}
```

The singular `facilitySet` property is obsolete.

## Code-list values

The WMDR2 JSON output stores compact values, not full code-list URLs.

```json
"observedVariable": 12006,
"observedDomain": "atmosphere",
"facilityType": "landFixed",
"wmoRegion": "europe"
```

Validation against WMO code lists is expected to be handled by a validator that knows which code list applies to each property.

## Observations, deployments, instruments, and schedules

### Observations

Observations contain observation-specific metadata and references to deployments.

```json
{
  "id": "observation:12006",
  "title": "domain: atmosphere; geometry: point; variable: 12006 Horizontal wind speed at specified distance from reference surface",
  "observedVariable": 12006,
  "observedDomain": "atmosphere",
  "observedGeometryType": "point",
  "programAffiliations": ["GAWregional"],
  "deployments": ["deployment:abc123"]
}
```

Observation-level program affiliation is intentionally non-temporal and plural: use `programAffiliations: ["GAWregional"]`. Do not use the old singular temporal-object form:

```json
"programAffiliation": [
  {"programAffiliation": "GAWregional", "date": ".."}
]
```

Facility-level program affiliation remains temporal under `properties.temporalProgramAffiliation`, because it can carry `reportingStatus`, `programSpecificFacilityId`, and `programSpecificFacilityTitle`.

Observation reporting uses aligned arrays. Reporting information is sourced from the WMDR1 `dataGeneration.reporting` block and belongs to the observation, not to the deployment schedule:

```json
"reporting": {
  "internationalExchange": [false],
  "temporalAggregate": ["P1M"],
  "uom": ["DU"],
  "dataPolicy": [
    {
      "dataPolicy": "noLimitation",
      "attribution": {
        "originator": {
          "role": null
        }
      }
    }
  ],
  "levelOfData": ["level1"],
  "temporalTimeliness": [
    {"timeliness": "PT30M", "date": "1982-03-13"}
  ]
}
```

### Observing schedules

Schedules are first-class reusable objects in the WMDR2 full-record model. They are stored under `properties.schedules` as JSCalendar / RFC 8984 `Event` objects with a small WMDR2 extension profile. Observations do not embed schedule objects directly.

The schedule applicability history belongs under `deployments[].temporalObservingSchedule`, because the deployment is the atomic data-collection unit. Each deployment can use a different schedule, or several deployments can reuse the same schedule `uid`.

```json
"schedules": [
  {
    "@type": "Event",
    "uid": "schedule_df3ec3dc94b9",
    "start": "0001-01-01T00:00:00",
    "timeZone": "UTC",
    "duration": "P1D",
    "recurrenceRules": [
      {"@type": "RecurrenceRule", "frequency": "daily"}
    ],
    "wmo.int:aggregation": {
      "temporalAggregate": "P1M",
      "diurnalBaseTime": "00:00:00"
    }
  }
],
"deployments": [
  {
    "id": "deployment:abc123",
    "temporalObservingSchedule": [
      {"observingSchedule": "schedule_df3ec3dc94b9", "date": "1982-03-13"}
    ]
  }
]
```

### Deployments

Deployments are referenceable objects. Their `id` is preserved from the WMDR1 XML source when the source provides a useful deployment identifier.

```json
{
  "id": "deployment:abc123",
  "observingMethod": "automaticWeatherStation",
  "localReferenceSurface": "localGround",
  "verticalDistanceFromReferenceSurface": 2.0,
  "instrument": ["instrument:def456"],
  "serialNumbers": {
    "serialNumber": ["S123"],
    "dates": ["2020-01-01"]
  },
  "temporalObservingSchedule": [
    {"observingSchedule": "schedule_daily_12", "date": "2025-01-01"}
  ]
}
```

Deployment records do not carry `title`, `type`, `manufacturer`, or `model` properties.

### Instruments

Instruments are reusable catalogue objects. Manufacturer and model are stored here, while serial-number histories remain with deployments. Optional `title`, `description`, and `verticalRange` properties are part of the schema, but are only emitted when suitable source values are available; WMDR 1.0 records often do not provide all of them. `verticalRange` is a WMDR2 object with numeric `min` and `max` limits.

```json
{
  "id": "instrument:def456",
  "title": "Weather transmitter",
  "description": "Automatic weather instrument.",
  "manufacturer": "Vaisala",
  "model": "WXT536",
  "verticalRange": {
    "min": 0,
    "max": 30
  }
}
```

## Contacts and roles

Contacts are stored in `properties.contacts`. A contact may include an `id`, `organization`, `name`, `position`, `emails`, `phones`, `links`, and `roles`.

```json
{
  "id": "contact:owner:rmi",
  "organization": "Royal Meteorological Institute of Belgium",
  "roles": ["owner"]
}
```

Role values should be specific role codes, not URLs to a generic role code list.

## Keywords and themes

`keywords` are retained as lightweight discovery text. `themes` are not emitted in the current WMDR2 core representation. Controlled-vocabulary values are represented as explicit WMDR2 properties instead.

## Schemas and tests

The active schema files should live under `schemas/`:

```text
schemas/
  wmdr2-common.schema.json
  wmdr2-record-feature.schema.json
  wmdr2-facility-sets.schema.json
```

Run the schema tests with:

```bash
pytest -q tests/test_wmdr2_schemas.py
```

Run all tests with:

```bash
pytest -q
```

## Repository cleanup notes

The current WMDR2 workflow no longer uses the previous Records Part 1 GeoJSON conversion path. The following root-level files can be removed if they are not referenced by local branches or pending work:

- `convert_wmdr10_json_to_records_part1.py`
- `settings.geojson`
- `version.geojson`
- root-level `wmdr2-common.schema.json`
- root-level `wmdr2-feature-collection.schema.json`
- root-level `wmdr2-record-feature.schema.json`

Do not remove the active schema files under `schemas/`.
