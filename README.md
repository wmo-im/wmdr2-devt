# WMDR2 development

This repository contains experimental tooling for transforming legacy WMDR 1.0 XML records into a simplified WMDR10 JSON representation and then into a draft WMDR2 JSON representation.

The current WMDR2 output is a **facility-centric full record** encoded as a GeoJSON-like `Feature` with WMDR2-specific content in `properties`. Output files use the `.json` extension.

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
- `temporalGeometry`: optional WMDR2 coordinate history. Coordinates and `dates` are aligned by array index. It deliberately does not include a `type` member such as `MovingPoint`, because it is not claiming conformance to OGC Moving Features.
- `time`: facility lifecycle interval. This uses date resolution only. Unknown bounds are represented with `..`.
- `conformsTo`: declares OGC Records core and the WMDR2 full-record profile.
- `properties`: contains the facility, observation, deployment, and instrument content.

## Temporal-history convention

WMDR2 temporal histories use aligned arrays. A value at index `i` becomes valid on `dates[i]` and remains valid until the next entry.

Example:

```json
"temporalProgramAffiliation": {
  "programAffiliation": ["GOSGeneral", "GOSGeneral"],
  "reportingStatus": ["operational", "closed"],
  "dates": ["2000-08-17", "2025-05-28"]
}
```

The same convention is used for items such as:

- `temporalGeometry.coordinates` / `temporalGeometry.dates`
- `temporalTerritory.territory` / `temporalTerritory.dates`
- `temporalClimateZone.climateZone` / `temporalClimateZone.dates`
- `temporalSurfaceCover.surfaceCover` / `temporalSurfaceCover.dates`
- `deployments[].serialNumbers.serialNumber` / `deployments[].serialNumbers.dates`

`temporalGeometry` is a WMDR2 history object, not an OGC Moving Features object. The current root `geometry` member remains the GeoJSON-compliant current point geometry.

## Code-list values

The WMDR2 JSON output stores compact values, not full code-list URLs.

Examples:

```json
"observedVariable": 12006,
"observedDomain": "atmosphere",
"facilityType": "landFixed",
"wmoRegion": "europe"
```

Validation against WMO code lists is expected to be handled by a validator that knows which code list applies to each property.

## Observations, deployments, and instruments

### Observations

Observations contain observation-specific metadata and references to deployments.

```json
{
  "id": "observation:12006",
  "title": "domain: atmosphere; geometry: point; variable: 12006 Horizontal wind speed at specified distance from reference surface",
  "observedVariable": 12006,
  "observedDomain": "atmosphere",
  "observedGeometryType": "point",
  "deployments": ["deployment:abc123"]
}
```

Observation reporting uses aligned arrays. Reporting information is sourced from the WMDR1 `dataGeneration.reporting` block and belongs to the observation, not to the deployment schedule:

```json
"reporting": {
  "internationalExchange": [false],
  "temporalReportingInterval": ["P1M"],
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
  "levelOfData": ["level1"]
}
```

Values at the same array index describe the same reporting configuration. The current converter preserves `dataPolicy.attribution` from WMDR1 reporting blocks, including explicit `null` values where the XML/WMDR10 JSON has no role value.

The effective source path is:

```text
WMDR1 XML
  observation / deployment / dataGeneration
    reporting
      internationalExchange
      temporalReportingInterval
      uom
      dataPolicy
      levelOfData

WMDR10 JSON
  dataGeneration.reporting

WMDR2 JSON
  observations[].reporting
```

The WMDR1 `dataGeneration.coverage` block is different. It describes when data are available for a deployment, assuming some regular sampling, processing, and reporting. It is therefore mapped to reusable schedule entries plus deployment-level schedule references:

```text
WMDR1 XML
  observation / deployment / dataGeneration
    sampling
    coverage
    reporting.temporalReportingInterval

WMDR10 JSON
  dataGeneration.sampling
  dataGeneration.coverage
  dataGeneration.reporting.temporalReportingInterval

WMDR2 JSON
  properties.schedules[]
  properties.deployments[].temporalObservingSchedule
```

`coverage.diurnalBaseTime` is preserved under `wmo.int:aggregation.diurnalBaseTime` because it is relevant for aggregate alignment, especially for 24-hour aggregates. The deployment-specific effective date remains under `deployments[].temporalObservingSchedule.dates`.

### Observing schedules

Schedules are first-class reusable objects in the WMDR2 full-record model. They are stored under `properties.schedules` as JSCalendar / RFC 8984 `Event` objects with a small WMDR2 extension profile. Observations do not embed schedule objects directly. The schedule applicability history belongs under `deployments[].temporalObservingSchedule`, because the deployment is the atomic data-collection unit. Each deployment can use a different schedule, or several deployments can reuse the same schedule `uid`.

For a WMDR10 JSON `dataGeneration` block like:

```json
{
  "sampling": null,
  "reporting": {
    "internationalExchange": "false",
    "uom": "http://codes.wmo.int/wmdr/unit/DU",
    "temporalReportingInterval": "P1M"
  },
  "beginPosition": "1982-03-13T00:00:00Z",
  "coverage": {
    "startMonth": "1",
    "endMonth": "12",
    "startWeekday": "1",
    "endWeekday": "7",
    "startHour": "0",
    "endHour": "23",
    "startMinute": "0",
    "endMinute": "59",
    "diurnalBaseTime": "00:00:00Z"
  }
}
```

the WMDR2 schedule representation is:

```json
"schedules": [
  {
    "@type": "Event",
    "uid": "schedule_df3ec3dc94b9",
    "start": "0001-01-01T00:00:00",
    "timeZone": "UTC",
    "duration": "P1D",
    "recurrenceRules": [
      {
        "@type": "RecurrenceRule",
        "frequency": "daily"
      }
    ],
    "wmo.int:sampling": null,
    "wmo.int:aggregation": {
      "temporalAggregate": "P1M",
      "diurnalBaseTime": "00:00:00"
    }
  }
],
"deployments": [
  {
    "id": "deployment:abc123",
    "temporalObservingSchedule": {
      "observingSchedule": ["schedule_df3ec3dc94b9"],
      "dates": ["1982-03-13"]
    }
  }
]
```

Values at the same index in `temporalObservingSchedule.observingSchedule` and `temporalObservingSchedule.dates` belong together. A repeated schedule reference with a later date can express the history of schedule applicability for that deployment, while a single schedule object can be shared by several deployments.

This separation is important. The JSCalendar `Event.start` is required and anchors recurrence expansion, but it is not used as WMDR2 schedule-validity metadata. WMDR2 uses a documented canonical anchor date, `0001-01-01`, for reusable schedule patterns. Deployment-specific validity remains in the aligned `dates` array.

Notes:

- Schedule entries use JSCalendar / RFC 8984 objects, but WMDR2 currently allows only `@type: "Event"` in `properties.schedules[]`. Other JSCalendar top-level object types such as `Task` and `Group` are outside the WMDR2 observing-schedule profile.
- Recurrence rules inside a schedule use `@type: "RecurrenceRule"`.
- JSCalendar uses `recurrenceRules`, plural, as an array.
- `start` is a local date-time and is mandatory for a JSCalendar `Event`. In WMDR2 schedule catalog entries, it is a canonical recurrence anchor, not the date on which the schedule became valid for a deployment.
- The canonical date part of reusable schedule anchors is `0001-01-01`. The time part comes from the coverage start hour/minute when available, otherwise midnight. It does not come from `diurnalBaseTime`.
- The real applicability date of a schedule for a deployment is expressed only by `deployments[].temporalObservingSchedule.dates`.
- `coverage.diurnalBaseTime` is stored as `wmo.int:aggregation.diurnalBaseTime` on the reusable schedule object, not under the deployment schedule reference. It is used for aggregate alignment, not for the JSCalendar occurrence start.
- `wmo.int:sampling` is derived from `dataGeneration.sampling`; when the source value is absent, it is represented as `null`.
- `wmo.int:aggregation.temporalAggregate` is derived from explicit aggregate metadata when available. For legacy WMDR1 records, the converter uses `reporting.temporalReportingInterval` as the default. `wmo.int:aggregation.diurnalBaseTime` may be present when the source provides a diurnal base time.
- `wmo.int:aggregation.spatialResolution` is optional and omitted when unavailable. When spatial resolution is known, prefer a numeric value in metres, following the DCAT `spatialResolutionInMeters` convention.
- `wmo.int:aggregation.statistics` is optional and currently schema-only; the XML/WMDR10 examples do not provide this information. Allowed values are `mean`, `median`, `min`, `max`, and `sum`.
- A full coverage window such as months 1..12, weekdays 1..7, hours 0..23, and minutes 0..59 is represented as a daily event with duration `P1D`. Restricted weekday/month coverage is represented using JSCalendar recurrence-rule constraints such as `byDay` or `byMonth`.
- Schedule `uid` values are generated from the normalized schedule pattern, excluding deployment-specific effective dates. This allows deployments with the same pattern but different validity dates to reuse one schedule object.
- JSCalendar `uid` values should use RFC 8984-safe identifier characters. The converter therefore uses values such as `schedule_<hash>`, not colon-separated identifiers.
- `timeZone` should normally be an IANA time-zone identifier such as `UTC`, `Europe/Zurich`, or the station-local time zone. If missing in WMDR1, the converter uses `UTC` as the default.
- A recurring schedule is open-ended when the recurrence rule has neither `until` nor `count`; the WMDR2 `dates` array describes when that schedule definition became applicable.
- Recurrence exceptions are represented with `recurrenceOverrides`; an excluded occurrence uses `{ "excluded": true }`. Null override values are not used.
- Recurrence override keys should match the occurrence local date-time, for example `2025-07-14T12:00:00`, not only the date.

#### Schedule profile constraints

The schema intentionally validates a small WMDR2 observing-schedule profile of JSCalendar rather than every possible JSCalendar object. In the current profile:

```text
properties.schedules[].@type = "Event"
properties.schedules[].recurrenceRules[].@type = "RecurrenceRule"
properties.schedules[].uid uses letters, digits, `_`, or `-`
properties.schedules[].recurrenceOverrides values are patch objects, not null
properties.schedules[] may include WMDR2 extension keys under wmo.int:*
deployments[].temporalObservingSchedule contains only observingSchedule and dates
```

### Observing schedules

Schedules are first-class reusable objects in the WMDR2 full-record model. They are stored under `properties.schedules` as JSCalendar / RFC 8984 `Event` objects. Observations do not embed schedule objects directly. The schedule applicability history belongs under `deployments[].temporalObservingSchedule`, because the deployment is the atomic data-collection unit. Each deployment can use a different schedule, or several deployments can reuse the same schedule `uid`.

```json
"schedules": [
  {
    "@type": "Event",
    "uid": "schedule_daily_12",
    "start": "0001-01-01T12:00:00",
    "timeZone": "UTC",
    "duration": "PT0S",
    "recurrenceRules": [
      {
        "@type": "RecurrenceRule",
        "frequency": "daily"
      }
    ],
    "recurrenceOverrides": {
      "2025-07-14T12:00:00": {
        "excluded": true
      }
    },
    "wmo.int:aggregation": {
      "diurnalBaseTime": "12:00:00"
    }
  }
],
"deployments": [
  {
    "id": "deployment:abc123",
    "temporalObservingSchedule": {
      "observingSchedule": ["schedule_daily_12"],
      "dates": ["2025-01-01"]
    }
  }
]
```

Values at the same index in `temporalObservingSchedule.observingSchedule` and `temporalObservingSchedule.dates` belong together. `diurnalBaseTime` is not part of the deployment schedule reference; when present, it belongs under the reusable schedule object at `wmo.int:aggregation.diurnalBaseTime`. A repeated schedule reference with a later date can express the history of schedule applicability for that deployment, while a single schedule object can be shared by several deployments.

This separation is important. The JSCalendar `Event.start` is required and anchors recurrence expansion, but it is not used as WMDR2 schedule-validity metadata. WMDR2 uses a documented canonical anchor date, `0001-01-01`, for reusable schedule patterns. Deployment-specific validity remains in the aligned `dates` array.

Notes:

- Schedule entries use JSCalendar / RFC 8984 objects, but WMDR2 currently allows only `@type: "Event"` in `properties.schedules[]`. Other JSCalendar top-level object types such as `Task` and `Group` are outside the WMDR2 observing-schedule profile.
- Recurrence rules inside a schedule use `@type: "RecurrenceRule"`.
- JSCalendar uses `recurrenceRules`, plural, as an array.
- `start` is a local date-time and is mandatory for a JSCalendar `Event`. In WMDR2 schedule catalog entries, it is a canonical recurrence anchor, not the date on which the schedule became valid for a deployment.
- The canonical date part of reusable schedule anchors is `0001-01-01`. The time part carries the start of one coverage/availability occurrence, for example `0001-01-01T09:00:00` for a window beginning at 09:00.
- The real applicability date of a schedule for a deployment is expressed only by `deployments[].temporalObservingSchedule.dates`.
- `coverage.diurnalBaseTime` is stored as `wmo.int:aggregation.diurnalBaseTime` on the reusable schedule object, not under the deployment schedule reference. It is used for aggregate alignment, not for the JSCalendar occurrence start.
- Schedule `uid` values are generated from the normalized schedule pattern, excluding deployment-specific effective dates. This allows deployments with the same pattern but different validity dates to reuse one schedule object.
- JSCalendar `uid` values should use RFC 8984-safe identifier characters. The converter therefore uses values such as `schedule_daily_12` or `schedule_<hash>`, not colon-separated identifiers.
- `timeZone` should normally be an IANA time-zone identifier such as `UTC`, `Europe/Zurich`, or the station-local time zone. If omitted, JSCalendar treats the event as floating time.
- `duration` describes the duration of one scheduled Event occurrence. For coverage-derived WMDR1 schedules it represents the covered/available window, for example `P1D` for all-day availability. For sampling-derived schedules it can represent sampling duration. `PT0S` is used only when no coverage or sampling duration can be inferred.
- A recurring schedule is open-ended when the recurrence rule has neither `until` nor `count`; the WMDR2 `dates` array describes when that schedule definition became applicable.
- Recurrence exceptions are represented with `recurrenceOverrides`; an excluded occurrence uses `{ "excluded": true }`. Null override values are not used.
- Recurrence override keys should match the occurrence local date-time, for example `2025-07-14T12:00:00`, not only the date.

#### Schedule profile constraints

The schema intentionally validates a small WMDR2 observing-schedule profile of JSCalendar rather than every possible JSCalendar object. In the current profile:

```text
properties.schedules[].@type = "Event"
properties.schedules[].recurrenceRules[].@type = "RecurrenceRule"
properties.schedules[].uid uses letters, digits, `_`, or `-`
properties.schedules[].recurrenceOverrides values are patch objects, not null
```

This keeps schedule objects reusable and predictable while remaining compatible with JSCalendar's event and recurrence-rule model.

### Deployments

Deployments are referenceable objects. Their `id` is preserved from the WMDR1 XML source when the source provides a useful deployment identifier.

```json
{
  "id": "deployment:abc123",
  "observingMethod": "automaticWeatherStation",
  "instrument": ["instrument:def456"],
  "serialNumbers": {
    "serialNumber": ["S123"],
    "dates": ["2020-01-01"]
  },
  "temporalObservingSchedule": {
    "observingSchedule": ["schedule_daily_12"],
    "dates": ["2025-01-01"]
  }
}
```

Deployment records do not carry `title` or `type` properties.

### Instruments

Instruments are reusable catalogue objects. Manufacturer and model are stored here, while serial-number histories remain with deployments.

```json
{
  "id": "instrument:def456",
  "manufacturer": "Vaisala",
  "model": "WXT536"
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

Role values should be **specific role codes**, not URLs to a generic role code list. For example:

```json
"roles": ["owner", "pointOfContact", "custodian"]
```

The converter drops generic ISO role-vocabulary references such as `CI_RoleCode` because they identify the vocabulary, not the actual role assignment.

The current role handling follows this policy:

- keep specific role values such as `owner`, `pointOfContact`, `custodian`, `originator`, `publisher`, or `distributor` when present;
- remove generic role-code-list references;
- do not emit `contactInstructions` in WMDR2 core records.

## Keywords and themes

`keywords` are retained as lightweight discovery text.

`themes` are not emitted in the current WMDR2 core representation. Controlled-vocabulary values are represented as explicit WMDR2 properties instead.

## Schemas and tests

The active schema files should live under `schemas/`:

```text
schemas/
  wmdr2-common.schema.json
  wmdr2-record-feature.schema.json
  wmdr2-feature-collection.schema.json
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
