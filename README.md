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

Observation reporting uses aligned arrays:

```json
"reporting": {
  "internationalExchange": [true, false],
  "temporalReportingInterval": ["PT1H", "PT10M"],
  "uom": [null, "mm"]
}
```

Values at the same array index describe the same reporting configuration.

In the current conversion, `reporting.temporalReportingInterval` is copied from the legacy WMDR1 reporting block, not derived from observation or deployment validity periods. The effective source path is:

```text
WMDR1 XML
  observation / deployment / dataGeneration
    reporting
      internationalExchange
      temporalReportingInterval

WMDR10 JSON
  dataGeneration.reporting.temporalReportingInterval

WMDR2 JSON
  observations[].reporting.temporalReportingInterval[]
```

This is distinct from the observing schedule. WMDR1 did not clearly separate observing schedule and reporting schedule in all cases, so WMDR2 should make the observing schedule explicit.

For legacy WMDR1 inputs, the converter derives the observing schedule primarily from the `sampling` branch, not from the `reporting` branch. In the WMDR10 JSON this is typically found as:

```text
WMDR1 XML
  observation / deployment / dataGeneration
    sampling
      beginPosition
      startMonth / endMonth
      startWeekday / endWeekday
      startHour / endHour
      startMinute / endMinute
      samplingDuration / samplingInterval-like ISO durations

WMDR10 JSON
  dataGeneration.sampling

WMDR2 JSON
  properties.schedules[]
  properties.deployments[].temporalObservingSchedule
```

When two ISO 8601 durations are present in a sampling fragment and their semantic names are ambiguous, the converter interprets the shorter duration as the sampling duration (`Event.duration`) and the longer duration as the recurrence interval (`recurrenceRules`). For example, `PT5S` plus `PT1M` becomes a five-second sampling occurrence repeated every minute.

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

Values at the same index in `temporalObservingSchedule.observingSchedule` and `temporalObservingSchedule.dates` belong together. A repeated schedule reference with a later date can express the history of schedule applicability for that deployment, while a single schedule object can be shared by several deployments.

This separation is important. The JSCalendar `Event.start` is required and anchors recurrence expansion, but it is not used as WMDR2 schedule-validity metadata. WMDR2 uses a documented canonical anchor date, `0001-01-01`, for reusable schedule patterns. Deployment-specific validity remains in the aligned `dates` array.

Notes:

- Schedule entries use JSCalendar / RFC 8984 objects, but WMDR2 currently allows only `@type: "Event"` in `properties.schedules[]`. Other JSCalendar top-level object types such as `Task` and `Group` are outside the WMDR2 observing-schedule profile.
- Recurrence rules inside a schedule use `@type: "RecurrenceRule"`.
- JSCalendar uses `recurrenceRules`, plural, as an array.
- `start` is a local date-time and is mandatory for a JSCalendar `Event`. In WMDR2 schedule catalog entries, it is a canonical recurrence anchor, not the date on which the schedule became valid for a deployment.
- The canonical date part of reusable schedule anchors is `0001-01-01`. The time part still carries the schedule's local time-of-day, for example `0001-01-01T12:00:00` for a daily schedule at 12:00.
- The real applicability date of a schedule for a deployment is expressed only by `deployments[].temporalObservingSchedule.dates`.
- Schedule `uid` values are generated from the normalized schedule pattern, excluding deployment-specific effective dates. This allows deployments with the same pattern but different validity dates to reuse one schedule object.
- JSCalendar `uid` values should use RFC 8984-safe identifier characters. The converter therefore uses values such as `schedule_daily_12` or `schedule_<hash>`, not colon-separated identifiers.
- `timeZone` should normally be an IANA time-zone identifier such as `UTC`, `Europe/Zurich`, or the station-local time zone. If omitted, JSCalendar treats the event as floating time.
- `duration` describes the duration of one scheduled observation occurrence. In WMDR2 this can be interpreted as the sampling duration, for example `PT10M` for a 10-minute integrated observation. Use `PT0S` for an instantaneous observation.
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

Deployment records do not carry `title`, `type` and `description` properties.

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

The active schema files live under `schemas/`:

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
