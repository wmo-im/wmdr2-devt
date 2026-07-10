# WMDR2 development model v0.3.1

This repository contains the current development version of the simplified WMDR2 JSON representation, converter utilities, JSON Schemas, generated examples, and tests.  The format is intended to represent WIGOS station metadata in an OGC Records / GeoJSON-oriented structure while preserving the information that can be recovered from WMDR 1.0 source records without inventing missing metadata.

The current model version described here is **WMDR2 v0.3.1**.

## Design principles

The v0.3.1 model follows these principles.

1. A WMDR2 station record is a GeoJSON `Feature` whose root `id` is the primary WIGOS Station Identifier.
2. Facility names and identifiers are normalized to one primary value plus explicit additional values.
3. Time-varying properties use a `time` object with an interval; older source-specific temporal field names are not part of the public model.
4. The useful content of observing-location and instrument-placement history is represented directly in `observingConfigurations[]`.
5. Reusable contacts, instruments, and schedules are registries in the facility record and are referenced from the places where they are used.
6. The converter must preserve recorded information and must not fabricate validity dates, phone country codes, instrument serial numbers, observing methods, or programme affiliations.
7. Source examples that cannot validate without inventing information are explicitly commented in the end-to-end tests rather than being silently “fixed”.

## Record shape

A WMDR2 facility record has this top-level shape:

```json
{
  "id": "0-20008-0-THE",
  "conformsTo": ["http://wigos.wmo.int/spec/wmdr/2/conf/core"],
  "type": "Feature",
  "geometry": {
    "type": "Point",
    "coordinates": [22.957, 40.631, 60.0]
  },
  "temporalGeometry": {
    "type": "MovingPoint",
    "coordinates": [[22.957, 40.631, 60.0]],
    "dates": ["1982-03-13"],
    "methods": [[]]
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
| `temporalGeometry` | Optional movement or location-history extension. It uses aligned `coordinates`, `dates`, and optional `methods` arrays. |
| `time` | Overall validity or temporal extent of the record. |
| `properties` | Facility metadata and related WMDR metadata blocks. |
| `links` | OGC-style links about the record. |

## Facility properties

`properties.type` is always `facility`.  The facility object carries the primary description of the station and registries for reusable objects.

```json
{
  "type": "facility",
  "title": "Flüela permafrost",
  "additionalTitles": ["Flüelapass"],
  "additionalIds": ["0-756-1-387493"],
  "facilityType": "landFixed",
  "wmoRegion": "southWestPacific",
  "description": "Example station description.",
  "keywords": ["GCW", "permafrost"],
  "contacts": [],
  "contactAssignments": [],
  "instruments": [],
  "observationSeries": [],
  "schedules": []
}
```

### Facility names and identifiers

The converter applies deterministic primary/additional rules.

| Source concept | WMDR2 output |
| --- | --- |
| First recorded facility identifier | root `id` |
| Further recorded WSI values | `properties.additionalIds[]` |
| First recorded facility name | `properties.title` |
| Further recorded facility names | `properties.additionalTitles[]` |

`additionalIds[]` contains only values that match the WSI pattern:

```text
^(0|1|2|3)-([1-9]\d*)-([0-9]+)-([A-Za-z0-9._-]+)$
```

This rule avoids hiding alternate official station identifiers while keeping the root feature identifier single-valued.

## Time model

Temporal metadata is represented with OGC-style `time` objects.

```json
{
  "time": {
    "interval": ["2020-01-01", ".."],
    "resolution": "P1D"
  }
}
```

The interval is a two-element array.  Each endpoint is either a date-like value (`YYYY`, `YYYY-MM`, `YYYY-MM-DD`) or `..` for open or unknown.  `time.resolution`, where present, is an ISO 8601 duration such as `P1D`, `PT1H`, or `PT10M`.

The same structure is used for facility histories, territories, programme affiliations, observing configurations, observing procedures, and official status entries.  When a source record does not provide a required time anchor for a time-varying object, the converter should not invent one.  Such source-derived examples are treated as intentionally non-validating in the end-to-end test policy until the source metadata is corrected.

## Spatial model

The root `geometry` is the current or representative facility position.  The optional root `temporalGeometry` records location history:

```json
{
  "type": "MovingPoint",
  "coordinates": [
    [7.0, 46.0, 100.0],
    [7.1, 46.1, 101.0]
  ],
  "dates": ["2000-01-01", "2020-01-01"],
  "methods": [[], ["gps"]]
}
```

`coordinates`, `dates`, and `methods` are aligned by array index.  Empty method arrays are allowed when the source does not record the position method.

## Contacts

Reusable contacts are stored in `properties.contacts[]` using the OGC Records Contact model.  Contact roles in WMDR are contextual, so they are represented separately through `contactAssignments[]` at the facility or observation-series level.

```json
{
  "contacts": [
    {
      "identifier": "contact:met-service-example",
      "organization": "Example Meteorological Service",
      "emails": [{"value": "ops@example.org"}],
      "phones": [{"value": "+41123456789"}],
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

Phone values follow the strict OGC Contact schema used in this repository and must be E.164-style values when emitted.  The converter may normalize clearly international numbers, for example `00...` to `+...`, but it must not infer a country code for local-only source values.

## Environment, territory, and programme affiliations

Facility-level environmental and administrative histories are arrays of time-bound objects.

```json
{
  "environment": [
    {
      "time": {"interval": ["2020-01-01", ".."]},
      "climateZone": "temperate",
      "surfaceCover": "grass",
      "surfaceRoughness": "low",
      "population": [10000, 50000],
      "perimeter_km": [10, 50]
    }
  ],
  "territory": [
    {
      "time": {"interval": ["2020-01-01", ".."]},
      "territory": "CHE"
    }
  ],
  "programAffiliations": [
    {
      "time": {"interval": ["2020-01-01", ".."]},
      "program": "GAWregional",
      "programSpecificFacilityId": "GAW-TEST",
      "reportingStatus": "operational"
    }
  ]
}
```

Programme affiliations at facility level are temporal objects because the station relationship with a programme may change.  Observation-series programme memberships may be emitted as plain code values where the source only records membership without a temporal association.

## Instruments

`properties.instruments[]` is a reusable instrument-type registry, not a list of individual physical instances.  It may contain manufacturer, model, observing-method metadata, and vertical range where these are known.

```json
{
  "instruments": [
    {
      "id": "instrument:thermo--49i",
      "manufacturer": "Thermo",
      "model": "49i",
      "observingMethods": [266],
      "verticalRange": {
        "min": 0.0,
        "max": 30.0
      }
    }
  ]
}
```

Serial numbers are not part of the instrument catalogue because they identify individual items, not catalogue entries.  They are optional instance metadata on `observingConfigurations[]`.  An instrument can therefore be documented through the catalogue reference even when the serial number is unknown.  In that case, omit `serialNumber`; do not create a catalogue-specific instrument instance.

## Observation series

An observation series describes observations of one property or closely related property/feature/geometry combination at the facility.

```json
{
  "id": "observationSeries:0-20008-0-THE--12006",
  "title": "Air temperature",
  "observedProperty": 12006,
  "observedFeature": {
    "domain": "atmosphere",
    "domainFeature": "nearSurface",
    "featureName": "air"
  },
  "observedGeometry": "point",
  "applicationArea": "weather",
  "representativeness": "local",
  "programAffiliations": ["GBON"],
  "observingConfigurations": [],
  "observingProcedures": [],
  "reportingProcedures": [],
  "officialStatus": [],
  "contactAssignments": []
}
```

`observedFeature.domain` is required when `observedFeature` is present.  `domainFeature` and `featureName` allow more specific description where the source contains it.

## Observing configurations

`observingConfigurations[]` is the time-bound history of how and where an observation series is made.  It is the place for observing method, optional operating status, source of observation, instrument reference, optional serial number, exposure, local geometry, reference surface, and vertical distance.

```json
{
  "time": {"interval": ["2020-01-01", ".."]},
  "observingMethod": 266,
  "operatingStatus": "operational",
  "sourceOfObservation": "automaticReading",
  "instrument": "instrument:thermo--49i",
  "serialNumber": "SN-001",
  "exposure": "good",
  "geometry": {
    "type": "Point",
    "coordinates": [7.0, 46.0, 2.0]
  },
  "referenceSurface": "localGround",
  "verticalDistanceFromReferenceSurface": {
    "value": 2.0,
    "uom": "m"
  }
}
```

An observing configuration requires `observingMethod` and a `time` interval.  `operatingStatus` has cardinality 0..1 and is emitted only when recorded.  `serialNumber` also has cardinality 0..1 and is emitted only when the instrument instance serial number is known; a missing serial number does not prevent documenting the instrument via `instrument`.  Use `{"nilReason": "unknown"}` for an explicitly unknown method.  Do not emit discovery keywords or a nested location wrapper here; the relevant location fields are represented directly on the configuration.

## Observing procedures

`observingProcedures[]` contains time-bound procedure history for the observation series.  It references reusable schedules from `properties.schedules[]` through `observingSchedules[]`.

```json
{
  "time": {"interval": ["2020-01-01", ".."]},
  "strategy": "continuous",
  "observingSchedules": ["schedule_001"]
}
```

The actual validity of the procedure is carried by `time.interval`.  The schedule object itself describes a reusable temporal pattern.

## Reporting procedures

`reportingProcedures[]` contains the reporting procedure metadata for an observation series.  Reporting procedures are not time-bound objects in this version.  They reference reusable schedules through `reportingSchedules[]`.

```json
{
  "dataFormat": ["BUFR"],
  "dataPolicy": "open",
  "internationalExchange": true,
  "levelOfData": "level1",
  "numberOfObservationsInReportingInterval": 6,
  "referenceDatum": "meanSeaLevel",
  "referenceTimeSource": ["utc"],
  "spatialReportingInterval": "point",
  "strategy": "automatic",
  "timeliness": "PT30M",
  "timeStampMeaning": "endOfPeriod",
  "uom": "K",
  "reportingSchedules": ["schedule_002"],
  "contactAssignments": []
}
```

The reporting interval and aggregation interval belong in the associated schedule as `wmo.int:aggregationInterval`, not as temporal properties on the reporting procedure.

## Official status

`officialStatus[]` is a time-bound observation-series history.

```json
{
  "time": {"interval": ["2020-01-01", ".."]},
  "officialStatus": "primary"
}
```

When converting from a boolean official-status source value, the intended mapping is:

| Source value | WMDR2 value |
| --- | --- |
| `true` | `primary` |
| `false` | `additional` |
| absent | no `officialStatus` entry |

## Schedules

Reusable schedules are stored once in `properties.schedules[]` and referenced from observing and reporting procedures.

```json
{
  "uid": "schedule_001",
  "@type": "Event",
  "start": "0001-01-01T06:00:00",
  "duration": "PT12H",
  "recurrenceRules": [],
  "recurrenceOverrides": {},
  "timeZone": "UTC",
  "wmo.int:samplingFrequency": "PT10M",
  "wmo.int:aggregationInterval": "PT1H",
  "wmo.int:diurnalBaseTime": "06:00:00"
}
```

The schedule fields are intentionally JSCalendar-like, with WMO extension members for sampling, aggregation, and diurnal base time.

`duration` is reserved for a within-day coverage window.  When a source gives a daily window, the converter anchors `start` to the dummy date `0001-01-01T<time>`.  Real-world validity remains on the procedure through `time.interval`.

A single schedule may be referenced by both observing and reporting procedures when the normalized pattern is truly the same.  Distinct patterns must have distinct `uid` values.  No schedule-type discriminator is needed because the referencing property gives the context.

## Catalogues and derived views

The main facility record may be transformed into catalogue-oriented views.  In a catalogue view, reusable contacts and instruments can be externalized to separate catalogue files while the facility record keeps lightweight references.

Typical catalogue outputs are:

```text
results/wmdr2_json_examples/
results/wmdr2_json_examples/catalogues/contacts.json
results/wmdr2_json_examples/catalogues/instruments.json
```

Instrument catalogue entries remain type-level entries: manufacturer, model, observing methods, and similar metadata.  Instance-level information should not be introduced into the catalogue.

## Converter workflow

The main converter is:

```text
convert_wmdr10_json_to_wmdr2_json.py
```

It converts intermediate WMDR10 JSON into WMDR2 facility records.

### Configuration

From the repository root, run:

```bash
python convert_wmdr10_json_to_wmdr2_json.py
```

With no arguments, the converter discovers `config.yaml` or `config.yml`, reads the `convert_wmdr10_json_to_wmdr2_json` section, and uses configured source and target paths.

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
python convert_wmdr10_json_to_wmdr2_json.py \
  --source resources/wmdr10_json_examples \
  --target results/wmdr2_json_examples
```

`--source` is an alias for `--input`, and `--target` is an alias for `--output`.

### Output reporting

The converter reports written files on stdout, for example:

```text
wrote results/wmdr2_json_examples/20241211_0-20008-0-THE.json
wrote results/wmdr2_json_examples/catalogues/contacts.json
```

## Schemas

The primary validation schema is:

```text
schemas/wmdr2-record-feature.schema.json
```

This schema validates the full WMDR2 facility record as a GeoJSON Feature.  It uses JSON Schema draft 2020-12.

Other schemas in the repository may validate catalogue views or discovery profiles.  They are not substitutes for validating a full WMDR2 facility record.

## Test policy

The canonical test set verifies converter helpers, record conversion, CLI behaviour, schema validation, temporal geometry alignment, and XML-to-WMDR2 end-to-end conversion.

Recommended checks:

```bash
python -m py_compile convert_wmdr10_json_to_wmdr2_json.py
pytest -q
```

The end-to-end tests deliberately distinguish between:

1. source XML files that are marked invalid at XML level;
2. source-derived WMDR2 outputs that are expected to fail the strict WMDR2 JSON schema because required information is not recorded;
3. source-derived WMDR2 outputs that must validate.

A source record being marked invalid at XML level does not automatically mean its WMDR2 projection must fail schema validation.  Conversely, if a WMDR2 record lacks required metadata and the source does not provide it, the test should comment that example as non-validating instead of making the converter invent values.

## Current non-goals

The v0.3.1 development model deliberately does not try to solve the following by inference:

- deriving country codes for local phone numbers;
- constructing validity intervals where no time anchor is recorded;
- turning individual serial-numbered items into catalogue entries;
- guessing observing methods, exposure, or source of observation;
- forcing observing and reporting procedures to share schedules when their temporal patterns differ.

These constraints keep the converter faithful to the source metadata and keep schema validation meaningful.
