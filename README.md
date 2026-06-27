# WMDR2 development v0.2.4.2

This repository contains experimental tooling for transforming legacy WMDR 1.0 XML records into a simplified WMDR10 JSON representation and then into the draft WMDR2 v0.2.4.2 JSON representation.

The WMDR2 output is a facility-centric full record encoded as a GeoJSON-like `Feature`. WMDR2-specific content is stored in `properties`. Output files use the `.json` extension.

## References

- [OGC API - Records - Part 1: Core](https://docs.ogc.org/is/20-004r1/20-004r1.html)
- [WMO-No. 1192 WIGOS Metadata Standard](https://library.wmo.int/documents/wmo-1192)
- [WMDR 1.0 schemas](https://schemas.wmo.int/wmdr/1.0/)
- [WMDR2 draft UML aligned to OMS](https://wmo-im.github.io/wmdr2-devt/oms/html/)

## Conversion workflow

The workflow has two active stages.

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

This stage writes one WMDR2 full-record JSON file per facility. The converter accepts `observationSeries` input and also accepts older intermediate inputs that still use `observations`; output is always written as `properties.observationSeries`.

## Full-record structure

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
    "type": "MovingPoint",
    "coordinates": [
      [7.823197, 46.420453, 1538],
      [7.8232, 46.4204, 1540]
    ],
    "dates": ["2000-08-17", "2024-01-17"],
    "methods": [[], ["gps"]]
  },
  "time": {
    "interval": ["2000-08-17", "2025-05-28"]
  },
  "conformsTo": [
    "http://wigos.wmo.int/spec/wmdr/2/conf/core"
  ],
  "properties": {
    "type": "facility",
    "title": "Blatten",
    "created": "2025-07-29T00:00:00Z",
    "updated": "2025-07-29T00:00:00Z",
    "facilitySets": ["facilitySet:gaw"],
    "keywords": ["0-20000-0-06725", "Blatten"],
    "programAffiliation": [],
    "territory": [],
    "environment": [],
    "observationSeries": [],
    "deployments": [],
    "reporting": [],
    "schedules": [],
    "instruments": []
  }
}
```

### Root members

- `type`: always `Feature`.
- `id`: facility identifier, usually based on the WIGOS station identifier.
- `geometry`: current GeoJSON point geometry, derived from the most recent known coordinates.
- `temporalGeometry`: optional WMDR2 `MovingPoint` coordinate history. It is the only temporal object that uses aligned `coordinates`, `dates`, and optional `methods` arrays.
- `time`: facility lifecycle interval using date resolution only. Unknown bounds are represented with `..`.
- `conformsTo`: declares the WMDR2 core conformance class. The only allowed value is `http://wigos.wmo.int/spec/wmdr/2/conf/core`.
- `properties`: contains facility-level metadata and reusable registries for observation series, deployments, reporting definitions, schedules, instruments, contacts, environment, territory, and programme affiliation.

`externalIds` is not emitted because it repeats the feature `id`.

## Time-varying object convention

`temporalGeometry` is the special trajectory object and remains an aligned-array `MovingPoint`.

```json
"temporalGeometry": {
  "type": "MovingPoint",
  "coordinates": [
    [7.823197, 46.420453, 1538],
    [7.8232, 46.4204, 1540]
  ],
  "dates": ["2000-08-17", "2024-01-17"],
  "methods": [[], ["gps"]]
}
```

Other time-varying concepts use arrays of dated objects. Examples include `environment`, `programAffiliation`, `territory`, `deployments`, `reporting`, `observingProcedures`, and `officialStatus`.

## Facility-level properties

### Facility sets

A facility record references one or more logical facility groupings with `facilitySets`.

```json
"facilitySets": ["facilitySet:oscar-station-cluster-5"]
```

The referenced facility-set records are kept in a separate facility-set catalogue and are validated by `schemas/wmdr2-facility-sets.schema.json`. A facility set is useful when several facilities need to be treated as one logical entity, for example when a dataset or observing record spans nearby sites or when downstream discovery metadata should point to a group rather than enumerate each facility separately.

A real example is OSCAR/Surface station-cluster report 5, which groups facilities in the Arosa and Davos area. In WMDR2 this can be represented as a facility set to highlight the continuity of the total column ozone observations that started in Arosa and are being continued in Davos.

```json
{
  "facilitySets": [
    {
      "id": "facilitySet:oscar-station-cluster-5",
      "title": "Arosa/Davos total column ozone facilities",
      "description": "Grouping of facilities in either the Arosa or Davos area, and to highlight the link between the total column ozone observations started in Arosa and being continued in Davos.",
      "links": [
        {
          "rel": "canonical",
          "type": "text/html",
          "title": "OSCAR/Surface station cluster report 5",
          "href": "https://oscar.wmo.int/surface/#/search/stationClusterReport/5"
        }
      ]
    }
  ]
}
```

The facility-set identifier is the value used by facility records. The external OSCAR/Surface URL remains a link on the facility-set catalogue entry. The singular `facilitySet` property is obsolete.

### Programme affiliation

Facility-level programme affiliation is represented by `properties.programAffiliation`, an array of dated objects.

```json
"programAffiliation": [
  {
    "date": "2000-08-17",
    "programAffiliation": "GOSGeneral",
    "reportingStatus": "operational",
    "programSpecificFacilityId": "GOS-06725"
  },
  {
    "date": "2022-09-08",
    "programAffiliation": "GBON",
    "reportingStatus": "operational"
  }
]
```

`programSpecificFacilityId` is retained when present.

### Territory

Facility territory history is represented by `properties.territory`.

```json
"territory": [
  {
    "date": "2000-08-17",
    "territory": "CHE"
  }
]
```

### Environment

Facility environmental context is represented by `properties.environment`, an array of dated `Environment` objects.

```json
"environment": [
  {
    "date": "1980-01-01",
    "climateZone": "Cfb",
    "surfaceCover": "urbanBuiltup"
  },
  {
    "date": "1990-01-01",
    "population": [1000.0, null],
    "perimeter_km": [10.0, 50.0]
  },
  {
    "date": "1991-01-01",
    "surfaceRoughness": "rough"
  },
  {
    "date": "..",
    "topographyBathymetry": {
      "localTopography": "flat",
      "relativeElevation": "middle",
      "topographicContext": "rises",
      "altitudeOrDepth": "veryHighAltitude"
    }
  }
]
```

Obsolete names such as `historicalEnvironment`, `temporalPopulation`, `temporalPopulationDensities`, and `temporalTopographyBathymetry` are not emitted.

## Observation series

Observation-series records are stored under `properties.observationSeries`.

```json
{
  "id": "observationSeries:12006",
  "title": "domain: atmosphere; geometry: point; variable: 12006",
  "time": {"interval": ["2016-04-29", ".."]},
  "observedProperty": 12006,
  "observedGeometry": "point",
  "observedFeature": {
    "domain": "atmosphere",
    "domainFeature": "near-surface-air",
    "featureName": "2 m air"
  },
  "applicationArea": ["weatherForecasting"],
  "programAffiliation": ["GAWregional"],
  "sourceOfObservation": "automaticReading",
  "referenceSurface": "localGround",
  "representativeness": "local",
  "verticalDistanceFromReferenceSurface": {
    "value": 2.0,
    "uom": "m"
  },
  "observingMethods": [
    {"date": "1980-01-01", "observingMethod": 266},
    {"date": "2001-01-01", "observingMethod": 267}
  ],
  "officialStatus": [
    {
      "date": "2020-01-01",
      "officialStatus": "primary"
    }
  ],
  "deployments": ["deployment:dep-1"],
  "reporting": [
    {
      "date": "2020-01-01",
      "strategy": "unknown",
      "reporting": "reporting:hourly-level1-open",
      "uom": "K"
    }
  ],
  "observingProcedures": [
    {
      "date": "2020-01-01",
      "strategy": "continuous",
      "observingSchedules": ["schedule_8fd3e0f1094a"]
    }
  ]
}
```

Important conventions:

- The identifier prefix is `observationSeries:`.
- `observedFeature.domain` is used for the broad domain such as `atmosphere`, `terrestrial`, `ocean`, or `earth`.
- `observedFeature.domainFeature` and `observedFeature.featureName` are optional enrichment fields.
- Observation-series programme affiliation uses `programAffiliation`, not `programAffiliations`.
- `officialStatus` is an array of dated objects. WMDR10 boolean `officialStatus` values map to `primary` for `true` and `additional` for `false`.
- `deployments` is an array of references to `properties.deployments[*].id`.
- `reporting` is an array of dated `ReportingProcedure` objects that reference reusable reporting definitions.
- `observingMethods` is the dated observing-method history for the observation series. Each entry has `date` and mandatory `observingMethod`; use a nil-reason object when the method is unknown.
- `observingProcedures` is an array of dated `ObservingProcedure` objects. Each observing procedure carries a `strategy` and references one or more reusable schedules through `observingSchedules`.

Obsolete observation-series names such as `observations`, `observedVariable`, `observedDomain`, `domain`, `domainName`, `historicalDeployments`, `historicalReporting`, `historicalOfficialStatus`, `observingSchedules`, and `programAffiliations` are not emitted.

## Reporting

Reporting definitions are reusable facility-level objects stored under `properties.reporting`.

```json
"reporting": [
  {
    "id": "reporting:hourly-level1-open",
    "internationalExchange": true,
    "temporalAggregate": "PT1H",
    "levelOfData": "level1",
    "dataPolicy": {
      "dataPolicy": "noLimitation",
      "attribution": {
        "originator": {
          "role": null
        }
      }
    },
    "timeliness": "PT30M"
  }
]
```

Observation-series reporting history is stored in `observationSeries[*].reporting` as dated `ReportingProcedure` objects.

```json
"reporting": [
  {
    "date": "2020-01-01",
    "strategy": "unknown",
    "reporting": "reporting:hourly-level1-open",
    "uom": "K"
  }
]
```

This split allows one reporting definition to be reused by multiple observation series. Observation-series-specific values such as `uom` and `links` remain on the dated `ReportingProcedure` object. The v0.2.4.2 model requires a reporting procedure `strategy`; the converter uses a source strategy when one is available and otherwise emits `unknown`.

## Deployments

Deployments are reusable, dated instrument-instance/state objects stored under `properties.deployments`.

```json
"deployments": [
  {
    "id": "deployment:dep-1",
    "date": "2020-01-01",
    "instrument": "instrument:aws-001-temperature-sensor",
    "serialNumber": "SN-001",
    "operatingStatus": "operational",
    "exposure": "good",
    "geometry": {
      "type": "Point",
      "coordinates": [7.0, 46.0, 502]
    }
  }
]
```

The same deployment may be referenced by several observation series.

```json
"observationSeries": [
  {
    "id": "observationSeries:12006",
    "deployments": ["deployment:dep-1"]
  },
  {
    "id": "observationSeries:12001",
    "deployments": ["deployment:dep-1"]
  }
]
```

Important conventions:

- `deployment.instrument` is a scalar reference to one instrument identifier, not a one-element array.
- `deployment.date` is required.
- `serialNumber`, `operatingStatus`, `exposure`, and `geometry` describe the dated deployment state.
- Deployment records do not carry `title`, `type`, `manufacturer`, or `model` properties.
- Obsolete names such as `temporalSerialNumber`, `serialNumbers`, `temporalOfficialStatus`, and `temporalGeometry` are not emitted on deployments.

## Instruments

Instruments are reusable catalogue objects stored under `properties.instruments`.

```json
"instruments": [
  {
    "id": "instrument:aws-001-temperature-sensor",
    "title": "Temperature sensor",
    "description": "Automatic air-temperature sensor.",
    "manufacturer": "Vaisala",
    "model": "HMP155",
    "observingMethods": [266],
    "verticalRange": {
      "min": 0,
      "max": 30
    },
    "observedProperty": [12006],
    "observedGeometry": "point"
  }
]
```

Manufacturer, model, optional title, optional description, optional vertical range, and optional instrument capability information belong on the instrument. Serial number belongs on the deployment. Instrument `observingMethods` is a compact list of method values only when the catalogue instrument type is known and the method capability is known:

```json
"observingMethods": [266, 267]
```

Do not create an instrument catalogue entry merely to carry an observing method. If make/model are unknown and the serial number is not documented, the observing method can be represented on the observation series alone.

## Observing methods

The authoritative observing-method information for users is the dated history on `ObservationSeries`:

```json
"observingMethods": [
  {"date": "1980-01-01", "observingMethod": 266},
  {"date": "2001-01-01", "observingMethod": 267}
]
```

This answers the important question of how the observation series was generated and when the method changed. In v0.2.4.2 the `observingMethod` value in each history entry is mandatory. When the method is unknown for a known period, use a nil-reason object:

```json
"observingMethods": [
  {"date": "1980-01-01", "observingMethod": {"nilReason": "unknown"}}
]
```

`ObservingMethod` is not carried by `Deployment`. A deployment is the act/state of placing an instrument instance and making it contribute to one or more observation series. The same deployment can support two observation series that use different methods for different observed properties. `ObservingProcedure` is also not used to carry the observing method; it is the dated procedure/strategy object that links an observation series to one or more observing schedules.

## Observing schedules and observing procedures

Schedules are reusable JSCalendar-like objects stored under `properties.schedules`. Observation series use them through dated `ObservingProcedure` objects under `observationSeries[*].observingProcedures`.

```json
"schedules": [
  {
    "@type": "Event",
    "uid": "schedule_8fd3e0f1094a",
    "start": "0001-01-01T00:00:00",
    "timeZone": "UTC",
    "duration": "PT1H",
    "recurrenceRules": [
      {
        "@type": "RecurrenceRule",
        "frequency": "hourly"
      }
    ],
    "wmi.int:samplingFrequency": "PT1H",
    "wmo.int:aggregationInterval": "PT1H"
  }
]
```

The v0.2.4.2 XMI names the sampling extension `wmi.int:samplingFrequency`; the converter follows that spelling. Aggregation interval is emitted as `wmo.int:aggregationInterval`. The older nested `wmo.int:aggregation` object is not emitted.

```json
"observingProcedures": [
  {
    "date": "2020-01-01",
    "strategy": "continuous",
    "observingSchedules": ["schedule_8fd3e0f1094a"]
  }
]
```

`ObservingProcedure.strategy` is required by the model. The converter uses source `samplingStrategy` / `observingStrategy` when available and otherwise emits `unknown`.

## Contacts and catalogue representation

Contacts are stored in `properties.contacts`. A contact may include an `id`, `organization`, `name`, `position`, `emails`, `phones`, `links`, and `roles`.

```json
{
  "id": "contact:owner:rmi",
  "organization": "Royal Meteorological Institute of Belgium",
  "roles": ["owner"]
}
```

When catalogue post-processing is enabled, contacts and instruments can be externalized to catalogue files. The rewritten facility records retain minimal inline contact references and remove inline `properties.instruments`. Deployment `instrument` references remain scalar strings.

## Code-list values

The WMDR2 JSON output stores compact code-list values where possible, not full code-list URLs.

```json
{
  "observedProperty": 12006,
  "observedFeature": {"domain": "atmosphere"},
  "facilityType": "landFixed",
  "wmoRegion": "europe"
}
```

Validation against WMO code lists is expected to be handled by validators that know which code list applies to each property.

## Keywords and discovery policy

`keywords` are retained as lightweight discovery text only when configured. If the converter section has no `discovery` block, built-in defaults emit facility keywords from `identifier` and `name`, and deployment keywords from selected instrument/deployment fields.

As soon as a `discovery` block is present in `config.yaml`, it is authoritative: omitted buckets and empty lists suppress extraction.

```yaml
convert_wmdr10_json_to_wmdr2_json:
  source: resources/wmdr10_json_examples
  target: results/wmdr2_json_examples
  discovery:
    facility:
      keywords: []
      links: []
    observation:
      keywords: []
      links: []
    deployment:
      keywords: []
      links: []
```

To retain the default facility keywords explicitly, use:

```yaml
convert_wmdr10_json_to_wmdr2_json:
  discovery:
    facility:
      keywords: [identifier, name]
```

`themes` are intentionally not emitted in the current WMDR2 core representation.

## Active schema files

The active schema files live under `schemas/`.

```text
schemas/
  wmdr2-common.schema.json
  wmdr2-record-feature.schema.json
  wmdr2-facility-sets.schema.json
```

Run schema tests with:

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
