from __future__ import annotations

import copy

import convert_wmdr10_json_to_wmdr2_json as converter

OBS = 'http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/12006'
GEOM = 'http://codes.wmo.int/wmdr/Geometry/point'
PROGRAM = 'http://codes.wmo.int/wmdr/ProgramAffiliation/GAW'
METHOD = 'http://codes.wmo.int/wmdr/ObservingMethodAtmosphere/266'
SOURCE = 'http://codes.wmo.int/wmdr/SourceOfObservation/automaticReading'
STATUS = 'http://codes.wmo.int/wmdr/ReportingStatus/operational'


def base_payload():
    return {
        'header': {'dateStamp': '2020-01-02'},
        'facility': {
            'identifier': '0-20000-0-TEST',
            'name': ['Primary', 'Alias'],
            'geospatialLocation': '46 7 500',
            'beginPosition': '2020-01-01',
            'facilityType': 'http://codes.wmo.int/wmdr/FacilityType/landFixed',
            'wmoRegion': 'http://codes.wmo.int/wmdr/WMORegion/europe',
            'onlineResource': {'url': 'https://example.org/station'},
            'programAffiliation': [{
                'programAffiliation': [PROGRAM],
                'programSpecificFacilityId': 'JFJ',
                'programSpecificFacilityTitle': 'Jungfraujoch',
                'reportingStatus': {
                    'reportingStatus': STATUS,
                    'beginPosition': '2020-01-01',
                },
            }],
        },
        'observations': [{
            'observedProperty': OBS,
            'type': GEOM,
            'programAffiliation': [PROGRAM],
            'deployments': [{
                'id': 'dep-1',
                'beginPosition': '2020-01-01',
                'observingMethod': METHOD,
                'sourceOfObservation': SOURCE,
                'referenceSurface': 'http://codes.wmo.int/wmdr/ReferenceSurfaceType/localGround',
                'heightAboveLocalReferenceSurface': {'@uom': 'm', '#text': '2.0'},
                'manufacturer': 'Vaisala',
                'model': 'HMP155',
                'serialNumber': 'SN1',
            }],
        }],
    }


def test_new_official_names_and_concepts():
    source = base_payload()
    before = copy.deepcopy(source)
    record = converter.convert_record(source)
    assert source == before
    props = record['properties']
    assert 'observationSeries' not in props
    obs = props['observations'][0]
    assert obs['observedProperty'] == {'id': OBS}
    assert obs['observedGeometry'] == {'id': GEOM}
    assert 'observingConfigurations' not in obs
    cfg = obs['configurations'][0]
    assert cfg['id'] == 'dep-1'
    assert cfg['observingMethod'] == {'id': METHOD}
    assert cfg['sourceOfObservation'] == {'id': SOURCE}
    assert cfg['instrumentSerialNumber'] == 'SN1'
    assert 'serialNumber' not in cfg
    assert props['wmoRegion'] == {
        'id': 'https://codes.wmo.int/wmdr/WMORegion/europe'
    }


def test_programme_mapping_and_facility_identity():
    record = converter.convert_record(base_payload())
    props = record['properties']
    assert props['externalIds'] == [{'scheme': 'GAW', 'value': 'JFJ'}]
    assert props['additionalTitles'] == ['Alias', 'Jungfraujoch']
    assert 'programAffiliations' not in props
    aff = props['observations'][0]['programAffiliations'][0]
    assert aff['programAffiliation'] == {'id': PROGRAM}
    assert aff['reportingStatus'] == {'id': STATUS}
    assert aff['dates'] == ['2020-01-01', '..']


def test_reusable_instrument_registry_not_duplicated_in_configuration():
    record = converter.convert_record(base_payload())
    props = record['properties']
    cfg = props['observations'][0]['configurations'][0]
    assert cfg['instrument'] == 'vaisala-hmp155'
    assert props['instruments'] == [{
        'id': 'vaisala-hmp155',
        'manufacturer': 'Vaisala',
        'model': 'HMP155',
        'observingMethods': [{'id': METHOD}],
    }]
    assert 'manufacturer' not in cfg
    assert 'model' not in cfg


def test_vertical_distance_uses_official_shape_and_full_unit_uri():
    record = converter.convert_record(base_payload())
    vertical = record['properties']['observations'][0]['configurations'][0]['verticalDistance']
    assert vertical == {
        'distances': [2.0],
        'unit': {'id': 'http://codes.wmo.int/wmdr/unit/m'},
        'referenceSurface': {'id': 'http://codes.wmo.int/wmdr/ReferenceSurfaceType/localGround'},
    }


def test_facility_links_are_feature_level():
    record = converter.convert_record(base_payload())
    assert record['links'][0]['href'] == 'https://example.org/station'
    assert 'links' not in record['properties']


def test_territory_uses_plural_and_dates():
    payload = base_payload()
    payload['facility']['territory'] = {
        'territoryName': 'http://codes.wmo.int/wmdr/TerritoryName/CHE',
        'beginPosition': '2020-01-01',
    }
    record = converter.convert_record(payload)
    assert record['properties']['territories'] == [{
        'territory': {'id': 'http://codes.wmo.int/wmdr/TerritoryName/CHE'},
        'dates': ['2020-01-01', '..'],
    }]


def test_programme_dates_not_invented_when_absent():
    payload = base_payload()
    payload['facility']['programAffiliation'][0].pop('reportingStatus')
    record = converter.convert_record(payload)
    affiliation = record['properties']['observations'][0]['programAffiliations'][0]
    assert affiliation == {'programAffiliation': {'id': PROGRAM}}


def test_status_history_splits_configuration_and_keeps_unique_ids():
    payload = base_payload()
    dep = payload['observations'][0]['deployments'][0]
    dep['instrumentOperatingStatus'] = [
        {'instrumentOperatingStatus': 'http://codes.wmo.int/wmdr/InstrumentOperatingStatus/operational', 'beginPosition': '2020-01-01', 'endPosition': '2020-12-31'},
        {'instrumentOperatingStatus': 'http://codes.wmo.int/wmdr/InstrumentOperatingStatus/inactive', 'beginPosition': '2021-01-01'},
    ]
    configs = converter.convert_record(payload)['properties']['observations'][0]['configurations']
    assert [c['id'] for c in configs] == ['dep-1-1', 'dep-1-2']
    assert [c['operatingStatus']['id'].rsplit('/',1)[-1] for c in configs] == ['operational', 'inactive']


def test_required_nillable_configuration_concept_maps_unknown_to_null():
    payload = base_payload()
    payload['observations'][0]['deployments'][0]['sourceOfObservation'] = 'unknown'
    cfg = converter.convert_record(payload)['properties']['observations'][0]['configurations'][0]
    assert 'sourceOfObservation' in cfg
    assert cfg['sourceOfObservation'] is None

