from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

MODULE_PATH = Path('convert_wmdr10_json_to_wmdr2_geojson.py')
spec = spec_from_file_location('conv_v7', MODULE_PATH)
module = module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset mutable module globals between tests."""
    old_labels = dict(module.CODE_LIST_LABELS)
    old_policy = {
        key: {bucket: list(values) for bucket, values in buckets.items()}
        for key, buckets in module.DISCOVERY_POLICY.items()
    }
    try:
        module.CODE_LIST_LABELS.clear()
        module.DISCOVERY_POLICY.clear()
        module.DISCOVERY_POLICY.update({
            key: {bucket: list(values) for bucket, values in buckets.items()}
            for key, buckets in module.DEFAULT_DISCOVERY_POLICY.items()
        })
        yield
    finally:
        module.CODE_LIST_LABELS.clear()
        module.CODE_LIST_LABELS.update(old_labels)
        module.DISCOVERY_POLICY.clear()
        module.DISCOVERY_POLICY.update(old_policy)


OBSERVED_179 = 'http://codes.wmo.int/wmdr/ObservedVariableAtmosphere/179'


def test_observation_title_uses_label_and_domain_when_labels_available():
    module.CODE_LIST_LABELS.update({'ObservedVariableAtmosphere': {'179': 'Cloud amount'}})

    title = module._format_observation_title(OBSERVED_179)

    assert title == 'variable 179: Cloud amount; domain: Atmosphere'


def test_observation_title_falls_back_to_code_and_domain_without_label():
    title = module._format_observation_title(OBSERVED_179)

    assert title == 'variable 179; domain: Atmosphere'


def test_observation_description_collapses_unknown_unknown():
    obs = {
        'observedProperty': OBSERVED_179,
        'type': 'point',
    }
    deployments = [
        {'manufacturer': '(unknown)', 'model': 'unknown', 'observingMethod': None},
    ]

    desc = module._observation_description(obs, deployments)

    assert desc == 'Observed property 179; geometry type point; deployment procedure unknown'


def test_observation_description_humanizes_observing_method():
    obs = {
        'observedProperty': OBSERVED_179,
        'type': 'point',
    }
    deployments = [
        {'observingMethod': 'instrumentAutomaticReading'},
    ]

    desc = module._observation_description(obs, deployments)

    assert desc == (
        'Observed property 179; geometry type point; '
        'deployment procedure instrument automatic reading'
    )


def test_temporal_observing_schedule_defaults_interval_unknown_when_only_id_present():
    data_generation = [
        {'@gml:id': 'dg-1'},
    ]

    schedule = module._normalize_temporal_observing_schedule(data_generation)

    assert schedule == [{'id': 'dg-1', 'interval': 'unknown'}]


def test_temporal_reporting_schedule_defaults_interval_unknown_when_only_id_and_reporting_present():
    data_generation = [
        {
            '@gml:id': 'dg-1',
            'reporting': {'internationalExchange': 'true'},
        },
    ]

    schedule = module._normalize_temporal_reporting_schedule(data_generation)

    assert schedule == [{'id': 'dg-1', 'interval': 'unknown'}]


def test_default_discovery_policy_matches_current_agreement():
    assert module.DEFAULT_DISCOVERY_POLICY['facility']['themes'][-2:] == [
        'programAffiliation',
        'reportingStatus',
    ]
    assert module.DEFAULT_DISCOVERY_POLICY['observation']['themes'] == ['programAffiliation']
