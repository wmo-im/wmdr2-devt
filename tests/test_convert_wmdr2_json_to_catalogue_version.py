import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from convert_wmdr2_json_to_catalogue_version import CataloguePaths, contact_identifier, convert_to_catalogue_version


def test_contact_identifier_prefers_email():
    contact = {
        "name": "Jane Smith",
        "organization": "Example Org",
        "emails": ["Jane.Smith@Example.ORG"],
    }
    assert contact_identifier(contact) == "contact:jane.smith@example.org"


def test_contact_identifier_falls_back_to_name_and_organization():
    contact = {"name": "Jane Smith", "organization": "Example Org"}
    assert contact_identifier(contact) == "contact:jane-smith--example-org"


def test_externalize_contacts_and_instruments(tmp_path: Path):
    source = tmp_path / "source"
    records = tmp_path / "records"
    catalogues = tmp_path / "catalogues"
    source.mkdir()

    feature_a = {
        "type": "Feature",
        "id": "wsi:a",
        "properties": {
            "type": "facility",
            "contacts": [
                {
                    "name": "Jane Smith",
                    "organization": "Example Org",
                    "roles": ["pointOfContact"],
                    "emails": ["jane.smith@example.org"],
                    "phones": ["+41 1 234 56 78"],
                }
            ],
            "observationSeries": [{"uid": "observationSeries:a", "observingConfigurations": [{"validFrom": "2020-01-01", "instrument": "instrument:thermo-49i", "observingMethod": {"nilReason": "unknown"}}]}],
            "instruments": [{"uid": "instrument:thermo-49i", "manufacturer": "Thermo", "model": "49i"}],
        },
    }
    feature_b = {
        "type": "Feature",
        "id": "wsi:b",
        "properties": {
            "type": "facility",
            "contacts": [
                {
                    "name": "Jane Smith",
                    "organization": "Example Org",
                    "emails": ["JANE.SMITH@example.org"],
                },
                {"name": "No Email", "organization": "Example Org"},
            ],
            "observationSeries": [{"uid": "observationSeries:b", "observingConfigurations": [{"validFrom": "2020-01-01", "instrument": "instrument:thermo-49i", "observingMethod": {"nilReason": "unknown"}}]}],
            "instruments": [{"uid": "instrument:thermo-49i", "manufacturer": "Thermo", "model": "49i"}],
        },
    }
    (source / "a.json").write_text(json.dumps(feature_a), encoding="utf-8")
    (source / "b.json").write_text(json.dumps(feature_b), encoding="utf-8")

    convert_to_catalogue_version(
        CataloguePaths(
            source=source,
            records_path=records,
            contacts_path=catalogues / "contacts.json",
            instruments_path=catalogues / "instruments.json",
        )
    )

    contacts = json.loads((catalogues / "contacts.json").read_text(encoding="utf-8"))["contacts"]
    instruments = json.loads((catalogues / "instruments.json").read_text(encoding="utf-8"))["instruments"]
    rewritten = json.loads((records / "a.json").read_text(encoding="utf-8"))

    assert [c["uid"] for c in contacts] == [
        "contact:jane.smith@example.org",
        "contact:no-email--example-org",
    ]
    assert contacts[0]["phones"] == ["+41 1 234 56 78"]
    assert instruments == [{"uid": "instrument:thermo--49i", "manufacturer": "Thermo", "model": "49i"}]

    inline_contact = rewritten["properties"]["contacts"][0]
    assert inline_contact == {
        "uid": "contact:jane.smith@example.org",
        "name": "Jane Smith",
        "organization": "Example Org",
        "roles": ["pointOfContact"],
        "links": [
            {
                "rel": "about",
                "href": "../catalogues/contacts.json#contact:jane.smith@example.org",
                "type": "application/json",
            }
        ],
    }
    assert "phones" not in inline_contact
    assert "instruments" not in rewritten["properties"]
    assert rewritten["properties"]["observationSeries"][0]["observingConfigurations"][0]["instrument"] == "instrument:thermo--49i"
    assert "deployment" not in rewritten["properties"]["observationSeries"][0]["observingConfigurations"][0]



def test_externalize_preserves_temporal_geometry_methods_alignment(tmp_path: Path):
    source = tmp_path / "source"
    records = tmp_path / "records"
    catalogues = tmp_path / "catalogues"
    source.mkdir()

    feature = {
        "type": "Feature",
        "id": "wsi:a",
        "geometry": {"type": "Point", "coordinates": [7.0, 46.0, 100]},
        "temporalGeometry": {
            "type": "MovingPoint",
            "coordinates": [[6.0, 45.0, 99], [7.0, 46.0, 100]],
            "dates": ["2020-01-01", "2021-01-01"],
            "methods": [[], ["gps"]],
        },
        "time": {"interval": ["2020-01-01", ".."]},
        "conformsTo": ["http://wigos.wmo.int/spec/wmdr/2/conf/core"],
        "properties": {
            "type": "facility",
            "title": "A",
            "contacts": [
                {
                    "name": "Jane Smith",
                    "organization": "Example Org",
                    "emails": ["jane.smith@example.org"],
                }
            ],
            "observationSeries": [{"uid": "observationSeries:a", "observingConfigurations": [{"validFrom": "2020-01-01", "instrument": "instrument:thermo-49i", "observingMethod": {"nilReason": "unknown"}}]}],
            "instruments": [{"uid": "instrument:thermo-49i", "manufacturer": "Thermo", "model": "49i"}],
        },
    }
    (source / "a.json").write_text(json.dumps(feature), encoding="utf-8")

    convert_to_catalogue_version(
        CataloguePaths(
            source=source,
            records_path=records,
            contacts_path=catalogues / "contacts.json",
            instruments_path=catalogues / "instruments.json",
        )
    )

    rewritten = json.loads((records / "a.json").read_text(encoding="utf-8"))
    assert rewritten["temporalGeometry"] == feature["temporalGeometry"]


def test_externalize_preserves_single_position_temporal_geometry_with_methods(tmp_path: Path):
    source = tmp_path / "source"
    records = tmp_path / "records"
    catalogues = tmp_path / "catalogues"
    source.mkdir()

    feature = {
        "type": "Feature",
        "id": "wsi:a",
        "geometry": {"type": "Point", "coordinates": [35.3833, -0.714732, 2156]},
        "temporalGeometry": {
            "type": "MovingPoint",
            "coordinates": [[35.3833, -0.714732, 2156]],
            "dates": ["2020-09-21"],
            "methods": [["gps"]],
        },
        "time": {"interval": ["2020-09-21", ".."]},
        "conformsTo": ["http://wigos.wmo.int/spec/wmdr/2/conf/core"],
        "properties": {
            "type": "facility",
            "title": "A",
            "contacts": [{"name": "Jane Smith", "organization": "Example Org"}],
        },
    }
    (source / "a.json").write_text(json.dumps(feature), encoding="utf-8")

    convert_to_catalogue_version(
        CataloguePaths(
            source=source,
            records_path=records,
            contacts_path=catalogues / "contacts.json",
            instruments_path=catalogues / "instruments.json",
        )
    )

    rewritten = json.loads((records / "a.json").read_text(encoding="utf-8"))
    assert rewritten["temporalGeometry"] == feature["temporalGeometry"]


def test_catalogue_externalizer_does_not_catalogue_serial_number_only_instruments(tmp_path: Path):
    source = tmp_path / "source"
    records = tmp_path / "records"
    catalogues = tmp_path / "catalogues"
    source.mkdir()
    feature = {
        "type": "Feature",
        "id": "wsi:a",
        "properties": {
            "type": "facility",
            "observationSeries": [
                {
                    "uid": "observationSeries:a",
                    "observingConfigurations": [
                        {"validFrom": "2020-01-01", "instrument": "instrument:instance-a", "serialNumber": "SN-001", "observingMethod": {"nilReason": "unknown"}}
                    ],
                }
            ],
            "instruments": [{"uid": "instrument:instance-a", "serialNumber": "SN-001"}],
        },
    }
    (source / "a.json").write_text(json.dumps(feature), encoding="utf-8")

    convert_to_catalogue_version(
        CataloguePaths(
            source=source,
            records_path=records,
            contacts_path=catalogues / "contacts.json",
            instruments_path=catalogues / "instruments.json",
        )
    )

    instruments = json.loads((catalogues / "instruments.json").read_text(encoding="utf-8"))["instruments"]
    rewritten = json.loads((records / "a.json").read_text(encoding="utf-8"))
    assert instruments == []
    cfg = rewritten["properties"]["observationSeries"][0]["observingConfigurations"][0]
    assert cfg["serialNumber"] == "SN-001"
    assert "instrument" not in cfg


def test_externalize_drops_generic_iso_role_codelist_reference(tmp_path: Path):
    source = tmp_path / "source"
    records = tmp_path / "records"
    catalogues = tmp_path / "catalogues"
    source.mkdir()
    feature = {
        "type": "Feature",
        "id": "wsi:a",
        "properties": {
            "type": "facility",
            "contacts": [
                {
                    "organization": "Org",
                    "emails": ["oscar@wmo.int"],
                    "roles": ["gmxCodelists.xml#CI_RoleCode"],
                }
            ],
        },
    }
    (source / "a.json").write_text(json.dumps(feature), encoding="utf-8")

    convert_to_catalogue_version(
        CataloguePaths(
            source=source,
            records_path=records,
            contacts_path=catalogues / "contacts.json",
            instruments_path=catalogues / "instruments.json",
        )
    )

    contacts = json.loads((catalogues / "contacts.json").read_text(encoding="utf-8"))["contacts"]
    rewritten = json.loads((records / "a.json").read_text(encoding="utf-8"))
    assert contacts == [{"organization": "Org", "emails": ["oscar@wmo.int"], "uid": "contact:oscar@wmo.int"}]
    assert rewritten["properties"]["contacts"][0] == {
        "uid": "contact:oscar@wmo.int",
        "organization": "Org",
        "links": [{"rel": "about", "href": "../catalogues/contacts.json#contact:oscar@wmo.int", "type": "application/json"}],
    }
