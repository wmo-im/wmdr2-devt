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
        "emails": [{"value": "Jane.Smith@Example.ORG"}],
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
        "id": "facility:a",
        "properties": {
            "type": "facility",
            "contacts": [
                {
                    "name": "Jane Smith",
                    "organization": "Example Org",
                    "roles": ["pointOfContact"],
                    "emails": [{"value": "jane.smith@example.org"}],
                    "phones": [{"value": "+41 1 234 56 78"}],
                }
            ],
            "deployments": [{"id": "deployment:a", "instrument": "instrument:thermo-49i"}],
            "observations": [{"id": "observations:a", "historicalDeployments": [{"date": "2020-01-01", "deployment": "deployment:a"}]}],
            "instruments": [{"id": "instrument:thermo-49i", "manufacturer": "Thermo", "model": "49i"}],
        },
    }
    feature_b = {
        "type": "Feature",
        "id": "facility:b",
        "properties": {
            "type": "facility",
            "contacts": [
                {
                    "name": "Jane Smith",
                    "organization": "Example Org",
                    "emails": [{"value": "JANE.SMITH@example.org"}],
                },
                {"name": "No Email", "organization": "Example Org"},
            ],
            "deployments": [{"id": "deployment:b", "instrument": "instrument:thermo-49i"}],
            "observations": [{"id": "observations:b", "historicalDeployments": [{"date": "2020-01-01", "deployment": "deployment:b"}]}],
            "instruments": [{"id": "instrument:thermo-49i", "manufacturer": "Thermo", "model": "49i"}],
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

    assert [c["identifier"] for c in contacts] == [
        "contact:jane.smith@example.org",
        "contact:no-email--example-org",
    ]
    assert contacts[0]["phones"] == [{"value": "+41 1 234 56 78"}]
    assert instruments == [{"id": "instrument:thermo-49i", "manufacturer": "Thermo", "model": "49i"}]

    inline_contact = rewritten["properties"]["contacts"][0]
    assert inline_contact == {
        "identifier": "contact:jane.smith@example.org",
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
    assert rewritten["properties"]["deployments"][0]["instrument"] == "instrument:thermo-49i"
    assert rewritten["properties"]["observations"][0]["historicalDeployments"][0]["deployment"] == "deployment:a"



def test_externalize_preserves_temporal_geometry_methods_alignment(tmp_path: Path):
    source = tmp_path / "source"
    records = tmp_path / "records"
    catalogues = tmp_path / "catalogues"
    source.mkdir()

    feature = {
        "type": "Feature",
        "id": "facility:a",
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
                    "emails": [{"value": "jane.smith@example.org"}],
                }
            ],
            "deployments": [{"id": "deployment:a", "instrument": "instrument:thermo-49i"}],
            "observations": [{"id": "observations:a", "historicalDeployments": [{"date": "2020-01-01", "deployment": "deployment:a"}]}],
            "instruments": [{"id": "instrument:thermo-49i", "manufacturer": "Thermo", "model": "49i"}],
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
        "id": "facility:a",
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
