#!/usr/bin/env python3
"""Generate an Enterprise Architect-oriented UML/XMI 2.1 model for the
WMDR2 schema at https://github.com/wmo-im/wmdr2/ on branch wigosbox-catalogue1.99-dev.

The model intentionally represents JSON Schema composition as UML classes,
attributes, associations, comments and constraints. It does not treat JSON
Schema allOf/oneOf as UML inheritance unless that is semantically warranted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable
from uuid import NAMESPACE_URL, uuid5

from lxml import etree

XMI_NS = "http://schema.omg.org/spec/XMI/2.1"
UML_NS = "http://schema.omg.org/spec/UML/2.1"
NSMAP = {"xmi": XMI_NS, "uml": UML_NS}

SOURCE_URL = (
    "https://github.com/wmo-im/wmdr2/blob/"
    "wigosbox-catalogue1.99-dev/schemas/wmdrRecordGeoJSON.yaml"
)
GENERATED_DATE = "2026-08-05"


def q(ns: str, local: str) -> str:
    return f"{{{ns}}}{local}"


def stable_id(prefix: str, key: str) -> str:
    """Return an EA-safe deterministic XMI identifier.

    Enterprise Architect stores GUIDs in repository fields limited to 40
    characters.  EA's XMI notation therefore uses exactly one five-character
    prefix (EAID_ or EAPK_) followed by the 36-character UUID.  Earlier
    generator versions used identifiers such as EAID_AT_<uuid>; those are
    three characters too long and can trigger DAO.Field 3163 during import.
    The semantic kind remains part of *key*, so identifiers remain unique.
    """
    u = str(uuid5(NAMESPACE_URL, f"wmdr2-xmi:{prefix}:{key}")).upper().replace("-", "_")
    if prefix == "EAPK":
        return f"EAPK_{u}"
    if prefix == "DUID":
        return f"DUID_{u}"
    return f"EAID_{u}"


@dataclass(frozen=True)
class Attribute:
    name: str
    type_name: str
    lower: int = 0
    upper: str = "1"
    description: str = ""
    constraints: tuple[str, ...] = ()
    default: str | None = None


@dataclass(frozen=True)
class UmlClass:
    name: str
    description: str = ""
    attributes: tuple[Attribute, ...] = ()
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class Association:
    owner: str
    role: str
    target: str
    lower: int
    upper: str
    aggregation: str = "none"
    description: str = ""
    constraints: tuple[str, ...] = ()


PRIMITIVES = ("String", "Integer", "Real", "Boolean", "Date", "DateTime", "URI")

CLASSES: tuple[UmlClass, ...] = (
    UmlClass(
        "WMDRRecord",
        "Root GeoJSON Feature representing a WMDR station metadata record.",
        (
            Attribute("id", "String", 1, "1", "WIGOS station identifier", (
                r"pattern: ^(0|1|2|3)-([1-9]\\d*)-([0-9]+)-([A-Za-z0-9\\._-]+)$",
            )),
            Attribute("conformsTo", "URI", 1, "*", "Conformance declarations", (
                "contains: http://wigos.wmo.int/spec/wmdr/2/conf/core",
            )),
            Attribute("type", "String", 1, "1", "GeoJSON feature type", ("const: Feature",)),
        ),
        (
            "JSON Schema draft 2020-12",
            "required: id, conformsTo, type, geometry, properties, links",
            f"source: {SOURCE_URL}",
        ),
    ),
    UmlClass(
        "PointGeometry",
        "GeoJSON Point geometry.",
        (
            Attribute("type", "String", 1, "1", "Geometry type", ("const: Point",)),
            Attribute("coordinates", "Real", 2, "*", "Coordinate tuple", ("minItems: 2",)),
        ),
        ("required: type, coordinates",),
    ),
    UmlClass(
        "TemporalExtent",
        "WCMP temporal representation; the schema allows null or an object.",
        (
            Attribute("date", "Date", 0, "1", constraints=(r"pattern: ^\\d{4}-\\d{2}-\\d{2}$",)),
            Attribute("timestamp", "DateTime", 0, "1", constraints=("UTC timestamp ending in Z",)),
            Attribute("interval", "String", 0, "2", "Two temporal boundary values", (
                "when present: exactly 2 items",
                "item: date, year-month, year, UTC timestamp, time-only, or '..'",
            )),
            Attribute("resolution", "String", 0, "1", "Minimum resolvable period", (
                "ISO 8601 duration",
            )),
        ),
        ("oneOf: null or object",),
    ),
    UmlClass(
        "FacilityProperties",
        "Facility metadata carried in the GeoJSON properties member.",
        (
            Attribute("title", "String", 1, "1", "Facility name (primary)"),
            Attribute("description", "String", 0, "1", "Facility description"),
            Attribute("additionalTitles", "String", 0, "*", "Additional facility names", (
                "if present: minItems 1",
            )),
            Attribute("additionalIds", "String", 0, "*", "Additional WIGOS Station Identifiers", (
                "if present: minItems 1",
                r"item pattern: ^(0|1|2|3)-([1-9]\\d*)-([0-9]+)-([A-Za-z0-9\\._-]+)$",
            )),
            Attribute("facilityType", "String", 0, "1", "Station/platform type", (
                "vocabulary: https://codes.wmo.int/wmdr/FacilityType",
            )),
            Attribute("wmoRegion", "String", 0, "1", "WMO Region", (
                "vocabulary: https://codes.wmo.int/wmdr/WMORegion",
            )),
            Attribute("created", "DateTime", 1, "1", "Record creation time"),
            Attribute("updated", "DateTime", 0, "1", "Most recent record update time"),
            Attribute("keywords", "String", 0, "*", "Free-form keywords"),
        ),
        ("required: contacts, created, title, observingCapabilities",),
    ),
    UmlClass(
        "Contact",
        "OGC Records contact, restricted by WMDR2 to require organization.",
        (
            Attribute("identifier", "String", 0, "1", "Unique contact identifier"),
            Attribute("name", "String", 0, "1", "Responsible person"),
            Attribute("position", "String", 0, "1", "Role or position in the organization"),
            Attribute("organization", "String", 1, "1", "Organization/affiliation"),
            Attribute("hoursOfService", "String", 0, "1"),
            Attribute("contactInstructions", "String", 0, "1"),
            Attribute("roles", "String", 0, "*", constraints=("if present: minItems 1",)),
        ),
        (
            "additionalProperties: false",
            "OGC base schema anyOf(name, organization); WMDR2 additionally requires organization",
        ),
    ),
    UmlClass(
        "Phone",
        "Telephone number and its roles.",
        (
            Attribute("value", "String", 1, "1", "Phone number", (r"pattern: ^\\+[1-9][0-9]{3,14}$",)),
            Attribute("roles", "String", 0, "*"),
        ),
        ("required: value",),
    ),
    UmlClass(
        "Email",
        "Email address and its roles.",
        (
            Attribute("value", "String", 1, "1", "Email address", ("format: email",)),
            Attribute("roles", "String", 0, "*"),
        ),
        ("required: value",),
    ),
    UmlClass(
        "Address",
        "Physical contact address.",
        (
            Attribute("deliveryPoint", "String", 0, "*", "Address lines"),
            Attribute("city", "String", 0, "1"),
            Attribute("administrativeArea", "String", 0, "1"),
            Attribute("postalCode", "String", 0, "1"),
            Attribute("country", "String", 0, "1", "ISO 3166-1 recommended"),
            Attribute("roles", "String", 0, "*"),
        ),
    ),
    UmlClass(
        "Link",
        "Superset of link properties used by OGC Records contacts and GeoJSON record links.",
        (
            Attribute("href", "URI", 1, "1", "Link target", ("format: uri where required by source schema",)),
            Attribute("rel", "String", 1, "1", "Link relation"),
            Attribute("type", "String", 0, "1", "Media type hint"),
            Attribute("hreflang", "String", 0, "1"),
            Attribute("title", "String", 0, "1"),
            Attribute("length", "Integer", 0, "1"),
            Attribute("profile", "URI", 0, "*"),
            Attribute("created", "DateTime", 0, "1"),
            Attribute("updated", "DateTime", 0, "1"),
        ),
        (
            "contact.logo additionally requires rel='icon' and type",
            "contact.links additionally requires type",
            "record links require href and rel",
        ),
    ),
    UmlClass(
        "ExternalIdentifier",
        "Identifier assigned by an entity external to the catalogue.",
        (
            Attribute("scheme", "URI", 0, "1"),
            Attribute("value", "String", 1, "1"),
        ),
        ("required: value",),
    ),
    UmlClass(
        "Theme",
        "Knowledge organization system used to classify the resource.",
        (Attribute("scheme", "URI", 1, "1"),),
        ("required: concepts, scheme", "if present in FacilityProperties: minItems 1"),
    ),
    UmlClass(
        "Concept",
        "Concept from a knowledge organization system.",
        (
            Attribute("id", "String", 1, "1"),
            Attribute("title", "String", 0, "1"),
            Attribute("description", "String", 0, "1"),
            Attribute("url", "URI", 0, "1", constraints=("format: uri",)),
        ),
        ("required: id",),
    ),
    UmlClass(
        "Territory",
        "Time series entry for territory of origin.",
        (
            Attribute("territory", "String", 1, "1", "Territory of origin of data", (
                "vocabulary: https://codes.wmo.int/wmdr/TerritoryName",
            )),
            Attribute("dates", "String", 0, "*", constraints=(
                "if present: minItems 1",
                "item oneOf: date or '..'",
            )),
        ),
        ("required: territory",),
    ),
    UmlClass(
        "ProgramAffiliation",
        "Programme/network affiliation over a date range.",
        (
            Attribute("facilityId", "String", 0, "1", "Programme-specific facility identifier"),
            Attribute("facilityTitle", "String", 0, "1", "Programme-specific facility name"),
            Attribute("programAffiliation", "String", 0, "1", constraints=(
                "vocabulary: https://codes.wmo.int/wmdr/ProgramAffiliation",
            )),
            Attribute("reportingStatus", "String", 0, "1", constraints=(
                "vocabulary: https://codes.wmo.int/wmdr/ReportingStatus",
            )),
            Attribute("dates", "String", 0, "*", constraints=(
                "if present: minItems 1",
                "item oneOf: date or '..'",
            )),
        ),
    ),
    UmlClass(
        "ObservingCapability",
        "Observed property, method, status and optional deployment.",
        (
            Attribute("observedProperty", "String", 1, "1", "Observed property", (
                "vocabularies: ObservedVariableAtmosphere, Earth, Ocean, OuterSpace, Terrestrial",
            )),
            Attribute("observingMethod", "String", 0, "1", constraints=(
                "vocabularies: ObservingMethodAtmosphere, Terrestrial, Earth",
            )),
            Attribute("operatingStatus", "String", 0, "1", constraints=(
                "vocabulary: https://codes.wmo.int/wmdr/InstrumentOperatingStatus",
            )),
            Attribute("sourceOfObservation", "String", 0, "1", constraints=(
                "vocabulary: https://codes.wmo.int/wmdr/SourceOfObservation",
            )),
        ),
        ("required: observedProperty", "deployments: maxItems 1"),
    ),
    UmlClass(
        "Deployment",
        "Placement/location of specific equipment.",
        (
            Attribute("id", "String", 1, "1", "Instrument identifier", ("JSON type: string or integer",)),
            Attribute("keywords", "String", 0, "*"),
        ),
        (
            "required: id, instrument",
            "instrument oneOf: id-only reference object or inline Instrument",
            "referenceSurfaces if present: minItems 1",
        ),
    ),
    UmlClass(
        "Instrument",
        "Instrument description; may be embedded or referenced from Deployment.",
        (
            Attribute("id", "String", 1, "1", "Instrument identifier", ("JSON type: string or integer",)),
            Attribute("manufacturer", "String", 0, "1"),
            Attribute("model", "String", 0, "1"),
            Attribute("serialNumber", "String", 0, "1"),
        ),
        ("required: id",),
    ),
    UmlClass(
        "ReferenceSurface",
        "Reference surface and vertical distance used by a deployment.",
        (
            Attribute("referenceSurface", "String", 1, "1", "Type of reference surface", (
                "vocabulary: https://codes.wmo.int/wmdr/ReferenceSurfaceType",
            )),
            Attribute("verticalDistance", "Real", 1, "1", "Height above reference surface"),
        ),
        ("required: referenceSurface, verticalDistance",),
    ),
)

ASSOCIATIONS: tuple[Association, ...] = (
    Association("WMDRRecord", "geometry", "PointGeometry", 1, "1"),
    Association("WMDRRecord", "time", "TemporalExtent", 0, "1"),
    Association("WMDRRecord", "properties", "FacilityProperties", 1, "1"),
    Association("WMDRRecord", "links", "Link", 1, "*"),
    Association("FacilityProperties", "contacts", "Contact", 1, "*"),
    Association("FacilityProperties", "externalIds", "ExternalIdentifier", 0, "*"),
    Association("FacilityProperties", "themes", "Theme", 0, "*", constraints=("if present: minItems 1",)),
    Association("FacilityProperties", "territories", "Territory", 0, "*", constraints=("if present: minItems 1",)),
    Association("FacilityProperties", "programAffiliations", "ProgramAffiliation", 0, "*", constraints=("if present: minItems 1",)),
    Association("FacilityProperties", "observingCapabilities", "ObservingCapability", 1, "*"),
    Association("Contact", "logo", "Link", 0, "1", constraints=("rel='icon' and type required",)),
    Association("Contact", "phones", "Phone", 0, "*"),
    Association("Contact", "emails", "Email", 0, "*"),
    Association("Contact", "addresses", "Address", 0, "*"),
    Association("Contact", "links", "Link", 0, "*", constraints=("type required",)),
    Association("Theme", "concepts", "Concept", 1, "*"),
    Association("ObservingCapability", "time", "TemporalExtent", 0, "1"),
    Association("ObservingCapability", "deployments", "Deployment", 0, "1"),
    Association("Deployment", "geometry", "PointGeometry", 0, "1"),
    Association("Deployment", "time", "TemporalExtent", 0, "1"),
    Association("Deployment", "instrument", "Instrument", 1, "1", "none", constraints=(
        "id-only reference or inline Instrument",
    )),
    Association("Deployment", "referenceSurfaces", "ReferenceSurface", 0, "*", constraints=(
        "if present: minItems 1",
    )),
)

CORE_DIAGRAM = (
    "WMDRRecord", "PointGeometry", "TemporalExtent", "FacilityProperties",
    "Contact", "Link", "Territory", "ProgramAffiliation",
    "ObservingCapability", "Deployment", "Instrument", "ReferenceSurface",
)
SUPPORT_DIAGRAM = (
    "Contact", "Phone", "Email", "Address", "Link", "ExternalIdentifier",
    "Theme", "Concept", "TemporalExtent", "PointGeometry",
)


def add_comment(parent: etree._Element, text: str, key: str) -> None:
    if not text:
        return
    c = etree.SubElement(parent, "ownedComment", {q(XMI_NS, "type"): "uml:Comment", q(XMI_NS, "id"): stable_id("EAID_CM", key)})
    etree.SubElement(c, "body").text = text


def add_constraint(parent: etree._Element, name: str, body: str, key: str) -> None:
    rule = etree.SubElement(parent, "ownedRule", {
        q(XMI_NS, "type"): "uml:Constraint",
        q(XMI_NS, "id"): stable_id("EAID_CT", key),
        "name": name,
    })
    spec = etree.SubElement(rule, "specification", {
        q(XMI_NS, "type"): "uml:OpaqueExpression",
        q(XMI_NS, "id"): stable_id("EAID_OE", key),
    })
    etree.SubElement(spec, "language").text = "JSON Schema"
    etree.SubElement(spec, "body").text = body


def add_multiplicity(parent: etree._Element, lower: int, upper: str, key: str) -> None:
    """Write multiplicity in the form emitted by Enterprise Architect.

    EA writes finite upper bounds as uml:LiteralInteger.  Only the unbounded
    value is written as uml:LiteralUnlimitedNatural, using -1 rather than *.
    Although other serializations can be valid UML/XMI, this EA-native form
    is required for reliable import into EA's Source/Target Role fields.
    """
    etree.SubElement(parent, "lowerValue", {
        q(XMI_NS, "type"): "uml:LiteralInteger",
        q(XMI_NS, "id"): stable_id("EAID_LO", key),
        "value": str(lower),
    })
    if str(upper) == "*":
        upper_type = "uml:LiteralUnlimitedNatural"
        upper_value = "-1"
    else:
        upper_type = "uml:LiteralInteger"
        upper_value = str(upper)
    etree.SubElement(parent, "upperValue", {
        q(XMI_NS, "type"): upper_type,
        q(XMI_NS, "id"): stable_id("EAID_UP", key),
        "value": upper_value,
    })


def make_model() -> tuple[etree._ElementTree, dict[str, str], dict[str, str]]:
    root = etree.Element(q(XMI_NS, "XMI"), nsmap=NSMAP, attrib={q(XMI_NS, "version"): "2.1"})
    etree.SubElement(root, q(XMI_NS, "Documentation"), {
        "exporter": "Enterprise Architect",
        "exporterVersion": "6.5",
        "exporterID": "1710",
    })
    model = etree.SubElement(root, q(UML_NS, "Model"), {
        q(XMI_NS, "type"): "uml:Model",
        q(XMI_NS, "id"): stable_id("EAID", "model"),
        "name": "WMDR2 Official Schema",
        "visibility": "public",
    })
    package_id = stable_id("EAPK", "package:WMDR2 Official Schema")
    pkg = etree.SubElement(model, "packagedElement", {
        q(XMI_NS, "type"): "uml:Package",
        q(XMI_NS, "id"): package_id,
        "name": "WMDR2 Official Schema",
        "visibility": "public",
    })
    add_comment(pkg, f"Generated {GENERATED_DATE} from {SOURCE_URL}", "package-source")

    type_ids: dict[str, str] = {}
    for p in PRIMITIVES:
        pid = stable_id("EAID_DT", f"primitive:{p}")
        type_ids[p] = pid
        etree.SubElement(pkg, "packagedElement", {
            q(XMI_NS, "type"): "uml:PrimitiveType",
            q(XMI_NS, "id"): pid,
            "name": p,
            "visibility": "public",
        })

    class_ids = {c.name: stable_id("EAID", f"class:{c.name}") for c in CLASSES}
    class_nodes: dict[str, etree._Element] = {}
    attr_ids: dict[tuple[str, str], str] = {}

    for cls in CLASSES:
        node = etree.SubElement(pkg, "packagedElement", {
            q(XMI_NS, "type"): "uml:Class",
            q(XMI_NS, "id"): class_ids[cls.name],
            "name": cls.name,
            "visibility": "public",
        })
        class_nodes[cls.name] = node
        add_comment(node, cls.description, f"class-comment:{cls.name}")
        for i, constraint in enumerate(cls.constraints, 1):
            add_constraint(node, f"schemaRule{i}", constraint, f"class:{cls.name}:constraint:{i}")
        for attr in cls.attributes:
            aid = stable_id("EAID_AT", f"attribute:{cls.name}.{attr.name}")
            attr_ids[(cls.name, attr.name)] = aid
            a = etree.SubElement(node, "ownedAttribute", {
                q(XMI_NS, "type"): "uml:Property",
                q(XMI_NS, "id"): aid,
                "name": attr.name,
                "visibility": "public",
                "isStatic": "false",
                "isReadOnly": "false",
                "isDerived": "false",
                "isOrdered": "true" if attr.upper not in ("0", "1") else "false",
                "isUnique": "true",
                "isDerivedUnion": "false",
            })
            etree.SubElement(a, "type", {q(XMI_NS, "idref"): type_ids[attr.type_name]})
            add_multiplicity(a, attr.lower, attr.upper, f"attribute:{cls.name}.{attr.name}")
            if attr.default is not None:
                dv = etree.SubElement(a, "defaultValue", {
                    q(XMI_NS, "type"): "uml:LiteralString",
                    q(XMI_NS, "id"): stable_id("EAID_DV", f"attribute:{cls.name}.{attr.name}"),
                    "value": attr.default,
                })
            add_comment(a, attr.description, f"attribute-comment:{cls.name}.{attr.name}")
            for i, constraint in enumerate(attr.constraints, 1):
                add_constraint(a, f"schemaRule{i}", constraint, f"attribute:{cls.name}.{attr.name}:constraint:{i}")

    assoc_ids: dict[str, str] = {}
    for assoc in ASSOCIATIONS:
        key = f"association:{assoc.owner}.{assoc.role}->{assoc.target}"
        assoc_id = stable_id("EAID", key)
        assoc_ids[key] = assoc_id
        owner_end_id = stable_id("EAID_AT", f"assoc-end:{assoc.owner}.{assoc.role}")
        target_end_id = stable_id("EAID_AT", f"assoc-opposite:{assoc.owner}.{assoc.role}->{assoc.target}")

        owner_node = class_nodes[assoc.owner]
        end = etree.SubElement(owner_node, "ownedAttribute", {
            q(XMI_NS, "type"): "uml:Property",
            q(XMI_NS, "id"): owner_end_id,
            "name": assoc.role,
            "visibility": "public",
            "association": assoc_id,
            "aggregation": assoc.aggregation,
            "isStatic": "false",
            "isReadOnly": "false",
            "isDerived": "false",
            "isOrdered": "true" if assoc.upper == "*" else "false",
            "isUnique": "true",
            "isDerivedUnion": "false",
        })
        etree.SubElement(end, "type", {q(XMI_NS, "idref"): class_ids[assoc.target]})
        add_multiplicity(end, assoc.lower, assoc.upper, f"assoc-end:{assoc.owner}.{assoc.role}")
        add_comment(end, assoc.description, f"association-comment:{key}")
        for i, constraint in enumerate(assoc.constraints, 1):
            add_constraint(end, f"schemaRule{i}", constraint, f"association:{key}:constraint:{i}")

        ae = etree.SubElement(pkg, "packagedElement", {
            q(XMI_NS, "type"): "uml:Association",
            q(XMI_NS, "id"): assoc_id,
            "visibility": "public",
        })
        # EA's own XMI 2.1 exporter serializes memberEnd as child elements,
        # not as a whitespace-separated XML attribute.
        etree.SubElement(ae, "memberEnd", {q(XMI_NS, "idref"): owner_end_id})
        etree.SubElement(ae, "memberEnd", {q(XMI_NS, "idref"): target_end_id})
        opposite = etree.SubElement(ae, "ownedEnd", {
            q(XMI_NS, "type"): "uml:Property",
            q(XMI_NS, "id"): target_end_id,
            "name": assoc.owner[0].lower() + assoc.owner[1:],
            "visibility": "public",
            "association": assoc_id,
            "aggregation": "none",
            "isStatic": "false",
            "isReadOnly": "false",
            "isDerived": "false",
            "isOrdered": "false",
            "isUnique": "true",
            "isDerivedUnion": "false",
        })
        etree.SubElement(opposite, "type", {q(XMI_NS, "idref"): class_ids[assoc.owner]})
        add_multiplicity(opposite, 1, "1", f"assoc-opposite:{assoc.owner}.{assoc.role}->{assoc.target}")

    add_ea_extension(root, package_id, class_ids, assoc_ids)
    return etree.ElementTree(root), class_ids, assoc_ids


def ea_multiplicity(lower: int, upper: str) -> str:
    """Return EA connector-role multiplicity text."""
    upper_s = str(upper)
    if str(lower) == upper_s:
        return upper_s
    return f"{lower}..{upper_s}"


def add_ea_extension(root: etree._Element, package_id: str, class_ids: dict[str, str], assoc_ids: dict[str, str]) -> None:
    ext = etree.SubElement(root, q(XMI_NS, "Extension"), {"extender": "Enterprise Architect", "extenderID": "6.5"})
    elements = etree.SubElement(ext, "elements")

    # Package metadata
    pe = etree.SubElement(elements, "element", {
        q(XMI_NS, "idref"): package_id,
        q(XMI_NS, "type"): "uml:Package",
        "name": "WMDR2 Official Schema",
        "scope": "public",
    })
    etree.SubElement(pe, "model", {"package2": package_id.replace("EAPK_", "EAID_"), "tpos": "0", "ea_localid": "1", "ea_eleType": "package"})
    etree.SubElement(pe, "properties", {"isSpecification": "false", "sType": "Package", "nType": "0", "scope": "public"})
    etree.SubElement(pe, "project", {"author": "OpenAI", "version": "1.0", "phase": "1.0", "created": GENERATED_DATE + " 00:00:00", "modified": GENERATED_DATE + " 00:00:00", "complexity": "1", "status": "Proposed"})
    etree.SubElement(pe, "code", {"gentype": "<none>"})
    etree.SubElement(pe, "style", {"appearance": "BackColor=-1;BorderColor=-1;BorderWidth=-1;FontColor=-1;"})
    etree.SubElement(pe, "tags")
    etree.SubElement(pe, "xrefs")
    etree.SubElement(pe, "extendedProperties", {"tagged": "0", "package_name": "Model"})
    etree.SubElement(pe, "packageproperties", {"version": "1.0"})
    etree.SubElement(pe, "paths")
    etree.SubElement(pe, "times", {"created": GENERATED_DATE + " 00:00:00", "modified": GENERATED_DATE + " 00:00:00"})
    etree.SubElement(pe, "flags", {"iscontrolled": "FALSE", "isprotected": "FALSE", "usedtd": "FALSE", "logxml": "FALSE", "packageFlags": "isModel=1;VICON=2;"})

    local_ids: dict[str, int] = {}
    for idx, cls in enumerate(CLASSES, start=2):
        local_ids[cls.name] = idx
        ce = etree.SubElement(elements, "element", {
            q(XMI_NS, "idref"): class_ids[cls.name],
            q(XMI_NS, "type"): "uml:Class",
            "name": cls.name,
            "scope": "public",
        })
        etree.SubElement(ce, "model", {"package": package_id, "tpos": "0", "ea_localid": str(idx), "ea_eleType": "element"})
        etree.SubElement(ce, "properties", {"isSpecification": "false", "sType": "Class", "nType": "0", "scope": "public"})
        etree.SubElement(ce, "project", {"author": "OpenAI", "version": "1.0", "phase": "1.0", "created": GENERATED_DATE + " 00:00:00", "modified": GENERATED_DATE + " 00:00:00", "complexity": "1", "status": "Proposed"})
        etree.SubElement(ce, "code", {"gentype": "<none>"})
        etree.SubElement(ce, "style", {"appearance": "BackColor=-1;BorderColor=-1;BorderWidth=-1;FontColor=-1;VSwimLanes=1;HSwimLanes=1;BorderStyle=0;"})
        etree.SubElement(ce, "tags")
        etree.SubElement(ce, "xrefs")
        etree.SubElement(ce, "extendedProperties", {"tagged": "0", "package_name": "WMDR2 Official Schema"})

    connectors = etree.SubElement(ext, "connectors")
    for i, assoc in enumerate(ASSOCIATIONS, start=1):
        key = f"association:{assoc.owner}.{assoc.role}->{assoc.target}"
        cid = assoc_ids[key]
        c = etree.SubElement(connectors, "connector", {q(XMI_NS, "idref"): cid})
        src = etree.SubElement(c, "source", {q(XMI_NS, "idref"): class_ids[assoc.owner]})
        etree.SubElement(src, "model", {"ea_localid": str(local_ids[assoc.owner]), "type": "Class", "name": assoc.owner})
        etree.SubElement(src, "role", {"visibility": "Public", "targetScope": "instance"})
        etree.SubElement(src, "type", {"multiplicity": "1", "aggregation": "none", "containment": "Unspecified"})
        etree.SubElement(src, "constraints")
        etree.SubElement(src, "modifiers", {"isOrdered": "false", "changeable": "none", "isNavigable": "false"})
        etree.SubElement(src, "style", {"value": "Union=0;Derived=0;AllowDuplicates=0;Owned=0;Navigable=Unspecified;"})
        etree.SubElement(src, "documentation")
        etree.SubElement(src, "xrefs")
        etree.SubElement(src, "tags")

        tgt = etree.SubElement(c, "target", {q(XMI_NS, "idref"): class_ids[assoc.target]})
        etree.SubElement(tgt, "model", {"ea_localid": str(local_ids[assoc.target]), "type": "Class", "name": assoc.target})
        etree.SubElement(tgt, "role", {"name": assoc.role, "visibility": "Public", "targetScope": "instance"})
        etree.SubElement(tgt, "type", {"multiplicity": ea_multiplicity(assoc.lower, assoc.upper), "aggregation": "none", "containment": "Unspecified"})
        etree.SubElement(tgt, "constraints")
        etree.SubElement(tgt, "modifiers", {"isOrdered": "true" if assoc.upper == "*" else "false", "changeable": "none", "isNavigable": "false"})
        etree.SubElement(tgt, "style", {"value": "Union=0;Derived=0;AllowDuplicates=0;Owned=0;Navigable=Unspecified;"})
        etree.SubElement(tgt, "documentation")
        etree.SubElement(tgt, "xrefs")
        etree.SubElement(tgt, "tags")

        etree.SubElement(c, "model", {"ea_localid": str(i)})
        etree.SubElement(c, "properties", {"ea_type": "Association", "direction": "Unspecified"})
        etree.SubElement(c, "documentation")
        etree.SubElement(c, "appearance", {"linemode": "1", "linecolor": "-1", "linewidth": "0", "seqno": "0", "headStyle": "0", "lineStyle": "0"})
        etree.SubElement(c, "labels", {"lb": "1", "rb": ea_multiplicity(assoc.lower, assoc.upper)})
        etree.SubElement(c, "extendedProperties", {"virtualInheritance": "0"})
        etree.SubElement(c, "style")
        etree.SubElement(c, "xrefs")
        etree.SubElement(c, "tags")

    primitive_types = etree.SubElement(ext, "primitivetypes")
    etree.SubElement(primitive_types, "packagedElement", {
        q(XMI_NS, "type"): "uml:Package",
        q(XMI_NS, "id"): "EAPrimitiveTypesPackage",
        "name": "EA_PrimitiveTypes_Package",
        "visibility": "public",
    })
    etree.SubElement(ext, "profiles")
    diagrams = etree.SubElement(ext, "diagrams")

    add_diagram(diagrams, package_id, class_ids, assoc_ids, "WMDR2 Core", CORE_DIAGRAM, 1)
    add_diagram(diagrams, package_id, class_ids, assoc_ids, "WMDR2 Supporting Types", SUPPORT_DIAGRAM, 2)


def add_diagram(diagrams: etree._Element, package_id: str, class_ids: dict[str, str], assoc_ids: dict[str, str], name: str, class_names: Iterable[str], local_id: int) -> None:
    names = tuple(class_names)
    did = stable_id("EAID", f"diagram:{name}")
    d = etree.SubElement(diagrams, "diagram", {q(XMI_NS, "id"): did})
    etree.SubElement(d, "model", {"package": package_id, "localID": str(local_id), "owner": package_id})
    etree.SubElement(d, "properties", {"name": name, "type": "Logical"})
    etree.SubElement(d, "project", {"author": "OpenAI", "version": "1.0", "created": GENERATED_DATE + " 00:00:00", "modified": GENERATED_DATE + " 00:00:00"})
    etree.SubElement(d, "style1", {"value": "ShowPrivate=1;ShowProtected=1;ShowPublic=1;HideRelationships=0;Locked=0;Border=1;HighlightForeign=1;PackageContents=1;ScalePrintImage=0;PPgs.cx=2;PPgs.cy=1;DocSize.cx=1650;DocSize.cy=1100;ShowDetails=0;Orientation=L;Zoom=80;ShowTags=0;OpParams=1;VisibleAttributeDetail=1;ShowOpRetType=1;ShowIcons=1;HideProps=0;ShowReqs=0;ShowCons=0;PaperSize=9;HideParents=0;UseAlias=0;HideAtts=0;HideOps=1;HideStereo=0;HideElemStereo=0;ConnectorNotation=UML 2.1;ExplicitNavigability=0;ShowShape=1;AdvancedElementProps=1;AdvancedFeatureProps=1;AdvancedConnectorProps=1;ShowNotes=0;SuppressBrackets=0;SuppConnectorLabels=0;PrintPageHeadFoot=0;ShowAsList=0;"})
    etree.SubElement(d, "style2", {"value": "ExcludeRTF=0;DocAll=0;HideQuals=0;AttPkg=1;SuppressFOC=0;MatrixActive=0;SwimlanesActive=1;KanbanActive=0;MatrixLineWidth=1;MatrixLineClr=0;MatrixLocked=0;TConnectorNotation=UML 2.1;TExplicitNavigability=0;AdvancedElementProps=1;AdvancedFeatureProps=1;AdvancedConnectorProps=1;ShowNotes=0;VisibleAttributeDetail=1;ShowOpRetType=1;SuppressBrackets=0;SuppConnectorLabels=0;PrintPageHeadFoot=0;ShowAsList=0;"})
    etree.SubElement(d, "swimlanes", {"value": "locked=false;orientation=0;width=0;inbar=false;names=false;color=-1;bold=false;fcol=0;tcol=-1;"})
    etree.SubElement(d, "matrixitems", {"value": "locked=false;matrixactive=false;swimlanesactive=true;kanbanactive=false;width=1;clrLine=0;"})
    etree.SubElement(d, "extendedProperties")
    de = etree.SubElement(d, "elements")

    # Layout on a regular grid, with wider boxes for attribute-rich classes.
    cols = 4 if len(names) > 10 else 3
    box_w, box_h = 300, 230
    x0, y0, gap_x, gap_y = 50, 50, 90, 80
    for idx, cname in enumerate(names):
        row, col = divmod(idx, cols)
        left = x0 + col * (box_w + gap_x)
        top = y0 + row * (box_h + gap_y)
        right = left + box_w
        bottom = top + box_h
        etree.SubElement(de, "element", {
            "geometry": f"Left={left};Top={top};Right={right};Bottom={bottom};",
            "subject": class_ids[cname],
            "seqno": str(idx + 1),
            "style": f"DUID={stable_id('DUID', f'{name}:{cname}').replace('DUID_', '')};",
        })

    visible = set(names)
    for assoc in ASSOCIATIONS:
        if assoc.owner in visible and assoc.target in visible:
            key = f"association:{assoc.owner}.{assoc.role}->{assoc.target}"
            etree.SubElement(de, "element", {
                "geometry": "SX=0;SY=0;EX=0;EY=0;Path=;",
                "subject": assoc_ids[key],
                "style": ";Hidden=0;",
            })


def write_plantuml(path: Path) -> None:
    lines = [
        "@startuml",
        "hide methods",
        "skinparam classAttributeIconSize 0",
        "title WMDR2 official schema (wigosbox-catalogue1.99-dev)",
    ]
    for cls in CLASSES:
        lines.append(f'class {cls.name} {{')
        for a in cls.attributes:
            mult = "" if (a.lower, a.upper) == (0, "1") else f" [{a.lower}..{a.upper}]"
            lines.append(f"  {a.name}: {a.type_name}{mult}")
        lines.append("}")
    for a in ASSOCIATIONS:
        arrow = "--"
        lines.append(f'{a.owner} {arrow} "{a.lower}..{a.upper}" {a.target} : {a.role}')
    lines.append("@enduml")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_references(tree: etree._ElementTree) -> list[str]:
    ids = set(tree.xpath("//@xmi:id", namespaces={"xmi": XMI_NS}))
    errors: list[str] = []
    for attr in ("type", "association", "memberEnd"):
        for value in tree.xpath(f"//@{attr}"):
            if attr == "type" and value.startswith("uml:"):
                continue
            for token in value.split():
                if token.startswith(("EAID_", "EAPK_")) and token not in ids:
                    errors.append(f"unresolved {attr}: {token}")
    for value in tree.xpath("//@xmi:idref", namespaces={"xmi": XMI_NS}):
        # EA extension may refer to package and model items; all should exist here.
        if value.startswith(("EAID_", "EAPK_")) and value not in ids:
            errors.append(f"unresolved xmi:idref: {value}")
    return sorted(set(errors))


def validate_ea_identifier_lengths(tree: etree._ElementTree) -> list[str]:
    """Reject XMI identifiers that cannot round-trip through EA GUID fields."""
    errors: list[str] = []
    ns = {"xmi": XMI_NS}
    ids = tree.xpath("//@xmi:id", namespaces=ns)
    if len(ids) != len(set(ids)):
        errors.append("duplicate xmi:id values")
    for attr_name in ("id", "idref"):
        for value in tree.xpath(f"//@xmi:{attr_name}", namespaces=ns):
            if value.startswith(("EAID_", "EAPK_")) and len(value) != 41:
                errors.append(
                    f"EA identifier must be exactly 41 characters in XMI: {value!r} "
                    f"({len(value)} characters)"
                )
    return sorted(set(errors))


def validate_ea_multiplicity_encoding(tree: etree._ElementTree) -> list[str]:
    """Check the EA-compatible association-end serialization."""
    errors: list[str] = []
    ns = {"xmi": XMI_NS}
    associations = tree.xpath('//*[local-name()="packagedElement"][@xmi:type="uml:Association"]', namespaces=ns)
    for association in associations:
        aid = association.get(q(XMI_NS, "id"), "<unknown>")
        member_ends = association.xpath('./memberEnd/@xmi:idref', namespaces=ns)
        if len(member_ends) != 2:
            errors.append(f"{aid}: expected two child memberEnd elements")
        if association.get("memberEnd") is not None:
            errors.append(f"{aid}: memberEnd must not be serialized as an attribute")
    for upper in tree.xpath('//*[local-name()="upperValue"]'):
        value = upper.get("value")
        xtype = upper.get(q(XMI_NS, "type"))
        if value == "-1" and xtype != "uml:LiteralUnlimitedNatural":
            errors.append("unbounded upperValue must be LiteralUnlimitedNatural -1")
        elif value != "-1" and xtype != "uml:LiteralInteger":
            errors.append(f"finite upperValue {value!r} must be LiteralInteger")
    return sorted(set(errors))


def validate_ea_extension_multiplicities(tree: etree._ElementTree) -> list[str]:
    """Ensure multiplicity is encoded where EA's native XMI exporter puts it."""
    errors: list[str] = []
    ns = {"xmi": XMI_NS}
    connectors = tree.xpath('//*[local-name()="connectors"]/*[local-name()="connector"]')
    for connector in connectors:
        cid = connector.get(q(XMI_NS, "idref"), "<unknown>")
        for end_name in ("source", "target"):
            ends = connector.xpath(f'./*[local-name()="{end_name}"]')
            if len(ends) != 1:
                errors.append(f"{cid}: expected one {end_name}")
                continue
            end = ends[0]
            roles = end.xpath('./*[local-name()="role"]')
            types = end.xpath('./*[local-name()="type"]')
            if len(roles) != 1 or len(types) != 1:
                errors.append(f"{cid}: malformed {end_name} role/type")
                continue
            if roles[0].get("multiplicity") is not None:
                errors.append(f"{cid}: {end_name} multiplicity must not be on role")
            if not types[0].get("multiplicity"):
                errors.append(f"{cid}: {end_name} type lacks multiplicity")
        labels = connector.xpath('./*[local-name()="labels"]')
        if len(labels) != 1 or not labels[0].get("lb") or not labels[0].get("rb"):
            errors.append(f"{cid}: connector labels lack lb/rb multiplicities")
    return sorted(set(errors))


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    xmi_path = out_dir / "wmdr2-pr22.xmi"
    puml_path = out_dir / "wmdr2-pr22.puml"
    tree, _, _ = make_model()
    tree.write(str(xmi_path), encoding="UTF-8", xml_declaration=True, pretty_print=True)
    # Parse it again to ensure well-formedness.
    parsed = etree.parse(str(xmi_path))
    errors = (
        validate_references(parsed)
        + validate_ea_identifier_lengths(parsed)
        + validate_ea_multiplicity_encoding(parsed)
        + validate_ea_extension_multiplicities(parsed)
    )
    if errors:
        raise SystemExit("XMI validation failed:\n" + "\n".join(sorted(set(errors))))
    write_plantuml(puml_path)
    print(f"Wrote {xmi_path}")
    print(f"Wrote {puml_path}")
    print(f"Classes: {len(CLASSES)}; associations: {len(ASSOCIATIONS)}")


if __name__ == "__main__":
    main()
