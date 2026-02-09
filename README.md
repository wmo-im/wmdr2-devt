# References
- OGC API - Records - Part 1: Core (https://docs.ogc.org/is/22-079r3/22-079r3.html)
- WMO 1192 WIGOS Metadata Standard (https://library.wmo.int/documents/wmo-1192)
- WMDR1.0 (https://schemas.wmo.int/wmdr/1.0/)
- WMDR2 draft UML aligned to OMS (https://wmo-im.github.io/wmdr2-devt/oms/html/)

# mappings
## WMDR1.0 vs OGC Records - Part 1:Core
### Facility identity
facility.identifier is mapped both to id (the primary record identifier; required) and to properties.externalIds[*].value (to preserve all identifiers). The record identifier requirement comes from the GeoJSON record schema.

### Facility location
facility.geospatialLocation[*].geoLocation maps to the record geometry (required in the GeoJSON record encoding), while the full list is also preserved under an extension property (properties.wmdr:geospatialLocationHistory[...]).

### Classification / controlled vocabularies
WMO code-list URIs (e.g., region, territory, surface cover, topography) map naturally into properties.themes[*] (scheme + concepts), which is intended for classification systems. For example, wmdr:wmoRegion and wmdr:territory map into separate themes.

### Contacts
facility.responsibleParty and header.recordOwner map into properties.contacts[*] (organization/name, identifier, emails, links, roles).

### Extensions
Where WMDR has rich and explicit history objects (beginPosition, endPosition, @gml:id, event logs), Records Part 1 explicitly allows “any number of additional properties”, and recommends advertising them via conformsTo

# Useage
## Command line

### Extract elements specifiedin WMO 1192 WIGOS Metadata Standard
```bash
python extract_wmd_elements_from_wmo_1192.py --config config.yaml --pdf resources/wmo_1192_wigos_metadata_standard.pdf --out results/wmd_elements_extracted_from_wmo_1192.csv
```

### Convert WMDR1.0 XML to a full WMDR1.0 JSON representation, and/or create JSON records of parts of the WMDR1.0 data model. 
```bash
python convert_wmdr10_xml_to_json.py --config config.yaml --section convert_wmdr10_xml_to_json --source resources/wmdr10_xml_examples --target results/wmdr10_json_examples
``` 

### Convert WMDR1.0 JSON to OGC Records - Part 1: Core GeoJSON
```bash
python convert_wmdr10_json_to_records_part1.py --config config.yaml --section convert_wmdr10_json_to_records_part1 --source resources/wmdr10_json_examples --target results/records_part1_geojson_examples --mapping mappings/wmdr10_vs_records_part1.csv
``` 
