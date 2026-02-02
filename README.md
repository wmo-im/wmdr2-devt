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