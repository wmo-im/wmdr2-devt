#!/usr/bin/env python3
"""
Convert WMDR10 XML files into a lean JSON representation, shedding off the GML overhead. 
Relevant gml:id elements are maintained to allow cross-referencing.
This conversion is intended to be loss-less for the WMDR-relevant information.
The output JSON files can be used for easier processing in downstream conversions.

author: joerg.klausen@meteoswiss.ch

config.yaml snippet:
-------------------
convert_wmdr10_xml_to_lean_json:
  source: <path to WMDR10 XML files>
  target: <path to output JSON files>


Output
------
- For each WMDR10 XML file `<name>.xml` in the source directory, a corresponding
  JSON file `<name>.json` is created in the target directory.
- Each JSON file contains the following top-level keys:
    - header: Metadata from the WMDR10 header section.
    - facility: Information about the facility.
    - observations: List of observation records.
    - deployments: List of deployment records.
- Optionally, individual observation or deployment records can be extracted
  by specifying an index.

Dependencies
------------
    pip install -r requirements.txt
"""

from pathlib import Path

from utils.config import load_config
from wmdr10.wmdr10 import WMDR10


def main():
    config = load_config(Path("config.yaml"))
    source_path = Path(config['convert_wmdr10_xml_to_wmdr10_json']['source'])
    target_path = Path(config['convert_wmdr10_xml_to_wmdr10_json']['target'])
    target_path.mkdir(parents=True, exist_ok=True)

    # read WMDR10 XML files and convert to lean json
    xml_files = list(source_path.glob("*.xml"))
    for xml_file in xml_files:
        wmdr10 = WMDR10(xml_file)
        wmdr10.export(path=target_path / xml_file.name.replace('.xml', ''))

        # extract parts of WMDR10 object and store as lean json
        print(f"{wmdr10.export(parts='header', path=target_path / xml_file.name)} created.")
        print(f"{wmdr10.export(parts='facility', path=target_path / xml_file.name)} created.")
        print(f"{wmdr10.export(parts='observations', path=target_path / xml_file.name)} created.")
        print(f"{wmdr10.export(parts='deployments', path=target_path / xml_file.name)} created.")

        # extract specific observation and deployment by index
        print(f"{wmdr10.export(parts='observations', index=1, path=target_path / xml_file.name)} created.")
        print(f"{wmdr10.export(parts='observations', index=5, path=target_path / xml_file.name)} created.")
        print(f"{wmdr10.export(parts='deployments', index=1, path=target_path / xml_file.name)} created.")
        print(f"{wmdr10.export(parts='deployments', index=3, path=target_path / xml_file.name)} created.")

        print(f"Finished processing '{xml_file.name}'.")

if __name__ == "__main__":
    main()
