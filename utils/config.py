from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML config file."""
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping/dict: {config_path}")
    return data


