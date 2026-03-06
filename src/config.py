"""Load and expose ClinSense configuration."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_data_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    """Return raw and processed data directory paths."""
    base = Path(__file__).parent.parent
    raw = base / config["data"]["raw_dir"]
    processed = base / config["data"]["processed_dir"]
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    return raw, processed
