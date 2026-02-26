"""
config/config.py

Reproducible YAML config loader.

- Loads config/config.yaml (or a given path)
- Validates required keys
- Resolves relative paths relative to project root
"""

from __future__ import annotations

from pathlib import Path
import yaml


def _project_root_from_config_path(config_path: Path) -> Path:
    """
    Given .../config/config.yaml, return the project root (parent of config/).
    """
    return config_path.resolve().parent.parent


def load_yaml_config(config_path: str | Path = "config/config.yaml") -> dict:
    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a mapping (dict) at top-level.")

    return cfg


def get_paths(cfg: dict, config_path: str | Path = "config/config.yaml"):
    if "paths" not in cfg:
        raise KeyError("Missing required config section: paths")

    paths = cfg["paths"]
    if not isinstance(paths, dict):
        raise ValueError("Config key 'paths' must be a mapping (dict).")

    for k in ("data_folder", "clinical_data"):
        if k not in paths or not isinstance(paths[k], str) or not paths[k].strip():
            raise KeyError(f"Missing or invalid required config key: paths.{k}")

    config_path = Path(config_path)
    project_root = _project_root_from_config_path(config_path)

    def resolve(p: str) -> Path:
        pp = Path(p).expanduser()
        if pp.is_absolute():
            return pp
        return (project_root / pp).resolve()

    data_folder = resolve(paths["data_folder"])
    clinical_data = resolve(paths["clinical_data"])

    if not data_folder.is_dir():
        raise FileNotFoundError(f"paths.data_folder is not a directory: {data_folder}")

    if not clinical_data.is_file():
        raise FileNotFoundError(f"paths.clinical_data is not a file: {clinical_data}")

    return str(data_folder), str(clinical_data)


def load_paths(config_path: str | Path = "config/config.yaml"):
    cfg = load_yaml_config(config_path)
    return get_paths(cfg, config_path=config_path)