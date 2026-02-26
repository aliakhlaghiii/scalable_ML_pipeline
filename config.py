"""
config.py

Minimal YAML config loader for this project.

- Loads config.yaml
- Validates required keys
- Ensures paths exist
"""

import os
import yaml


def load_yaml_config(config_path="config.yaml"):
    """
    Load a YAML configuration file and return it as a dict.

    Parameters
    ----------
    config_path : str
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If YAML content is empty or not a mapping.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError("Config file not found: {}".format(config_path))

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        raise ValueError("Config is empty: {}".format(config_path))

    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a YAML mapping (dict).")

    return cfg


def get_paths(cfg):
    """
    Extract and validate the 'paths' section from the loaded config.

    Parameters
    ----------
    cfg : dict
        Parsed configuration.

    Returns
    -------
    tuple
        (data_folder, clinical_data) as strings.

    Raises
    ------
    KeyError
        If required keys are missing.
    ValueError
        If values are invalid types/empty.
    FileNotFoundError
        If configured paths do not exist.
    """
    if "paths" not in cfg:
        raise KeyError("Missing required config section: paths")

    paths = cfg["paths"]
    if not isinstance(paths, dict):
        raise ValueError("Config key 'paths' must be a mapping (dict).")

    if "data_folder" not in paths:
        raise KeyError("Missing required config key: paths.data_folder")
    if "clinical_data" not in paths:
        raise KeyError("Missing required config key: paths.clinical_data")

    data_folder = paths["data_folder"]
    clinical_data = paths["clinical_data"]

    if not isinstance(data_folder, str) or not data_folder.strip():
        raise ValueError("paths.data_folder must be a non-empty string.")
    if not isinstance(clinical_data, str) or not clinical_data.strip():
        raise ValueError("paths.clinical_data must be a non-empty string.")

    # These checks prevent silent errors later (monkey-proof).
    if not os.path.isdir(data_folder):
        raise FileNotFoundError("paths.data_folder is not a directory: {}".format(data_folder))

    if not os.path.isfile(clinical_data):
        raise FileNotFoundError("paths.clinical_data is not a file: {}".format(clinical_data))

    return data_folder, clinical_data


def load_paths(config_path="config.yaml"):
    """
    Convenience wrapper: load config.yaml and return validated paths.

    Parameters
    ----------
    config_path : str
        Path to YAML config.

    Returns
    -------
    tuple
        (data_folder, clinical_data) as strings.
    """
    cfg = load_yaml_config(config_path)
    return get_paths(cfg)
