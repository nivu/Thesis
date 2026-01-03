import yaml
from config.paths import relative_to_root, resolve_paths

def load_config(path: str) -> dict:
    """
    Load a YAML configuration file from the project root.

    Args:
        path (str): Relative path to the YAML file.

    Returns:
        dict: Loaded configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the YAML is malformed.
    """
    config_path = relative_to_root(path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise RuntimeError(f"YAML parsing error in {config_path}: {e}")

    config = resolve_paths(config)

    return config