from pathlib import Path
from typing import Union, Dict, Any

# Define the root of your project relative to this file
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

def relative_to_root(*path_parts: str) -> Path:
    """
    Resolve a path relative to the project root.

    Example:
        relative_to_root("data", "example.txt")
        => /path/to/project/data/example.txt
    """
    return PROJECT_ROOT.joinpath(*path_parts)

def resolve_path(path_value: Union[str, Path]) -> Path:
    """Convert a string or Path to an absolute Path."""
    if isinstance(path_value, str):
        path = Path(path_value)
    else:
        path = path_value
    return path if path.is_absolute() else relative_to_root(str(path))

def resolve_paths(config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve all path values in a dictionary (keys containing 'path' or 'dir')."""
    resolved_config = config.copy()
    
    for key, value in config.items():
        if isinstance(value, str) and ('path' in key.lower() or 'dir' in key.lower()):
            resolved_config[key] = resolve_path(value)
        elif isinstance(value, dict):
            resolved_config[key] = resolve_paths(value)
    
    return resolved_config