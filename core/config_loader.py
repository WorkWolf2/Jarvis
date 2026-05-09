"""
Configuration loader with dot-notation access and runtime overrides.
"""

import json
import os
from pathlib import Path
from typing import Any, Optional
from copy import deepcopy


class ConfigLoader:
    """
    Loads and manages configuration from JSON file.
    Supports dot-notation access: config.get("llm.model")
    Supports environment variable overrides: JARVIS_LLM_MODEL
    """

    def __init__(self, config_path: Path) -> None:
        self._path = config_path
        self._data: dict = {}
        self._overrides: dict = {}
        self._load()
        self._apply_env_overrides()

    def _load(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(f"Config file not found: {self._path}")
        with open(self._path, "r") as f:
            self._data = json.load(f)

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides (JARVIS_SECTION_KEY format)."""
        prefix = "JARVIS_"
        for key, value in os.environ.items():
            if key.startswith(prefix):
                dot_key = key[len(prefix):].lower().replace("_", ".", 1)
                self.set(dot_key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value by dot-notation key."""
        # Check overrides first
        if key in self._overrides:
            return self._overrides[key]

        parts = key.split(".")
        current = self._data
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def set(self, key: str, value: Any) -> None:
        """Set a runtime override by dot-notation key."""
        self._overrides[key] = value

    def get_section(self, section: str) -> dict:
        """Get an entire config section as a dict."""
        result = deepcopy(self._data.get(section, {}))
        # Apply any matching overrides
        prefix = section + "."
        for key, val in self._overrides.items():
            if key.startswith(prefix):
                subkey = key[len(prefix):]
                result[subkey] = val
        return result

    def reload(self) -> None:
        """Reload config from disk."""
        self._load()

    def to_dict(self) -> dict:
        """Get full config as dict."""
        result = deepcopy(self._data)
        for key, val in self._overrides.items():
            parts = key.split(".")
            current = result
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = val
        return result

    def __repr__(self) -> str:
        return f"ConfigLoader(path={self._path})"