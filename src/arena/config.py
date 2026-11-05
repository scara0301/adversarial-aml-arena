"""Config loading. One YAML file, dotted-path access, CLI-overridable.

An overlay can be laid on top by setting ``ARENA_CONFIG`` to another YAML file;
its keys replace the defaults, section by section. Phase 5 uses this to run the
unchanged Phase 1-3 scripts on the real IBM data (``config/ibm.yaml``), so the
real-data results come from exactly the code that produced the synthetic ones.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

#: Repo root, resolved from this file's location so scripts work from anywhere.
ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CONFIG = ROOT / "config" / "default.yaml"


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Config:
    """Nested dict with dotted-path lookup.

    Thin on purpose: the value of a config object here is that every script
    reads the same defaults, not that it validates schemas.
    """

    __slots__ = ("_data",)

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        p = Path(path) if path else DEFAULT_CONFIG
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        overlay = os.environ.get("ARENA_CONFIG")
        if overlay:
            o = Path(overlay)
            o = o if o.is_absolute() else ROOT / o
            data = _merge(data, yaml.safe_load(o.read_text(encoding="utf-8")) or {})
        return cls(data)

    def get(self, dotted: str, default: Any = None) -> Any:
        """``cfg.get("detector.lightgbm.num_leaves")``."""
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def section(self, dotted: str) -> dict:
        value = self.get(dotted, {})
        return dict(value) if isinstance(value, dict) else {}

    def path(self, dotted: str) -> Path:
        """A config path resolved against the repo root."""
        value = self.get(dotted)
        if value is None:
            raise KeyError(f"no path configured at {dotted!r}")
        p = Path(value)
        return p if p.is_absolute() else ROOT / p

    def as_dict(self) -> dict:
        return dict(self._data)
