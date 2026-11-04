"""Cached loaders for the report artifacts the dashboard reads.

Everything here reads files that scripts/train_detector.py and
scripts/evolve_adversary.py already wrote to reports/. The dashboard never
retrains or re-evolves anything from this module -- it explains what those
two scripts already produced. See live.py for the one place that touches the
trained detector directly, for the interactive genome explorer.

Caching note: results are cached by file path string, not by file content or
mtime. If you regenerate reports/ while the dashboard is running, restart
Streamlit (or press "r" in the running app) to pick up the new numbers --
otherwise you'll keep seeing the cached ones. Fine for a demo dashboard that
is launched after the pipeline finishes; not fine for a live-updating one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from arena.config import Config


@st.cache_resource
def get_config() -> Config:
    return Config.load()


@st.cache_data
def load_json(path: str) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data
def load_csv(path: str) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, parse_dates=["timestamp"])


def phase1_metrics() -> dict | None:
    cfg = get_config()
    return load_json(str(cfg.path("paths.reports") / "phase1_metrics.json"))


def phase2_metrics() -> dict | None:
    cfg = get_config()
    return load_json(str(cfg.path("paths.reports") / "phase2_metrics.json"))


def phase3_metrics() -> dict | None:
    cfg = get_config()
    return load_json(str(cfg.path("paths.reports") / "phase3_metrics.json"))


def phase2_example_strategy() -> pd.DataFrame | None:
    cfg = get_config()
    return load_csv(str(cfg.path("paths.reports") / "phase2_best_strategy_example.csv"))


def report_image_path(name: str) -> Path:
    cfg = get_config()
    return cfg.path("paths.reports") / name


def report_image(name: str) -> str | None:
    """The image path as a string, or None if it hasn't been generated yet."""
    p = report_image_path(name)
    return str(p) if p.exists() else None
