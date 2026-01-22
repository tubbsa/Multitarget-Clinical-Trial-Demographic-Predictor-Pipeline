# src/harry_ml/io.py
"""
Data I/O utilities for the Harry ML pipeline.

This module is intentionally conservative:
- It does NOT call the ClinicalTrials.gov API (that belongs in a separate ingestion layer)
- It provides safe, predictable loaders/savers for local datasets used in training/inference
- It avoids mutating inputs and normalizes common file formats

Supported:
- CSV
- Parquet
- Feather
- Pickle/joblib (generic objects)
- JSON (dict config/metadata)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import pandas as pd


# -----------------------------
# Path helpers
# -----------------------------
def as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def ensure_parent_dir(path: str | Path) -> Path:
    p = as_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------
# DataFrame loaders
# -----------------------------
def load_dataframe(path: str | Path, *, dtype_backend: Optional[str] = None) -> pd.DataFrame:
    """
    Load a DataFrame from CSV/Parquet/Feather.

    dtype_backend: pandas >=2.0 option, e.g. "pyarrow" for memory/NA robustness.
    """
    p = as_path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in (".parquet", ".pq"):
        # dtype_backend supported in pandas read_parquet in newer versions; safe to ignore if not available
        try:
            return pd.read_parquet(p, dtype_backend=dtype_backend)  # type: ignore[arg-type]
        except TypeError:
            return pd.read_parquet(p)
    if suffix == ".feather":
        return pd.read_feather(p)

    raise ValueError(f"Unsupported dataframe file type: {suffix} ({p})")


def save_dataframe(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    """
    Save a DataFrame to CSV/Parquet/Feather based on file extension.
    """
    p = ensure_parent_dir(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        df.to_csv(p, index=index)
        return p
    if suffix in (".parquet", ".pq"):
        df.to_parquet(p, index=index)
        return p
    if suffix == ".feather":
        if index:
            # feather doesn't store index reliably; reset
            df.reset_index(drop=False).to_feather(p)
        else:
            df.reset_index(drop=True).to_feather(p)
        return p

    raise ValueError(f"Unsupported dataframe file type: {suffix} ({p})")


# -----------------------------
# Generic object IO
# -----------------------------
def save_joblib(obj: Any, path: str | Path) -> Path:
    p = ensure_parent_dir(path)
    joblib.dump(obj, p)
    return p


def load_joblib(path: str | Path) -> Any:
    p = as_path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return joblib.load(p)


def save_json(data: Dict[str, Any], path: str | Path) -> Path:
    p = ensure_parent_dir(path)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return p


def load_json(path: str | Path) -> Dict[str, Any]:
    p = as_path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Convenience: project paths
# -----------------------------
def project_root(start: str | Path = ".") -> Path:
    """
    Walk upwards to find the repo root (heuristic: contains pyproject.toml or .git).
    Useful when running scripts from arbitrary working directories.
    """
    p = as_path(start)
    if p.is_file():
        p = p.parent

    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent

    # fallback to start dir
    return p


def data_dir(start: str | Path = ".") -> Path:
    """
    Returns <repo_root>/data
    """
    return project_root(start) / "data"


def artifacts_dir(start: str | Path = ".") -> Path:
    """
    Returns <repo_root>/artifacts
    """
    return project_root(start) / "artifacts"
