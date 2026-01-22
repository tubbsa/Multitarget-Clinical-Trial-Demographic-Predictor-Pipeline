# src/harry_ml/artifacts.py
"""
Artifact IO for the Harry ML pipeline.

Purpose
- Save *trained* preprocessing + models + schema into a single artifact directory
- Load them back deterministically for inference / evaluation
- Write a schema/manifest JSON so feature order & invariants are explicit

What this module supports (generic, robust):
- joblib for sklearn-like objects (OneHotEncoder, scalers, etc.)
- pickle fallback for odd objects
- numpy arrays (optional)
- JSON manifest for schema and metadata

It does NOT:
- train models
- compute features
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np


# -----------------------------
# Filenames (canonical)
# -----------------------------
ENCODER_FNAME = "encoder.pkl"
SCHEMA_FNAME = "schema_manifest.json"
META_FNAME = "artifact_meta.json"

# If you have a single multitarget CatBoost model:
MODEL_FNAME = "catboost_multitarget.cbm"

# If you have hurdle components, store them under a folder
HURDLE_DIR = "hurdle_models"  # e.g., hurdle_models/AIAN_presence.pkl etc.


# -----------------------------
# Dataclasses
# -----------------------------
@dataclass(frozen=True)
class LoadedArtifacts:
    """
    Bundle returned by load_artifacts().
    """
    model: Any
    encoder: Any
    schema: Dict[str, Any]
    meta: Dict[str, Any]
    hurdle: Optional[Dict[str, Any]] = None


# -----------------------------
# Low-level IO helpers
# -----------------------------
def ensure_dir(path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_pickle(obj: Any, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(in_path: str | Path) -> Any:
    with open(in_path, "rb") as f:
        return pickle.load(f)


def save_joblib(obj: Any, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, out_path)


def load_joblib(in_path: str | Path) -> Any:
    return joblib.load(in_path)


def save_json(data: Dict[str, Any], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def load_json(in_path: str | Path) -> Dict[str, Any]:
    with open(in_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_numpy(array: np.ndarray, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, array)


def load_numpy(in_path: str | Path) -> np.ndarray:
    return np.load(in_path, allow_pickle=False)


# -----------------------------
# Schema / manifest helpers
# -----------------------------
def build_schema_manifest(
    *,
    cat_cols: list[str],
    num_cols: list[str],
    cat_feature_names: Optional[list[str]] = None,
    text_model_name: Optional[str] = None,
    text_cols: Optional[list[str]] = None,
    embedding_dim: Optional[int] = None,
    geo_numeric_cols: Optional[list[str]] = None,
    targets: Optional[list[str]] = None,
) -> Dict[str, Any]:
    """
    Create a schema manifest that makes feature ordering explicit.

    cat_feature_names should come from encoder.get_feature_names_out(cat_cols).
    """
    manifest: Dict[str, Any] = {
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "cat_cols": list(cat_cols),
        "num_cols": list(num_cols),
        "cat_feature_names": list(cat_feature_names) if cat_feature_names is not None else None,
        "text_model_name": text_model_name,
        "text_cols": list(text_cols) if text_cols is not None else None,
        "embedding_dim": embedding_dim,
        "geo_numeric_cols": list(geo_numeric_cols) if geo_numeric_cols is not None else None,
        "targets": list(targets) if targets is not None else None,
    }
    return manifest


def validate_schema_manifest(schema: Dict[str, Any]) -> None:
    """
    Hard-fail if schema is missing key fields needed for stable inference.
    """
    required = ["cat_cols", "num_cols"]
    missing = [k for k in required if k not in schema]
    if missing:
        raise ValueError(f"Schema manifest missing required fields: {missing}")

    if not isinstance(schema["cat_cols"], list) or not isinstance(schema["num_cols"], list):
        raise ValueError("Schema manifest fields cat_cols/num_cols must be lists.")

    # Optional but strongly recommended sanity checks
    geo = schema.get("geo_numeric_cols")
    if geo is not None:
        bad = set(geo).intersection(set(schema["cat_cols"]))
        if bad:
            raise ValueError(f"Schema invalid: geographic cols appear in cat_cols: {sorted(bad)}")


# -----------------------------
# Model IO (CatBoost optional)
# -----------------------------
def save_catboost_model(model: Any, out_path: str | Path) -> None:
    """
    Saves a CatBoost model if it supports .save_model(), else joblib.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save_model"):
        model.save_model(str(out_path))
    else:
        # fallback
        save_joblib(model, out_path.with_suffix(".pkl"))


def load_catboost_model(in_path: str | Path) -> Any:
    """
    Loads a CatBoost model if CatBoost is installed and file is .cbm.
    Otherwise attempts joblib.
    """
    in_path = Path(in_path)
    if in_path.suffix.lower() == ".cbm":
        try:
            from catboost import CatBoost  # type: ignore
        except Exception as e:
            raise ImportError("CatBoost not available to load .cbm model.") from e
        m = CatBoost()
        m.load_model(str(in_path))
        return m

    return load_joblib(in_path)


# -----------------------------
# High-level save/load API
# -----------------------------
def save_artifacts(
    out_dir: str | Path,
    *,
    model: Any,
    encoder: Any,
    schema_manifest: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
    hurdle_models: Optional[Dict[str, Any]] = None,
    model_fname: str = MODEL_FNAME,
) -> Path:
    """
    Save all artifacts needed for inference.

    Writes:
      - model file (CatBoost .cbm if supported, otherwise joblib .pkl)
      - encoder.pkl (joblib)
      - schema_manifest.json
      - artifact_meta.json
      - optional hurdle models under hurdle_models/
    """
    out_dir = ensure_dir(out_dir)

    # Schema
    validate_schema_manifest(schema_manifest)
    save_json(schema_manifest, out_dir / SCHEMA_FNAME)

    # Meta
    meta_out = dict(meta or {})
    meta_out.setdefault("saved_utc", datetime.utcnow().isoformat(timespec="seconds") + "Z")
    save_json(meta_out, out_dir / META_FNAME)

    # Encoder
    save_joblib(encoder, out_dir / ENCODER_FNAME)

    # Main model
    model_path = out_dir / model_fname
    # If caller passes ".cbm" and model supports, it'll save as cbm
    save_catboost_model(model, model_path)

    # Hurdle models (optional)
    if hurdle_models:
        hdir = ensure_dir(out_dir / HURDLE_DIR)
        # Expect a dict like {"AIAN": {"presence": mdl1, "reg": mdl2}, ...}
        for target, bundle in hurdle_models.items():
            if not isinstance(bundle, dict):
                # Save as one object
                save_joblib(bundle, hdir / f"{target}.pkl")
                continue
            for k, obj in bundle.items():
                # Use joblib for sklearn/catboost-like objects
                save_joblib(obj, hdir / f"{target}_{k}.pkl")

    return out_dir


def load_artifacts(
    in_dir: str | Path,
    *,
    model_fname: str = MODEL_FNAME,
    load_hurdles: bool = True,
) -> LoadedArtifacts:
    """
    Load artifacts from directory produced by save_artifacts().
    """
    in_dir = Path(in_dir).expanduser().resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"Artifact directory not found: {in_dir}")

    schema = load_json(in_dir / SCHEMA_FNAME)
    validate_schema_manifest(schema)

    meta = load_json(in_dir / META_FNAME) if (in_dir / META_FNAME).exists() else {}

    encoder = load_joblib(in_dir / ENCODER_FNAME)

    # main model: try cbm first, else fallback
    model_path = in_dir / model_fname
    if model_path.exists():
        model = load_catboost_model(model_path)
    else:
        # fallback to joblib model (if you saved with fallback)
        pkl_guess = (in_dir / model_fname).with_suffix(".pkl")
        if not pkl_guess.exists():
            raise FileNotFoundError(f"Model file not found: {model_path} (or {pkl_guess})")
        model = load_joblib(pkl_guess)

    hurdle: Optional[Dict[str, Any]] = None
    if load_hurdles:
        hdir = in_dir / HURDLE_DIR
        if hdir.exists() and hdir.is_dir():
            hurdle = {}
            for p in sorted(hdir.glob("*.pkl")):
                key = p.stem  # e.g., "AIAN_presence"
                hurdle[key] = load_joblib(p)

    return LoadedArtifacts(
        model=model,
        encoder=encoder,
        schema=schema,
        meta=meta,
        hurdle=hurdle,
    )


def print_artifact_summary(artifacts_dir: str | Path) -> None:
    """
    Convenience function to show what exists in an artifact directory.
    """
    p = Path(artifacts_dir).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(p)

    files = sorted([x.name for x in p.iterdir() if x.is_file()])
    print("Artifacts:", str(p))
    for f in files:
        print(" -", f)
    hdir = p / HURDLE_DIR
    if hdir.exists():
        print("Hurdle models:", str(hdir))
        for f in sorted(hdir.glob("*.pkl")):
            print(" -", f.name)
