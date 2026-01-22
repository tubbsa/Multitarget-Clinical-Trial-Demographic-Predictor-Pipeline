# src/harry_ml/infer.py
"""
Inference wrapper for the Harry ML pipeline.

This module performs deterministic inference given:
- a cleaned trial-level dataframe (df_clean)
- a saved artifacts directory containing:
    - main multi-target model
    - OneHotEncoder
    - schema_manifest.json (cat_cols, num_cols, optional cat_feature_names, text config)
    - optional hurdle models for sparse targets

It builds features using:
- harry_ml.features_structured (categorical + numeric)
- harry_ml.features_text (merged protocol narrative + embeddings)

Then assembles the final feature matrix and runs:
- main model prediction for dense targets
- hurdle prediction for sparse targets (if present in artifacts)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from harry_ml.artifacts import LoadedArtifacts, load_artifacts
from harry_ml.features_structured import build_numeric_frame, build_categorical_frame, encode_categoricals
from harry_ml.features_text import build_text_embeddings, DEFAULT_MODEL_NAME, DEFAULT_TEXT_COLS
from harry_ml.hurdle import HurdleBundle, predict_hurdle_targets, predict_hurdle


# -----------------------------
# Result container
# -----------------------------
@dataclass(frozen=True)
class InferenceResult:
    preds: pd.DataFrame
    features_shape: Tuple[int, int]
    used_hurdles: bool
    artifact_dir: str


# -----------------------------
# Feature assembly
# -----------------------------
def _assemble_feature_matrix(
    df_num: pd.DataFrame,
    df_cat_enc: pd.DataFrame,
    text_emb: np.ndarray,
) -> np.ndarray:
    """
    Concatenate numeric + categorical + text embeddings into one dense matrix.

    Notes:
    - If your training used sparse concatenation, update this accordingly.
    - This implementation uses dense concat for simplicity and portability.
    """
    X_num = df_num.to_numpy(dtype=float)
    X_cat = df_cat_enc.to_numpy(dtype=float)

    if text_emb.ndim != 2:
        raise ValueError(f"text_emb must be 2D, got shape {text_emb.shape}")

    if X_num.shape[0] != X_cat.shape[0] or X_num.shape[0] != text_emb.shape[0]:
        raise ValueError(
            "Row count mismatch in features:\n"
            f"X_num: {X_num.shape}\n"
            f"X_cat: {X_cat.shape}\n"
            f"text:  {text_emb.shape}"
        )

    X = np.concatenate([X_num, X_cat, text_emb], axis=1)
    return X


# -----------------------------
# Hurdle model loading
# -----------------------------
def _parse_hurdle_dict(hurdle_flat: Optional[Dict[str, Any]]) -> Dict[str, HurdleBundle]:
    """
    artifacts.hurdle is saved as flat dict keys like:
      - "AIAN_presence", "AIAN_reg"
    Convert to HurdleBundle per target.
    """
    if not hurdle_flat:
        return {}

    grouped: Dict[str, Dict[str, Any]] = {}
    for key, obj in hurdle_flat.items():
        # key like "AIAN_presence"
        if "_" not in key:
            continue
        tgt, part = key.rsplit("_", 1)
        grouped.setdefault(tgt, {})[part] = obj

    bundles: Dict[str, HurdleBundle] = {}
    for tgt, parts in grouped.items():
        if "presence" in parts and "reg" in parts:
            bundles[tgt] = HurdleBundle(
                presence_model=parts["presence"],
                reg_model=parts["reg"],
                threshold=0.5,
                zero_value=0.0,
            )
    return bundles


# -----------------------------
# Main inference API
# -----------------------------
def predict_from_artifacts(
    df_clean: pd.DataFrame,
    artifacts_dir: str,
    *,
    dense_cats: bool = True,
    hard_gate_hurdles: bool = False,
    clip01: bool = True,
) -> InferenceResult:
    """
    Run end-to-end inference.

    Args:
      df_clean: cleaned dataframe (output of harry_ml.cleaning.clean_trials)
      artifacts_dir: directory created by harry_ml.artifacts.save_artifacts
      dense_cats: encode categoricals to dense dataframe (recommended if you trained dense)
      hard_gate_hurdles: whether to hard gate sparse targets (default False uses p*y_cond)
      clip01: clip predictions into [0,1]

    Returns:
      InferenceResult containing per-target predictions as a DataFrame.
    """
    artifacts: LoadedArtifacts = load_artifacts(artifacts_dir, load_hurdles=True)
    schema = artifacts.schema

    # ----- schema fields -----
    cat_cols: List[str] = schema.get("cat_cols") or []
    num_cols: List[str] = schema.get("num_cols") or []

    # text config (optional; fallback to defaults)
    text_cols = schema.get("text_cols") or list(DEFAULT_TEXT_COLS)
    text_model_name = schema.get("text_model_name") or DEFAULT_MODEL_NAME

    targets = schema.get("targets")  # optional list[str]

    # ----- build structured features (transform using fitted encoder) -----
    df_cat = build_categorical_frame(df_clean, cat_cols)
    df_cat_enc = encode_categoricals(df_cat, artifacts.encoder, dense_dataframe=dense_cats)

    df_num = build_numeric_frame(df_clean, num_cols)

    # ----- build text embeddings -----
    text_res = build_text_embeddings(
        df_clean,
        text_cols=text_cols,
        model_name=text_model_name,
    )
    text_emb = text_res.embeddings

    # ----- assemble final feature matrix -----
    X = _assemble_feature_matrix(df_num, df_cat_enc, text_emb)

    # ----- main model prediction -----
    if not hasattr(artifacts.model, "predict"):
        raise TypeError("Loaded main model does not implement predict().")

    y_pred = np.asarray(artifacts.model.predict(X))

    # CatBoost multi-target can return shape (n, t) or list-like; coerce
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # Build target column names
    if isinstance(targets, list) and len(targets) == y_pred.shape[1]:
        colnames = targets
    else:
        colnames = [f"target_{i}" for i in range(y_pred.shape[1])]

    preds_df = pd.DataFrame(y_pred, columns=colnames)

    # ----- optional hurdle override for sparse targets -----
    hurdle_bundles = _parse_hurdle_dict(artifacts.hurdle)
    used_hurdles = False
    if hurdle_bundles:
        used_hurdles = True
        hurdle_preds = predict_hurdle_targets(
            hurdle_bundles,
            X,
            hard_gate=hard_gate_hurdles,
            clip01=clip01,
        )
        # Replace columns if names match
        # If your schema uses exact target names, set those keys accordingly when saving.
        for tgt, hp in hurdle_preds.items():
            # Try direct match
            if tgt in preds_df.columns:
                preds_df[tgt] = hp
            else:
                # also try common naming variants
                candidates = [c for c in preds_df.columns if tgt.lower() in c.lower()]
                if candidates:
                    preds_df[candidates[0]] = hp

    # ----- clip / renormalize -----
    if clip01:
        preds_df = preds_df.clip(lower=0.0, upper=1.0)

    # If your pipeline enforces male+female=1, do it here if target names exist
    sex_cols = [c for c in preds_df.columns if c.lower() in ("male", "female", "male %", "female %", "male_pct", "female_pct")]
    if len(sex_cols) >= 2:
        # heuristic: pick first two
        a, b = sex_cols[0], sex_cols[1]
        denom = preds_df[a] + preds_df[b]
        denom = denom.replace(0.0, 1.0)
        preds_df[a] = preds_df[a] / denom
        preds_df[b] = preds_df[b] / denom

    return InferenceResult(
        preds=preds_df,
        features_shape=X.shape,
        used_hurdles=used_hurdles,
        artifact_dir=artifacts_dir,
    )
