# src/harry_ml/train.py
"""
Training entry points for the Harry ML pipeline.

This module:
1) builds structured features (numeric + one-hot categoricals)
2) builds text embeddings (MiniLM)
3) concatenates into final feature matrix X
4) trains a multi-target CatBoost regressor for dense targets
5) optionally trains hurdle models for sparse targets
6) writes artifacts (model + encoder + schema manifest + meta)

Assumptions (match your pipeline):
- Inputs are already cleaned (use harry_ml.cleaning.clean_trials first)
- Targets are proportions in [0,1]
- CatBoost supports multi-target regression via CatBoostRegressor with loss_function="MultiRMSE"
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from harry_ml.config import (
    RANDOM_SEED,
    TRAIN_FRAC,
    TEST_FRAC,
    HOLDOUT_FRAC,
    CAT_COLS,
    NUMERIC_COLS,
    REQUIRED_GEO_NUMERIC_COLS,
    TEXT_COLS,
    TEXT_EMBEDDING_MODEL,
    TEXT_EMBEDDING_BATCH_SIZE,
    TEXT_EMBEDDING_MAX_LENGTH,
    TEXT_EMBEDDING_NORMALIZE,
    validate_schema,
)
from harry_ml.features_structured import (
    build_structured_features_fit,
    build_categorical_frame,
    build_numeric_frame,
    encode_categoricals,
)
from harry_ml.features_text import build_text_embeddings
from harry_ml.split import split_indices, SplitResult
from harry_ml.hurdle import fit_hurdle_models, HurdleBundle
from harry_ml.artifacts import (
    save_artifacts,
    build_schema_manifest,
)


# -----------------------------
# Dataclasses
# -----------------------------
@dataclass(frozen=True)
class TrainOutputs:
    model: Any
    encoder: Any
    schema_manifest: Dict[str, Any]
    split: SplitResult
    feature_shape: Tuple[int, int]
    target_names: List[str]
    hurdle_models: Optional[Dict[str, Dict[str, Any]]]
    artifacts_dir: Optional[str]


# -----------------------------
# Feature assembly
# -----------------------------
def assemble_features_fit(
    df_clean: pd.DataFrame,
    *,
    cat_cols: Sequence[str] = CAT_COLS,
    num_cols: Sequence[str] = NUMERIC_COLS,
    text_cols: Sequence[str] = TEXT_COLS,
    text_model_name: str = TEXT_EMBEDDING_MODEL,
) -> Tuple[np.ndarray, Any, Dict[str, Any]]:
    """
    Train-time feature build:
    - fit OneHotEncoder on categoricals
    - build numeric frame in canonical order
    - build text embeddings
    - concatenate dense: [X_num | X_cat | X_text]

    Returns:
      X: np.ndarray (n, d)
      encoder: fitted OneHotEncoder
      feature_meta: dict with cat_feature_names, embedding_dim
    """
    # structured (fit)
    sf = build_structured_features_fit(
        df_clean,
        cat_cols=list(cat_cols),
        num_cols=list(num_cols),
        handle_unknown="ignore",
        sparse_output=True,
        dense_cats=True,      # dense cat DF so concat is easy & deterministic
        fillna_value=0.0,
    )
    df_cat_enc = sf.df_cat_enc
    df_num = sf.df_num
    encoder = sf.encoder

    # text
    text_res = build_text_embeddings(
        df_clean,
        text_cols=list(text_cols),
        model_name=text_model_name,
        batch_size=TEXT_EMBEDDING_BATCH_SIZE,
        max_length=TEXT_EMBEDDING_MAX_LENGTH,
        normalize=TEXT_EMBEDDING_NORMALIZE,
    )
    X_text = text_res.embeddings
    emb_dim = int(X_text.shape[1]) if X_text.ndim == 2 else 0

    # concat
    X_num = df_num.to_numpy(dtype=float)
    X_cat = df_cat_enc.to_numpy(dtype=float)

    if X_text.ndim != 2:
        raise ValueError(f"Text embeddings must be 2D, got {X_text.shape}.")
    if X_num.shape[0] != X_cat.shape[0] or X_num.shape[0] != X_text.shape[0]:
        raise ValueError(
            "Row count mismatch assembling features:\n"
            f"X_num:  {X_num.shape}\n"
            f"X_cat:  {X_cat.shape}\n"
            f"X_text: {X_text.shape}"
        )

    X = np.concatenate([X_num, X_cat, X_text], axis=1)

    feature_meta = {
        "cat_feature_names": list(encoder.get_feature_names_out(list(cat_cols))),
        "embedding_dim": emb_dim,
    }
    return X, encoder, feature_meta


def assemble_features_transform(
    df_clean: pd.DataFrame,
    *,
    encoder: Any,
    cat_cols: Sequence[str] = CAT_COLS,
    num_cols: Sequence[str] = NUMERIC_COLS,
    text_cols: Sequence[str] = TEXT_COLS,
    text_model_name: str = TEXT_EMBEDDING_MODEL,
) -> np.ndarray:
    """
    Inference/eval-time feature build using a pre-fit encoder.
    """
    df_cat = build_categorical_frame(df_clean, list(cat_cols))
    df_cat_enc = encode_categoricals(df_cat, encoder, dense_dataframe=True)

    df_num = build_numeric_frame(df_clean, list(num_cols))

    text_res = build_text_embeddings(
        df_clean,
        text_cols=list(text_cols),
        model_name=text_model_name,
        batch_size=TEXT_EMBEDDING_BATCH_SIZE,
        max_length=TEXT_EMBEDDING_MAX_LENGTH,
        normalize=TEXT_EMBEDDING_NORMALIZE,
    )
    X_text = text_res.embeddings

    X_num = df_num.to_numpy(dtype=float)
    X_cat = df_cat_enc.to_numpy(dtype=float)

    if X_text.ndim != 2:
        raise ValueError(f"Text embeddings must be 2D, got {X_text.shape}.")
    if X_num.shape[0] != X_cat.shape[0] or X_num.shape[0] != X_text.shape[0]:
        raise ValueError(
            "Row count mismatch assembling features:\n"
            f"X_num:  {X_num.shape}\n"
            f"X_cat:  {X_cat.shape}\n"
            f"X_text: {X_text.shape}"
        )

    return np.concatenate([X_num, X_cat, X_text], axis=1)


# -----------------------------
# Target extraction
# -----------------------------
def extract_targets(
    df_clean: pd.DataFrame,
    target_cols: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Returns:
      Y: np.ndarray (n, t) clipped to [0,1]
      target_names: list[str]
    """
    missing = [c for c in target_cols if c not in df_clean.columns]
    if missing:
        raise KeyError(f"Missing target columns in df_clean: {missing}")

    Y = df_clean[list(target_cols)].copy()
    for c in target_cols:
        Y[c] = pd.to_numeric(Y[c], errors="coerce")
    Y = Y.fillna(0.0)

    Y_arr = Y.to_numpy(dtype=float)
    Y_arr = np.clip(Y_arr, 0.0, 1.0)

    return Y_arr, list(target_cols)


# -----------------------------
# Model training
# -----------------------------
def _make_catboost_regressor(params: Optional[Dict[str, Any]] = None):
    """
    Creates a CatBoostRegressor configured for multi-target regression.
    """
    try:
        from catboost import CatBoostRegressor  # type: ignore
    except Exception as e:
        raise ImportError("CatBoost is required for train.py. Install catboost.") from e

    base = {
        "loss_function": "MultiRMSE",
        "random_seed": RANDOM_SEED,
        "verbose": False,
        "iterations": 2000,
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 3.0,
        "task_type": "CPU",
    }
    if params:
        base.update(params)

    return CatBoostRegressor(**base)


def train_multitarget_catboost(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    *,
    params: Optional[Dict[str, Any]] = None,
):
    model = _make_catboost_regressor(params=params)
    model.fit(X_train, Y_train)
    return model


# -----------------------------
# Hurdle training
# -----------------------------
def train_hurdles(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    target_names: List[str],
    *,
    sparse_targets: Sequence[str],
    presence_model_factory,
    reg_model_factory,
    eps: float = 0.0,
) -> Dict[str, Dict[str, Any]]:
    """
    Fit hurdle models for specified sparse targets.

    Returns dict suitable for save_artifacts(hurdle_models=...):
      { "AIAN": {"presence": mdl1, "reg": mdl2}, ... }
    """
    name_to_idx = {n: i for i, n in enumerate(target_names)}
    hurdles: Dict[str, Dict[str, Any]] = {}

    for tgt in sparse_targets:
        if tgt not in name_to_idx:
            continue
        j = name_to_idx[tgt]
        y = Y_train[:, j]

        presence_model = presence_model_factory()
        reg_model = reg_model_factory()

        bundle = fit_hurdle_models(
            X_train,
            y,
            presence_model=presence_model,
            reg_model=reg_model,
            eps=eps,
            reg_fit_only_on_present=True,
        )
        hurdles[tgt] = {"presence": bundle.presence_model, "reg": bundle.reg_model}

    return hurdles


# -----------------------------
# Top-level training pipeline
# -----------------------------
def train_pipeline(
    df_clean: pd.DataFrame,
    *,
    target_cols: Sequence[str],
    key_col: Optional[str] = "nct_id",
    catboost_params: Optional[Dict[str, Any]] = None,
    # Hurdle config
    sparse_targets: Optional[Sequence[str]] = None,
    hurdle_presence_model_factory=None,
    hurdle_reg_model_factory=None,
    hurdle_eps: float = 0.0,
    # Artifact output
    artifacts_out_dir: Optional[str] = None,
) -> TrainOutputs:
    """
    End-to-end training with optional artifact writing.

    If artifacts_out_dir is provided, writes artifacts immediately.
    """
    validate_schema()

    # Ensure index is positional so split_arrays logic is safe
    df_clean = df_clean.reset_index(drop=True)

    # Split (trial-level if key exists)
    split = split_indices(
        df_clean,
        key_col=key_col,
        seed=RANDOM_SEED,
        train_frac=TRAIN_FRAC,
        test_frac=TEST_FRAC,
        holdout_frac=HOLDOUT_FRAC,
    )

    # Build features for all rows (so encoder & embeddings align)
    X, encoder, feature_meta = assemble_features_fit(df_clean)

    # Targets
    Y, target_names = extract_targets(df_clean, target_cols)

    # Slice splits by positional indices (we reset_index above)
    tr = split.train_idx
    te = split.test_idx
    ho = split.holdout_idx

    X_train, Y_train = X[tr], Y[tr]
    X_test, Y_test = X[te], Y[te]
    X_hold, Y_hold = X[ho], Y[ho]

    # Train main model
    model = train_multitarget_catboost(X_train, Y_train, params=catboost_params)

    # Optional hurdles
    hurdle_models: Optional[Dict[str, Dict[str, Any]]] = None
    if sparse_targets and hurdle_presence_model_factory and hurdle_reg_model_factory:
        hurdle_models = train_hurdles(
            X_train,
            Y_train,
            target_names,
            sparse_targets=sparse_targets,
            presence_model_factory=hurdle_presence_model_factory,
            reg_model_factory=hurdle_reg_model_factory,
            eps=hurdle_eps,
        )

    # Build schema manifest for inference reproducibility
    schema_manifest = build_schema_manifest(
        cat_cols=list(CAT_COLS),
        num_cols=list(NUMERIC_COLS),
        cat_feature_names=feature_meta.get("cat_feature_names"),
        text_model_name=TEXT_EMBEDDING_MODEL,
        text_cols=list(TEXT_COLS),
        embedding_dim=feature_meta.get("embedding_dim"),
        geo_numeric_cols=list(REQUIRED_GEO_NUMERIC_COLS),
        targets=target_names,
    )

    out_dir_written: Optional[str] = None
    if artifacts_out_dir:
        meta = {
            "pipeline": "harry_ml",
            "saved_by": "train_pipeline",
            "saved_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "n_rows": int(len(df_clean)),
            "feature_dim": int(X.shape[1]),
            "n_targets": int(Y.shape[1]),
        }
        save_artifacts(
            artifacts_out_dir,
            model=model,
            encoder=encoder,
            schema_manifest=schema_manifest,
            meta=meta,
            hurdle_models=hurdle_models,
        )
        out_dir_written = str(Path(artifacts_out_dir).expanduser().resolve())

    return TrainOutputs(
        model=model,
        encoder=encoder,
        schema_manifest=schema_manifest,
        split=split,
        feature_shape=X.shape,
        target_names=target_names,
        hurdle_models=hurdle_models,
        artifacts_dir=out_dir_written,
    )


# -----------------------------
# Optional factories you can reuse
# -----------------------------
def default_hurdle_presence_factory():
    """
    Default presence classifier factory (CatBoostClassifier).
    """
    try:
        from catboost import CatBoostClassifier  # type: ignore
    except Exception as e:
        raise ImportError("CatBoost is required for hurdle presence model.") from e

    return CatBoostClassifier(
        loss_function="Logloss",
        random_seed=RANDOM_SEED,
        verbose=False,
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        task_type="CPU",
    )


def default_hurdle_reg_factory():
    """
    Default conditional regressor factory (CatBoostRegressor).
    """
    try:
        from catboost import CatBoostRegressor  # type: ignore
    except Exception as e:
        raise ImportError("CatBoost is required for hurdle reg model.") from e

    return CatBoostRegressor(
        loss_function="RMSE",
        random_seed=RANDOM_SEED,
        verbose=False,
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        task_type="CPU",
    )
