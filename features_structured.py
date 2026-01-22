# src/harry_ml/features_structured.py
"""
Structured (non-text) feature engineering for the Harry pipeline.

This module is intentionally strict:
- Geography MUST be numeric-only (no geo columns in categoricals)
- Column order is canonical and must not drift between train/infer
- OneHotEncoder is fit on training data and reused for inference

It produces:
- df_cat_enc : encoded categorical features (DataFrame, deterministic columns)
- df_num     : numeric features (DataFrame, deterministic columns)
- encoder    : fitted OneHotEncoder
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------------------
# Canonical schemas (EXTRACTED FROM Harry_3-11_clean.py)
# ---------------------------------------------------------------------
CAT_COLS: List[str] = [
    "eligibility_sex",
    "sponsor",
    "collaborators",
    "phases",
    "funder_type",
    "study_type",
    "allocation",
    "intervention_model",
    "masking",
    "primary_purpose",
]

NUM_COLS: List[str] = [
    # Eligibility
    "eligibility_min_age_yrs",
    "eligibility_max_age_yrs",
    "min_age_missing",
    "max_age_missing",
    # Geography (U.S.-only, numeric)
    "n_sites",
    "n_us_states",
    "n_us_regions",
]

# Guardrail set used in your notebook
REQUIRED_GEO_NUMS = {"n_sites", "n_us_states", "n_us_regions"}


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class StructuredFeatureSet:
    """Outputs of structured feature build."""
    df_cat_enc: pd.DataFrame
    df_num: pd.DataFrame
    encoder: OneHotEncoder
    cat_cols: List[str]
    num_cols: List[str]


# ---------------------------------------------------------------------
# Helpers / validation
# ---------------------------------------------------------------------
def _assert_columns_present(df: pd.DataFrame, cols: Sequence[str], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {name}: {missing}")


def _coerce_numeric(df: pd.DataFrame, cols: Sequence[str], *, fillna_value: float = 0.0) -> pd.DataFrame:
    out = df[list(cols)].copy()
    # Deterministic numeric coercion
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.fillna(fillna_value)
    return out


def _validate_no_geo_in_categoricals(cat_cols: Sequence[str]) -> None:
    # Strict rule from your notebook: Geography MUST NEVER be categorical
    geo_like = set(REQUIRED_GEO_NUMS)
    bad = [c for c in cat_cols if c in geo_like]
    if bad:
        raise ValueError(f"Geographic columns detected in CAT_COLS (not allowed): {bad}")


def _validate_numeric_guardrails(df_num: pd.DataFrame, num_cols: Sequence[str]) -> None:
    # 1) Ensure required geo numeric features exist
    if not REQUIRED_GEO_NUMS.issubset(df_num.columns):
        raise ValueError(
            "Geographic numeric features missing from df_num.\n"
            f"Expected: {REQUIRED_GEO_NUMS}\n"
            f"Found: {set(df_num.columns)}"
        )

    # 2) Ensure schema matches exactly (set-level)
    if set(num_cols) != set(df_num.columns):
        raise ValueError(
            "NUM_COLS and df_num columns mismatch.\n"
            f"NUM_COLS: {list(num_cols)}\n"
            f"df_num:  {list(df_num.columns)}"
        )

    # 3) Basic sanity: geography should not be empty
    if "n_sites" in df_num.columns and float(df_num["n_sites"].max()) <= 0:
        raise ValueError("Geography features appear empty (n_sites max <= 0).")

    # 4) Optional non-negativity check for geo counts
    for c in ["n_sites", "n_us_states", "n_us_regions"]:
        if c in df_num.columns and (df_num[c] < 0).any():
            raise ValueError(f"Numeric guardrail failed: {c} contains negative values.")


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def build_categorical_frame(df_clean: pd.DataFrame, cat_cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Returns a categorical feature frame (copy) without mutating df_clean.
    """
    cols = list(cat_cols) if cat_cols is not None else list(CAT_COLS)
    _validate_no_geo_in_categoricals(cols)
    _assert_columns_present(df_clean, cols, name="df_clean (categoricals)")
    return df_clean[cols].copy()


def fit_categorical_encoder(
    df_cat: pd.DataFrame,
    *,
    handle_unknown: str = "ignore",
    sparse_output: bool = True,
) -> OneHotEncoder:
    """
    Fits a OneHotEncoder on the categorical frame.
    """
    enc = OneHotEncoder(handle_unknown=handle_unknown, sparse_output=sparse_output)
    enc.fit(df_cat)
    return enc


def encode_categoricals(
    df_cat: pd.DataFrame,
    encoder: OneHotEncoder,
    *,
    dense_dataframe: bool = True,
) -> pd.DataFrame:
    """
    Encodes categoricals using a fitted encoder and returns a DataFrame with stable column names.
    """
    X = encoder.transform(df_cat)
    feat_names = encoder.get_feature_names_out(df_cat.columns)

    if dense_dataframe:
        # Match your notebook behavior (dense DataFrame for easy concat)
        if hasattr(X, "toarray"):
            X_arr = X.toarray()
        else:
            X_arr = np.asarray(X)
        return pd.DataFrame(X_arr, columns=feat_names)
    else:
        # Still return DataFrame, but with sparse dtype (pandas >= 1.0 supports sparse arrays)
        # This keeps memory lower but may affect downstream concat behavior.
        try:
            return pd.DataFrame.sparse.from_spmatrix(X, columns=feat_names)
        except Exception:
            # Fallback to dense if sparse conversion fails
            if hasattr(X, "toarray"):
                X_arr = X.toarray()
            else:
                X_arr = np.asarray(X)
            return pd.DataFrame(X_arr, columns=feat_names)


def build_numeric_frame(
    df_clean: pd.DataFrame,
    num_cols: Optional[Sequence[str]] = None,
    *,
    fillna_value: float = 0.0,
) -> pd.DataFrame:
    """
    Returns numeric feature frame (copy) in canonical order, coerced to numeric, NaNs filled.
    """
    cols = list(num_cols) if num_cols is not None else list(NUM_COLS)
    _assert_columns_present(df_clean, cols, name="df_clean (numerics)")
    df_num = _coerce_numeric(df_clean, cols, fillna_value=fillna_value)

    # Enforce canonical order
    df_num = df_num[cols].copy()
    _validate_numeric_guardrails(df_num, cols)
    return df_num


def build_structured_features_fit(
    df_clean: pd.DataFrame,
    *,
    cat_cols: Optional[Sequence[str]] = None,
    num_cols: Optional[Sequence[str]] = None,
    handle_unknown: str = "ignore",
    sparse_output: bool = True,
    dense_cats: bool = True,
    fillna_value: float = 0.0,
) -> StructuredFeatureSet:
    """
    Train-time: builds df_cat, fits encoder, encodes cats, builds numeric frame.
    Returns StructuredFeatureSet with fitted encoder.
    """
    cat_cols_ = list(cat_cols) if cat_cols is not None else list(CAT_COLS)
    num_cols_ = list(num_cols) if num_cols is not None else list(NUM_COLS)

    df_cat = build_categorical_frame(df_clean, cat_cols_)
    encoder = fit_categorical_encoder(df_cat, handle_unknown=handle_unknown, sparse_output=sparse_output)
    df_cat_enc = encode_categoricals(df_cat, encoder, dense_dataframe=dense_cats)

    df_num = build_numeric_frame(df_clean, num_cols_, fillna_value=fillna_value)

    return StructuredFeatureSet(
        df_cat_enc=df_cat_enc.reset_index(drop=True),
        df_num=df_num.reset_index(drop=True),
        encoder=encoder,
        cat_cols=cat_cols_,
        num_cols=num_cols_,
    )


def build_structured_features_transform(
    df_clean: pd.DataFrame,
    *,
    encoder: OneHotEncoder,
    cat_cols: Optional[Sequence[str]] = None,
    num_cols: Optional[Sequence[str]] = None,
    dense_cats: bool = True,
    fillna_value: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Inference-time: uses a pre-fit encoder to encode categoricals and builds numeric frame.

    Returns:
      (df_cat_enc, df_num)
    """
    cat_cols_ = list(cat_cols) if cat_cols is not None else list(CAT_COLS)
    num_cols_ = list(num_cols) if num_cols is not None else list(NUM_COLS)

    df_cat = build_categorical_frame(df_clean, cat_cols_)
    df_cat_enc = encode_categoricals(df_cat, encoder, dense_dataframe=dense_cats)

    df_num = build_numeric_frame(df_clean, num_cols_, fillna_value=fillna_value)

    return df_cat_enc.reset_index(drop=True), df_num.reset_index(drop=True)
