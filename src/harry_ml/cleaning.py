# src/harry_ml/cleaning.py
"""
Centralized cleaning for Harry ML.

Design goals (non-negotiable):
- Safe to run exactly once
- Feature-neutral: does NOT create engineered features beyond missingness flags
- Geography-safe: does NOT modify geographic fields or encode geography
- Deterministic parsing of age fields
- Leaves schema predictable for downstream feature builders

Public API:
- clean_trials(df: pd.DataFrame) -> pd.DataFrame
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Defaults aligned with your pipeline modules
DEFAULT_CAT_COLS: Tuple[str, ...] = (
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
)

DEFAULT_TEXT_COLS: Tuple[str, ...] = (
    "inclusion_text",
    "exclusion_text",
    "conditions_text",
    "interventions_text",
    "primary_outcome_text",
    "secondary_outcome_text",
)


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _find_column_case_insensitive(df: pd.DataFrame, exact_lower: str) -> Optional[str]:
    """
    Returns the actual column name whose lowercase matches exact_lower,
    or None if not found.
    """
    lookup = {c.lower(): c for c in df.columns}
    return lookup.get(exact_lower.lower())


def _parse_age_to_years(x) -> float:
    """
    Parses age inputs to a float representing years.
    Accepts numeric, strings like '18 Years', '18', '18 yrs', etc.
    Returns np.nan when unparseable.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan

    # if already numeric
    if isinstance(x, (int, float, np.integer, np.floating)):
        try:
            return float(x)
        except Exception:
            return np.nan

    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return np.nan

    # Strip common unit tokens
    tokens = ["years", "year", "yrs", "yr", "y"]
    s_clean = s
    for t in tokens:
        s_clean = s_clean.replace(t, "")
        s_clean = s_clean.replace(t.capitalize(), "")
        s_clean = s_clean.replace(t.upper(), "")

    s_clean = s_clean.strip()

    # Some sources can include '+', '>', etc. Strip non-numeric edges.
    # Keep digits, decimal, minus (rare), and whitespace.
    # If it's like "18+" or ">18", strip to "18".
    s_clean = s_clean.lstrip("><=~+ ").rstrip("+ ")

    try:
        return float(s_clean)
    except Exception:
        return np.nan


def _normalize_categorical_series(s: pd.Series) -> pd.Series:
    """
    Normalizes a categorical series to strings with 'Unknown' for missing.
    """
    # Convert to string but preserve real missing values first
    s2 = s.copy()
    s2 = s2.replace({None: np.nan})
    s2 = s2.astype("object")

    # Fill missing
    s2 = s2.where(~pd.isna(s2), other="Unknown")

    # Convert to str and normalize common text-missing tokens
    s2 = s2.astype(str)
    s2 = s2.replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
    return s2


def _normalize_text_series(s: pd.Series) -> pd.Series:
    """
    Normalizes text series to string, missing -> "".
    """
    s2 = s.copy()
    s2 = s2.replace({None: ""})
    s2 = s2.where(~pd.isna(s2), other="")
    s2 = s2.astype(str)
    return s2


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def clean_trials(
    df_raw: pd.DataFrame,
    *,
    min_age_col: str = "eligibility_min_age",
    max_age_col: str = "eligibility_max_age",
    cat_cols: Sequence[str] = DEFAULT_CAT_COLS,
    text_cols: Sequence[str] = DEFAULT_TEXT_COLS,
    unknown_token: str = "Unknown",
) -> pd.DataFrame:
    """
    Centralized cleaning (final, geo-safe, feature-neutral).

    Performs:
      - parses age columns into numeric years
      - creates min_age_missing / max_age_missing flags
      - imputes missing age values using column medians (after parsing)
      - normalizes NON-geographic categoricals ONLY (to strings; missing -> 'Unknown')
      - fills text fields with empty strings

    Does NOT:
      - modify or create geographic features
      - encode geography
      - create one-hot, scaling, embeddings, or other ML features
    """
    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError("clean_trials expects a pandas DataFrame.")

    df = df_raw.copy()

    # ---- resolve age column names case-insensitively (deterministic) ----
    min_age_actual = _find_column_case_insensitive(df, min_age_col)
    max_age_actual = _find_column_case_insensitive(df, max_age_col)

    if not min_age_actual or not max_age_actual:
        raise KeyError(
            f"Expected age columns '{min_age_col}' and '{max_age_col}' (case-insensitive). "
            f"Found columns include: {list(df.columns)[:25]}{'...' if len(df.columns) > 25 else ''}"
        )

    # ---- parse ages ----
    df[min_age_actual] = df[min_age_actual].apply(_parse_age_to_years)
    df[max_age_actual] = df[max_age_actual].apply(_parse_age_to_years)

    # ---- missingness flags (these ARE allowed; used later as numeric inputs) ----
    df["min_age_missing"] = pd.isna(df[min_age_actual]).astype(int)
    df["max_age_missing"] = pd.isna(df[max_age_actual]).astype(int)

    # ---- impute ages with medians (after parsing) ----
    # If entire column is NaN, median is NaN; fall back to safe defaults.
    min_med = float(df[min_age_actual].median()) if not df[min_age_actual].dropna().empty else 0.0
    max_med = float(df[max_age_actual].median()) if not df[max_age_actual].dropna().empty else 100.0

    df[min_age_actual] = df[min_age_actual].fillna(min_med)
    df[max_age_actual] = df[max_age_actual].fillna(max_med)

    # Optional: sanity constraints (non-breaking, but prevents nonsense)
    # Clamp to plausible range; adjust if your dataset includes pediatrics.
    df[min_age_actual] = df[min_age_actual].clip(lower=0.0, upper=120.0)
    df[max_age_actual] = df[max_age_actual].clip(lower=0.0, upper=120.0)

    # Ensure min <= max; if not, swap where needed (deterministic correction)
    bad = df[min_age_actual] > df[max_age_actual]
    if bad.any():
        tmp = df.loc[bad, min_age_actual].copy()
        df.loc[bad, min_age_actual] = df.loc[bad, max_age_actual]
        df.loc[bad, max_age_actual] = tmp

    # ---- normalize NON-geographic categoricals ----
    for col in cat_cols:
        if col in df.columns:
            df[col] = _normalize_categorical_series(df[col]).replace({"Unknown": unknown_token})
        else:
            # If missing, create it as Unknown (prevents downstream KeyErrors)
            df[col] = unknown_token

    # ---- normalize text fields ----
    for col in text_cols:
        if col in df.columns:
            df[col] = _normalize_text_series(df[col])
        else:
            # If missing, create empty string column
            df[col] = ""

    return df


# Backwards-compatible alias if you already used this name elsewhere
clean_missing = clean_trials


