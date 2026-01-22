# src/harry_ml/split.py
"""
Deterministic train/test/holdout splitting for the Harry ML pipeline.

Key requirements for your dissertation pipeline:
- Fixed 64/16/20 split (train/test/holdout)
- Reproducible with a single random seed
- Stable at the trial level (no leakage across splits)
- Works with a unique key column if present (recommended)

This module provides:
- split_indices(): returns integer row indices for each split
- split_dataframe(): returns three DataFrames
- split_arrays(): returns split arrays for X/Y

If you have a trial identifier column (e.g., "nct_id"), use it so that
duplicate rows or derived rows do not leak across splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitResult:
    train_idx: np.ndarray
    test_idx: np.ndarray
    holdout_idx: np.ndarray


def _validate_fracs(train_frac: float, test_frac: float, holdout_frac: float) -> None:
    total = train_frac + test_frac + holdout_frac
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split fractions must sum to 1.0 (got {total}).")
    for name, v in [("train_frac", train_frac), ("test_frac", test_frac), ("holdout_frac", holdout_frac)]:
        if v <= 0:
            raise ValueError(f"{name} must be > 0 (got {v}).")


def split_indices(
    df: pd.DataFrame,
    *,
    key_col: Optional[str] = "nct_id",
    seed: int = 42,
    train_frac: float = 0.64,
    test_frac: float = 0.16,
    holdout_frac: float = 0.20,
) -> SplitResult:
    """
    Create deterministic split indices.

    If key_col exists:
      - split by unique key values (trial-level split)
      - then map back to row indices

    If key_col is None or missing:
      - split by row index directly

    Returns:
      SplitResult with integer indices into df.
    """
    _validate_fracs(train_frac, test_frac, holdout_frac)
    n = len(df)
    if n == 0:
        raise ValueError("Cannot split empty dataframe.")

    rng = np.random.default_rng(seed)

    # Trial-level split (preferred)
    if key_col and key_col in df.columns:
        keys = df[key_col].astype(str)
        unique_keys = keys.dropna().unique()
        unique_keys = np.array(unique_keys, dtype=object)

        rng.shuffle(unique_keys)

        n_keys = len(unique_keys)
        n_train = int(round(train_frac * n_keys))
        n_test = int(round(test_frac * n_keys))
        # remainder goes to holdout
        n_train = min(n_train, n_keys)
        n_test = min(n_test, n_keys - n_train)
        n_hold = n_keys - n_train - n_test

        train_keys = set(unique_keys[:n_train])
        test_keys = set(unique_keys[n_train : n_train + n_test])
        hold_keys = set(unique_keys[n_train + n_test :])

        train_idx = df.index[keys.isin(train_keys)].to_numpy()
        test_idx = df.index[keys.isin(test_keys)].to_numpy()
        hold_idx = df.index[keys.isin(hold_keys)].to_numpy()

        # Safety: ensure disjoint
        if set(train_idx) & set(test_idx) or set(train_idx) & set(hold_idx) or set(test_idx) & set(hold_idx):
            raise RuntimeError("Split indices are not disjoint (unexpected).")

        return SplitResult(train_idx=train_idx, test_idx=test_idx, holdout_idx=hold_idx)

    # Row-level fallback
    idx = df.index.to_numpy()
    rng.shuffle(idx)

    n_train = int(round(train_frac * n))
    n_test = int(round(test_frac * n))
    n_train = min(n_train, n)
    n_test = min(n_test, n - n_train)
    n_hold = n - n_train - n_test

    train_idx = idx[:n_train]
    test_idx = idx[n_train : n_train + n_test]
    hold_idx = idx[n_train + n_test :]

    return SplitResult(train_idx=train_idx, test_idx=test_idx, holdout_idx=hold_idx)


def split_dataframe(
    df: pd.DataFrame,
    *,
    key_col: Optional[str] = "nct_id",
    seed: int = 42,
    train_frac: float = 0.64,
    test_frac: float = 0.16,
    holdout_frac: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (df_train, df_test, df_holdout) in deterministic order.
    """
    s = split_indices(
        df,
        key_col=key_col,
        seed=seed,
        train_frac=train_frac,
        test_frac=test_frac,
        holdout_frac=holdout_frac,
    )
    return df.loc[s.train_idx].copy(), df.loc[s.test_idx].copy(), df.loc[s.holdout_idx].copy()


def split_arrays(
    X,
    Y,
    idx: SplitResult,
):
    """
    Split arrays/matrices X and Y by indices.
    Works for numpy arrays and pandas objects, and scipy sparse matrices that support [].

    Args:
      X: feature matrix
      Y: target matrix/vector
      idx: SplitResult from split_indices()

    Returns:
      (X_train, X_test, X_holdout, Y_train, Y_test, Y_holdout)
    """
    # If df.index isn't 0..n-1, convert to positional indices
    # Expect idx arrays contain actual df indices; convert to positions if needed by caller.
    # Here we assume caller has already aligned X/Y to df order and uses positional indexing.
    # If your df index is not range(n), do: df = df.reset_index(drop=True) before building X/Y.

    tr = idx.train_idx
    te = idx.test_idx
    ho = idx.holdout_idx

    return X[tr], X[te], X[ho], Y[tr], Y[te], Y[ho]
