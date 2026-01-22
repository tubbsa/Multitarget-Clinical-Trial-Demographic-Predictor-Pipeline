# src/harry_ml/ood.py
"""
Out-of-distribution (OOD) utilities for Harry ML.

Implements a simple, transparent OOD score:
- For specified numeric features, compute z = (x - mean) / std using TRAIN stats
- OOD score = max(|z|) across the selected features

This matches "lightweight screening" style checks common in papers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class OODStats:
    cols: List[str]
    mean: Dict[str, float]
    std: Dict[str, float]


def fit_ood_stats(df_train: pd.DataFrame, cols: Sequence[str]) -> OODStats:
    """
    Fit mean/std on the training set for the specified columns.
    """
    cols = list(cols)
    missing = [c for c in cols if c not in df_train.columns]
    if missing:
        raise KeyError(f"Missing OOD columns in training df: {missing}")

    mean = {}
    std = {}
    for c in cols:
        x = pd.to_numeric(df_train[c], errors="coerce")
        m = float(x.mean(skipna=True))
        s = float(x.std(skipna=True, ddof=0))
        # avoid division by zero: clamp tiny std
        if not np.isfinite(s) or s < 1e-12:
            s = 1e-12
        mean[c] = m
        std[c] = s

    return OODStats(cols=cols, mean=mean, std=std)


def score_ood(df: pd.DataFrame, stats: OODStats) -> pd.Series:
    """
    Compute max-abs z-score across stats.cols for each row.
    """
    missing = [c for c in stats.cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing OOD columns in df: {missing}")

    Z = []
    for c in stats.cols:
        x = pd.to_numeric(df[c], errors="coerce").fillna(stats.mean[c]).to_numpy(dtype=float)
        z = (x - stats.mean[c]) / stats.std[c]
        Z.append(np.abs(z))

    Z = np.vstack(Z).T  # (n, k)
    score = np.max(Z, axis=1)
    return pd.Series(score, index=df.index, name="ood_score")


def flag_ood(score: pd.Series, *, warn_thresh: float = 2.0, crit_thresh: float = 3.0) -> pd.DataFrame:
    """
    Convert a score series into boolean flags.
    """
    s = pd.to_numeric(score, errors="coerce").fillna(0.0)
    return pd.DataFrame(
        {
            "ood_score": s,
            "ood_warn": s >= warn_thresh,
            "ood_crit": s >= crit_thresh,
        },
        index=score.index,
    )
