# src/harry_ml/metrics.py
"""
Metrics utilities for the Harry ML pipeline.

Primary use:
- Regenerate Table V: per-target RMSE/MAE on train/test/holdout splits.

Notes:
- Accepts numpy arrays or pandas DataFrames/Series.
- Handles clipping to [0,1] optionally (consistent with your pipeline outputs).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, pd.DataFrame, pd.Series]


def _to_2d(a: ArrayLike) -> np.ndarray:
    if isinstance(a, pd.DataFrame):
        x = a.to_numpy(dtype=float)
    elif isinstance(a, pd.Series):
        x = a.to_numpy(dtype=float).reshape(-1, 1)
    else:
        x = np.asarray(a, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
    return x


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    yt = _to_2d(y_true)
    yp = _to_2d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    return np.sqrt(np.mean((yt - yp) ** 2, axis=0))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> np.ndarray:
    yt = _to_2d(y_true)
    yp = _to_2d(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")
    return np.mean(np.abs(yt - yp), axis=0)


@dataclass(frozen=True)
class PerTargetMetrics:
    target_names: List[str]
    rmse: np.ndarray
    mae: np.ndarray

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Target": self.target_names,
                "RMSE": self.rmse,
                "MAE": self.mae,
            }
        )


def per_target_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    target_names: Optional[Sequence[str]] = None,
    clip01: bool = True,
) -> PerTargetMetrics:
    yt = _to_2d(y_true)
    yp = _to_2d(y_pred)

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")

    if clip01:
        yt = np.clip(yt, 0.0, 1.0)
        yp = np.clip(yp, 0.0, 1.0)

    t = yt.shape[1]
    if target_names is None:
        names = [f"target_{i}" for i in range(t)]
    else:
        if len(target_names) != t:
            raise ValueError(f"target_names length {len(target_names)} != n_targets {t}")
        names = list(target_names)

    return PerTargetMetrics(
        target_names=names,
        rmse=rmse(yt, yp),
        mae=mae(yt, yp),
    )


def table_v_macro(
    train: PerTargetMetrics,
    test: PerTargetMetrics,
    holdout: PerTargetMetrics,
) -> pd.DataFrame:
    """
    Table V format:
      Target | Train RMSE | Train MAE | Test RMSE | Test MAE | Holdout RMSE | Holdout MAE
    """
    if train.target_names != test.target_names or train.target_names != holdout.target_names:
        raise ValueError("Target name mismatch across splits.")

    df = pd.DataFrame({"Target": train.target_names})
    df["Train RMSE"] = train.rmse
    df["Train MAE"] = train.mae
    df["Test RMSE"] = test.rmse
    df["Test MAE"] = test.mae
    df["Holdout RMSE"] = holdout.rmse
    df["Holdout MAE"] = holdout.mae
    return df


def save_table(df: pd.DataFrame, out_path: str) -> None:
    """
    Save metrics table. Uses extension to choose format.
    """
    if out_path.lower().endswith(".csv"):
        df.to_csv(out_path, index=False)
    elif out_path.lower().endswith(".xlsx"):
        df.to_excel(out_path, index=False)
    else:
        # default to csv
        df.to_csv(out_path, index=False)
