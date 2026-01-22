#!/usr/bin/env python3
"""
Run baseline models and compute per-target RMSE/MAE across train/test/holdout.

Baselines implemented (paper-typical):
1) Mean baseline: predict training-set mean per target
2) Zero baseline for sparse targets: force selected targets to 0 (others use mean baseline)
3) Structured-only CatBoost (no text embeddings)
4) Text-only kNN baseline in embedding space (average y over k nearest neighbors)

Example:
  python scripts/run_baselines.py \
    --data data/trials_for_eval.csv \
    --targets "White %" "Black %" "Asian %" "AIAN %" "NHPI %" "Male %" "Female %" "Age 65+ %" \
    --sparse-targets "AIAN %" "NHPI %" \
    --key-col nct_id \
    --out baselines_table.csv \
    --knn-k 25
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from harry_ml.cleaning import clean_trials
from harry_ml.io import load_dataframe
from harry_ml.metrics import per_target_metrics, table_v_macro
from harry_ml.split import split_indices
from harry_ml.config import RANDOM_SEED, TRAIN_FRAC, TEST_FRAC, HOLDOUT_FRAC, CAT_COLS, NUMERIC_COLS
from harry_ml.features_structured import build_structured_features_fit, build_structured_features_transform
from harry_ml.features_text import build_text_embeddings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baselines and regenerate baseline metric tables.")
    p.add_argument("--data", required=True, help="Dataset path (csv/parquet/feather).")
    p.add_argument("--targets", nargs="+", required=True, help="Target columns (space-separated).")
    p.add_argument("--sparse-targets", nargs="*", default=[], help="Targets to force-zero in the zero baseline.")
    p.add_argument("--key-col", default="nct_id", help="Trial ID column for split stability.")
    p.add_argument("--out", default="baselines_table.csv", help="Output CSV for baseline metrics.")
    p.add_argument("--knn-k", type=int, default=25, help="k for text-only kNN baseline.")
    return p.parse_args()


def _catboost_multitarget_regressor():
    try:
        from catboost import CatBoostRegressor  # type: ignore
    except Exception as e:
        raise ImportError("catboost is required for structured-only baseline.") from e

    return CatBoostRegressor(
        loss_function="MultiRMSE",
        random_seed=RANDOM_SEED,
        verbose=False,
        iterations=1500,
        learning_rate=0.05,
        depth=8,
        task_type="CPU",
    )


def _nearest_neighbors_predict(X_train: np.ndarray, Y_train: np.ndarray, X_query: np.ndarray, k: int) -> np.ndarray:
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception as e:
        raise ImportError("scikit-learn is required for kNN baseline.") from e

    nn = NearestNeighbors(n_neighbors=min(k, len(X_train)), metric="cosine")
    nn.fit(X_train)
    dists, idxs = nn.kneighbors(X_query, return_distance=True)
    # average labels of neighbors
    preds = np.zeros((X_query.shape[0], Y_train.shape[1]), dtype=float)
    for i in range(X_query.shape[0]):
        preds[i] = Y_train[idxs[i]].mean(axis=0)
    return np.clip(preds, 0.0, 1.0)


def main() -> None:
    args = parse_args()

    df = load_dataframe(args.data)
    df_clean = clean_trials(df).reset_index(drop=True)

    missing = [c for c in args.targets if c not in df_clean.columns]
    if missing:
        raise KeyError(f"Missing target columns: {missing}")

    Y = df_clean[args.targets].copy()
    Y = Y.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    Y = np.clip(Y, 0.0, 1.0)

    split = split_indices(
        df_clean,
        key_col=args.key_col if args.key_col else None,
        seed=RANDOM_SEED,
        train_frac=TRAIN_FRAC,
        test_frac=TEST_FRAC,
        holdout_frac=HOLDOUT_FRAC,
    )
    tr, te, ho = split.train_idx, split.test_idx, split.holdout_idx

    # ---------- Baseline 1: mean ----------
    mean_vec = Y[tr].mean(axis=0)
    pred_mean = np.tile(mean_vec, (len(df_clean), 1))

    # ---------- Baseline 2: mean + forced zero on sparse targets ----------
    pred_zero = pred_mean.copy()
    sparse_idx = [args.targets.index(t) for t in args.sparse_targets if t in args.targets]
    if sparse_idx:
        pred_zero[:, sparse_idx] = 0.0

    # ---------- Baseline 3: structured-only CatBoost ----------
    sf = build_structured_features_fit(
        df_clean,
        cat_cols=list(CAT_COLS),
        num_cols=list(NUMERIC_COLS),
        handle_unknown="ignore",
        sparse_output=True,
        dense_cats=True,
        fillna_value=0.0,
    )
    X_struct = np.concatenate(
        [sf.df_num.to_numpy(dtype=float), sf.df_cat_enc.to_numpy(dtype=float)],
        axis=1,
    )
    model_struct = _catboost_multitarget_regressor()
    model_struct.fit(X_struct[tr], Y[tr])
    pred_struct = np.asarray(model_struct.predict(X_struct))
    if pred_struct.ndim == 1:
        pred_struct = pred_struct.reshape(-1, 1)
    pred_struct = np.clip(pred_struct, 0.0, 1.0)

    # ---------- Baseline 4: text-only kNN in embedding space ----------
    text_res = build_text_embeddings(df_clean)
    X_text = text_res.embeddings
    pred_knn = _nearest_neighbors_predict(X_text[tr], Y[tr], X_text, k=args.knn_k)

    # ---------- Compute Table V-style metrics for each baseline ----------
    def compute_table(pred: np.ndarray) -> pd.DataFrame:
        m_tr = per_target_metrics(Y[tr], pred[tr], target_names=args.targets, clip01=True)
        m_te = per_target_metrics(Y[te], pred[te], target_names=args.targets, clip01=True)
        m_ho = per_target_metrics(Y[ho], pred[ho], target_names=args.targets, clip01=True)
        return table_v_macro(m_tr, m_te, m_ho)

    tables = {
        "mean": compute_table(pred_mean),
        "mean_plus_zero_sparse": compute_table(pred_zero),
        "structured_only_catboost": compute_table(pred_struct),
        "text_only_knn": compute_table(pred_knn),
    }

    # Flatten into one CSV: add Baseline column and stack
    out_rows = []
    for name, tdf in tables.items():
        tdf2 = tdf.copy()
        tdf2.insert(0, "Baseline", name)
        out_rows.append(tdf2)
    out_df = pd.concat(out_rows, axis=0, ignore_index=True)

    out_path = Path(args.out).expanduser().resolve()
    out_df.to_csv(out_path, index=False)

    print("\n=== Baselines complete ===")
    print("Saved:", str(out_path))
    print("Baselines:", ", ".join(tables.keys()))
    print("\nPreview:")
    print(out_df.head(25).to_string(index=False))


if __name__ == "__main__":
    main()
