#!/usr/bin/env python3
"""
Evaluate a trained artifact directory and regenerate Table V (per-target RMSE/MAE)
across train/test/holdout splits.

This script assumes:
- You have a dataset containing the target columns.
- You have an artifacts directory produced by scripts/train_model.py (or equivalent).
- Split logic matches your pipeline: harry_ml.split.split_indices() (64/16/20) and RANDOM_SEED.

Example:
  python scripts/evaluate.py \
    --data data/trials_for_eval.csv \
    --artifacts artifacts/harry_v1 \
    --targets "White %" "Black %" "Asian %" "AIAN %" "NHPI %" "Male %" "Female %" "Age 65+ %" \
    --key-col nct_id \
    --out table_v.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import pandas as pd

from harry_ml.cleaning import clean_trials
from harry_ml.io import load_dataframe
from harry_ml.infer import predict_from_artifacts
from harry_ml.metrics import per_target_metrics, table_v_macro
from harry_ml.split import split_indices
from harry_ml.config import RANDOM_SEED, TRAIN_FRAC, TEST_FRAC, HOLDOUT_FRAC


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Regenerate Table V (RMSE/MAE per target).")
    p.add_argument("--data", required=True, help="Path to dataset (csv/parquet/feather).")
    p.add_argument("--artifacts", required=True, help="Artifacts directory from training.")
    p.add_argument("--targets", nargs="+", required=True, help="Target columns (space-separated).")
    p.add_argument("--key-col", default="nct_id", help="Trial ID column for split stability.")
    p.add_argument("--out", default="table_v.csv", help="Output path for Table V (csv or xlsx).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = load_dataframe(args.data)
    df_clean = clean_trials(df).reset_index(drop=True)

    # Create deterministic splits
    split = split_indices(
        df_clean,
        key_col=args.key_col if args.key_col else None,
        seed=RANDOM_SEED,
        train_frac=TRAIN_FRAC,
        test_frac=TEST_FRAC,
        holdout_frac=HOLDOUT_FRAC,
    )

    # Run inference once for all rows, then slice by split
    res = predict_from_artifacts(df_clean, artifacts_dir=args.artifacts, clip01=True)
    preds = res.preds

    # Ground truth
    missing = [c for c in args.targets if c not in df_clean.columns]
    if missing:
        raise KeyError(f"Missing target columns in dataset: {missing}")

    y_true = df_clean[args.targets].copy()
    y_pred = preds[args.targets].copy() if all(t in preds.columns for t in args.targets) else None

    if y_pred is None:
        raise KeyError(
            "Predictions do not contain all requested target columns. "
            "Check artifacts schema_manifest 'targets' and inference output columns."
        )

    # Slice
    tr, te, ho = split.train_idx, split.test_idx, split.holdout_idx

    m_train = per_target_metrics(y_true.iloc[tr], y_pred.iloc[tr], target_names=args.targets, clip01=True)
    m_test = per_target_metrics(y_true.iloc[te], y_pred.iloc[te], target_names=args.targets, clip01=True)
    m_hold = per_target_metrics(y_true.iloc[ho], y_pred.iloc[ho], target_names=args.targets, clip01=True)

    table = table_v_macro(m_train, m_test, m_hold)

    out_path = Path(args.out).expanduser().resolve()
    if out_path.suffix.lower() == ".xlsx":
        table.to_excel(out_path, index=False)
    else:
        table.to_csv(out_path, index=False)

    print("\n=== Table V regenerated ===")
    print("Rows:", len(df_clean))
    print("Artifacts:", str(Path(args.artifacts).expanduser().resolve()))
    print("Saved:", str(out_path))
    print("\nPreview:")
    print(table.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
