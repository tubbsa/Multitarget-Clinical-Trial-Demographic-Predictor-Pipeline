#!/usr/bin/env python3
"""
Train the Harry ML pipeline and write artifacts.

Usage examples:

  python scripts/train_model.py \
    --data data/trials_clean_input.csv \
    --out artifacts/harry_v1 \
    --key-col nct_id \
    --targets "White %" "Black %" "Asian %" "AIAN %" "NHPI %" "Male %" "Female %" "Age 65+ %" \
    --sparse-targets "AIAN %" "NHPI %" "Asian %"

Notes:
- The input dataset must already contain the structured + text columns expected by cleaning/features.
- This script calls harry_ml.cleaning.clean_trials() first, then trains.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from harry_ml.cleaning import clean_trials  # must exist in your cleaning.py
from harry_ml.io import load_dataframe
from harry_ml.train import (
    train_pipeline,
    default_hurdle_presence_factory,
    default_hurdle_reg_factory,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Harry ML pipeline and save artifacts.")
    p.add_argument("--data", required=True, help="Path to input dataset (csv/parquet/feather).")
    p.add_argument("--out", required=True, help="Output artifact directory (will be created).")
    p.add_argument("--key-col", default="nct_id", help="Trial-level key column for split stability.")
    p.add_argument(
        "--targets",
        nargs="+",
        required=True,
        help="Target column names (space-separated).",
    )
    p.add_argument(
        "--sparse-targets",
        nargs="*",
        default=[],
        help="Subset of targets to train with hurdle models (space-separated).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    df = load_dataframe(data_path)
    df_clean = clean_trials(df)

    outputs = train_pipeline(
        df_clean,
        target_cols=args.targets,
        key_col=args.key_col if args.key_col else None,
        sparse_targets=args.sparse_targets if args.sparse_targets else None,
        hurdle_presence_model_factory=default_hurdle_presence_factory if args.sparse_targets else None,
        hurdle_reg_model_factory=default_hurdle_reg_factory if args.sparse_targets else None,
        artifacts_out_dir=str(out_dir),
    )

    print("\n=== TRAIN COMPLETE ===")
    print("Rows:", len(df_clean))
    print("Features shape:", outputs.feature_shape)
    print("Targets:", outputs.target_names)
    print("Artifacts dir:", outputs.artifacts_dir)
    print("Hurdles:", "yes" if outputs.hurdle_models else "no")


if __name__ == "__main__":
    main()
