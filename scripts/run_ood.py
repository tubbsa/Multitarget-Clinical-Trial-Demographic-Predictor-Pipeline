#!/usr/bin/env python3
"""
Fit OOD stats on TRAIN split and score all rows.

Outputs a CSV with:
- ood_score
- ood_warn (>=2)
- ood_crit (>=3)

Example:
  python scripts/run_ood.py \
    --data data/trials_for_eval.csv \
    --key-col nct_id \
    --ood-cols "eligibility_min_age" "eligibility_max_age" "num_sites" "enrollment_count" "trial_duration_days" \
    --out ood_scores.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from harry_ml.cleaning import clean_trials
from harry_ml.io import load_dataframe
from harry_ml.split import split_indices
from harry_ml.config import RANDOM_SEED, TRAIN_FRAC, TEST_FRAC, HOLDOUT_FRAC
from harry_ml.ood import fit_ood_stats, score_ood, flag_ood


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit OOD stats on TRAIN split and score dataset.")
    p.add_argument("--data", required=True, help="Dataset path (csv/parquet/feather).")
    p.add_argument("--key-col", default="nct_id", help="Trial ID column for split stability.")
    p.add_argument("--ood-cols", nargs="+", required=True, help="Numeric columns to use for OOD z-score.")
    p.add_argument("--out", default="ood_scores.csv", help="Output CSV path.")
    p.add_argument("--warn", type=float, default=2.0, help="Warn threshold on max-|z|.")
    p.add_argument("--crit", type=float, default=3.0, help="Critical threshold on max-|z|.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = load_dataframe(args.data)
    df_clean = clean_trials(df).reset_index(drop=True)

    split = split_indices(
        df_clean,
        key_col=args.key_col if args.key_col else None,
        seed=RANDOM_SEED,
        train_frac=TRAIN_FRAC,
        test_frac=TEST_FRAC,
        holdout_frac=HOLDOUT_FRAC,
    )
    tr = split.train_idx

    stats = fit_ood_stats(df_clean.iloc[tr], args.ood_cols)
    score = score_ood(df_clean, stats)
    flags = flag_ood(score, warn_thresh=args.warn, crit_thresh=args.crit)

    out = flags.copy()
    if "nct_id" in df_clean.columns:
        out.insert(0, "nct_id", df_clean["nct_id"].astype(str).values)

    out_path = Path(args.out).expanduser().resolve()
    out.to_csv(out_path, index=False)

    print("\n=== OOD scoring complete ===")
    print("Saved:", str(out_path))
    print("Warn count:", int(flags["ood_warn"].sum()))
    print("Crit count:", int(flags["ood_crit"].sum()))


if __name__ == "__main__":
    main()
