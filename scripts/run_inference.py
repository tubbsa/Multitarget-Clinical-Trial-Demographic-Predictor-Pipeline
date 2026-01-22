#!/usr/bin/env python3
"""
Run inference using saved artifacts.

Usage examples:

  python scripts/run_inference.py \
    --data data/trials_for_inference.csv \
    --artifacts artifacts/harry_v1 \
    --out predictions.csv

This script:
- loads the dataset
- cleans it with harry_ml.cleaning.clean_trials()
- runs harry_ml.infer.predict_from_artifacts()
- writes a CSV of predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from harry_ml.cleaning import clean_trials  # must exist
from harry_ml.io import load_dataframe, save_dataframe
from harry_ml.infer import predict_from_artifacts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run Harry ML inference from artifacts.")
    p.add_argument("--data", required=True, help="Path to input dataset (csv/parquet/feather).")
    p.add_argument("--artifacts", required=True, help="Artifact directory created by training.")
    p.add_argument("--out", required=True, help="Output predictions CSV/Parquet/Feather.")
    p.add_argument("--hard-gate-hurdles", action="store_true", help="Hard-gate hurdle targets (optional).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data).expanduser().resolve()
    artifacts_dir = str(Path(args.artifacts).expanduser().resolve())
    out_path = Path(args.out).expanduser().resolve()

    df = load_dataframe(data_path)
    df_clean = clean_trials(df)

    res = predict_from_artifacts(
        df_clean,
        artifacts_dir=artifacts_dir,
        hard_gate_hurdles=args.hard_gate_hurdles,
        clip01=True,
    )

    preds = res.preds.copy()
    # If nct_id exists, keep it alongside predictions for traceability
    if "nct_id" in df_clean.columns and "nct_id" not in preds.columns:
        preds.insert(0, "nct_id", df_clean["nct_id"].astype(str).values)

    # Save predictions
    save_dataframe(preds, out_path, index=False)

    print("\n=== INFERENCE COMPLETE ===")
    print("Rows:", len(df_clean))
    print("Feature shape used:", res.features_shape)
    print("Used hurdles:", res.used_hurdles)
    print("Wrote:", str(out_path))


if __name__ == "__main__":
    main()
