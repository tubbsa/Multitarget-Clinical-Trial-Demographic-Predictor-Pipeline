#!/usr/bin/env python3
"""
Compute SHAP summaries for a trained CatBoost model and write figure-ready CSVs.

Outputs (CSV):
- shap_mean_abs_per_feature.csv  (global mean|SHAP| per feature, averaged over rows and targets)
- shap_group_mass.csv            (sum mean|SHAP| within feature groups)
- shap_top_structured_features.csv (top structured (non-text) features)
- shap_group_by_target.csv       (mean|SHAP| per group x target)

Example:
  python scripts/run_shap.py \
    --data data/trials_for_eval.csv \
    --artifacts artifacts/harry_v1 \
    --targets "White %" "Black %" "Asian %" "AIAN %" "NHPI %" "Male %" "Female %" "Age 65+ %" \
    --outdir shap_outputs \
    --sample-n 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from harry_ml.cleaning import clean_trials
from harry_ml.artifacts import load_artifacts
from harry_ml.features_structured import build_categorical_frame, build_numeric_frame, encode_categoricals
from harry_ml.features_text import build_text_embeddings
from harry_ml.io import load_dataframe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute SHAP summaries and write CSV outputs.")
    p.add_argument("--data", required=True, help="Dataset path (csv/parquet/feather).")
    p.add_argument("--artifacts", required=True, help="Artifacts directory.")
    p.add_argument("--targets", nargs="+", required=True, help="Target names (must match manifest targets).")
    p.add_argument("--outdir", default="shap_outputs", help="Directory to write CSV outputs.")
    p.add_argument("--sample-n", type=int, default=2000, help="Row sample for SHAP (speed).")
    return p.parse_args()


def _assemble_X(df_clean: pd.DataFrame, artifacts_dir: str) -> Tuple[np.ndarray, List[str], Dict[str, List[str]]]:
    art = load_artifacts(artifacts_dir, load_hurdles=False)
    schema = art.schema

    cat_cols = schema.get("cat_cols") or []
    num_cols = schema.get("num_cols") or []
    text_cols = schema.get("text_cols") or None
    text_model = schema.get("text_model_name") or None

    # Structured
    df_cat = build_categorical_frame(df_clean, cat_cols)
    df_cat_enc = encode_categoricals(df_cat, art.encoder, dense_dataframe=True)
    df_num = build_numeric_frame(df_clean, num_cols)

    # Text
    text_res = build_text_embeddings(
        df_clean,
        text_cols=text_cols if text_cols is not None else None,
        model_name=text_model if text_model is not None else None,
    )
    X_text = text_res.embeddings

    X = np.concatenate(
        [df_num.to_numpy(dtype=float), df_cat_enc.to_numpy(dtype=float), X_text],
        axis=1,
    )

    # Feature names
    cat_feature_names = schema.get("cat_feature_names")
    if cat_feature_names is None:
        # fallback
        cat_feature_names = list(art.encoder.get_feature_names_out(cat_cols))

    text_dim = X_text.shape[1]
    text_feature_names = [f"text_emb_{i:03d}" for i in range(text_dim)]

    feature_names = list(num_cols) + list(cat_feature_names) + text_feature_names

    groups = {
        "numeric": list(num_cols),
        "categorical_onehot": list(cat_feature_names),
        "text_embeddings": text_feature_names,
    }
    return X, feature_names, groups


def main() -> None:
    args = parse_args()

    df = load_dataframe(args.data)
    df_clean = clean_trials(df).reset_index(drop=True)

    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    art = load_artifacts(args.artifacts, load_hurdles=False)
    model = art.model

    # Build X and names
    X, feature_names, groups = _assemble_X(df_clean, args.artifacts)

    # Subsample for SHAP if needed
    n = len(df_clean)
    if args.sample_n and n > args.sample_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=args.sample_n, replace=False)
        X_shap = X[idx]
    else:
        idx = np.arange(n)
        X_shap = X

    # CatBoost SHAP
    try:
        from catboost import Pool  # type: ignore
    except Exception as e:
        raise ImportError("catboost is required for SHAP extraction via get_feature_importance.") from e

    if not hasattr(model, "get_feature_importance"):
        raise TypeError("Loaded model does not support CatBoost get_feature_importance().")

    pool = Pool(X_shap, feature_names=feature_names)

    shap_vals = model.get_feature_importance(pool, type="ShapValues")

    shap_vals = np.asarray(shap_vals)

    # Expected shapes:
    # - single target: (n, d+1) where last col is expected value
    # - multi-target: (n, d+1, t) OR (t, n, d+1) depending on CatBoost version
    d = X_shap.shape[1]

    # Normalize to (n, d, t)
    if shap_vals.ndim == 2:
        # (n, d+1)
        if shap_vals.shape[1] != d + 1:
            raise ValueError(f"Unexpected SHAP shape {shap_vals.shape}, expected (n, d+1).")
        S = shap_vals[:, :d].reshape(shap_vals.shape[0], d, 1)
        targets = args.targets[:1]
    elif shap_vals.ndim == 3:
        if shap_vals.shape[1] == d + 1:
            # (n, d+1, t)
            S = shap_vals[:, :d, :]
            targets = args.targets
        elif shap_vals.shape[2] == d + 1:
            # (t, n, d+1) -> transpose
            S = np.transpose(shap_vals[:, :, :d], (1, 2, 0))
            targets = args.targets
        else:
            raise ValueError(f"Unexpected SHAP shape {shap_vals.shape} for d={d}.")
    else:
        raise ValueError(f"Unexpected SHAP ndim {shap_vals.ndim} with shape {shap_vals.shape}")

    if S.shape[1] != d:
        raise ValueError(f"Internal SHAP normalization failed; got S shape {S.shape} for d={d}.")

    # mean abs shap per feature per target
    mean_abs = np.mean(np.abs(S), axis=0)  # (d, t)
    # global mean abs per feature (average across targets)
    mean_abs_global = mean_abs.mean(axis=1)  # (d,)

    df_feat = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_global,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    df_feat.to_csv(outdir / "shap_mean_abs_per_feature.csv", index=False)

    # Group mass: sum mean|shap| within each group (global)
    group_rows = []
    feat_to_val = dict(zip(feature_names, mean_abs_global))
    for gname, flist in groups.items():
        group_rows.append(
            {
                "group": gname,
                "aggregated_shap_mass": float(sum(feat_to_val.get(f, 0.0) for f in flist)),
                "n_features": len(flist),
                "mean_per_feature": float(
                    (sum(feat_to_val.get(f, 0.0) for f in flist) / max(len(flist), 1))
                ),
            }
        )
    df_group = pd.DataFrame(group_rows).sort_values("aggregated_shap_mass", ascending=False)
    df_group.to_csv(outdir / "shap_group_mass.csv", index=False)

    # Top structured features (exclude text embeddings)
    structured = df_feat[~df_feat["feature"].str.startswith("text_emb_")].copy()
    structured.head(50).to_csv(outdir / "shap_top_structured_features.csv", index=False)

    # Group by target heatmap data
    # mean_abs is (d, t); sum within group for each target
    group_by_target = []
    feat_index = {f: i for i, f in enumerate(feature_names)}
    t_count = mean_abs.shape[1]

    for gname, flist in groups.items():
        idxs = [feat_index[f] for f in flist if f in feat_index]
        if not idxs:
            continue
        vals = mean_abs[idxs, :].sum(axis=0)  # (t,)
        for j in range(t_count):
            group_by_target.append(
                {
                    "group": gname,
                    "target": targets[j] if j < len(targets) else f"target_{j}",
                    "sum_mean_abs_shap": float(vals[j]),
                }
            )
    df_gbt = pd.DataFrame(group_by_target)
    df_gbt.to_csv(outdir / "shap_group_by_target.csv", index=False)

    print("\n=== SHAP outputs written ===")
    print("Outdir:", str(outdir))
    print("Rows used:", X_shap.shape[0])
    print("Features:", d)
    print("Targets:", S.shape[2])


if __name__ == "__main__":
    main()
