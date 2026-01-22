# Multitarget-Clinical-Trial-Demographic-Predictor-Pipeline



The code implements a modular, reproducible pipeline that maps clinical trial protocol metadata to **predicted demographic enrollment proportions** across race, sex, and age groups.

This repository is intended to support **methodological transparency and reproducibility** for the associated manuscript. It is not a production system.

---

## Overview

The pipeline consists of the following stages:

1. **Centralized Cleaning**  
   Deterministic, feature-neutral preprocessing of trial metadata  
   (`harry_ml.cleaning`)

2. **Feature Construction**  
   - Structured protocol features (eligibility, design, recruitment scale)  
   - Text embeddings from protocol narratives  
   (`harry_ml.features_structured`, `harry_ml.features_text`)

3. **Deterministic Data Splitting**  
   Fixed 64 / 16 / 20 train–test–holdout splits  
   (`harry_ml.split`)

4. **Multi-Target Learning**  
   - CatBoost multi-output regression  
   - Two-stage hurdle models for sparse demographic targets  
   (`harry_ml.train`, `harry_ml.hurdle`)

5. **Frozen-Artifact Inference**  
   Predictions generated strictly from saved preprocessing and model artifacts  
   (`harry_ml.infer`, `harry_ml.artifacts`)

6. **Evaluation and Analysis**  
   - Per-target RMSE / MAE (Table V in paper)  
   - Baseline comparisons  
   - SHAP-based interpretability  
   - Out-of-distribution (OOD) screening  
   (`harry_ml.metrics`, `harry_ml.ood`, `scripts/`)

---

## Repository Structure
├── src/harry_ml/ # Core ML pipeline (installable package)
│ ├── cleaning.py
│ ├── features_structured.py
│ ├── features_text.py
│ ├── split.py
│ ├── train.py
│ ├── hurdle.py
│ ├── infer.py
│ ├── metrics.py
│ ├── ood.py
│ ├── artifacts.py
│ └── io.py
│
├── scripts/ # Reproducible runners
│ ├── train_model.py
│ ├── evaluate.py # Regenerates Table V
│ ├── run_baselines.py
│ ├── run_shap.py
│ └── run_ood.py
│
├── notebooks/ # Exploratory / development notebooks
├── data/ # (not tracked)
├── artifacts/ # (not tracked)
├── README.md
├── LICENSE
└── pyproject.toml

---

## Installation

Python ≥ 3.9 is required.

```bash
pip install -e .
Training
Train the full pipeline and write frozen artifacts:
python scripts/train_model.py \
  --data path/to/trials.csv \
  --out artifacts/harry_v1 \
  --targets "White %" "Black %" "Asian %" "AIAN %" "NHPI %" \
            "Male %" "Female %" "Age 65+ %" \
  --sparse-targets "AIAN %" "NHPI %"
Artifacts include trained models, encoders, schema manifests, and split metadata.
Evaluation (Table V)
Regenerate per-target RMSE / MAE across train, test, and holdout splits:
python scripts/evaluate.py \
  --data path/to/trials.csv \
  --artifacts artifacts/harry_v1 \
  --targets "White %" "Black %" "Asian %" "AIAN %" "NHPI %" \
            "Male %" "Female %" "Age 65+ %" \
  --out table_v.csv
Baselines
Run baseline models described in the paper:
python scripts/run_baselines.py \
  --data path/to/trials.csv \
  --targets "White %" "Black %" "Asian %" "AIAN %" "NHPI %" \
            "Male %" "Female %" "Age 65+ %" \
  --sparse-targets "AIAN %" "NHPI %" \
  --out baselines.csv
Interpretability (SHAP)
Compute SHAP summaries used in the interpretability analysis:
python scripts/run_shap.py \
  --data path/to/trials.csv \
  --artifacts artifacts/harry_v1 \
  --targets "White %" "Black %" "Asian %" "AIAN %" "NHPI %" \
            "Male %" "Female %" "Age 65+ %" \
  --outdir shap_outputs
Outputs are figure-ready CSVs.
Out-of-Distribution Screening
Fit OOD statistics on the training split and score all trials:
python scripts/run_ood.py \
  --data path/to/trials.csv \
  --ood-cols eligibility_min_age eligibility_max_age num_sites enrollment_count trial_duration_days \
  --out ood_scores.csv
