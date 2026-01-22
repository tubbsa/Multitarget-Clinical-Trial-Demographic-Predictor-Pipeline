# Multitarget-Clinical-Trial-Demographic-Predictor-Pipeline
# Multitarget Clinical Trial Demographic Predictor Pipeline


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

