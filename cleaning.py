#!/usr/bin/env python
# coding: utf-8

# In[4]:


# ============================================================
# 1. IMPORTS & GLOBAL SETTINGS
# ============================================================

import numpy as np
import pandas as pd
import torch

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
from catboost import Pool, CatBoostRegressor

RANDOM_STATE = 42






# In[5]:


import os
import pickle
import numpy as np

# MUST match app structure
PROJECT_ROOT = "/Users/abigailtubbs/Downloads/digital_twin"
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------
# Shared artifact paths (TRAIN ↔ PREDICTOR CONTRACT)
# ------------------------------------------------------------
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "FEATURE_NAMES.pkl")

print("MODEL_DIR:", MODEL_DIR)
print("FEATURE_NAMES_PATH:", FEATURE_NAMES_PATH)



# In[6]:


# ============================================================
# 2. LOAD RAW DATA
# ============================================================

RAW_DATA_PATH = "/Users/abigailtubbs/Desktop/Fall 2025/Dissertation F25/Harry/cleaned_dataset.xlsx"

df_raw = pd.read_excel(RAW_DATA_PATH)

print("Raw shape:", df_raw.shape)
df_raw.head()


# In[7]:


# ============================================================
# 2B. CENTRALIZED CLEANING (FINAL, GEO-SAFE, FEATURE-NEUTRAL)
# ============================================================

def clean_missing(df):
    """
    Cleans raw trial fields:
      - parses age columns
      - creates missingness flags
      - normalizes NON-geographic categoricals
      - fills text fields

    IMPORTANT:
      • Does NOT create features
      • Does NOT modify geography
      • Safe to run exactly once
    """

    df = df.copy()

    # ----------------------------------------------
    # IDENTIFY AGE COLUMNS (DETERMINISTIC)
    # ----------------------------------------------
    min_age_candidates = [c for c in df.columns if c.lower() == "eligibility_min_age"]
    max_age_candidates = [c for c in df.columns if c.lower() == "eligibility_max_age"]

    if not min_age_candidates or not max_age_candidates:
        raise KeyError("Expected eligibility_min_age / eligibility_max_age columns")

    min_age_col = min_age_candidates[0]
    max_age_col = max_age_candidates[0]

    # ----------------------------------------------
    # PARSE AGE STRINGS → FLOAT
    # ----------------------------------------------
    def parse_age(x):
        if x is None or pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)

        s = str(x)
        for word in ["Years", "Year", "years", "year", "yrs", "yr", "Yrs", "Y"]:
            s = s.replace(word, "")
        try:
            return float(s.strip())
        except ValueError:
            return np.nan

    df[min_age_col] = df[min_age_col].apply(parse_age)
    df[max_age_col] = df[max_age_col].apply(parse_age)

    # ----------------------------------------------
    # MISSINGNESS FLAGS (NOT FEATURES YET)
    # ----------------------------------------------
    df["min_age_missing"] = df[min_age_col].isna().astype(int)
    df["max_age_missing"] = df[max_age_col].isna().astype(int)

    # ----------------------------------------------
    # IMPUTE MEDIANS (TEMPORARY VALUES)
    # ----------------------------------------------
    df[min_age_col] = df[min_age_col].fillna(df[min_age_col].median())
    df[max_age_col] = df[max_age_col].fillna(df[max_age_col].median())

    # ----------------------------------------------
    # NON-GEOGRAPHIC CATEGORICAL CLEANING ONLY
    # ----------------------------------------------
    cat_cols = [
        "eligibility_sex", "sponsor", "collaborators",
        "phases", "funder_type", "study_type", "allocation",
        "intervention_model", "masking", "primary_purpose"
    ]

    for col in cat_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .replace({"nan": "Unknown", "None": "Unknown"})
                .fillna("Unknown")
            )

    # ----------------------------------------------
    # TEXT FIELDS
    # ----------------------------------------------
    text_cols = [
        "inclusion_text", "exclusion_text", "conditions_text",
        "interventions_text", "primary_outcome_text", "secondary_outcome_text"
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("")

    return df




# In[8]:


# ============================================================
# 3. APPLY CLEANING PIPELINE
# ============================================================

df_clean = clean_missing(df_raw)

print("Cleaned shape:", df_clean.shape)
df_clean.head()




# In[9]:

