# src/harry_ml/config.py
"""
Global configuration and canonical schemas for the Harry ML pipeline.

This file is the SINGLE SOURCE OF TRUTH for:
- Feature schemas (categorical vs numeric)
- Text embedding configuration
- Split ratios
- Random seeds
- Invariants that must never drift between training and inference

Nothing in this file should depend on runtime data.
"""

from __future__ import annotations

# ============================================================
# Reproducibility
# ============================================================

RANDOM_SEED: int = 42

# Train / test / holdout split ratios (must sum to 1.0)
TRAIN_FRAC: float = 0.64
TEST_FRAC: float = 0.16
HOLDOUT_FRAC: float = 0.20


# ============================================================
# CATEGORICAL FEATURE SCHEMA (NO GEOGRAPHY — EVER)
# ============================================================

# These columns are one-hot encoded.
# Geography is explicitly forbidden here.
CAT_COLS = [
    "eligibility_sex",
    "sponsor",
    "collaborators",
    "phases",
    "funder_type",
    "study_type",
    "allocation",
    "intervention_model",
    "masking",
    "primary_purpose",
]


# ============================================================
# NUMERIC FEATURE SCHEMA (CANONICAL ORDER — DO NOT CHANGE)
# ============================================================

# Numeric columns MUST be in this exact order for both train and inference.
NUMERIC_COLS = [
    # Eligibility (numeric)
    "eligibility_min_age_yrs",
    "eligibility_max_age_yrs",
    "min_age_missing",
    "max_age_missing",

    # Geography (U.S.-only, numeric by design)
    "n_sites",
    "n_us_states",
    "n_us_regions",
]

# Geography columns are REQUIRED and MUST be numeric
REQUIRED_GEO_NUMERIC_COLS = {
    "n_sites",
    "n_us_states",
    "n_us_regions",
}


# ============================================================
# TEXT FEATURE CONFIGURATION
# ============================================================

# Columns used to build the unified protocol narrative
TEXT_COLS = (
    "inclusion_text",
    "exclusion_text",
    "conditions_text",
    "interventions_text",
    "primary_outcome_text",
    "secondary_outcome_text",
)

# Embedding model configuration
TEXT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_EMBEDDING_BATCH_SIZE = 16
TEXT_EMBEDDING_MAX_LENGTH = 256
TEXT_EMBEDDING_NORMALIZE = False


# ============================================================
# MODEL / PIPELINE GUARDRAILS
# ============================================================

def validate_schema() -> None:
    """
    Hard-fail if the configuration violates core invariants.
    This should be called once at startup (train and inference).
    """
    # 1) Splits must sum to 1
    total = TRAIN_FRAC + TEST_FRAC + HOLDOUT_FRAC
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    # 2) Geography must NEVER be categorical
    overlap = set(CAT_COLS).intersection(REQUIRED_GEO_NUMERIC_COLS)
    if overlap:
        raise ValueError(
            "Geographic columns detected in CAT_COLS (forbidden): "
            f"{sorted(overlap)}"
        )

    # 3) Numeric schema must include all required geo columns
    if not REQUIRED_GEO_NUMERIC_COLS.issubset(NUMERIC_COLS):
        raise ValueError(
            "NUMERIC_COLS missing required geographic columns.\n"
            f"Required: {REQUIRED_GEO_NUMERIC_COLS}\n"
            f"Found: {set(NUMERIC_COLS)}"
        )


# ============================================================
# OPTIONAL: call validation immediately on import
# ============================================================

validate_schema()
