"""
harry_ml

End-to-end machine learning pipeline for clinical trial demographic
enrollment prediction.

Public API:
- Configuration and schemas
- Structured + text feature builders
- Artifact save/load utilities

Internal modules (train, split, hurdle, infer) are intentionally NOT
auto-imported to avoid side effects.
"""

# Package version
__version__ = "0.1.0"

# ------------------------------------------------------------
# Configuration / schemas
# ------------------------------------------------------------
from .config import (
    RANDOM_SEED,
    TRAIN_FRAC,
    TEST_FRAC,
    HOLDOUT_FRAC,
    CAT_COLS,
    NUMERIC_COLS,
    REQUIRED_GEO_NUMERIC_COLS,
    TEXT_COLS,
    TEXT_EMBEDDING_MODEL,
    TEXT_EMBEDDING_BATCH_SIZE,
    TEXT_EMBEDDING_MAX_LENGTH,
    TEXT_EMBEDDING_NORMALIZE,
    validate_schema,
)

# ------------------------------------------------------------
# Structured features
# ------------------------------------------------------------
from .features_structured import (
    StructuredFeatureSet,
    build_categorical_frame,
    build_numeric_frame,
    build_structured_features_fit,
    build_structured_features_transform,
)

# ------------------------------------------------------------
# Text features
# ------------------------------------------------------------
from .features_text import (
    TextEmbeddingResult,
    build_protocol_text,
    build_text_embeddings,
)

# ------------------------------------------------------------
# Artifacts
# ------------------------------------------------------------
from .artifacts import (
    LoadedArtifacts,
    save_artifacts,
    load_artifacts,
    build_schema_manifest,
    print_artifact_summary,
)

# ------------------------------------------------------------
# Define explicit public surface
# ------------------------------------------------------------
__all__ = [
    # config
    "RANDOM_SEED",
    "TRAIN_FRAC",
    "TEST_FRAC",
    "HOLDOUT_FRAC",
    "CAT_COLS",
    "NUMERIC_COLS",
    "REQUIRED_GEO_NUMERIC_COLS",
    "TEXT_COLS",
    "TEXT_EMBEDDING_MODEL",
    "TEXT_EMBEDDING_BATCH_SIZE",
    "TEXT_EMBEDDING_MAX_LENGTH",
    "TEXT_EMBEDDING_NORMALIZE",
    "validate_schema",

    # structured features
    "StructuredFeatureSet",
    "build_categorical_frame",
    "build_numeric_frame",
    "build_structured_features_fit",
    "build_structured_features_transform",

    # text features
    "TextEmbeddingResult",
    "build_protocol_text",
    "build_text_embeddings",

    # artifacts
    "LoadedArtifacts",
    "save_artifacts",
    "load_artifacts",
    "build_schema_manifest",
    "print_artifact_summary",
]
