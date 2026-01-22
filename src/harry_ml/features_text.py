# src/harry_ml/features_text.py
"""
Text feature engineering + transformer embeddings for the Harry pipeline.

Matches the logic in Harry_3-11_clean.py:
- Builds a single merged protocol text field from up to 6 narrative columns
- Uses HuggingFace "sentence-transformers/all-MiniLM-L6-v2"
- Computes sentence embeddings via mean pooling over token embeddings (with attention mask)
- Runs safely on CPU (macOS-friendly)

Public API:
- build_protocol_text(df) -> pd.Series
- embed_texts(texts, ...) -> np.ndarray
- build_text_embeddings(df, ...) -> TextEmbeddingResult
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# Default model used in your extracted script
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Columns used to build the merged narrative field (as in your script)
DEFAULT_TEXT_COLS: Tuple[str, ...] = (
    "inclusion_text",
    "exclusion_text",
    "conditions_text",
    "interventions_text",
    "primary_outcome_text",
    "secondary_outcome_text",
)


@dataclass(frozen=True)
class TextEmbeddingResult:
    merged_text: pd.Series            # length n
    embeddings: np.ndarray            # shape (n, d)
    model_name: str
    max_length: int
    batch_size: int


def build_protocol_text(
    df_clean: pd.DataFrame,
    text_cols: Sequence[str] = DEFAULT_TEXT_COLS,
) -> pd.Series:
    """
    Build the merged protocol text used for embeddings, WITHOUT mutating df_clean.

    Behavior mirrors your notebook/script:
    merged_text = inclusion + exclusion + conditions + interventions + primary_outcome + secondary_outcome
    and then normalize whitespace.

    Missing columns are treated as empty strings.
    """
    parts: List[pd.Series] = []
    for col in text_cols:
        if col in df_clean.columns:
            s = df_clean[col]
            # Defensive: convert to string and fill NaN
            s = s.fillna("").astype(str)
        else:
            # Column missing => empty string series
            s = pd.Series([""] * len(df_clean), index=df_clean.index, dtype="object")
        parts.append(s)

    merged = parts[0]
    for s in parts[1:]:
        merged = merged + " " + s

    merged = merged.str.replace(r"\s+", " ", regex=True).str.strip()
    return merged


@lru_cache(maxsize=4)
def _load_hf_model(model_name: str):
    """
    Cached load for tokenizer/model. Keeps CPU for stability.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    return tokenizer, model, device


def _mean_pool(last_hidden_state, attention_mask):
    """
    Mean pooling with attention mask.
    last_hidden_state: (batch, seq_len, hidden)
    attention_mask: (batch, seq_len)
    """
    import torch

    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (batch, seq_len, 1)
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)                                      # (batch, hidden)
    counts = mask.sum(dim=1).clamp(min=1e-9)                        # (batch, 1)
    return summed / counts


def embed_texts(
    texts: Iterable[str],
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 16,
    max_length: int = 256,
    normalize: bool = False,
) -> np.ndarray:
    """
    Encode texts into embeddings using the HF MiniLM model + mean pooling.

    Returns:
      embeddings: np.ndarray shape (n_texts, embedding_dim)

    Notes:
    - Runs on CPU (device fixed to CPU)
    - Deterministic for inference (dropout disabled via model.eval())
    """
    import torch

    tokenizer, model, device = _load_hf_model(model_name)

    # Materialize iterable once; keep stable ordering
    text_list = list(texts)
    all_embs: List[np.ndarray] = []

    for start in range(0, len(text_list), batch_size):
        batch = text_list[start : start + batch_size]

        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # Move to device
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            output = model(**enc)

        pooled = _mean_pool(output.last_hidden_state, enc["attention_mask"])
        emb = pooled.cpu().numpy()
        all_embs.append(emb)

    X = np.vstack(all_embs) if all_embs else np.zeros((0, 0), dtype=float)

    if normalize and X.size:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        X = X / norms

    return X


def build_text_embeddings(
    df_clean: pd.DataFrame,
    *,
    text_cols: Sequence[str] = DEFAULT_TEXT_COLS,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 16,
    max_length: int = 256,
    normalize: bool = False,
) -> TextEmbeddingResult:
    """
    Convenience wrapper:
      - build merged_text from df_clean
      - embed merged_text
      - return both in a single result object
    """
    merged = build_protocol_text(df_clean, text_cols=text_cols)
    X = embed_texts(
        merged.astype(str).tolist(),
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        normalize=normalize,
    )
    return TextEmbeddingResult(
        merged_text=merged,
        embeddings=X,
        model_name=model_name,
        max_length=max_length,
        batch_size=batch_size,
    )

