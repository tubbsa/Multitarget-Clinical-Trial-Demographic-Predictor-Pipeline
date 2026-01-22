# src/harry_ml/hurdle.py
"""
Two-stage (hurdle) modeling utilities for sparse demographic targets.

Use case (matches your dissertation pipeline):
- Some targets (e.g., AIAN, NHPI, sometimes Asian depending on filtering) are
  zero-inflated / sparse.
- Stage 1: presence classifier predicts whether target proportion > 0.
- Stage 2: conditional regressor predicts value given presence.
- Final prediction: p(present)*E[value | present] OR hard-gated variant.

This module is model-agnostic and works with:
- CatBoostClassifier / CatBoostRegressor
- sklearn-like estimators supporting fit/predict(_proba)

It does NOT assemble features; you pass X (numpy array or scipy sparse).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ---------------------------
# Data structures
# ---------------------------
@dataclass(frozen=True)
class HurdleBundle:
    """
    A single target's hurdle model:
      - presence_model: classifier (predict_proba or decision_function/predict)
      - reg_model: regressor (predict)
      - threshold: decision boundary for presence (on probability unless specified)
      - zero_value: value to output when predicted absent
    """
    presence_model: Any
    reg_model: Any
    threshold: float = 0.5
    zero_value: float = 0.0


@dataclass(frozen=True)
class HurdlePrediction:
    """
    Container for debugging/analysis.
    """
    y_pred: np.ndarray
    p_present: np.ndarray
    y_cond: np.ndarray
    present_mask: np.ndarray


# ---------------------------
# Helpers
# ---------------------------
def _to_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1:
        return x[:, 0]
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {x.shape}")
    return x


def _clip01(y: np.ndarray) -> np.ndarray:
    return np.clip(y, 0.0, 1.0)


def _predict_presence_proba(model: Any, X) -> np.ndarray:
    """
    Returns P(present) as float array shape (n,).
    Prefers predict_proba[:, 1]. Falls back to decision_function->sigmoid,
    else uses predict() as 0/1.
    """
    # sklearn / catboost typically have predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return _to_1d(proba[:, 1])
        # Some libs may return (n,) already
        return _to_1d(proba)

    # decision_function fallback
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        scores = _to_1d(scores)
        # sigmoid
        return 1.0 / (1.0 + np.exp(-scores))

    # last resort: predict as hard labels
    if hasattr(model, "predict"):
        pred = np.asarray(model.predict(X))
        pred = _to_1d(pred)
        # if labels are not 0/1, coerce
        return (pred > 0).astype(float)

    raise TypeError("Presence model must implement predict_proba, decision_function, or predict.")


def make_presence_labels(y: np.ndarray, *, eps: float = 0.0) -> np.ndarray:
    """
    Converts a continuous proportion target into 0/1 presence labels.
    present = y > eps.
    """
    y = _to_1d(np.asarray(y))
    return (y > eps).astype(int)


# ---------------------------
# Fit
# ---------------------------
def fit_hurdle_models(
    X_train,
    y_train: np.ndarray,
    *,
    presence_model: Any,
    reg_model: Any,
    eps: float = 0.0,
    reg_fit_only_on_present: bool = True,
) -> HurdleBundle:
    """
    Fit a hurdle model for a single target.

    Args:
      X_train: feature matrix
      y_train: target vector in [0,1]
      presence_model: classifier instance (will be fit)
      reg_model: regressor instance (will be fit)
      eps: presence threshold for label creation (default 0)
      reg_fit_only_on_present: fit regressor only on samples where y > eps

    Returns:
      HurdleBundle with fitted models.
    """
    y_train = _to_1d(np.asarray(y_train))
    y_train = _clip01(y_train)

    y_presence = make_presence_labels(y_train, eps=eps)
    presence_model.fit(X_train, y_presence)

    if reg_fit_only_on_present:
        mask = y_train > eps
        if mask.sum() == 0:
            # Nothing present; fit a degenerate regressor if possible
            # but safest is to still call fit on empty? Many models can't.
            # Instead, keep reg_model unfit and handle downstream.
            # Here we fit on a single dummy point at zero to avoid crashes.
            X_one = X_train[:1]
            y_one = np.array([0.0])
            reg_model.fit(X_one, y_one)
        else:
            reg_model.fit(X_train[mask], y_train[mask])
    else:
        reg_model.fit(X_train, y_train)

    return HurdleBundle(presence_model=presence_model, reg_model=reg_model, threshold=0.5, zero_value=0.0)


# ---------------------------
# Predict
# ---------------------------
def predict_hurdle(
    bundle: HurdleBundle,
    X,
    *,
    hard_gate: bool = False,
    clip01: bool = True,
) -> HurdlePrediction:
    """
    Predict using a hurdle model.

    If hard_gate=False (recommended):
      y_pred = p_present * y_cond
      (smooth; works better for calibration)

    If hard_gate=True:
      y_pred = y_cond if p_present >= threshold else 0
      (more literal hurdle; may be brittle)

    Returns:
      HurdlePrediction with intermediates.
    """
    p_present = _predict_presence_proba(bundle.presence_model, X)

    # Conditional regression
    if hasattr(bundle.reg_model, "predict"):
        y_cond = np.asarray(bundle.reg_model.predict(X))
        y_cond = _to_1d(y_cond)
    else:
        raise TypeError("Regressor must implement predict().")

    if clip01:
        y_cond = _clip01(y_cond)

    present_mask = p_present >= bundle.threshold

    if hard_gate:
        y_pred = np.where(present_mask, y_cond, bundle.zero_value)
    else:
        y_pred = p_present * y_cond

    if clip01:
        y_pred = _clip01(y_pred)

    return HurdlePrediction(
        y_pred=y_pred,
        p_present=p_present,
        y_cond=y_cond,
        present_mask=present_mask,
    )


def predict_hurdle_targets(
    bundles: Dict[str, HurdleBundle],
    X,
    *,
    hard_gate: bool = False,
    clip01: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Convenience for multiple sparse targets.

    Returns dict target_name -> y_pred (n,)
    """
    out: Dict[str, np.ndarray] = {}
    for name, b in bundles.items():
        out[name] = predict_hurdle(b, X, hard_gate=hard_gate, clip01=clip01).y_pred
    return out
