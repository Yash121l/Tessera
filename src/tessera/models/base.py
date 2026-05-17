"""Model interface, model card, and evaluation utilities."""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel


class CVScores(BaseModel):
    """Cross-validation performance scores for the model card."""

    mean_sharpe: float
    std_sharpe: float
    deflated_sharpe: float
    n_trials: int


class ModelCard(BaseModel):
    """Model metadata JSON sidecar (AFML §8)."""

    model_name: str
    model_type: str
    training_date: str
    git_commit: str
    data_version: str
    features: list[str]
    hyperparameters: dict[str, Any]
    cv_scores: CVScores | None = None


def get_git_commit() -> str:
    """Return the current HEAD commit hash, or 'unknown' on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S603, S607
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def compute_deflated_sharpe(
    sharpe_ratios: np.ndarray,
    n_obs: int,
    n_trials: int,
) -> float:
    """Deflated Sharpe ratio per Bailey & López de Prado (2014).

    Returns PSR(SR*) where SR* is the expected maximum Sharpe under the
    null hypothesis of zero skill, corrected for the number of independent
    trials (Optuna configurations tested).
    """
    from scipy.stats import norm

    if n_trials <= 1 or len(sharpe_ratios) == 0:
        return float(sharpe_ratios.max()) if len(sharpe_ratios) > 0 else 0.0

    sr_hat = float(sharpe_ratios.max())
    sr_std = float(sharpe_ratios.std(ddof=1)) if len(sharpe_ratios) > 1 else 1.0
    if sr_std < 1e-12:
        return sr_hat

    euler_mascheroni = 0.5772156649
    sr_0 = sr_std * (
        (1 - euler_mascheroni) * norm.ppf(1 - 1.0 / n_trials)
        + euler_mascheroni * norm.ppf(1 - 1.0 / (n_trials * np.e))
    )

    denom = sr_std / np.sqrt(max(n_obs - 1, 1))
    return float(norm.cdf((sr_hat - sr_0) / denom)) if denom > 1e-12 else 0.0


class Model(ABC):
    """Abstract base for all Tessera models."""

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,  # type: ignore[type-arg]
        sample_weight: np.ndarray | None = None,
        X_val: pd.DataFrame | None = None,  # noqa: N803
        y_val: pd.Series | None = None,  # type: ignore[type-arg]
    ) -> Self: ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...  # noqa: N803

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...  # noqa: N803

    @abstractmethod
    def save(self, path: Path) -> Path: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> Self: ...

    @abstractmethod
    def get_model_card(self) -> ModelCard: ...
