"""Convex-combination ensemble of Tessera models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import structlog

from tessera.models.base import Model, ModelCard

logger = structlog.get_logger(__name__)


class EnsembleModel(Model):
    """Weighted average of multiple model predictions.

    Weights are constrained to the probability simplex (non-negative,
    sum to 1) and chosen on a held-out validation fold by maximising
    the Sharpe ratio of the combined strategy.
    """

    def __init__(self, models: list[Model]) -> None:
        if len(models) < 2:
            msg = "EnsembleModel requires at least 2 component models"
            raise ValueError(msg)
        self.models = models
        self.weights = np.ones(len(models)) / len(models)
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Weight optimisation
    # ------------------------------------------------------------------

    def fit_weights(
        self,
        X_val: pd.DataFrame,  # noqa: N803
        forward_returns: pd.Series,  # type: ignore[type-arg]
    ) -> np.ndarray:
        """Find Sharpe-optimal convex weights on the validation set.

        Uses SLSQP with the simplex constraint w ≥ 0, Σw = 1.
        """
        from scipy.optimize import minimize

        if not self._feature_names:
            self._feature_names = list(X_val.columns)

        predictions = np.array([m.predict(X_val).astype(float) for m in self.models])
        ret = forward_returns.values.astype(float)

        def neg_sharpe(w: np.ndarray) -> float:
            combined = w @ predictions
            strat = combined * ret
            std = float(strat.std())
            return float(-(strat.mean() / std)) if std > 1e-12 else 0.0

        n = len(self.models)
        result = minimize(
            neg_sharpe,
            x0=np.ones(n) / n,
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints={"type": "eq", "fun": lambda w: float(np.sum(w) - 1)},
        )
        self.weights = result.x
        logger.info("ensemble_weights", weights=self.weights.tolist())
        return self.weights

    # ------------------------------------------------------------------
    # Model interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,  # type: ignore[type-arg]
        sample_weight: np.ndarray | None = None,
        X_val: pd.DataFrame | None = None,  # noqa: N803
        y_val: pd.Series | None = None,  # type: ignore[type-arg]
    ) -> Self:
        self._feature_names = list(X.columns)
        for model in self.models:
            model.fit(X, y, sample_weight=sample_weight, X_val=X_val, y_val=y_val)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        predictions = np.array([m.predict(X).astype(float) for m in self.models])
        combined = self.weights @ predictions
        return np.round(combined).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        probas = np.array([m.predict_proba(X) for m in self.models])
        return np.tensordot(self.weights, probas, axes=([0], [0]))

    def save(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        for i, model in enumerate(self.models):
            model.save(path / f"component_{i}")
        meta = {
            "weights": self.weights.tolist(),
            "n_components": len(self.models),
            "component_types": [type(m).__name__ for m in self.models],
        }
        (path / "ensemble_meta.json").write_text(json.dumps(meta, indent=2))
        card = self.get_model_card()
        (path / "model_card.json").write_text(card.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> Self:
        from tessera.models.lightgbm_model import MetaLightGBMModel, PrimaryLightGBMModel

        _registry: dict[str, type[Model]] = {
            "PrimaryLightGBMModel": PrimaryLightGBMModel,
            "MetaLightGBMModel": MetaLightGBMModel,
        }

        meta = json.loads((path / "ensemble_meta.json").read_text())
        models: list[Model] = []
        for i in range(meta["n_components"]):
            cls_name = meta["component_types"][i]
            model_cls = _registry[cls_name]
            models.append(model_cls.load(path / f"component_{i}"))

        instance = cls(models)
        instance.weights = np.array(meta["weights"])
        return instance

    def get_model_card(self) -> ModelCard:
        return ModelCard(
            model_name="ensemble",
            model_type="ensemble",
            training_date="",
            git_commit="",
            data_version="",
            features=self._feature_names,
            hyperparameters={"weights": self.weights.tolist()},
        )
