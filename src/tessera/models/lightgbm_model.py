"""LightGBM model implementations for primary and meta classification tasks."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import joblib
import numpy as np
import pandas as pd
import structlog

from tessera.config import TesseraSettings
from tessera.models.base import (
    CVScores,
    Model,
    ModelCard,
    compute_deflated_sharpe,
    get_git_commit,
)

logger = structlog.get_logger(__name__)


class _BaseLightGBMModel(Model):
    """Shared LightGBM logic for primary and meta models.

    Subclasses set ``_objective`` and ``_model_type`` class variables
    to configure the LightGBM objective and model-card type tag.
    """

    _objective: str
    _model_type: str

    def __init__(
        self,
        monotonic_constraints: dict[str, int] | None = None,
        seed: int | None = None,
        **lgb_params: Any,
    ) -> None:
        settings = TesseraSettings()
        self._seed = seed if seed is not None else settings.random_seed
        self._monotonic_constraints = monotonic_constraints or {}
        self._lgb_params = lgb_params
        self._model: Any = None
        self._feature_names: list[str] = []
        self._cv_scores: CVScores | None = None
        self._training_date: str = ""
        self._data_version: str = ""

    def _build_constraints_list(self, feature_names: list[str]) -> list[int]:
        return [self._monotonic_constraints.get(f, 0) for f in feature_names]

    def _make_lgbm(self, extra_params: dict[str, Any] | None = None) -> Any:
        import lightgbm as lgb

        params: dict[str, Any] = {
            "objective": self._objective,
            "random_state": self._seed,
            "n_jobs": 1,
            "verbose": -1,
        }
        params.update(self._lgb_params)
        if extra_params:
            params.update(extra_params)
        if self._feature_names and self._monotonic_constraints:
            params["monotone_constraints"] = self._build_constraints_list(self._feature_names)
        return lgb.LGBMClassifier(**params)

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
        import lightgbm as lgb

        self._feature_names = list(X.columns)
        if not self._training_date:
            self._training_date = datetime.now(UTC).isoformat()

        model = self._make_lgbm()

        fit_kwargs: dict[str, Any] = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["callbacks"] = [lgb.early_stopping(50, verbose=False)]

        model.fit(X, y, **fit_kwargs)
        self._model = model
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        return self._model.predict_proba(X)

    def save(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._model, path / "model.joblib")
        card = self.get_model_card()
        (path / "model_card.json").write_text(card.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> Self:
        card_data = json.loads((path / "model_card.json").read_text())
        card = ModelCard.model_validate(card_data)

        instance = cls.__new__(cls)
        instance._seed = 42
        instance._monotonic_constraints = {}
        instance._lgb_params = {}
        instance._model = joblib.load(path / "model.joblib")
        instance._feature_names = card.features
        instance._cv_scores = card.cv_scores
        instance._training_date = card.training_date
        instance._data_version = card.data_version
        return instance

    def get_model_card(self) -> ModelCard:
        params: dict[str, Any] = {}
        if self._model is not None:
            raw = self._model.get_params()
            params = {
                k: v
                for k, v in raw.items()
                if not callable(v) and k not in ("monotone_constraints",)
            }

        return ModelCard(
            model_name=self._model_type,
            model_type=self._model_type,
            training_date=self._training_date,
            git_commit=get_git_commit(),
            data_version=self._data_version,
            features=self._feature_names,
            hyperparameters=params,
            cv_scores=self._cv_scores,
        )

    # ------------------------------------------------------------------
    # Optuna hyper-parameter tuning
    # ------------------------------------------------------------------

    def tune(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,  # type: ignore[type-arg]
        t1: pd.Series,  # type: ignore[type-arg]
        sample_weight: np.ndarray | None = None,
        forward_returns: pd.Series | None = None,  # type: ignore[type-arg]
        n_trials: int = 100,
        n_splits: int = 5,
        pct_embargo: float = 0.01,
    ) -> Any:
        """Hyperparameter search via Optuna with PurgedKFold (AFML §7).

        After tuning, the final model is trained on the full dataset with
        the best parameters.  CV Sharpe statistics are stored in the model
        card for downstream reporting.
        """
        import lightgbm as lgb
        import optuna

        from tessera.cv.purged_kfold import PurgedKFold

        self._feature_names = list(X.columns)
        self._training_date = datetime.now(UTC).isoformat()

        cv = PurgedKFold(n_splits=n_splits, samples_info_sets=t1, pct_embargo=pct_embargo)
        constraints_list = self._build_constraints_list(self._feature_names)

        def objective(trial: Any) -> float:
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
                "bagging_freq": 1,
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            }

            fold_scores: list[float] = []
            for train_idx, val_idx in cv.split(X):
                base: dict[str, Any] = {
                    "objective": self._objective,
                    "random_state": self._seed,
                    "n_jobs": 1,
                    "verbose": -1,
                }
                if constraints_list:
                    base["monotone_constraints"] = constraints_list
                base.update(params)

                clf = lgb.LGBMClassifier(**base)
                x_tr, x_va = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
                sw = sample_weight[train_idx] if sample_weight is not None else None

                # Only use early-stopping eval_set when val labels ⊆ train labels.
                # If a rare class (e.g. label=0) is absent from the train fold,
                # LightGBM's internal LabelEncoder would raise ValueError on the
                # eval_set transform.
                train_cls = set(np.unique(y_tr.values.astype(int)))
                val_cls = set(np.unique(y_va.values.astype(int)))
                fit_kw: dict[str, Any] = {"sample_weight": sw}
                if val_cls <= train_cls:
                    fit_kw["eval_set"] = [(x_va, y_va)]
                    fit_kw["callbacks"] = [lgb.early_stopping(50, verbose=False)]

                clf.fit(x_tr, y_tr, **fit_kw)

                acc = float((clf.predict(x_va) == y_va).mean())
                fold_scores.append(acc)

            return float(np.mean(fold_scores))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=self._seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials)

        best = study.best_params
        best["bagging_freq"] = 1
        logger.info("optuna_complete", best_params=best, best_value=study.best_value)

        self._cv_scores = self._compute_cv_sharpe(
            X, y, t1, sample_weight, forward_returns, best, n_splits, pct_embargo, n_trials
        )

        self._lgb_params.update(best)
        self.fit(X, y, sample_weight=sample_weight)
        return study

    def _compute_cv_sharpe(
        self,
        X: pd.DataFrame,  # noqa: N803
        y: pd.Series,  # type: ignore[type-arg]
        t1: pd.Series,  # type: ignore[type-arg]
        sample_weight: np.ndarray | None,
        forward_returns: pd.Series | None,  # type: ignore[type-arg]
        params: dict[str, Any],
        n_splits: int,
        pct_embargo: float,
        n_trials: int,
    ) -> CVScores:
        import lightgbm as lgb

        from tessera.cv.purged_kfold import PurgedKFold

        cv = PurgedKFold(n_splits=n_splits, samples_info_sets=t1, pct_embargo=pct_embargo)
        constraints_list = self._build_constraints_list(self._feature_names)

        fold_sharpes: list[float] = []
        n_obs = 0

        for train_idx, val_idx in cv.split(X):
            base: dict[str, Any] = {
                "objective": self._objective,
                "random_state": self._seed,
                "n_jobs": 1,
                "verbose": -1,
            }
            if constraints_list:
                base["monotone_constraints"] = constraints_list
            base.update(params)

            clf = lgb.LGBMClassifier(**base)
            x_tr, x_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            sw = sample_weight[train_idx] if sample_weight is not None else None

            train_cls = set(np.unique(y_tr.values.astype(int)))
            val_cls = set(np.unique(y_va.values.astype(int)))
            fit_kw: dict[str, Any] = {"sample_weight": sw}
            if val_cls <= train_cls:
                fit_kw["eval_set"] = [(x_va, y_va)]
                fit_kw["callbacks"] = [lgb.early_stopping(50, verbose=False)]

            clf.fit(x_tr, y_tr, **fit_kw)

            preds = clf.predict(x_va)
            n_obs += len(val_idx)

            if forward_returns is not None:
                ret = forward_returns.iloc[val_idx].values
                strat = preds.astype(float) * ret
                std = float(strat.std())
                sharpe = float(strat.mean() / std) if std > 1e-12 else 0.0
            else:
                sharpe = float((preds == y_va).mean())

            fold_sharpes.append(sharpe)

        arr = np.array(fold_sharpes)
        return CVScores(
            mean_sharpe=float(arr.mean()),
            std_sharpe=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            deflated_sharpe=compute_deflated_sharpe(arr, n_obs, n_trials),
            n_trials=n_trials,
        )


class PrimaryLightGBMModel(_BaseLightGBMModel):
    """LightGBM classifier for the primary {-1, 0, +1} labeling task."""

    _objective = "multiclass"
    _model_type = "lightgbm_primary"


class MetaLightGBMModel(_BaseLightGBMModel):
    """LightGBM classifier for the binary {0, 1} meta-labeling task."""

    _objective = "binary"
    _model_type = "lightgbm_meta"

    def predict_bet_size(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Return P(primary correct) ∈ [0, 1]."""
        return self.predict_proba(X)[:, 1]
