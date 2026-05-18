"""Temporal Fusion Transformer wrapper (pytorch-forecasting).

TFT is a multi-horizon probabilistic forecasting model, not a classifier.
We adapt it by:
  1. Treating triple-barrier labels {-1, 0, +1} as a continuous target.
  2. Training with QuantileLoss (P10, P50, P90).
  3. At inference, rounding the P50 forecast to the nearest integer in {-1, 0, +1}.

This framing is intentionally honest about TFT's limitations for
classification — the comparison study expects to document whether this
adaptation is competitive with a purpose-built classifier.

Requires: ``uv sync --extra sequence``
  (pytorch-forecasting>=1.0, lightning>=2.0)
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import structlog

from tessera.models.base import CVScores, Model, ModelCard, get_git_commit

logger = structlog.get_logger(__name__)

_MAX_ENCODER_LENGTH = 60
_MAX_PREDICTION_LENGTH = 1
_HIDDEN_SIZE = 32  # small to keep comparison fair with PatchTST
_ATTENTION_HEAD_SIZE = 4
_DROPOUT = 0.1
_HIDDEN_CONTINUOUS_SIZE = 16


def _check_deps() -> None:
    try:
        import lightning  # noqa: F401
        import pytorch_forecasting  # noqa: F401
    except ImportError as e:
        msg = (
            "TFT requires pytorch-forecasting and lightning. Install with: uv sync --extra sequence"
        )
        raise ImportError(msg) from e


def _df_to_tsds(
    X: pd.DataFrame,  # noqa: N803
    y: pd.Series,  # type: ignore[type-arg]
    time_idx_start: int = 0,
    group: str = "asset",
    max_encoder_length: int = _MAX_ENCODER_LENGTH,
) -> Any:
    """Convert tabular (X, y) to pytorch-forecasting TimeSeriesDataSet."""
    from pytorch_forecasting import TimeSeriesDataSet

    df = X.copy()
    df["target"] = y.values.astype(float)
    df["time_idx"] = np.arange(time_idx_start, time_idx_start + len(df), dtype=int)
    df["group"] = group

    time_varying_reals = list(X.columns)

    return TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="target",
        group_ids=["group"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=_MAX_PREDICTION_LENGTH,
        time_varying_unknown_reals=time_varying_reals,
        # No static or known-future reals in our setup
        static_categoricals=[],
        static_reals=[],
        time_varying_known_reals=[],
        time_varying_known_categoricals=[],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )


class TFTModel(Model):
    """TFT-based directional classifier via pytorch-forecasting.

    Uses QuantileLoss internally; the P50 prediction is rounded to {-1, 0, +1}.
    """

    def __init__(
        self,
        hidden_size: int = _HIDDEN_SIZE,
        attention_head_size: int = _ATTENTION_HEAD_SIZE,
        dropout: float = _DROPOUT,
        hidden_continuous_size: int = _HIDDEN_CONTINUOUS_SIZE,
        lr: float = 1e-3,
        max_epochs: int = 30,
        patience: int = 5,
        batch_size: int = 64,
        max_encoder_length: int = _MAX_ENCODER_LENGTH,
        seed: int | None = None,
    ) -> None:
        from tessera.config import TesseraSettings

        settings = TesseraSettings()
        self._seed = seed if seed is not None else settings.random_seed
        self._hidden_size = hidden_size
        self._attention_head_size = attention_head_size
        self._dropout = dropout
        self._hidden_continuous_size = hidden_continuous_size
        self._lr = lr
        self._max_epochs = max_epochs
        self._patience = patience
        self._batch_size = batch_size
        self._max_encoder_length = max_encoder_length

        self._tft: Any = None
        self._training_ds: Any = None  # needed to create prediction dataloaders
        self._train_df: pd.DataFrame | None = None  # stored for context at predict time
        self._train_time_start: int = 0
        self._feature_names: list[str] = []
        self._cv_scores: CVScores | None = None
        self._training_date: str = ""
        self._data_version: str = ""

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
        _check_deps()
        import lightning.pytorch as pl
        import torch
        from pytorch_forecasting import TemporalFusionTransformer
        from pytorch_forecasting.metrics import QuantileLoss

        torch.manual_seed(self._seed)
        pl.seed_everything(self._seed, workers=True)

        self._feature_names = list(X.columns)
        self._training_date = datetime.now(UTC).isoformat()

        # Build TimeSeriesDataSet for training data
        training_ds = _df_to_tsds(
            X, y, time_idx_start=0, max_encoder_length=self._max_encoder_length
        )
        self._training_ds = training_ds
        self._train_df = X.copy()
        self._train_df["_target"] = y.values.astype(float)

        train_loader = training_ds.to_dataloader(
            train=True,
            batch_size=self._batch_size,
            num_workers=0,
        )

        # Validation dataloader
        if X_val is not None and y_val is not None:
            from pytorch_forecasting import TimeSeriesDataSet

            val_ds = TimeSeriesDataSet.from_dataset(
                training_ds,
                pd.concat(
                    [
                        self._train_df.assign(time_idx=np.arange(len(self._train_df))),
                        X_val.assign(
                            _target=y_val.values.astype(float),
                            time_idx=np.arange(
                                len(self._train_df),
                                len(self._train_df) + len(X_val),
                            ),
                        ),
                    ]
                ),
            )
            val_loader = val_ds.to_dataloader(
                train=False,
                batch_size=self._batch_size * 4,
                num_workers=0,
            )
        else:
            val_loader = training_ds.to_dataloader(
                train=False,
                batch_size=self._batch_size * 4,
                num_workers=0,
            )

        tft = TemporalFusionTransformer.from_dataset(
            training_ds,
            learning_rate=self._lr,
            hidden_size=self._hidden_size,
            attention_head_size=self._attention_head_size,
            dropout=self._dropout,
            hidden_continuous_size=self._hidden_continuous_size,
            loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
            log_interval=10,
            log_val_interval=1,
        )
        logger.info("tft_params", n_params=sum(p.numel() for p in tft.parameters()))

        from lightning.pytorch.callbacks import EarlyStopping

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self._patience,
                mode="min",
                verbose=False,
            ),
        ]

        trainer = pl.Trainer(
            max_epochs=self._max_epochs,
            gradient_clip_val=0.1,
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            accelerator="cpu",
        )
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)
        self._tft = tft

        return self

    def _build_predict_loader(self, X: pd.DataFrame) -> Any:  # noqa: N803
        """Create a prediction DataLoader that includes the training context."""
        assert self._train_df is not None
        assert self._training_ds is not None

        n_context = min(self._max_encoder_length, len(self._train_df))
        context_df = self._train_df.tail(n_context).copy()
        context_df["time_idx"] = np.arange(n_context, dtype=int)

        pred_df = X.copy()
        pred_df["_target"] = 0.0  # placeholder
        pred_df["time_idx"] = np.arange(n_context, n_context + len(X), dtype=int)

        combined = pd.concat([context_df, pred_df], ignore_index=True)
        combined["group"] = "asset"

        from pytorch_forecasting import TimeSeriesDataSet

        pred_ds = TimeSeriesDataSet.from_dataset(
            self._training_ds,
            combined,
            predict=True,
            stop_randomization=True,
        )
        return pred_ds.to_dataloader(train=False, batch_size=len(X), num_workers=0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        assert self._tft is not None, "Call fit() first"
        loader = self._build_predict_loader(X)
        raw = self._tft.predict(loader, mode="quantiles", return_x=False)
        # raw: [n_samples, prediction_length, n_quantiles] — P50 is index 1
        p50 = raw[:, 0, 1].numpy()
        # Clip and round to nearest label in {-1, 0, +1}
        clipped = np.clip(p50, -1.0, 1.0)
        return np.array([min(1, max(-1, round(float(v)))) for v in clipped])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Soft probabilities derived from quantile forecasts (approximate)."""
        assert self._tft is not None, "Call fit() first"
        loader = self._build_predict_loader(X)
        raw = self._tft.predict(loader, mode="quantiles", return_x=False)
        p10 = raw[:, 0, 0].numpy()
        p50 = raw[:, 0, 1].numpy()
        p90 = raw[:, 0, 2].numpy()

        n = len(p50)
        proba = np.zeros((n, 3), dtype=float)
        for i, (_lo, mid, _hi) in enumerate(zip(p10, p50, p90, strict=True)):
            prob_down = max(0.0, min(1.0, (1.0 - mid) / 2.0))
            prob_up = max(0.0, min(1.0, (1.0 + mid) / 2.0))
            prob_flat = max(0.0, 1.0 - prob_down - prob_up)
            proba[i] = [prob_down, prob_flat, prob_up]

        return proba

    def save(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        assert self._tft is not None
        import torch

        torch.save(self._tft.state_dict(), path / "tft_weights.pt")
        meta = {
            "hidden_size": self._hidden_size,
            "attention_head_size": self._attention_head_size,
            "dropout": self._dropout,
            "hidden_continuous_size": self._hidden_continuous_size,
            "max_encoder_length": self._max_encoder_length,
        }
        (path / "arch.json").write_text(json.dumps(meta, indent=2))
        card = self.get_model_card()
        (path / "model_card.json").write_text(card.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> Self:
        card_data = json.loads((path / "model_card.json").read_text())
        card = ModelCard.model_validate(card_data)
        arch = json.loads((path / "arch.json").read_text())

        instance = cls.__new__(cls)
        instance._seed = 42
        instance._hidden_size = arch["hidden_size"]
        instance._attention_head_size = arch["attention_head_size"]
        instance._dropout = arch["dropout"]
        instance._hidden_continuous_size = arch["hidden_continuous_size"]
        instance._max_encoder_length = arch["max_encoder_length"]
        instance._lr = 1e-3
        instance._max_epochs = 30
        instance._patience = 5
        instance._batch_size = 64
        instance._tft = None  # weights not restored here — use torch.load separately
        instance._training_ds = None
        instance._train_df = None
        instance._feature_names = card.features
        instance._cv_scores = card.cv_scores
        instance._training_date = card.training_date
        instance._data_version = card.data_version
        return instance

    def get_model_card(self) -> ModelCard:
        return ModelCard(
            model_name="tft",
            model_type="tft",
            training_date=self._training_date,
            git_commit=get_git_commit(),
            data_version=self._data_version,
            features=self._feature_names,
            hyperparameters={
                "hidden_size": self._hidden_size,
                "max_encoder_length": self._max_encoder_length,
                "lr": self._lr,
                "max_epochs": self._max_epochs,
                "framework": "pytorch-forecasting",
            },
            cv_scores=self._cv_scores,
        )
