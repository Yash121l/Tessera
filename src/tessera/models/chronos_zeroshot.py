"""Chronos zero-shot directional forecaster.

Uses Amazon's Chronos-Bolt-Base pre-trained model (zero-shot, no fine-tuning)
to forecast the next 5-minute BTCUSDT log-return. The forecast median is
thresholded into a {-1, 0, +1} signal.

Design:
  - Input feature: a column named ``log_return`` (or the first feature column).
  - ``fit(X, y)``: calibrates the threshold on the training period by
    minimising the difference between P50 forecast direction and true labels.
    All Chronos inference is zero-shot — the model weights are never updated.
  - ``predict(X)``: runs Chronos on the log-return column, applies threshold.

Reproducibility:
  - Model is pinned to ``CHRONOS_MODEL_REVISION`` (git commit on HuggingFace).
  - ``torch.manual_seed`` is called before each inference batch.

Requires: ``uv sync --extra sequence``
  (chronos-forecasting[torch])
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import structlog

from tessera.models.base import Model, ModelCard, get_git_commit

logger = structlog.get_logger(__name__)

# Pinned HuggingFace revision for reproducibility — update consciously
CHRONOS_MODEL_ID = "amazon/chronos-bolt-base"
CHRONOS_MODEL_REVISION = "main"  # pin to a commit SHA in production

_LOOKBACK = 60
_PREDICTION_LENGTH = 1


def _check_deps() -> None:
    try:
        from chronos import ChronosBoltPipeline  # noqa: F401
    except ImportError as e:
        msg = "Chronos requires chronos-forecasting. Install with: uv sync --extra sequence"
        raise ImportError(msg) from e


def _load_pipeline(device: str = "cpu") -> Any:
    """Load the pinned Chronos-Bolt pipeline (downloads on first call, cached thereafter)."""
    from chronos import ChronosBoltPipeline

    pipeline = ChronosBoltPipeline.from_pretrained(
        CHRONOS_MODEL_ID,
        revision=CHRONOS_MODEL_REVISION,
        device_map=device,
    )
    logger.info(
        "chronos_loaded",
        model_id=CHRONOS_MODEL_ID,
        revision=CHRONOS_MODEL_REVISION,
    )
    return pipeline


def _forecast_p50(
    pipeline: Any,
    series: np.ndarray,
    lookback: int,
    seed: int,
) -> np.ndarray:
    """Run Chronos-Bolt zero-shot; return P50 for each 1-step-ahead forecast.

    For T observations, returns T - lookback + 1 forecasts.
    Each forecast uses the preceding ``lookback`` values as context.

    ChronosBoltPipeline.predict() returns [batch, num_quantiles, prediction_length]
    where num_quantiles=9 for [0.1, 0.2, ..., 0.9]. Index 4 is the P50 (median).
    """
    import torch

    n = len(series)
    results = []

    for i in range(lookback - 1, n):
        context = series[i - lookback + 1 : i + 1]
        ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)

        torch.manual_seed(seed)
        with torch.no_grad():
            # forecast: [batch=1, num_quantiles=9, prediction_length=1]
            forecast = pipeline.predict(
                ctx_tensor,
                prediction_length=_PREDICTION_LENGTH,
            )

        # P50 = quantile index 4 (0.5 in [0.1, 0.2, ..., 0.9])
        p50 = float(forecast[0, 4, 0].item())
        results.append(p50)

    return np.array(results, dtype=float)


def _calibrate_threshold(p50: np.ndarray, labels: np.ndarray) -> float:
    """Find threshold τ that maximises direction accuracy on (p50, labels).

    Signal: +1 if p50 > τ, -1 if p50 < -τ, else 0.
    Searches τ in [0, 3σ] where σ = std(p50).
    """
    sigma = float(np.std(p50)) or 1e-6
    best_tau = 0.0
    best_acc = -1.0

    for tau in np.linspace(0, 3 * sigma, 30):
        signal = np.where(p50 > tau, 1, np.where(p50 < -tau, -1, 0))
        acc = float((signal == labels).mean())
        if acc > best_acc:
            best_acc = acc
            best_tau = tau

    logger.info("chronos_threshold_calibrated", tau=round(best_tau, 6), accuracy=round(best_acc, 4))
    return best_tau


class ChronosZeroShotModel(Model):
    """Zero-shot Chronos directional signal with calibrated threshold.

    Chronos weights are never updated — only the classification threshold
    is fitted on the training period.
    """

    def __init__(
        self,
        lookback: int = _LOOKBACK,
        return_col: str = "log_return",
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        self._lookback = lookback
        self._return_col = return_col
        self._seed = seed
        self._device = device

        self._pipeline: Any = None
        self._threshold: float = 0.0
        self._context_series: np.ndarray | None = None  # last lookback bars of training returns
        self._feature_names: list[str] = []
        self._training_date: str = ""
        self._data_version: str = ""

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_return_series(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        if self._return_col in X.columns:
            return X[self._return_col].values.astype(float)
        # Fallback: use the first column
        logger.warning(
            "chronos_return_col_missing",
            expected=self._return_col,
            using=X.columns[0],
        )
        return X.iloc[:, 0].values.astype(float)

    def _ensure_pipeline(self) -> None:
        if self._pipeline is None:
            _check_deps()
            self._pipeline = _load_pipeline(device=self._device)

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
        self._ensure_pipeline()
        self._feature_names = list(X.columns)
        self._training_date = datetime.now(UTC).isoformat()

        series = self._get_return_series(X)

        logger.info(
            "chronos_fit_start",
            n_bars=len(series),
            lookback=self._lookback,
        )

        p50 = _forecast_p50(self._pipeline, series, self._lookback, self._seed)

        # Align labels: first valid index = lookback - 1
        labels = y.values[self._lookback - 1 :]

        if X_val is not None and y_val is not None:
            # Calibrate on provided (purged) validation fold
            calib_series = self._get_return_series(X_val)
            calib_p50 = _forecast_p50(self._pipeline, calib_series, self._lookback, self._seed)
            calib_labels = y_val.values[self._lookback - 1 :]
            self._threshold = _calibrate_threshold(calib_p50, calib_labels)
        else:
            # Calibrate on training period (in-sample — less principled)
            self._threshold = _calibrate_threshold(p50, labels)

        # Store context so predict() works on short sequences (< lookback bars)
        self._context_series = series[-self._lookback :].copy()

        return self

    def _full_series(self, X: pd.DataFrame) -> tuple[np.ndarray, int]:  # noqa: N803
        """Return (full_series, n_context) where full_series = context + X returns.

        n_context is the number of context rows prepended from training.
        Results for X row i correspond to p50[i] (no extra padding needed).
        """
        series = self._get_return_series(X)
        if self._context_series is not None:
            ctx = self._context_series[-self._lookback :]
            return np.concatenate([ctx, series]), len(ctx)
        # No context: zero-pad so every X row has a full lookback window
        pad = np.zeros(self._lookback, dtype=float)
        return np.concatenate([pad, series]), self._lookback

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        self._ensure_pipeline()
        full, n_ctx = self._full_series(X)
        tau = self._threshold

        p50_all = _forecast_p50(self._pipeline, full, self._lookback, self._seed)
        # p50_all[i] = forecast given full[i : i+lookback]
        # full[n_ctx] = X[0], so p50_all[0] forecasts X[0]
        n = len(X)
        p50 = p50_all[:n]  # exactly one forecast per X row
        return np.where(p50 > tau, 1, np.where(p50 < -tau, -1, 0)).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:  # noqa: N803
        """Soft probabilities from the full Chronos forecast distribution."""
        self._ensure_pipeline()
        import torch

        full, n_ctx = self._full_series(X)
        n_full = len(full)
        tau = self._threshold
        n = len(X)

        results = []
        for i in range(self._lookback - 1, n_full):
            context = full[i - self._lookback + 1 : i + 1]
            ctx_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
            torch.manual_seed(self._seed)
            with torch.no_grad():
                # Bolt: [batch=1, num_quantiles=9, prediction_length=1]
                forecast = self._pipeline.predict(
                    ctx_tensor,
                    prediction_length=_PREDICTION_LENGTH,
                )
            # Use all 9 quantile values as a proxy distribution
            quantile_vals = forecast[0, :, 0].numpy()  # shape [9]
            p_down = float((quantile_vals < -tau).mean())
            p_up = float((quantile_vals > tau).mean())
            p_flat = max(0.0, 1.0 - p_down - p_up)
            results.append([p_down, p_flat, p_up])

        arr = np.array(results)  # shape [n_full - lookback + 1, 3]
        return arr[:n]  # first n rows correspond to X[0..n-1]

    def save(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "lookback": self._lookback,
            "return_col": self._return_col,
            "seed": self._seed,
            "device": self._device,
            "threshold": self._threshold,
            "model_id": CHRONOS_MODEL_ID,
            "model_revision": CHRONOS_MODEL_REVISION,
        }
        (path / "chronos_meta.json").write_text(json.dumps(meta, indent=2))
        if self._context_series is not None:
            np.save(str(path / "context_series.npy"), self._context_series)
        card = self.get_model_card()
        (path / "model_card.json").write_text(card.model_dump_json(indent=2))
        return path

    @classmethod
    def load(cls, path: Path) -> Self:
        meta = json.loads((path / "chronos_meta.json").read_text())
        card_data = json.loads((path / "model_card.json").read_text())
        card = ModelCard.model_validate(card_data)

        instance = cls.__new__(cls)
        instance._lookback = meta["lookback"]
        instance._return_col = meta["return_col"]
        instance._seed = meta["seed"]
        instance._device = meta["device"]
        instance._threshold = meta["threshold"]
        instance._pipeline = None  # loaded lazily

        ctx_path = path / "context_series.npy"
        instance._context_series = np.load(str(ctx_path)) if ctx_path.exists() else None

        instance._feature_names = card.features
        instance._training_date = card.training_date
        instance._data_version = card.data_version
        return instance

    def get_model_card(self) -> ModelCard:
        return ModelCard(
            model_name="chronos_zeroshot",
            model_type="chronos_zeroshot",
            training_date=self._training_date,
            git_commit=get_git_commit(),
            data_version=self._data_version,
            features=self._feature_names,
            hyperparameters={
                "model_id": CHRONOS_MODEL_ID,
                "model_revision": CHRONOS_MODEL_REVISION,
                "lookback": self._lookback,
                "threshold": self._threshold,
                "fine_tuned": False,
            },
        )
