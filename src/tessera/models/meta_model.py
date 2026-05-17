"""Meta-labeling model (AFML §3.6).

Combines a primary model's directional signal with the full feature set
to train a secondary classifier that predicts P(primary is correct).
The output ∈ [0, 1] feeds downstream as ``bet_size_raw`` for Kelly sizing.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import structlog

from tessera.models.lightgbm_model import MetaLightGBMModel, PrimaryLightGBMModel

if TYPE_CHECKING:
    from tessera.models.base import Model

logger = structlog.get_logger(__name__)


class MetaModel:
    """Wraps a primary model and a meta-labeling classifier.

    The meta model learns *when* to trust the primary signal.  During
    prediction it returns both the primary direction and a bet-size
    probability that can be passed to a Kelly criterion module.
    """

    def __init__(
        self,
        primary: Model,
        meta: MetaLightGBMModel | None = None,
        **lgb_params: Any,
    ) -> None:
        self.primary = primary
        self.meta = meta or MetaLightGBMModel(**lgb_params)

    def fit(  # noqa: PLR0913
        self,
        X: pd.DataFrame,  # noqa: N803
        y_true: pd.Series,  # type: ignore[type-arg]
        sample_weight: np.ndarray | None = None,
        X_val: pd.DataFrame | None = None,  # noqa: N803
        y_val: pd.Series | None = None,  # type: ignore[type-arg]
        t1: pd.Series | None = None,  # type: ignore[type-arg]
        forward_returns: pd.Series | None = None,  # type: ignore[type-arg]
        n_trials: int = 100,
    ) -> MetaModel:
        primary_preds = self.primary.predict(X)

        x_meta = X.copy()
        x_meta["primary_signal"] = primary_preds

        y_meta = pd.Series(
            (primary_preds == y_true.values).astype(int),
            index=y_true.index,
        )

        logger.info(
            "meta_label_distribution",
            n_correct=int(y_meta.sum()),
            n_total=len(y_meta),
            accuracy=f"{y_meta.mean():.3f}",
        )

        if t1 is not None:
            self.meta.tune(
                x_meta,
                y_meta,
                t1,
                sample_weight=sample_weight,
                forward_returns=forward_returns,
                n_trials=n_trials,
            )
        else:
            x_val_meta = None
            if X_val is not None:
                x_val_meta = X_val.copy()
                x_val_meta["primary_signal"] = self.primary.predict(X_val)
            self.meta.fit(
                x_meta,
                y_meta,
                sample_weight=sample_weight,
                X_val=x_val_meta,
                y_val=y_val,
            )

        return self

    def predict(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:  # noqa: N803
        """Return ``(direction, bet_size)``."""
        direction = self.primary.predict(X)
        x_meta = X.copy()
        x_meta["primary_signal"] = direction
        bet_size = self.meta.predict_bet_size(x_meta)
        return direction, bet_size

    def save(self, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        self.primary.save(path / "primary")
        self.meta.save(path / "meta")
        return path

    @classmethod
    def load(cls, path: Path) -> MetaModel:
        primary = PrimaryLightGBMModel.load(path / "primary")
        meta = MetaLightGBMModel.load(path / "meta")
        return cls(primary=primary, meta=meta)
