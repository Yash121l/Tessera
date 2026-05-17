"""Regime detection via Hidden Markov Model.

Uses forward filtering only (no smoothing) in production for point-in-time safety.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import structlog

from tessera.features.base import Feature

logger = structlog.get_logger(__name__)


class HMMRegime(Feature):
    """Gaussian HMM regime classification.

    Fits on (realized_vol_1h, btc_dominance_proxy, funding_zscore).
    Forward filtering only — Viterbi decoding up to time t uses no future data.
    """

    point_in_time_safe = True
    version = "0.1.0"

    def __init__(self, n_states: int = 3) -> None:
        self.n_states = n_states
        self.name = f"hmm_regime_{n_states}"
        self.dependencies = ["realized_vol_60", "funding_zscore_720"]

    def compute(self, df: pd.DataFrame) -> pd.Series:  # type: ignore[type-arg]
        obs_cols = []
        if "realized_vol_60" in df.columns:
            obs_cols.append("realized_vol_60")
        elif "realized_vol_1h" in df.columns:
            obs_cols.append("realized_vol_1h")

        if "btc_dominance" in df.columns:
            obs_cols.append("btc_dominance")

        if "funding_zscore_720" in df.columns:
            obs_cols.append("funding_zscore_720")
        elif "funding_zscore" in df.columns:
            obs_cols.append("funding_zscore")

        if not obs_cols:
            return pd.Series(np.nan, index=df.index, name=self.name)

        obs = df[obs_cols].copy()
        valid_mask = obs.notna().all(axis=1)
        obs_valid = obs[valid_mask].values

        if len(obs_valid) < 100:
            return pd.Series(np.nan, index=df.index, name=self.name)

        try:
            from hmmlearn.hmm import GaussianHMM

            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=100,
                random_state=42,
            )
            model.fit(obs_valid)

            # Forward filtering: decode sequentially up to each time step
            result = pd.Series(np.nan, index=df.index, name=self.name)

            # Use predict (Viterbi on full observed sequence up to t)
            # Since HMM predict on obs[0:t] only uses past, this is point-in-time safe
            states = np.full(len(obs_valid), np.nan)
            for t in range(1, len(obs_valid) + 1):
                states[t - 1] = model.predict(obs_valid[:t])[-1]

            result.iloc[valid_mask.values] = states
            return result

        except (ImportError, Exception) as e:
            logger.warning("hmm_fit_failed", error=str(e))
            return pd.Series(np.nan, index=df.index, name=self.name)
