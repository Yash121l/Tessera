"""Volatility-targeting position scalar.

Scales position sizes so the portfolio targets a constant annualised volatility
regardless of the prevailing market regime. Uses a 30-day EWMA of squared daily
returns to estimate current realised vol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_TRADING_DAYS_PER_YEAR = 252
_EWMA_HALFLIFE_DAYS = 30
_MAX_SCALAR = 3.0  # cap to avoid overlevering in quiet regimes


def vol_target_scalar(
    realized_vol: float | pd.Series | np.ndarray,
    target_vol_annual: float = 0.15,
) -> float:
    """Position size multiplier that targets a constant annualised volatility.

    If `realized_vol` is a scalar, it is treated as the already-annualised
    current volatility. If it is a Series/array of *daily* returns, the
    function computes a 30-day EWMA variance and annualises it.

    The multiplier is capped at 3.0 to prevent runaway leverage during
    unusually calm periods.

    Args:
        realized_vol: Annualised vol scalar, or a Series/array of daily returns
            from which EWMA vol is estimated.
        target_vol_annual: Target annualised volatility (e.g. 0.15 = 15% p.a.).

    Returns:
        Scalar multiplier: target_vol / current_vol, capped at _MAX_SCALAR.
    """
    if isinstance(realized_vol, pd.Series | np.ndarray):
        returns = np.asarray(realized_vol, dtype=float)
        if len(returns) == 0:
            return 1.0
        weights = _ewma_weights(len(returns), halflife=_EWMA_HALFLIFE_DAYS)
        ewma_var = float(np.dot(weights, returns**2))
        current_vol = float(np.sqrt(ewma_var * _TRADING_DAYS_PER_YEAR))
    else:
        current_vol = float(realized_vol)

    if current_vol < 1e-9:
        return 1.0

    return min(target_vol_annual / current_vol, _MAX_SCALAR)


def _ewma_weights(n: int, halflife: float) -> np.ndarray:
    """Normalised EWMA weights, ordered oldest → newest."""
    decay = np.exp(-np.log(2.0) / halflife)
    weights = decay ** np.arange(n - 1, -1, -1)
    return weights / weights.sum()
