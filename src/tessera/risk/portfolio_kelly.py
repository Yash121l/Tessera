"""Portfolio-level Kelly position sizing with Ledoit-Wolf covariance shrinkage.

The single-asset Kelly formula f* = μ/σ² generalises to multiple assets as:
    w* = Σ⁻¹ μ
where Σ is the covariance matrix and μ is the expected return vector
(see MacLean, Thorp & Ziemba 2011, §2.3).

Ledoit-Wolf analytical shrinkage (Ledoit & Wolf 2004) is applied to Σ because
sample covariance matrices are ill-conditioned at the asset counts common in
crypto portfolios (5–50 symbols with 252 daily observations).

Usage::

    weights = portfolio_kelly_weights(
        returns=daily_returns_df,          # T × N DataFrame of daily log-returns
        expected_returns=forecast_series,  # N Series of expected daily returns
        fraction=0.25,                     # quarter-Kelly
        max_gross_leverage=1.0,
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def portfolio_kelly_weights(
    returns: pd.DataFrame,
    expected_returns: pd.Series,
    fraction: float = 0.25,
    max_gross_leverage: float = 1.0,
) -> pd.Series:
    """Fractional Kelly weights for a multi-asset portfolio.

    Estimates the covariance matrix via Ledoit-Wolf analytical shrinkage, then
    solves w* = Σ⁻¹ μ and scales by ``fraction``. Gross leverage is clipped to
    ``max_gross_leverage``.

    Args:
        returns: T × N DataFrame of per-period returns (daily recommended).
            Must have at least N+1 observations for a full-rank estimate.
        expected_returns: N-element Series of expected per-period returns,
            aligned with ``returns.columns``.
        fraction: Kelly fraction in (0, 1]. Typical: 0.25 (quarter-Kelly).
        max_gross_leverage: Clip so that Σ|w_i| ≤ this value.

    Returns:
        N-element Series of position weights (signed; sum may be < 1).

    Raises:
        ValueError: If fewer than N+1 observations are provided.
    """
    if returns.empty:
        return pd.Series(0.0, index=expected_returns.index)

    # Align columns
    common = returns.columns.intersection(expected_returns.index)
    if common.empty:
        return pd.Series(0.0, index=expected_returns.index)

    r = returns[common].dropna()
    mu = expected_returns.reindex(common).fillna(0.0)

    n_assets = len(common)
    if len(r) < n_assets + 1:
        raise ValueError(
            f"Need at least {n_assets + 1} observations for a rank-{n_assets} "
            f"covariance estimate; got {len(r)}."
        )

    from sklearn.covariance import LedoitWolf

    lw = LedoitWolf()
    lw.fit(r.values)
    cov = lw.covariance_  # shape N × N

    # Regularise if near-singular (defensive)
    reg = 1e-8 * np.eye(n_assets)
    cov_inv = np.linalg.inv(cov + reg)

    raw_weights = fraction * cov_inv @ mu.values
    weights = pd.Series(raw_weights, index=common)

    # Clip gross leverage
    gross = float(weights.abs().sum())
    if gross > max_gross_leverage and gross > 1e-12:
        weights = weights * (max_gross_leverage / gross)

    return weights.reindex(expected_returns.index).fillna(0.0)


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> pd.DataFrame:
    """Return the Ledoit-Wolf shrunk covariance matrix as a DataFrame.

    Convenience wrapper for use outside the Kelly solver (e.g. risk reporting).

    Args:
        returns: T × N DataFrame of per-period returns.

    Returns:
        N × N DataFrame with the shrunk covariance matrix.
    """
    from sklearn.covariance import LedoitWolf

    r = returns.dropna()
    lw = LedoitWolf()
    lw.fit(r.values)
    return pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
