"""Probabilistic Sharpe Ratio (PSR) per Bailey & López de Prado (2014).

PSR(SR₀) = Φ[(SR̂ - SR₀) × √(T−1) / √(1 − γ₃·SR̂_p + (γ₄−1)/4·SR̂_p²)]

where γ₃ is skewness, γ₄ is ordinary kurtosis (= excess + 3), and SR̂_p is
the per-period Sharpe (SR̂_p = SR̂_annual / √annualization_factor).
"""

from __future__ import annotations

import numpy as np


def probabilistic_sharpe(
    observed_sr: float,
    sr_benchmark: float,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 0.0,
    annualization_factor: int = 252,
) -> float:
    """P(true SR > sr_benchmark) given the observed annualized SR over n_obs returns.

    Args:
        observed_sr: Observed annualized Sharpe ratio.
        sr_benchmark: Benchmark annualized Sharpe ratio (often 0).
        n_obs: Number of return observations used to compute observed_sr.
        skew: Skewness of the return series.
        kurt: Excess kurtosis of the return series (scipy default, fisher=True).
        annualization_factor: Periods per year (252 for daily, 52 for weekly, etc.).

    Returns:
        Probability in [0, 1].
    """
    from scipy.stats import norm

    if n_obs <= 1:
        return 0.5

    sr_p = observed_sr / np.sqrt(annualization_factor)
    sr_0_p = sr_benchmark / np.sqrt(annualization_factor)

    # Variance of per-period SR estimate (Lo 2002; Bailey & LdP eq. 1)
    # Ordinary kurtosis γ₄ = excess kurtosis + 3  →  (γ₄ − 1)/4 = (kurt + 2)/4
    variance_sr = (1.0 - skew * sr_p + (kurt + 2.0) / 4.0 * sr_p**2) / (n_obs - 1)
    if variance_sr <= 0.0:
        variance_sr = 1.0 / (n_obs - 1)

    z = (sr_p - sr_0_p) / np.sqrt(variance_sr)
    return float(norm.cdf(z))


def sharpe_skew_kurt(returns: np.ndarray | None = None) -> tuple[float, float]:
    """Compute skewness and excess kurtosis from a return array.

    Returns (skew, excess_kurt). Falls back to (0, 0) for fewer than 4 observations.
    """
    from scipy.stats import kurtosis as scipy_kurtosis
    from scipy.stats import skew as scipy_skew

    if returns is None or len(returns) < 4:
        return 0.0, 0.0
    return float(scipy_skew(returns)), float(scipy_kurtosis(returns, fisher=True))
