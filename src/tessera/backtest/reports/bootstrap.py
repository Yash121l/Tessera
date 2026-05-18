"""Stationary block bootstrap confidence interval for annualized Sharpe ratio.

Uses the Politis & Romano (1994) stationary bootstrap via arch.bootstrap,
which handles serial correlation in strategy returns without the boundary
artefacts of circular bootstrap.

Block size guidance
-------------------
Set block_size ≈ mean holding period in bars.  For a daily-bar strategy
with a 5-day mean holding period, use block_size=5.  A safe default is
⌈√T⌉ (Lahiri 2003 rule-of-thumb), which we use when block_size is None.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def block_bootstrap_sharpe(
    returns: pd.Series,
    block_size: int | None = None,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    annualization_factor: int = 252,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Stationary block bootstrap CI for annualized Sharpe ratio.

    Args:
        returns: Return series (daily or per-bar, already at the desired
            frequency — do NOT pass annualized returns).
        block_size: Block size in bars.  Defaults to ⌈√T⌉.
            Set to mean holding period for best coverage.
        n_resamples: Number of bootstrap samples (default 10 000).
        confidence: Confidence level (default 0.95 → 95% CI).
        annualization_factor: Periods per year used to annualize the SR.
        seed: RNG seed for reproducibility.

    Returns:
        (lower, point_estimate, upper) — all annualized Sharpe ratios.
    """
    try:
        from arch.bootstrap import StationaryBootstrap
    except ImportError as exc:
        raise ImportError(
            "arch is required for block_bootstrap_sharpe. Install it with: uv add arch"
        ) from exc

    r = np.asarray(returns.dropna(), dtype=float)
    n_obs = len(r)
    if n_obs < 4:
        pt = _annualized_sharpe(r, annualization_factor)
        return pt, pt, pt

    if block_size is None:
        block_size = max(1, int(np.ceil(np.sqrt(n_obs))))

    block_size = max(1, min(block_size, n_obs // 2))
    point_est = _annualized_sharpe(r, annualization_factor)

    bs = StationaryBootstrap(block_size, r, seed=seed)
    samples: list[float] = []
    for (boot_r,), _ in bs.bootstrap(n_resamples):
        samples.append(_annualized_sharpe(boot_r, annualization_factor))

    arr = np.array(samples)
    alpha = 1.0 - confidence
    lo = float(np.percentile(arr, 100.0 * alpha / 2.0))
    hi = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2.0)))
    return lo, point_est, hi


def _annualized_sharpe(r: np.ndarray, annualization_factor: int) -> float:
    if len(r) < 2:
        return 0.0
    mu = r.mean()
    sigma = r.std(ddof=1)
    if sigma < 1e-12:
        return 0.0
    return float(mu / sigma * np.sqrt(annualization_factor))
