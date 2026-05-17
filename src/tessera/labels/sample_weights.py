"""Concurrency-aware sample weighting (AFML §4).

Labels with overlapping time windows share information. These functions compute
weights that account for concurrency, preventing redundant samples from
dominating the training set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_concurrent_events(
    close_idx: pd.DatetimeIndex,
    t1: pd.Series,  # type: ignore[type-arg]
) -> pd.Series:  # type: ignore[type-arg]
    """Count label windows overlapping each bar (AFML §4.2).

    For each bar in close_idx, counts how many events have a label window
    [t0, t1] that spans that bar.

    Args:
        close_idx: DatetimeIndex of the price series.
        t1: Series mapping event start (index) to event end (value).

    Returns:
        Series indexed by close_idx with concurrency counts.
    """
    t1 = t1.dropna()
    concurrency = pd.Series(0, index=close_idx)

    for t0, end in t1.items():
        mask = (close_idx >= t0) & (close_idx <= end)
        concurrency[mask] += 1

    return concurrency


def _get_avg_uniqueness(
    t1: pd.Series,  # type: ignore[type-arg]
    concurrency: pd.Series,  # type: ignore[type-arg]
) -> pd.Series:  # type: ignore[type-arg]
    """Average uniqueness of each event's label window."""
    uniqueness = pd.Series(index=t1.index, dtype=float)
    for t0, end in t1.items():
        window = concurrency.loc[t0:end]
        if len(window) > 0:
            uniqueness[t0] = (1.0 / window).mean()
        else:
            uniqueness[t0] = 1.0
    return uniqueness


def get_sample_weights_by_return(
    t1: pd.Series,  # type: ignore[type-arg]
    close: pd.Series,  # type: ignore[type-arg]
    num_threads: int = 1,
) -> pd.Series:  # type: ignore[type-arg]
    """Compute sample weights by absolute return divided by concurrency (AFML §4.4).

    Args:
        t1: Series mapping event start to event end.
        close: Close price series.
        num_threads: Number of parallel workers (currently unused, kept for API compat).

    Returns:
        Series of sample weights indexed by event timestamps.
    """
    t1 = t1.dropna()
    concurrency = get_concurrent_events(close.index, t1)

    # Absolute log returns over each label window
    abs_returns = pd.Series(index=t1.index, dtype=float)
    for t0, end in t1.items():
        abs_returns[t0] = abs(np.log(close.loc[end] / close.loc[t0]))

    # Average uniqueness per event
    uniqueness = _get_avg_uniqueness(t1, concurrency)

    weights = abs_returns * uniqueness
    # Normalize to sum to len(weights)
    total = weights.sum()
    if total > 0:
        weights = weights * len(weights) / total
    return weights


def get_time_decay_weights(
    weights: pd.Series,  # type: ignore[type-arg]
    decay: float = 1.0,
) -> pd.Series:  # type: ignore[type-arg]
    """Apply time decay to sample weights (AFML §4.10).

    Args:
        weights: Sample weights from get_sample_weights_by_return.
        decay: Decay factor.
            decay=1.0: no decay (uniform).
            decay=0.0: linearly decaying to zero at the oldest sample.
            0 < decay < 1: linearly decaying to decay at the oldest sample.

    Returns:
        Weights multiplied by a time-decay factor.
    """
    n = len(weights)
    if n == 0:
        return weights

    factors = np.ones(n) if decay >= 1.0 else np.linspace(decay, 1.0, n)

    result = weights.copy()
    result.iloc[:] = weights.values * factors
    return result
