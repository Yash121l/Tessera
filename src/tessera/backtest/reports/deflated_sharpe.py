"""Deflated Sharpe Ratio (DSR) per Bailey & López de Prado (2014).

DSR replaces the zero benchmark in PSR with SR*, the expected maximum
Sharpe under H₀ (no skill) across N independent trials:

    SR* = σ_SR × [(1−γ_em)·Φ⁻¹(1−1/N) + γ_em·Φ⁻¹(1−1/(N·e))]

where γ_em = 0.5772… (Euler-Mascheroni), N = n_trials, and σ_SR = sr_std
is the standard deviation of Sharpe ratios across those trials.

DSR = PSR(observed_sr, SR*, n_obs, skew, kurt)
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np

from tessera.backtest.reports.probabilistic_sharpe import probabilistic_sharpe

_EULER_MASCHERONI = 0.5772156649015328


def deflated_sharpe(
    observed_sr: float,
    sr_std: float,
    n_trials: int,
    n_obs: int,
    skew: float = 0.0,
    kurt: float = 0.0,
    annualization_factor: int = 252,
) -> float:
    """Deflated Sharpe Ratio: P(true SR > SR*) adjusted for n_trials.

    Args:
        observed_sr: Best observed annualized SR across the trials.
        sr_std: Standard deviation of annualized SRs across the n_trials
            independent configurations. Use 1/√n_obs as a conservative fallback
            when the full trial distribution is unknown.
        n_trials: Total number of independent strategies / hyperparameter
            configs tested (Optuna trials + manual notebook experiments).
        n_obs: Number of return observations used to compute observed_sr.
        skew: Skewness of the return series.
        kurt: Excess kurtosis of the return series.
        annualization_factor: Periods per year.

    Returns:
        Probability in [0, 1]. Values below 0.95 suggest the observed SR
        is consistent with a lucky draw under the null of no skill.
    """
    from scipy.stats import norm

    if n_trials <= 1:
        return probabilistic_sharpe(observed_sr, 0.0, n_obs, skew, kurt, annualization_factor)

    effective_std = max(sr_std, 1.0 / np.sqrt(max(n_obs, 1)))

    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    sr_star = effective_std * ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2)

    return probabilistic_sharpe(observed_sr, sr_star, n_obs, skew, kurt, annualization_factor)


def compute_trial_count(
    optuna_study: Any = None,
    manual_configs: list[Any] | int | None = None,
) -> int:
    """Total trial count = completed Optuna trials + manual experiment count.

    Pass this as n_trials to deflated_sharpe so that all model selection
    decisions—automated and manual—inflate the multiple-testing penalty.

    Args:
        optuna_study: An optuna.Study object (optional). Completed trials
            are counted via len(study.trials).
        manual_configs: Either a list of config dicts (one per manual run)
            or an integer count of manual notebook experiments. Each
            hyperparameter grid search, each notebook model run, and each
            ablation variant counts as one trial.

    Returns:
        Total trial count (≥ 1).
    """
    total = 0

    if optuna_study is not None:
        with contextlib.suppress(AttributeError):
            total += len(optuna_study.trials)

    if isinstance(manual_configs, int):
        total += manual_configs
    elif isinstance(manual_configs, list):
        total += len(manual_configs)

    return max(total, 1)
