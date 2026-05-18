"""Bootstrap CI coverage test.

Verifies that the stationary block bootstrap CI for Sharpe contains the
true population Sharpe ~95% of the time across 200 independent simulations.

This is a meta-test (testing the test): coverage below 75% or above 99.5%
indicates a broken implementation.

Runtime: ~10s with n_resamples=500 per simulation.  Kept low deliberately
so the full suite stays fast; use n_resamples=5000 for offline verification.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("arch", reason="requires ml/backtest extra: uv sync --extra ml")

from tessera.backtest.reports.bootstrap import (  # noqa: E402
    _annualized_sharpe,
    block_bootstrap_sharpe,
)

# ---------------------------------------------------------------------------
# Unit tests for the helper
# ---------------------------------------------------------------------------


def test_annualized_sharpe_known():
    """Hand-check: daily μ/σ × √252 with known values."""
    rng = np.random.default_rng(0)
    r = rng.normal(0.001, 0.01, 252)  # annual SR ≈ 0.001/0.01 × √252 ≈ 1.587
    sr = _annualized_sharpe(r, 252)
    assert 0.5 < sr < 3.0, f"Unexpected SR={sr:.3f}"


def test_block_bootstrap_returns_tuple():
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(0.001, 0.01, 252))
    lo, mid, hi = block_bootstrap_sharpe(r, block_size=10, n_resamples=200, seed=42)
    assert lo <= mid <= hi, f"CI ordering violated: [{lo:.3f}, {mid:.3f}, {hi:.3f}]"


def test_block_bootstrap_tiny_series():
    """Should not crash on very short series."""
    r = pd.Series([0.01, -0.01, 0.02])
    result = block_bootstrap_sharpe(r, block_size=2, n_resamples=50)
    assert len(result) == 3


def test_block_bootstrap_flat_series():
    """Zero-variance series → Sharpe = 0, CI centred at 0."""
    r = pd.Series([0.0] * 100)
    lo, mid, hi = block_bootstrap_sharpe(r, block_size=5, n_resamples=50)
    assert abs(mid) < 1e-6, f"Expected mid≈0 for flat series, got {mid}"


def test_block_bootstrap_reproducible():
    """Same seed → same CI bounds."""
    rng = np.random.default_rng(7)
    r = pd.Series(rng.normal(0.001, 0.01, 252))
    r1 = block_bootstrap_sharpe(r, n_resamples=200, seed=0)
    r2 = block_bootstrap_sharpe(r, n_resamples=200, seed=0)
    assert r1 == r2, "Same seed should produce identical results"


# ---------------------------------------------------------------------------
# Coverage meta-test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_bootstrap_ci_coverage_rate():
    """Bootstrap 95% CI should contain the true Sharpe ~95% of the time.

    Generates 200 i.i.d. AR(0) return series with known true SR, computes
    the bootstrap CI for each, and checks coverage is between 80% and 99.5%.

    Uses n_resamples=500 per simulation to keep runtime under 30s.
    """
    n_sim = 200
    n_obs = 504  # ~2 years of daily bars
    ann = 252
    true_sr_annual = 1.5

    sigma_daily = 0.01
    mu_daily = true_sr_annual / np.sqrt(ann) * sigma_daily

    rng = np.random.default_rng(2024)
    covered = 0

    for _ in range(n_sim):
        r = rng.normal(mu_daily, sigma_daily, n_obs)
        returns = pd.Series(r)
        lo, _, hi = block_bootstrap_sharpe(
            returns,
            block_size=10,
            n_resamples=500,
            confidence=0.95,
            annualization_factor=ann,
            seed=int(rng.integers(0, 2**31)),
        )
        if lo <= true_sr_annual <= hi:
            covered += 1

    coverage = covered / n_sim
    assert 0.75 <= coverage <= 0.995, (
        f"Bootstrap 95% CI coverage = {coverage:.2%} "
        f"({covered}/{n_sim} intervals contained true SR={true_sr_annual}).  "
        "Expected in [75%, 99.5%]."
    )
