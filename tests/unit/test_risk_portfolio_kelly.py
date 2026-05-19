"""Unit tests for tessera.risk.portfolio_kelly."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tessera.risk.portfolio_kelly import ledoit_wolf_shrinkage, portfolio_kelly_weights


class TestPortfolioKellyWeights:
    def _make_returns(self, n_obs: int = 260, n_assets: int = 3, seed: int = 42) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        cols = [f"A{i}" for i in range(n_assets)]
        return pd.DataFrame(rng.normal(0.0, 0.01, (n_obs, n_assets)), columns=cols)

    def test_output_index_matches_expected_returns(self) -> None:
        returns = self._make_returns()
        mu = pd.Series({"A0": 0.001, "A1": 0.0005, "A2": -0.0002})
        weights = portfolio_kelly_weights(returns, mu)
        assert set(weights.index) == set(mu.index)

    def test_gross_leverage_clipped(self) -> None:
        returns = self._make_returns()
        mu = pd.Series({"A0": 0.01, "A1": 0.01, "A2": 0.01})
        weights = portfolio_kelly_weights(returns, mu, fraction=1.0, max_gross_leverage=0.5)
        assert float(weights.abs().sum()) <= 0.5 + 1e-9

    def test_positive_mu_positive_weight(self) -> None:
        rng = np.random.default_rng(0)
        cols = ["A0", "A1"]
        # Independent assets, strong positive expected return
        returns = pd.DataFrame(rng.normal(0.0, 0.01, (500, 2)), columns=cols)
        mu = pd.Series({"A0": 0.002, "A1": 0.002})
        weights = portfolio_kelly_weights(returns, mu, fraction=0.25, max_gross_leverage=1.0)
        assert float(weights["A0"]) > 0
        assert float(weights["A1"]) > 0

    def test_zero_expected_returns_zero_weights(self) -> None:
        returns = self._make_returns()
        mu = pd.Series({"A0": 0.0, "A1": 0.0, "A2": 0.0})
        weights = portfolio_kelly_weights(returns, mu)
        assert weights.abs().max() < 1e-9

    def test_empty_returns_returns_zeros(self) -> None:
        returns = pd.DataFrame(columns=["A0", "A1"])
        mu = pd.Series({"A0": 0.001, "A1": 0.001})
        weights = portfolio_kelly_weights(returns, mu)
        assert (weights == 0.0).all()

    def test_raises_on_insufficient_observations(self) -> None:
        returns = pd.DataFrame(
            np.random.default_rng(0).normal(0, 0.01, (3, 5)),
            columns=list("ABCDE"),
        )
        mu = pd.Series(0.001, index=list("ABCDE"))
        with pytest.raises(ValueError, match="observations"):
            portfolio_kelly_weights(returns, mu)

    def test_fraction_scales_weights(self) -> None:
        returns = self._make_returns()
        mu = pd.Series({"A0": 0.001, "A1": 0.001, "A2": 0.001})
        # Use a very large leverage cap so neither call is clipped — then
        # quarter-Kelly weights must be exactly 1/4 of full-Kelly weights.
        w_full = portfolio_kelly_weights(returns, mu, fraction=1.0, max_gross_leverage=1000.0)
        w_quarter = portfolio_kelly_weights(returns, mu, fraction=0.25, max_gross_leverage=1000.0)
        np.testing.assert_allclose(w_quarter.values, w_full.values * 0.25, rtol=1e-6)


class TestLedoitWolfShrinkage:
    def test_output_shape(self) -> None:
        rng = np.random.default_rng(0)
        returns = pd.DataFrame(rng.normal(0, 0.01, (100, 4)), columns=["A", "B", "C", "D"])
        cov = ledoit_wolf_shrinkage(returns)
        assert cov.shape == (4, 4)
        assert list(cov.columns) == list(cov.index)

    def test_positive_semidefinite(self) -> None:
        rng = np.random.default_rng(1)
        returns = pd.DataFrame(rng.normal(0, 0.01, (200, 3)), columns=["X", "Y", "Z"])
        cov = ledoit_wolf_shrinkage(returns)
        eigenvalues = np.linalg.eigvalsh(cov.values)
        assert np.all(eigenvalues >= -1e-10)

    def test_diagonal_positive(self) -> None:
        rng = np.random.default_rng(2)
        returns = pd.DataFrame(rng.normal(0, 0.01, (150, 2)), columns=["A", "B"])
        cov = ledoit_wolf_shrinkage(returns)
        assert float(cov.loc["A", "A"]) > 0
        assert float(cov.loc["B", "B"]) > 0
