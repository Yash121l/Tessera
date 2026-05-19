"""Unit tests for tessera.risk.vol_target."""

from __future__ import annotations

import numpy as np
import pytest

from tessera.risk.vol_target import vol_target_scalar


class TestVolTargetScalar:
    def test_scalar_at_target_returns_one(self) -> None:
        assert vol_target_scalar(0.15, target_vol_annual=0.15) == pytest.approx(1.0)

    def test_high_vol_returns_less_than_one(self) -> None:
        scalar = vol_target_scalar(0.30, target_vol_annual=0.15)
        assert scalar == pytest.approx(0.5)

    def test_low_vol_is_capped_at_three(self) -> None:
        # 0.01 annual vol with 0.15 target → 15x, capped at 3
        scalar = vol_target_scalar(0.01, target_vol_annual=0.15)
        assert scalar == pytest.approx(3.0)

    def test_zero_vol_returns_one(self) -> None:
        # Graceful fallback when vol is effectively zero
        assert vol_target_scalar(0.0, target_vol_annual=0.15) == pytest.approx(1.0)

    def test_series_of_returns(self) -> None:
        rng = np.random.default_rng(42)
        # Daily returns with ~1% daily vol → ~16% annual
        daily_returns = rng.normal(0, 0.01, 60)
        scalar = vol_target_scalar(daily_returns, target_vol_annual=0.15)
        # Should be around 15/16 ≈ 0.94, but allow wide range due to EWMA and randomness
        assert 0.1 < scalar <= 3.0

    def test_empty_series_returns_one(self) -> None:
        result = vol_target_scalar(np.array([], dtype=float), target_vol_annual=0.15)
        assert result == pytest.approx(1.0)

    def test_pandas_series_accepted(self) -> None:
        import pandas as pd

        returns = pd.Series(np.full(30, 0.01))
        scalar = vol_target_scalar(returns, target_vol_annual=0.15)
        assert 0.0 < scalar <= 3.0

    def test_inverse_relationship(self) -> None:
        low_vol = vol_target_scalar(0.10, target_vol_annual=0.15)
        high_vol = vol_target_scalar(0.20, target_vol_annual=0.15)
        assert low_vol > high_vol

    def test_bars_per_day_daily_vs_minute_consistency(self) -> None:
        """Minute and daily bar inputs should yield comparable scalars for same annualised vol."""
        rng = np.random.default_rng(99)
        daily_vol = 0.01  # 1% daily std → ~16% annual
        # 60 daily returns with roughly daily_vol std
        daily_returns = rng.normal(0, daily_vol, 60)
        # Expand to 1-min bars: 1440 bars/day, 60 days = 86 400 bars
        minute_returns = rng.normal(0, daily_vol / np.sqrt(1440), 60 * 1440)

        scalar_daily = vol_target_scalar(daily_returns, target_vol_annual=0.15, bars_per_day=1)
        scalar_minute = vol_target_scalar(minute_returns, target_vol_annual=0.15, bars_per_day=1440)
        # Both should be in the same ballpark (within 2×)
        assert 0.5 * scalar_daily <= scalar_minute <= 2.0 * scalar_daily

    def test_bars_per_day_parameter_accepted(self) -> None:
        rng = np.random.default_rng(0)
        returns_5min = rng.normal(0, 0.001, 288 * 10)  # 10 days of 5-min bars
        scalar = vol_target_scalar(returns_5min, target_vol_annual=0.15, bars_per_day=288)
        assert 0.0 < scalar <= 3.0
