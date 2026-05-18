"""Unit tests for tessera.risk.limits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tessera.risk.limits import PositionLimits, correlation_limit_check


class TestPositionLimitsCheck:
    def test_all_clear(self) -> None:
        limits = PositionLimits()
        positions = {"BTC": 10_000.0, "ETH": 8_000.0}
        assert limits.check(positions, nav=100_000.0) == {}

    def test_asset_limit_violation(self) -> None:
        limits = PositionLimits(max_asset_pct=0.20)
        positions = {"BTC": 25_000.0}
        violations = limits.check(positions, nav=100_000.0)
        assert "BTC" in violations

    def test_gross_limit_violation(self) -> None:
        limits = PositionLimits(max_gross_pct=2.0)
        positions = {"BTC": 110_000.0, "ETH": 110_000.0}
        violations = limits.check(positions, nav=100_000.0)
        assert "gross" in violations

    def test_net_limit_violation(self) -> None:
        limits = PositionLimits(max_net_pct=1.0)
        # All long; net = 120% of NAV
        positions = {"BTC": 80_000.0, "ETH": 40_000.0}
        violations = limits.check(positions, nav=100_000.0)
        assert "net" in violations

    def test_sector_cap_violation(self) -> None:
        limits = PositionLimits(sector_caps={"L1": 0.30})
        positions = {"BTC": 20_000.0, "ETH": 15_000.0}
        sector_map = {"BTC": "L1", "ETH": "L1"}
        violations = limits.check(positions, nav=100_000.0, sector_map=sector_map)
        assert "sector:L1" in violations

    def test_zero_nav_returns_error(self) -> None:
        limits = PositionLimits()
        violations = limits.check({"BTC": 1.0}, nav=0.0)
        assert "nav" in violations


class TestPositionLimitsClip:
    def test_clip_asset(self) -> None:
        limits = PositionLimits(max_asset_pct=0.20, max_gross_pct=2.0, max_net_pct=1.0)
        positions = {"BTC": 50_000.0}
        clipped = limits.clip_to_limits(positions, nav=100_000.0)
        assert clipped["BTC"] <= 20_000.0 + 1e-9

    def test_clip_gross(self) -> None:
        limits = PositionLimits(max_asset_pct=1.0, max_gross_pct=1.0, max_net_pct=2.0)
        positions = {"BTC": 80_000.0, "ETH": 80_000.0}
        clipped = limits.clip_to_limits(positions, nav=100_000.0)
        gross = sum(abs(v) for v in clipped.values())
        assert gross <= 100_000.0 + 1e-6

    def test_clip_net_long(self) -> None:
        limits = PositionLimits(max_asset_pct=1.0, max_gross_pct=5.0, max_net_pct=1.0)
        positions = {"BTC": 80_000.0, "ETH": 60_000.0}
        clipped = limits.clip_to_limits(positions, nav=100_000.0)
        net = sum(clipped.values())
        assert net <= 100_000.0 + 1e-6

    def test_no_op_when_within_limits(self) -> None:
        limits = PositionLimits()
        positions = {"BTC": 10_000.0, "ETH": -5_000.0}
        clipped = limits.clip_to_limits(positions, nav=100_000.0)
        assert clipped["BTC"] == pytest.approx(10_000.0)
        assert clipped["ETH"] == pytest.approx(-5_000.0)


class TestCorrelationLimitCheck:
    def _returns(self, n: int = 40) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        base = rng.standard_normal(n)
        # BTC and ETH1 perfectly correlated, ETH2 independent
        return pd.DataFrame(
            {
                "BTC": base,
                "ETH1": base + rng.standard_normal(n) * 0.01,
                "ETH2": rng.standard_normal(n),
            }
        )

    def test_correlated_pair_collapsed(self) -> None:
        positions = {"BTC": 50_000.0, "ETH1": 30_000.0, "ETH2": 20_000.0}
        effective = correlation_limit_check(positions, self._returns(), corr_threshold=0.7)
        # BTC and ETH1 should be merged; ETH2 separate
        assert "ETH2" in effective
        assert len(effective) == 2  # BTC+ETH1 group + ETH2

    def test_merged_notional_is_sum(self) -> None:
        positions = {"BTC": 50_000.0, "ETH1": 30_000.0}
        effective = correlation_limit_check(positions, self._returns(), corr_threshold=0.7)
        total = sum(effective.values())
        assert total == pytest.approx(80_000.0)

    def test_uncorrelated_positions_unchanged(self) -> None:
        positions = {"BTC": 50_000.0, "ETH2": 20_000.0}
        effective = correlation_limit_check(positions, self._returns(), corr_threshold=0.7)
        assert len(effective) == 2

    def test_symbols_missing_from_returns_pass_through(self) -> None:
        positions = {"UNKNOWN": 10_000.0}
        effective = correlation_limit_check(positions, self._returns(), corr_threshold=0.7)
        assert effective["UNKNOWN"] == 10_000.0

    def test_single_position_is_unchanged(self) -> None:
        positions = {"BTC": 50_000.0}
        effective = correlation_limit_check(positions, self._returns(), corr_threshold=0.7)
        assert effective["BTC"] == 50_000.0
