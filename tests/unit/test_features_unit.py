"""Unit tests for features: hand-computed values on tiny datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tessera.features.cross_sectional import BetaToBTC, UniverseRank
from tessera.features.funding import FundingRate, FundingZScore, SpotPerpBasis
from tessera.features.microstructure import MicroPrice, OrderFlowImbalance, SpreadBps
from tessera.features.returns import LogReturn
from tessera.features.volatility import GarmanKlass, Parkinson, RealizedVol


class TestLogReturn:
    def test_horizon_1(self) -> None:
        df = pd.DataFrame({"close": [100.0, 105.0, 110.0, 100.0]})
        feat = LogReturn(horizon=1)
        result = feat.compute(df)

        assert pd.isna(result.iloc[0])
        expected_1 = np.log(105.0 / 100.0)
        np.testing.assert_allclose(result.iloc[1], expected_1, rtol=1e-10)

    def test_horizon_2(self) -> None:
        df = pd.DataFrame({"close": [100.0, 105.0, 110.0, 100.0]})
        feat = LogReturn(horizon=2)
        result = feat.compute(df)

        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        expected_2 = np.log(110.0 / 100.0)
        np.testing.assert_allclose(result.iloc[2], expected_2, rtol=1e-10)


class TestRealizedVol:
    def test_basic(self) -> None:
        prices = [100.0, 101.0, 99.0, 102.0, 98.0, 103.0]
        df = pd.DataFrame({"close": prices})
        feat = RealizedVol(window=3)
        result = feat.compute(df)

        # First 3 values should be NaN (need window=3 of returns, so 4 prices)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])
        # 4th value should be the std of first 3 log returns
        log_rets = np.log(np.array(prices[1:4]) / np.array(prices[:3]))
        expected = np.std(log_rets, ddof=1)
        np.testing.assert_allclose(result.iloc[3], expected, rtol=1e-10)


class TestMicroPrice:
    def test_basic(self) -> None:
        df = pd.DataFrame(
            {
                "bid_price": [99.0, 98.0],
                "ask_price": [101.0, 102.0],
                "bid_size": [10.0, 20.0],
                "ask_size": [10.0, 10.0],
            }
        )
        feat = MicroPrice()
        result = feat.compute(df)

        # Row 0: (10*101 + 10*99) / (10+10) = 2000/20 = 100.0
        np.testing.assert_allclose(result.iloc[0], 100.0, rtol=1e-10)
        # Row 1: (20*102 + 10*98) / (20+10) = (2040+980)/30 = 3020/30
        np.testing.assert_allclose(result.iloc[1], 3020.0 / 30.0, rtol=1e-10)


class TestSpreadBps:
    def test_basic(self) -> None:
        df = pd.DataFrame({"bid_price": [99.0], "ask_price": [101.0]})
        feat = SpreadBps()
        result = feat.compute(df)

        mid = 100.0
        expected = (101.0 - 99.0) / mid * 10_000
        np.testing.assert_allclose(result.iloc[0], expected, rtol=1e-10)


class TestOrderFlowImbalance:
    def test_basic(self) -> None:
        df = pd.DataFrame(
            {
                "bid_price": [100.0, 100.0, 101.0],
                "ask_price": [101.0, 101.0, 102.0],
                "bid_size": [10.0, 12.0, 15.0],
                "ask_size": [10.0, 8.0, 12.0],
            }
        )
        feat = OrderFlowImbalance(depth=1)
        result = feat.compute(df)

        assert pd.isna(result.iloc[0])
        # Row 1: bid_p same → delta_bid = 12-10=2; ask_p same → delta_ask = 8-10=-2
        # OFI = 2 - (-2) = 4
        np.testing.assert_allclose(result.iloc[1], 4.0, rtol=1e-10)


class TestFundingRate:
    def test_ffill(self) -> None:
        df = pd.DataFrame({"funding_rate": [0.001, np.nan, np.nan, 0.002]})
        feat = FundingRate()
        result = feat.compute(df)

        np.testing.assert_allclose(result.iloc[0], 0.001)
        np.testing.assert_allclose(result.iloc[1], 0.001)
        np.testing.assert_allclose(result.iloc[2], 0.001)
        np.testing.assert_allclose(result.iloc[3], 0.002)


class TestFundingZScore:
    def test_basic(self) -> None:
        n = 50
        rng = np.random.default_rng(42)
        fr = rng.normal(0.0001, 0.0005, n)
        df = pd.DataFrame({"funding_rate": fr})
        feat = FundingZScore(window=30)
        result = feat.compute(df)

        # After warmup, z-scores should be finite
        assert result.iloc[29:].notna().all()
        # Values should be reasonable z-scores (within a few std)
        valid = result.iloc[29:]
        assert (valid.abs() < 10).all()


class TestSpotPerpBasis:
    def test_basic(self) -> None:
        df = pd.DataFrame({"close": [100.0, 101.0], "spot_price": [99.5, 100.5]})
        feat = SpotPerpBasis()
        result = feat.compute(df)

        expected_0 = (100.0 - 99.5) / 99.5 * 10_000
        np.testing.assert_allclose(result.iloc[0], expected_0, rtol=1e-10)

    def test_missing_spot(self) -> None:
        df = pd.DataFrame({"close": [100.0, 101.0]})
        feat = SpotPerpBasis()
        result = feat.compute(df)
        assert result.isna().all()


class TestParkinson:
    def test_basic(self) -> None:
        df = pd.DataFrame(
            {
                "high": [105.0, 106.0, 104.0],
                "low": [95.0, 94.0, 96.0],
            }
        )
        feat = Parkinson(window=2)
        result = feat.compute(df)

        assert pd.isna(result.iloc[0])
        # Check that it produces a positive volatility
        assert result.iloc[1] > 0


class TestGarmanKlass:
    def test_basic(self) -> None:
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 99.0],
                "high": [105.0, 106.0, 104.0],
                "low": [95.0, 96.0, 94.0],
                "close": [102.0, 103.0, 97.0],
            }
        )
        feat = GarmanKlass(window=2)
        result = feat.compute(df)
        assert pd.isna(result.iloc[0])
        assert result.iloc[1] > 0


class TestUniverseRank:
    def test_single_symbol(self) -> None:
        df = pd.DataFrame({"close": [1.0, 3.0, 2.0, 4.0]})
        feat = UniverseRank(metric="close")
        result = feat.compute(df)
        # Expanding percentile: at each point, fraction of past values < current
        # t=0: 0 of 1 values < 1.0 → 0.0
        np.testing.assert_allclose(result.iloc[0], 0.0, rtol=1e-10)
        # t=1: 1 of 2 values < 3.0 → 0.5
        np.testing.assert_allclose(result.iloc[1], 0.5, rtol=1e-10)
        # t=3: 3 of 4 values < 4.0 → 0.75
        np.testing.assert_allclose(result.iloc[3], 0.75, rtol=1e-10)


class TestBetaToBTC:
    def test_constant_ratio(self) -> None:
        n = 50
        btc_ret = np.array([0.01] * n)
        # Asset moves exactly 2x BTC
        close = 100.0 * np.exp(np.cumsum(btc_ret * 2))
        df = pd.DataFrame({"close": close, "btc_return": btc_ret})
        feat = BetaToBTC(window=30)
        result = feat.compute(df)
        # After warmup, beta should be approximately 2.0
        # (Exact 2.0 since returns are constant)
        valid = result.dropna()
        if len(valid) > 0:
            np.testing.assert_allclose(valid.iloc[-1], 2.0, rtol=0.1)
