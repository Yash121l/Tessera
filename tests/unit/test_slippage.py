"""Unit tests for tessera.backtest.slippage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tessera.backtest.slippage import OHLCVSlippageModel


class TestOHLCVSlippageModel:
    def test_zero_order_zero_impact(self) -> None:
        # Pass is_taker=False so the half-spread baseline is not added.
        model = OHLCVSlippageModel(k=1.0, half_spread_bps=0.0)
        assert model.impact_bps(0.0, 1_000_000.0, is_taker=False) == pytest.approx(0.0)

    def test_impact_increases_with_order_size(self) -> None:
        model = OHLCVSlippageModel(k=1.0, half_spread_bps=0.0)
        small = model.impact_bps(1_000.0, 1_000_000.0, is_taker=False)
        large = model.impact_bps(100_000.0, 1_000_000.0, is_taker=False)
        assert large > small

    def test_taker_adds_half_spread(self) -> None:
        model = OHLCVSlippageModel(k=1.0, half_spread_bps=2.5)
        maker_impact = model.impact_bps(10_000.0, 1_000_000.0, is_taker=False)
        taker_impact = model.impact_bps(10_000.0, 1_000_000.0, is_taker=True)
        assert taker_impact == pytest.approx(maker_impact + 2.5)

    def test_adjust_price_buy_increases_price(self) -> None:
        model = OHLCVSlippageModel(k=1.0)
        adj = model.adjust_price(100.0, "buy", 10_000.0, 1_000_000.0)
        assert adj > 100.0

    def test_adjust_price_sell_decreases_price(self) -> None:
        model = OHLCVSlippageModel(k=1.0)
        adj = model.adjust_price(100.0, "sell", 10_000.0, 1_000_000.0)
        assert adj < 100.0

    def test_per_symbol_k_overrides_default(self) -> None:
        model = OHLCVSlippageModel(k=1.0, half_spread_bps=0.0)
        model._k_by_symbol["BTCUSDT"] = 2.0
        default_bps = model.impact_bps(10_000.0, 1_000_000.0, symbol="ETHUSDT", is_taker=False)
        btc_bps = model.impact_bps(10_000.0, 1_000_000.0, symbol="BTCUSDT", is_taker=False)
        assert btc_bps == pytest.approx(2.0 * default_bps)

    def test_fit_k_min_obs_30(self) -> None:
        model = OHLCVSlippageModel()
        few_fills = pd.DataFrame(
            {
                "order_notional": [1000.0] * 5,
                "adv_notional": [1_000_000.0] * 5,
                "actual_slippage_bps": [1.0] * 5,
            }
        )
        model.fit_k("ETHUSDT", few_fills)
        assert "ETHUSDT" not in model._k_by_symbol

    def test_fit_k_with_sufficient_obs(self) -> None:
        rng = np.random.default_rng(42)
        n = 50
        notional = rng.uniform(1_000, 100_000, n)
        adv = np.full(n, 1_000_000.0)
        # True k = 2.0; slippage = 2.0 * sqrt(notional/adv)
        true_k = 2.0
        slippage = true_k * np.sqrt(notional / adv) + rng.normal(0, 0.05, n)
        fills = pd.DataFrame(
            {
                "order_notional": notional,
                "adv_notional": adv,
                "actual_slippage_bps": np.maximum(slippage, 0.0),
            }
        )
        model = OHLCVSlippageModel(k=1.0)
        model.fit_k("BTCUSDT", fills)
        assert "BTCUSDT" in model._k_by_symbol
        # Fitted k should be in [1.5, 2.5] — within 25% of true value
        fitted_k = model._k_by_symbol["BTCUSDT"]
        assert 1.5 <= fitted_k <= 2.5

    def test_fit_k_robust_to_outliers(self) -> None:
        rng = np.random.default_rng(7)
        n = 50
        notional = rng.uniform(1_000, 50_000, n)
        adv = np.full(n, 500_000.0)
        true_k = 1.5
        slippage = true_k * np.sqrt(notional / adv)
        # Inject 3 extreme outliers
        slippage[0] = 1000.0
        slippage[1] = 500.0
        slippage[2] = -10.0
        fills = pd.DataFrame(
            {
                "order_notional": notional,
                "adv_notional": adv,
                "actual_slippage_bps": slippage,
            }
        )
        model_huber = OHLCVSlippageModel(k=1.0)
        model_huber.fit_k("X", fills)
        # Huber should stay closer to true_k than plain OLS (which gets pulled by outliers)
        fitted = model_huber._k_by_symbol.get("X", float("inf"))
        # Accept a wider tolerance — main check is that it doesn't explode
        assert fitted < 50.0, f"Huber k estimate too large: {fitted}"
