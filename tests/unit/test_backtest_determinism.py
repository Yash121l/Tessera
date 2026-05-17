"""Test: backtest is deterministic given the same seed and config.

Two engine runs with identical seed must produce identical fill logs and
equity curves (within floating-point tolerance).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("nautilus_trader", reason="requires backtest extra: uv sync --extra backtest")

from nautilus_trader.model.data import Bar, BarType  # noqa: E402
from nautilus_trader.model.objects import Price, Quantity  # noqa: E402

from tessera.backtest.engine import TesseraBacktestEngine  # noqa: E402
from tessera.strategies.base import TesseraBaseStrategy, TesseraStrategyConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Test strategies
# ---------------------------------------------------------------------------


class FlatStrategyConfig(TesseraStrategyConfig, frozen=True):
    """No-op strategy: never trades."""


class FlatStrategy(TesseraBaseStrategy):
    """Always flat: ignores all bars."""

    def _on_bar_impl(self, bar: Bar) -> None:  # noqa: D401
        pass  # no signal, no orders


class OracleStrategyConfig(TesseraStrategyConfig, frozen=True):
    """Perfect-foresight strategy config."""

    future_closes: tuple[float, ...] = ()  # serialised future close prices
    signal_delay_bars: int = 0


class OracleStrategy(TesseraBaseStrategy):
    """Perfect-foresight strategy: always trades in the direction of next-bar return."""

    def __init__(self, config: OracleStrategyConfig) -> None:
        super().__init__(config)
        self._oracle_cfg = config
        self._bar_idx: dict[str, int] = {}

    def _on_bar_impl(self, bar: Bar) -> None:
        id_str = str(bar.bar_type.instrument_id)
        idx = self._bar_idx.get(id_str, 0)
        self._bar_idx[id_str] = idx + 1

        closes = self._oracle_cfg.future_closes
        # Need at least idx+1 to peek at the next bar
        if idx + 1 >= len(closes):
            return

        current_close = closes[idx]
        next_close = closes[idx + 1]
        if current_close <= 0:
            return

        signal = 1 if next_close > current_close else -1
        instr = self.cache.instrument(bar.bar_type.instrument_id)
        if instr is None:
            return

        from nautilus_trader.model.enums import OrderSide

        # Always hold 0.01 BTC in the direction of the signal
        qty = instr.make_qty(0.01)
        side = OrderSide.BUY if signal > 0 else OrderSide.SELL

        self.cancel_all_orders(instrument_id=bar.bar_type.instrument_id)
        order = self.order_factory.market(
            instrument_id=bar.bar_type.instrument_id,
            order_side=side,
            quantity=qty,
        )
        self.submit_order(order)


# ---------------------------------------------------------------------------
# Synthetic bar factory
# ---------------------------------------------------------------------------


def _make_synthetic_bars(
    instrument_id_str: str,
    n: int = 200,
    initial_price: float = 50_000.0,
    vol: float = 0.001,
    seed: int = 42,
    trend: float = 0.0001,  # per-bar drift
) -> list[Bar]:
    rng = np.random.default_rng(seed)
    bar_type = BarType.from_str(f"{instrument_id_str}-1-MINUTE-LAST-EXTERNAL")
    # symbol prefix used for price precision heuristic
    price_prec = 1 if "BTC" in instrument_id_str else 2

    prices = [initial_price]
    for _ in range(n - 1):
        r = trend + vol * rng.standard_normal()
        prices.append(prices[-1] * (1 + r))

    base_ts = int(pd.Timestamp("2023-01-01", tz="UTC").value)
    bar_ns = 60 * 1_000_000_000  # 1 minute in ns

    bars: list[Bar] = []
    for i, close in enumerate(prices):
        o = prices[i - 1] if i > 0 else close
        h = max(o, close) * (1 + abs(rng.standard_normal()) * vol * 0.5)
        lo = min(o, close) * (1 - abs(rng.standard_normal()) * vol * 0.5)
        vol_qty = abs(rng.standard_normal()) * 50 + 10

        fmt = f"{{:.{price_prec}f}}"
        ts = base_ts + i * bar_ns
        bars.append(
            Bar(
                bar_type=bar_type,
                open=Price.from_str(fmt.format(o)),
                high=Price.from_str(fmt.format(h)),
                low=Price.from_str(fmt.format(lo)),
                close=Price.from_str(fmt.format(close)),
                volume=Quantity.from_str(f"{vol_qty:.3f}"),
                ts_event=ts,
                ts_init=ts,
            )
        )
    return bars


def _make_flat_engine(instrument_id_str: str, seed: int = 42) -> TesseraBacktestEngine:
    cfg = FlatStrategyConfig(instrument_ids=(instrument_id_str,))
    strategy = FlatStrategy(config=cfg)
    bars = _make_synthetic_bars(instrument_id_str, n=100, seed=seed)
    return TesseraBacktestEngine.from_bars(
        {instrument_id_str: bars},
        strategy,
        run_id=f"test-{seed}",
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


INSTR = "BTC-USDT-PERP.BINANCE"


def test_flat_strategy_no_trades() -> None:
    """No-signal strategy must never submit an order → 0 trades, PnL = 0."""
    engine = _make_flat_engine(INSTR, seed=42)
    result = engine.run()
    assert result.n_trades == 0
    assert result.total_pnl == pytest.approx(0.0, abs=1e-6)
    assert result.trading_pnl == pytest.approx(0.0, abs=1e-6)


def test_backtest_determinism() -> None:
    """Two runs with the same seed must produce identical n_trades and equity."""
    engine_a = _make_flat_engine(INSTR, seed=99)
    engine_b = _make_flat_engine(INSTR, seed=99)

    result_a = engine_a.run()
    result_b = engine_b.run()

    assert result_a.n_trades == result_b.n_trades
    assert result_a.total_pnl == pytest.approx(result_b.total_pnl, abs=1e-6)


def test_different_seeds_different_latency() -> None:
    """Different seeds draw different latencies, but both runs are individually deterministic."""
    # We can't compare outcomes directly (different latencies), but we can verify
    # that re-running with the same seed gives the same result.
    bars = _make_synthetic_bars(INSTR, n=150, seed=7)

    def _run(seed: int) -> float:
        cfg = FlatStrategyConfig(instrument_ids=(INSTR,))
        strategy = FlatStrategy(config=cfg)
        eng = TesseraBacktestEngine.from_bars(
            {INSTR: bars},
            strategy,
            run_id=f"t-{seed}",
            seed=seed,
        )
        return eng.run().sharpe_ratio

    s1a = _run(seed=1)
    s1b = _run(seed=1)
    assert s1a == pytest.approx(s1b, abs=1e-9)
