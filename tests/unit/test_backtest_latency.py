"""Test: increasing signal delay degrades Sharpe monotonically.

We use the OracleStrategy (perfect foresight) on a strongly trending price path.
With signal_delay_bars=0 the signal is fresh; with delay=N the signal is N bars
stale. As N grows, correlation between signal and future return falls → Sharpe
degrades monotonically.

We test delays [0, 1, 3] and verify:
  sharpe(delay=0) >= sharpe(delay=1) >= sharpe(delay=3)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.objects import Price, Quantity

from tessera.backtest.engine import TesseraBacktestEngine
from tessera.strategies.base import TesseraBaseStrategy, TesseraStrategyConfig

# ---------------------------------------------------------------------------
# Oracle strategy (re-defined locally to avoid circular fixture concerns)
# ---------------------------------------------------------------------------


class OracleCfg(TesseraStrategyConfig, frozen=True):
    future_closes: tuple[float, ...] = ()
    signal_delay_bars: int = 0
    position_qty: float = 0.01


class OracleStrat(TesseraBaseStrategy):
    """Fixed-size oracle that only trades the delta to reach target position."""

    def __init__(self, config: OracleCfg) -> None:
        super().__init__(config)
        self._ocfg = config
        self._bar_idx: dict[str, int] = {}

    def _on_bar_impl(self, bar: Bar) -> None:
        id_str = str(bar.bar_type.instrument_id)
        delay = self._ocfg.signal_delay_bars
        idx = self._bar_idx.get(id_str, 0)
        self._bar_idx[id_str] = idx + 1

        closes = self._ocfg.future_closes
        target_idx = idx + 1 + delay
        if target_idx >= len(closes) or idx >= len(closes):
            return

        if closes[idx] <= 0:
            return
        signal = 1 if closes[target_idx] > closes[idx] else -1
        target_qty = signal * self._ocfg.position_qty

        instr = self.cache.instrument(bar.bar_type.instrument_id)
        if instr is None:
            return

        from nautilus_trader.model.enums import OrderSide

        current_qty = self._net_position(bar.bar_type.instrument_id)
        delta = target_qty - current_qty
        min_qty = float(instr.size_increment)
        if abs(delta) < min_qty:
            return

        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        qty = instr.make_qty(abs(delta))
        order = self.order_factory.market(
            instrument_id=bar.bar_type.instrument_id,
            order_side=side,
            quantity=qty,
        )
        self.submit_order(order)


# ---------------------------------------------------------------------------
# Synthetic bar factory with alternating trend segments
# ---------------------------------------------------------------------------


def _make_alt_trend_bars(
    instrument_id_str: str,
    n: int = 600,
    drift: float = 0.0015,
    noise: float = 0.0003,
    seed: int = 0,
) -> tuple[list[Bar], tuple[float, ...]]:
    """Alternating up/down trend segments (50-bar periods)."""
    rng = np.random.default_rng(seed)
    bar_type = BarType.from_str(f"{instrument_id_str}-1-MINUTE-LAST-EXTERNAL")
    period = 50
    prices = [50_000.0]
    for i in range(n - 1):
        seg_drift = drift if (i // period) % 2 == 0 else -drift
        r = seg_drift + noise * rng.standard_normal()
        prices.append(prices[-1] * (1 + r))

    base_ts = int(pd.Timestamp("2023-01-01", tz="UTC").value)
    bar_ns = 60 * 1_000_000_000

    bars: list[Bar] = []
    for i, close in enumerate(prices):
        o = prices[i - 1] if i > 0 else close
        h = max(o, close) * 1.0001
        lo = min(o, close) * 0.9999
        ts = base_ts + i * bar_ns
        bars.append(
            Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{o:.1f}"),
                high=Price.from_str(f"{h:.1f}"),
                low=Price.from_str(f"{lo:.1f}"),
                close=Price.from_str(f"{close:.1f}"),
                volume=Quantity.from_str("10.000"),
                ts_event=ts,
                ts_init=ts,
            )
        )
    return bars, tuple(prices)


INSTR = "BTC-USDT-PERP.BINANCE"


def _run_oracle(delay: int, bars: list[Bar], closes: tuple[float, ...]) -> float:
    cfg = OracleCfg(
        instrument_ids=(INSTR,),
        future_closes=closes,
        signal_delay_bars=delay,
        max_drawdown_pct=99.0,
    )
    strategy = OracleStrat(config=cfg)
    engine = TesseraBacktestEngine.from_bars(
        {INSTR: bars},
        strategy,
        run_id=f"latency-delay{delay}",
        seed=0,
        latency_ms=0,
    )
    result = engine.run()
    return result.sharpe_ratio


def test_increasing_delay_degrades_sharpe() -> None:
    """Sharpe must not increase as signal delay grows: sharpe(0) ≥ sharpe(1) ≥ sharpe(3)."""
    bars, closes = _make_alt_trend_bars(INSTR, n=600, drift=0.0015, noise=0.0003, seed=0)

    sharpes = {delay: _run_oracle(delay, bars, closes) for delay in (0, 1, 3)}
    s0, s1, s3 = sharpes[0], sharpes[1], sharpes[3]

    msg01 = f"Sharpe should not improve from delay=0 to delay=1: {s0:.3f} < {s1:.3f}"
    msg13 = f"Sharpe should not improve from delay=1 to delay=3: {s1:.3f} < {s3:.3f}"
    assert s0 >= s1 - 1e-6, msg01
    assert s1 >= s3 - 1e-6, msg13


def test_zero_delay_has_nonnegative_sharpe_on_trending_data() -> None:
    """Oracle with delay=0 on a trending path must have Sharpe ≥ 0."""
    bars, closes = _make_alt_trend_bars(INSTR, n=300, drift=0.002, noise=0.0002, seed=1)
    s = _run_oracle(delay=0, bars=bars, closes=closes)
    assert s >= 0.0, f"Oracle with no delay should have non-negative Sharpe; got {s:.3f}"
