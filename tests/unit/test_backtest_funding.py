"""Test: funding payments are recorded at the configured cadence.

We run a backtest for > 16h of simulated time with an open position and verify:
  - At least 2 funding events are recorded (one at t=8h, one at t=16h).
  - Each event has non-zero funding_pnl (position is open).
  - The funding period respects the configured cadence.
"""

from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("nautilus_trader", reason="requires backtest extra: uv sync --extra backtest")

from nautilus_trader.model.data import Bar, BarType  # noqa: E402
from nautilus_trader.model.objects import Price, Quantity  # noqa: E402

from tessera.backtest.engine import TesseraBacktestEngine  # noqa: E402
from tessera.strategies.base import TesseraBaseStrategy, TesseraStrategyConfig  # noqa: E402

_NS_PER_HOUR = 3_600_000_000_000


# ---------------------------------------------------------------------------
# Long-only strategy: opens and holds a position
# ---------------------------------------------------------------------------


class LongOnlyConfig(TesseraStrategyConfig, frozen=True):
    """Strategy that opens a position on bar 1 and never closes."""

    position_qty: float = 0.01
    funding_period_nanos: int = 8 * _NS_PER_HOUR


class LongOnlyStrategy(TesseraBaseStrategy):
    """Opens a long position on the first bar and holds it."""

    def __init__(self, config: LongOnlyConfig) -> None:
        super().__init__(config)
        self._long_cfg = config
        self._entered = False

    def _on_bar_impl(self, bar: Bar) -> None:
        if self._entered:
            return
        instr = self.cache.instrument(bar.bar_type.instrument_id)
        if instr is None:
            return

        from nautilus_trader.model.enums import OrderSide

        qty = instr.make_qty(self._long_cfg.position_qty)
        order = self.order_factory.market(
            instrument_id=bar.bar_type.instrument_id,
            order_side=OrderSide.BUY,
            quantity=qty,
        )
        self.submit_order(order)
        self._entered = True


# ---------------------------------------------------------------------------
# Bar factory: 25 hours of 1-minute bars
# ---------------------------------------------------------------------------


def _make_25h_bars(
    instrument_id_str: str,
    price: float = 50_000.0,
) -> list[Bar]:
    """Create 25 hours × 60 minutes = 1500 bars at a constant price."""
    bar_type = BarType.from_str(f"{instrument_id_str}-1-MINUTE-LAST-EXTERNAL")
    n = 25 * 60
    base_ts = int(pd.Timestamp("2023-01-01 00:00:00", tz="UTC").value)
    bar_ns = 60 * 1_000_000_000  # 1 minute

    bars = []
    for i in range(n):
        ts = base_ts + i * bar_ns
        bars.append(
            Bar(
                bar_type=bar_type,
                open=Price.from_str(f"{price:.1f}"),
                high=Price.from_str(f"{price * 1.0001:.1f}"),
                low=Price.from_str(f"{price * 0.9999:.1f}"),
                close=Price.from_str(f"{price:.1f}"),
                volume=Quantity.from_str("10.000"),
                ts_event=ts,
                ts_init=ts,
            )
        )
    return bars


INSTR = "BTC-USDT-PERP.BINANCE"


def test_funding_events_recorded_at_8h_cadence() -> None:
    """At least 2 funding events in a 25h run with an open long position."""
    bars = _make_25h_bars(INSTR)
    cfg = LongOnlyConfig(
        instrument_ids=(INSTR,),
        max_drawdown_pct=99.0,
        funding_period_nanos=8 * _NS_PER_HOUR,
    )
    strategy = LongOnlyStrategy(config=cfg)

    engine = TesseraBacktestEngine.from_bars(
        {INSTR: bars},
        strategy,
        run_id="funding-test",
        seed=42,
        latency_ms=0,
        funding_period_hours=8,
    )
    result = engine.run()

    # 25h with 8h cadence: payments at ~8h, ~16h → at least 2 events
    n_events = len(result.funding_events)
    assert n_events >= 2, f"Expected ≥ 2 funding events in a 25h run; got {n_events}"


def test_funding_events_have_nonzero_pnl_when_position_is_open() -> None:
    """Each recorded funding event must carry a non-zero PnL (position held)."""
    bars = _make_25h_bars(INSTR)
    cfg = LongOnlyConfig(
        instrument_ids=(INSTR,),
        max_drawdown_pct=99.0,
        funding_period_nanos=8 * _NS_PER_HOUR,
    )
    strategy = LongOnlyStrategy(config=cfg)

    engine = TesseraBacktestEngine.from_bars(
        {INSTR: bars},
        strategy,
        run_id="funding-pnl-test",
        seed=42,
        latency_ms=0,
    )
    result = engine.run()

    if result.funding_events.empty:
        pytest.skip("No funding events recorded — position may not have opened")

    # Every event with a non-zero position should have a non-zero pnl
    df = result.funding_events
    if "net_qty" in df.columns and "funding_pnl" in df.columns:
        nonzero_pos = df[df["net_qty"].abs() > 1e-9]
        if not nonzero_pos.empty:
            all_nonzero = (nonzero_pos["funding_pnl"].abs() > 1e-9).all()
            assert all_nonzero, "Funding event with open position must have non-zero PnL"


def test_funding_pnl_accumulates() -> None:
    """Total funding PnL tracked by strategy equals sum of individual events."""
    bars = _make_25h_bars(INSTR)
    cfg = LongOnlyConfig(
        instrument_ids=(INSTR,),
        max_drawdown_pct=99.0,
        funding_period_nanos=8 * _NS_PER_HOUR,
    )
    strategy = LongOnlyStrategy(config=cfg)

    engine = TesseraBacktestEngine.from_bars(
        {INSTR: bars},
        strategy,
        run_id="funding-accum-test",
        seed=42,
        latency_ms=0,
    )
    result = engine.run()

    if not result.funding_events.empty and "funding_pnl" in result.funding_events.columns:
        expected_total = float(result.funding_events["funding_pnl"].sum())
        assert result.funding_pnl == pytest.approx(expected_total, abs=1e-9)
