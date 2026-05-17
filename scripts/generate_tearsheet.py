"""Generate a QuantStats HTML tearsheet for the PHASE 7 REPORT.

Runs an oracle strategy on 1-year synthetic data and produces:
  docs/figures/phase7_tearsheet.html
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import quantstats as qs
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.objects import Price, Quantity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tessera.backtest.engine import TesseraBacktestEngine
from tessera.strategies.base import TesseraBaseStrategy, TesseraStrategyConfig

# ---------------------------------------------------------------------------
# Oracle strategy
# ---------------------------------------------------------------------------


class OracleConfig(TesseraStrategyConfig, frozen=True):
    future_closes: tuple[float, ...] = ()
    position_qty: float = 0.01


class OracleStrategy(TesseraBaseStrategy):
    def __init__(self, config: OracleConfig) -> None:
        super().__init__(config)
        self._ocfg = config
        self._bar_idx: dict[str, int] = {}

    def _on_bar_impl(self, bar: Bar) -> None:
        from nautilus_trader.model.enums import OrderSide

        id_str = str(bar.bar_type.instrument_id)
        idx = self._bar_idx.get(id_str, 0)
        self._bar_idx[id_str] = idx + 1

        closes = self._ocfg.future_closes
        if idx + 1 >= len(closes) or idx >= len(closes):
            return
        if closes[idx] <= 0:
            return

        signal = 1 if closes[idx + 1] > closes[idx] else -1
        target_qty = signal * self._ocfg.position_qty

        instr = self.cache.instrument(bar.bar_type.instrument_id)
        if instr is None:
            return

        current_qty = self._net_position(bar.bar_type.instrument_id)
        delta = target_qty - current_qty
        if abs(delta) < float(instr.size_increment):
            return

        side = OrderSide.BUY if delta > 0 else OrderSide.SELL
        order = self.order_factory.market(
            instrument_id=bar.bar_type.instrument_id,
            order_side=side,
            quantity=instr.make_qty(abs(delta)),
        )
        self.submit_order(order)


# ---------------------------------------------------------------------------
# Synthetic 1-year data (minute bars, alternating trend segments)
# ---------------------------------------------------------------------------


def make_year_bars(instrument_id_str: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    bar_type = BarType.from_str(f"{instrument_id_str}-1-MINUTE-LAST-EXTERNAL")

    n = 365 * 24 * 60  # 1 year of minute bars
    drift = 0.00015  # per-bar
    noise = 0.00008
    period = 1440  # 1-day segments

    prices = [50_000.0]
    for i in range(n - 1):
        seg_drift = drift if (i // period) % 2 == 0 else -drift
        r = seg_drift + noise * rng.standard_normal()
        prices.append(max(prices[-1] * (1 + r), 1.0))

    base_ts = int(pd.Timestamp("2023-01-01", tz="UTC").value)
    bar_ns = 60 * 1_000_000_000

    bars = []
    for i, close in enumerate(prices):
        o = prices[i - 1] if i > 0 else close
        h = max(o, close) * 1.00005
        lo = min(o, close) * 0.99995
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


def main():
    instr = "BTC-USDT-PERP.BINANCE"
    out = Path(__file__).parent.parent / "docs" / "figures" / "phase7_tearsheet.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Generating 1-year synthetic bars…")
    bars, closes = make_year_bars(instr, seed=7)

    cfg = OracleConfig(
        instrument_ids=(instr,),
        future_closes=closes,
        max_drawdown_pct=99.0,
    )
    strategy = OracleStrategy(config=cfg)
    engine = TesseraBacktestEngine.from_bars(
        {instr: bars},
        strategy,
        run_id="phase7-tearsheet",
        seed=7,
        latency_ms=0,
    )

    print("Running backtest…")
    import time

    t0 = time.monotonic()
    result = engine.run()
    elapsed = time.monotonic() - t0

    print(f"Done in {elapsed:.1f}s")
    print(f"  n_bars   = {result.n_bars:,}")
    print(f"  n_trades = {result.n_trades:,}")
    print(f"  Sharpe   = {result.sharpe_ratio:.3f}")
    print(f"  Sortino  = {result.sortino_ratio:.3f}")
    print(f"  Max DD   = {result.max_drawdown:.2%}")
    print(f"  Total PnL= ${result.total_pnl:,.2f}")

    # Build daily returns for QuantStats
    equity = result.equity_curve
    if len(equity) < 2:
        print("WARNING: equity curve too short for tearsheet")
        return

    returns = equity.pct_change().dropna()
    returns.index = pd.to_datetime(returns.index)
    returns.name = "Tessera Oracle (Phase 7)"

    print(f"Generating HTML tearsheet → {out}")
    qs.reports.html(
        returns,
        output=str(out),
        title="Tessera Phase 7 — Oracle Strategy Tearsheet",
        benchmark=None,
    )
    print("Tearsheet written.")

    # Estimate disk size (fills parquet would be this size in production)
    fills_mb = len(result.fills) * 200 / 1e6  # ~200 bytes per fill row
    print(f"\nDisk estimate: {fills_mb:.3f} MB for {len(result.fills)} fills")
    print(f"Wall-clock extrapolation: {elapsed:.1f}s for {result.n_bars:,} bars")
    bars_per_year = 365 * 24 * 60
    if result.n_bars > 0:
        four_year_s = elapsed * (4 * bars_per_year / result.n_bars)
        print(f"  → 4-year estimate: {four_year_s:.1f}s ({four_year_s / 60:.1f} min)")


if __name__ == "__main__":
    main()
