"""Generate the Phase 8 evaluation tearsheet.

Runs the oracle strategy on 1-year synthetic data (same seed=7 as Phase 7)
and produces:
  docs/figures/phase8_tearsheet.html

New in Phase 8 vs Phase 7:
  - BTC buy-and-hold synthetic benchmark alongside the strategy
  - Deflated Sharpe Ratio (DSR) with correct trial count
  - Probabilistic Sharpe Ratio (PSR)
  - Bootstrap 95% CI for the annualised Sharpe
  - Stress-window table (all OOS/none since data is synthetic 2023)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.objects import Price, Quantity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tessera.backtest.engine import TesseraBacktestEngine
from tessera.backtest.reports import compute_trial_count, generate_tearsheet
from tessera.strategies.base import TesseraBaseStrategy, TesseraStrategyConfig

# ---------------------------------------------------------------------------
# Oracle strategy (identical to Phase 7)
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
# Synthetic 1-year bars (same data-generating process as Phase 7)
# ---------------------------------------------------------------------------


def make_year_bars(instrument_id_str: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    bar_type = BarType.from_str(f"{instrument_id_str}-1-MINUTE-LAST-EXTERNAL")

    n = 365 * 24 * 60
    drift = 0.00015
    noise = 0.00008
    period = 1440

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
    return bars, prices


def build_btc_benchmark(prices: list[float]) -> pd.Series:
    """Synthetic BTC buy-and-hold daily returns from the same price path."""
    price_arr = np.array(prices)
    daily_samples = price_arr[::1440]  # one price per day
    if len(daily_samples) < 2:
        return pd.Series(dtype=float)
    daily_ret = np.diff(daily_samples) / daily_samples[:-1]
    dates = pd.date_range("2023-01-01", periods=len(daily_ret), freq="D")
    return pd.Series(daily_ret, index=dates, name="BTC Buy-and-Hold")


def main() -> None:
    instr = "BTC-USDT-PERP.BINANCE"
    out = Path(__file__).parent.parent / "docs" / "figures" / "phase8_tearsheet.html"
    out.parent.mkdir(parents=True, exist_ok=True)

    print("Generating 1-year synthetic bars (seed=7)…")
    bars, prices = make_year_bars(instr, seed=7)

    cfg = OracleConfig(
        instrument_ids=(instr,),
        future_closes=tuple(prices),
        max_drawdown_pct=99.0,
    )
    strategy = OracleStrategy(config=cfg)
    engine = TesseraBacktestEngine.from_bars(
        {instr: bars},
        strategy,
        run_id="phase8-tearsheet",
        seed=7,
        latency_ms=0,
    )

    print("Running backtest…")
    import time

    t0 = time.monotonic()
    result = engine.run()
    elapsed = time.monotonic() - t0
    sr = result.sharpe_ratio
    dd = result.max_drawdown
    print(f"Done in {elapsed:.1f}s  |  Sharpe={sr:.3f}  |  MaxDD={dd:.2%}")

    if result.equity_curve.empty or len(result.equity_curve) < 2:
        print("ERROR: equity curve too short — aborting tearsheet generation.")
        return

    strategy_returns = result.equity_curve.pct_change().dropna()
    strategy_returns.index = pd.to_datetime(strategy_returns.index)
    strategy_returns.name = "Tessera Oracle (Phase 8)"

    btc_benchmark = build_btc_benchmark(prices)
    # Align benchmark to strategy dates
    common_idx = strategy_returns.index.intersection(btc_benchmark.index)
    benchmark = btc_benchmark.reindex(common_idx) if len(common_idx) > 0 else None

    # Trial count: Phase 6 notebook 04 (20 trials) + Phase 6 notebook 05 (20 trials)
    # + ablation variants in notebook 11 (~40 configs) + production CLI budget (100)
    n_trials = compute_trial_count(manual_configs=180)  # 20 + 20 + 40 + 100
    print(f"Trial count for DSR: {n_trials}")

    # SR std estimate: std of fold Sharpes from notebook 04
    # Notebook 04 showed CV Sharpe std=0.033; use as a lower bound
    sr_std = 0.5  # conservative — typical spread across Optuna configs

    print(f"Generating Phase 8 tearsheet → {out}")
    generate_tearsheet(
        returns=strategy_returns,
        benchmark_returns=benchmark,
        output_path=out,
        trial_count=n_trials,
        sr_std=sr_std,
        n_obs_per_year=252,
        block_size=20,  # ~20-bar mean holding for this oracle strategy
        n_bootstrap=5_000,  # reduced for script speed; use 10_000 in production
        test_start_date="2023-01-01",  # all data is OOS (synthetic, no in-sample period)
        title=(
            "Tessera Phase 8 — Oracle Strategy | 2023-01-01 → 2023-12-31 | BTC-USDT-PERP | v0.8"
        ),
    )
    print(f"Tearsheet written: {out}")


if __name__ == "__main__":
    main()
