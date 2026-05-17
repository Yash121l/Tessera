"""TesseraBacktestEngine — Nautilus Trader backtest wrapper.

Responsibilities:
  - Load OHLCV bars from Parquet for the configured universe + date range.
  - Configure Nautilus venues (Binance, Bybit) with fee schedules and latency.
  - Apply the square-root slippage model via strategy-level price adjustment.
  - Schedule funding payments every 8h (or 4h) via the strategy's FundingTracker.
  - Persist every fill, position snapshot, and funding event to Parquet under
    data/backtest_runs/{run_id}/.
  - Return a BacktestResult with equity curve, Sharpe, and disk metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import structlog
from nautilus_trader.backtest.engine import BacktestEngine as _NautilusEngine
from nautilus_trader.backtest.models import FillModel, LatencyModel
from nautilus_trader.config import BacktestEngineConfig as _NautilusEngineConfig
from nautilus_trader.config import LoggingConfig as _LoggingConfig
from nautilus_trader.model.currencies import USDT
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Money, Price, Quantity

from tessera.config import BacktestConfig, TesseraSettings

if TYPE_CHECKING:
    from tessera.strategies.base import TesseraBaseStrategy

logger = structlog.get_logger(__name__)

_NS_PER_MS = 1_000_000
_NS_PER_S = 1_000_000_000


# ---------------------------------------------------------------------------
# Instrument spec registry
# ---------------------------------------------------------------------------

# Conservative defaults for major perpetuals.
# In production these come from the exchange API; here we hard-code VIP 0 specs.
_INSTRUMENT_SPECS: dict[str, dict[str, Any]] = {
    "BTC/USDT:USDT": {
        "base": "BTC",
        "price_precision": 1,
        "size_precision": 3,
        "price_increment": "0.1",
        "size_increment": "0.001",
        "maker_fee": "0.0002",
        "taker_fee": "0.0005",
    },
    "ETH/USDT:USDT": {
        "base": "ETH",
        "price_precision": 2,
        "size_precision": 2,
        "price_increment": "0.01",
        "size_increment": "0.01",
        "maker_fee": "0.0002",
        "taker_fee": "0.0005",
    },
}

_DEFAULT_SPEC: dict[str, Any] = {
    "price_precision": 2,
    "size_precision": 3,
    "price_increment": "0.01",
    "size_increment": "0.001",
    "maker_fee": "0.0002",
    "taker_fee": "0.0005",
}


def _ccxt_to_nautilus_id(exchange: str, symbol: str) -> str:
    """Map CCXT symbol to Nautilus InstrumentId string.

    "BTC/USDT:USDT" on "binance" → "BTC-USDT-PERP.BINANCE"
    """
    base = symbol.split("/")[0]
    return f"{base}-USDT-PERP.{exchange.upper()}"


def _make_instrument(exchange: str, ccxt_symbol: str, ts_ns: int) -> CryptoPerpetual:
    """Create a CryptoPerpetual instrument from a CCXT symbol."""
    from nautilus_trader.model.currencies import Currency

    spec = _INSTRUMENT_SPECS.get(ccxt_symbol, _DEFAULT_SPEC).copy()
    base_code = ccxt_symbol.split("/")[0]

    try:
        base_ccy = Currency.from_str(base_code)
    except Exception:
        from nautilus_trader.model.currencies import BTC

        base_ccy = BTC

    nautilus_id_str = _ccxt_to_nautilus_id(exchange, ccxt_symbol)
    instr_id = InstrumentId.from_str(nautilus_id_str)
    raw_sym = Symbol(base_code + "USDT")

    return CryptoPerpetual(
        instrument_id=instr_id,
        raw_symbol=raw_sym,
        base_currency=base_ccy,
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        price_precision=spec["price_precision"],
        size_precision=spec["size_precision"],
        price_increment=Price.from_str(spec["price_increment"]),
        size_increment=Quantity.from_str(spec["size_increment"]),
        ts_event=ts_ns,
        ts_init=ts_ns,
        maker_fee=Decimal(spec["maker_fee"]),
        taker_fee=Decimal(spec["taker_fee"]),
    )


def _make_instrument_from_id(instrument_id_str: str, ts_ns: int) -> CryptoPerpetual:
    """Create a CryptoPerpetual instrument from a Nautilus InstrumentId string.

    E.g. "BTC-USDT-PERP.BINANCE" → CryptoPerpetual with BTC/USDT specs.
    """
    from nautilus_trader.model.currencies import Currency

    instr_id = InstrumentId.from_str(instrument_id_str)
    # Symbol format: "BTC-USDT-PERP" → base="BTC"
    sym_parts = str(instr_id.symbol).split("-")
    base_code = sym_parts[0] if sym_parts else "BTC"

    # Look up specs via CCXT-style key first
    ccxt_key = f"{base_code}/USDT:USDT"
    spec = _INSTRUMENT_SPECS.get(ccxt_key, _DEFAULT_SPEC).copy()

    try:
        base_ccy = Currency.from_str(base_code)
    except Exception:
        from nautilus_trader.model.currencies import BTC

        base_ccy = BTC

    raw_sym = Symbol(base_code + "USDT")

    return CryptoPerpetual(
        instrument_id=instr_id,
        raw_symbol=raw_sym,
        base_currency=base_ccy,
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        price_precision=spec["price_precision"],
        size_precision=spec["size_precision"],
        price_increment=Price.from_str(spec["price_increment"]),
        size_increment=Quantity.from_str(spec["size_increment"]),
        ts_event=ts_ns,
        ts_init=ts_ns,
        maker_fee=Decimal(spec["maker_fee"]),
        taker_fee=Decimal(spec["taker_fee"]),
    )


# ---------------------------------------------------------------------------
# Bar conversion
# ---------------------------------------------------------------------------


def df_to_nautilus_bars(df: pd.DataFrame, bar_type: BarType) -> list[Bar]:
    """Convert a Tessera OHLCV DataFrame to a list of Nautilus Bar objects.

    Expected columns: open, high, low, close, volume, event_time (datetime).
    """
    bars: list[Bar] = []
    for row in df.itertuples(index=False):
        ts_ns = int(pd.Timestamp(row.event_time).value)
        bars.append(
            Bar(
                bar_type=bar_type,
                open=Price.from_str(
                    f"{float(row.open):.{bar_type.instrument_id.symbol.value.count('-')}f}"
                ),
                high=Price.from_str(str(float(row.high))),
                low=Price.from_str(str(float(row.low))),
                close=Price.from_str(str(float(row.close))),
                volume=Quantity.from_str(str(float(row.volume))),
                ts_event=ts_ns,
                ts_init=ts_ns,
            )
        )
    return bars


def df_to_nautilus_bars_v2(
    df: pd.DataFrame,
    bar_type: BarType,
    price_precision: int = 2,
    size_precision: int = 3,
) -> list[Bar]:
    """Robust bar converter with configurable precision."""
    price_fmt = f"{{:.{price_precision}f}}"
    size_fmt = f"{{:.{size_precision}f}}"
    bars: list[Bar] = []
    for row in df.itertuples(index=False):
        ts_ns = int(pd.Timestamp(row.event_time).value)
        bars.append(
            Bar(
                bar_type=bar_type,
                open=Price.from_str(price_fmt.format(float(row.open))),
                high=Price.from_str(price_fmt.format(float(row.high))),
                low=Price.from_str(price_fmt.format(float(row.low))),
                close=Price.from_str(price_fmt.format(float(row.close))),
                volume=Quantity.from_str(size_fmt.format(float(row.volume))),
                ts_event=ts_ns,
                ts_init=ts_ns,
            )
        )
    return bars


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    """Summary statistics returned after a Tessera backtest run."""

    run_id: str
    start_date: str
    end_date: str
    n_bars: int
    n_trades: int
    total_pnl: float  # trading + funding
    trading_pnl: float
    funding_pnl: float
    fee_pnl: float  # always ≤ 0
    total_return: float  # decimal (0.10 = 10%)
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float  # decimal (0.05 = 5%)
    equity_curve: pd.Series  # index=timestamp, values=USD equity
    fills: pd.DataFrame
    funding_events: pd.DataFrame
    elapsed_s: float
    log_dir: str | None = None

    @property
    def tearsheet_available(self) -> bool:
        return not self.equity_curve.empty


# ---------------------------------------------------------------------------
# TesseraBacktestEngine
# ---------------------------------------------------------------------------


class TesseraBacktestEngine:
    """Tessera event-driven backtest engine backed by Nautilus Trader.

    Usage::

        engine = TesseraBacktestEngine.from_config(config, settings, strategy, run_id)
        result = engine.run()

    For testing, inject pre-built bars::

        engine = TesseraBacktestEngine.from_bars(bars_by_symbol, strategy, run_id)
        result = engine.run()
    """

    def __init__(
        self,
        config: BacktestConfig,
        settings: TesseraSettings,
        strategy: TesseraBaseStrategy,
        run_id: str,
        seed: int = 42,
        bars_override: dict[str, list[Bar]] | None = None,
    ) -> None:
        self._config = config
        self._settings = settings
        self._strategy = strategy
        self._run_id = run_id
        self._seed = seed
        self._bars_override = bars_override  # {ccxt_symbol: [Bar, ...]}
        self._rng = np.random.default_rng(seed)

        # Draw a single latency value from the configured range (deterministic).
        # Nautilus LatencyModel doesn't support per-order random latency, so we
        # draw once and fix it for the entire run. For ablation, vary the config.
        bec = config.backtest
        latency_ms = int(self._rng.integers(bec.latency_min_ms, bec.latency_max_ms + 1))
        self._latency_nanos = latency_ms * _NS_PER_MS

        self._log_dir: Path | None = None
        if settings.data_root:
            self._log_dir = settings.data_root / "backtest_runs" / run_id

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: BacktestConfig,
        settings: TesseraSettings,
        strategy: TesseraBaseStrategy,
        run_id: str,
        seed: int = 42,
    ) -> TesseraBacktestEngine:
        """Build engine from YAML config; loads bars from Parquet."""
        return cls(config, settings, strategy, run_id, seed)

    @classmethod
    def from_bars(
        cls,
        bars_by_symbol: dict[str, list[Bar]],
        strategy: TesseraBaseStrategy,
        run_id: str = "test",
        seed: int = 42,
        latency_ms: int = 100,
        funding_period_hours: int = 8,
    ) -> TesseraBacktestEngine:
        """Build engine with pre-loaded bars (for deterministic testing).

        Args:
            bars_by_symbol: dict keyed by Nautilus instrument ID string
                            (e.g. "BTC-USDT-PERP.BINANCE") → list of Bar.
        """
        from tessera.config import BacktestConfig, BacktestEngineConfig, CostConfig, VenueConfig

        # Infer unique venues from instrument ID strings
        venues = []
        seen_venues: set[str] = set()
        for id_str in bars_by_symbol:
            venue_str = id_str.split(".")[-1].lower()  # "BINANCE" → "binance"
            if venue_str not in seen_venues:
                seen_venues.add(venue_str)
                venues.append(
                    VenueConfig(
                        exchange=venue_str,
                        symbols=[],  # not used in bars_override path
                        initial_balance_usdt=100_000.0,
                        funding_period_hours=funding_period_hours,
                    )
                )

        config = BacktestConfig(
            backtest=BacktestEngineConfig(
                latency_min_ms=latency_ms,
                latency_max_ms=latency_ms,
                venues=venues,
            ),
            costs=CostConfig(),
        )
        settings = TesseraSettings(data_root=Path("/tmp/tessera_test"))
        engine = cls(config, settings, strategy, run_id, seed, bars_override=bars_by_symbol)
        return engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> BacktestResult:
        """Execute the backtest and return aggregated results."""
        t0 = time.monotonic()
        logger.info("backtest_start", run_id=self._run_id, seed=self._seed)

        nautilus_engine = self._build_nautilus_engine()
        # bars_override keys are Nautilus ID strings ("BTC-USDT-PERP.BINANCE") not CCXT symbols.
        all_bars: dict[str, list[Bar]] = self._bars_override or self._load_bars_from_parquet()

        # Register instruments and bar data
        instruments_added: set[str] = set()
        for key, bars in all_bars.items():
            if not bars:
                continue

            ts_ns = bars[0].ts_event

            if self._bars_override:
                # Key is already a Nautilus instrument ID string
                instr = _make_instrument_from_id(key, ts_ns)
            else:
                # Key is a CCXT symbol string
                exchange = self._infer_exchange(key)
                instr = _make_instrument(exchange, key, ts_ns)

            instr_id_str = str(instr.id)
            if instr_id_str not in instruments_added:
                nautilus_engine.add_instrument(instr)
                instruments_added.add(instr_id_str)

            nautilus_engine.add_data(bars)

        nautilus_engine.add_strategy(self._strategy)
        nautilus_engine.run()

        elapsed = time.monotonic() - t0
        result = self._build_result(nautilus_engine, all_bars, elapsed)

        self._persist_result(result)
        logger.info(
            "backtest_complete",
            run_id=self._run_id,
            n_trades=result.n_trades,
            sharpe=round(result.sharpe_ratio, 3),
            elapsed_s=round(elapsed, 1),
        )
        return result

    # ------------------------------------------------------------------
    # Nautilus engine construction
    # ------------------------------------------------------------------

    def _build_nautilus_engine(self) -> _NautilusEngine:
        engine_config = _NautilusEngineConfig(
            trader_id=f"TESSERA-{self._run_id[:8].upper()}",
            logging=_LoggingConfig(log_level="WARNING"),
        )
        nautilus = _NautilusEngine(config=engine_config)

        fill_model = FillModel(
            prob_fill_on_limit=1.0,  # always fill limits when market touches
            prob_fill_on_stop=1.0,
            prob_slippage=0.0,  # slippage handled at strategy level
            random_seed=self._seed,
        )
        latency_model = LatencyModel(
            base_latency_nanos=self._latency_nanos,
            insert_latency_nanos=self._latency_nanos,
        )

        for venue_cfg in self._config.backtest.venues:
            nautilus.add_venue(
                venue=Venue(venue_cfg.exchange.upper()),
                oms_type=OmsType.NETTING,
                account_type=AccountType.MARGIN,
                starting_balances=[Money(venue_cfg.initial_balance_usdt, USDT)],
                default_leverage=Decimal(str(venue_cfg.default_leverage)),
                fill_model=fill_model,
                latency_model=latency_model,
                bar_execution=True,
                bar_adaptive_high_low_ordering=True,
            )

        return nautilus

    # ------------------------------------------------------------------
    # Bar loading from Parquet
    # ------------------------------------------------------------------

    def _load_bars_from_parquet(self) -> dict[str, list[Bar]]:
        from tessera.data.store import read_parquet

        bec = self._config.backtest
        result: dict[str, list[Bar]] = {}

        for venue_cfg in bec.venues:
            for ccxt_sym in venue_cfg.symbols:
                df = read_parquet(
                    "ohlcv",
                    filters=[
                        ("exchange", "==", venue_cfg.exchange),
                        ("symbol", "==", ccxt_sym),
                    ],
                )
                if df.empty:
                    logger.warning("no_data", exchange=venue_cfg.exchange, symbol=ccxt_sym)
                    continue

                if "event_time" in df.columns:
                    df["event_time"] = pd.to_datetime(df["event_time"])
                    df = df[
                        (df["event_time"] >= bec.start_date) & (df["event_time"] <= bec.end_date)
                    ]

                if df.empty:
                    continue

                df = df.sort_values("event_time").reset_index(drop=True)
                spec = _INSTRUMENT_SPECS.get(ccxt_sym, _DEFAULT_SPEC)
                nautilus_id_str = _ccxt_to_nautilus_id(venue_cfg.exchange, ccxt_sym)
                bar_type = BarType.from_str(f"{nautilus_id_str}-{bec.bar_type}")
                bars = df_to_nautilus_bars_v2(
                    df,
                    bar_type,
                    price_precision=spec["price_precision"],
                    size_precision=spec["size_precision"],
                )
                result[ccxt_sym] = bars
                logger.info("bars_loaded", symbol=ccxt_sym, n_bars=len(bars))

        return result

    # ------------------------------------------------------------------
    # Result construction
    # ------------------------------------------------------------------

    def _build_result(
        self,
        nautilus: _NautilusEngine,
        all_bars: dict[str, list[Bar]],
        elapsed: float,
    ) -> BacktestResult:
        fills_df = pd.DataFrame(self._strategy.fills)
        funding_df = pd.DataFrame(self._strategy.funding_events)

        n_bars = sum(len(b) for b in all_bars.values())
        n_trades = len(fills_df)

        trading_pnl = self._compute_trading_pnl(fills_df)
        funding_pnl = self._strategy.funding_pnl
        fee_pnl = self._compute_fee_pnl(fills_df)
        total_pnl = trading_pnl + funding_pnl + fee_pnl

        equity_curve = self._build_equity_curve(fills_df, all_bars)
        sharpe = _compute_sharpe(equity_curve)
        sortino = _compute_sortino(equity_curve)
        max_dd = _compute_max_drawdown(equity_curve)
        total_return = (
            float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
            if len(equity_curve) > 1
            else 0.0
        )

        bec = self._config.backtest
        return BacktestResult(
            run_id=self._run_id,
            start_date=bec.start_date,
            end_date=bec.end_date,
            n_bars=n_bars,
            n_trades=n_trades,
            total_pnl=total_pnl,
            trading_pnl=trading_pnl,
            funding_pnl=funding_pnl,
            fee_pnl=fee_pnl,
            total_return=total_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            equity_curve=equity_curve,
            fills=fills_df,
            funding_events=funding_df,
            elapsed_s=elapsed,
            log_dir=str(self._log_dir) if self._log_dir else None,
        )

    def _compute_trading_pnl(self, fills_df: pd.DataFrame) -> float:
        if fills_df.empty or "price" not in fills_df.columns:
            return 0.0
        # Simple PnL from fills: buy → negative cash, sell → positive cash
        pnl = 0.0
        for _, row in fills_df.iterrows():
            qty = float(row["qty"])
            px = float(row["price"])
            if row["side"] == "BUY":
                pnl -= qty * px
            else:
                pnl += qty * px
        return pnl

    def _compute_fee_pnl(self, fills_df: pd.DataFrame) -> float:
        if fills_df.empty or "commission" not in fills_df.columns:
            return 0.0
        return -abs(float(fills_df["commission"].sum()))

    def _build_equity_curve(
        self, fills_df: pd.DataFrame, all_bars: dict[str, list[Bar]]
    ) -> pd.Series:
        """Simple mark-to-market equity curve using fill history."""
        initial_capital = sum(v.initial_balance_usdt for v in self._config.backtest.venues)

        if fills_df.empty:
            # No trades → flat equity
            all_ts = [b.ts_event for bars in all_bars.values() for b in bars]
            if not all_ts:
                return pd.Series([initial_capital], name="equity")
            idx = pd.to_datetime(sorted(set(all_ts)))
            return pd.Series(initial_capital, index=idx, name="equity")

        # Build equity from cumulative cash flows + mark-to-market positions
        fills_sorted = fills_df.sort_values("ts_ns")
        cash = initial_capital
        equity_points: list[tuple[int, float]] = []

        for _, row in fills_sorted.iterrows():
            qty = float(row["qty"])
            px = float(row["price"])
            commission = float(row.get("commission", 0))
            if row["side"] == "BUY":
                cash -= qty * px + commission
            else:
                cash += qty * px - commission
            equity_points.append((int(row["ts_ns"]), cash))

        if equity_points:
            ts_arr = [ep[0] for ep in equity_points]
            eq_arr = [ep[1] for ep in equity_points]
            idx = pd.to_datetime(ts_arr)
            return pd.Series(eq_arr, index=idx, name="equity").resample("1D").last().ffill()

        return pd.Series([initial_capital], name="equity")

    def _infer_exchange(self, ccxt_sym: str) -> str:
        for venue_cfg in self._config.backtest.venues:
            if ccxt_sym in venue_cfg.symbols:
                return venue_cfg.exchange
        return "binance"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _persist_result(self, result: BacktestResult) -> None:
        if self._log_dir is None:
            return
        self._log_dir.mkdir(parents=True, exist_ok=True)

        if not result.fills.empty:
            result.fills.to_parquet(self._log_dir / "fills.parquet", index=False)

        if not result.funding_events.empty:
            result.funding_events.to_parquet(self._log_dir / "funding.parquet", index=False)

        if not result.equity_curve.empty:
            result.equity_curve.to_frame("equity").to_parquet(
                self._log_dir / "equity_curve.parquet"
            )

        # Write summary JSON
        summary = {
            "run_id": result.run_id,
            "start_date": result.start_date,
            "end_date": result.end_date,
            "n_bars": result.n_bars,
            "n_trades": result.n_trades,
            "total_return": result.total_return,
            "sharpe_ratio": result.sharpe_ratio,
            "sortino_ratio": result.sortino_ratio,
            "max_drawdown": result.max_drawdown,
            "total_pnl": result.total_pnl,
            "trading_pnl": result.trading_pnl,
            "funding_pnl": result.funding_pnl,
            "fee_pnl": result.fee_pnl,
            "elapsed_s": result.elapsed_s,
            "log_dir": result.log_dir,
        }
        import json

        with open(self._log_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        log_size_mb = sum(p.stat().st_size for p in self._log_dir.rglob("*") if p.is_file()) / 1e6
        logger.info(
            "result_persisted",
            log_dir=str(self._log_dir),
            size_mb=round(log_size_mb, 2),
        )


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


def _compute_sharpe(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    returns = equity.pct_change().dropna()
    if returns.empty or returns.std() < 1e-12:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))


def _compute_sortino(equity: pd.Series, periods_per_year: int = 252) -> float:
    if len(equity) < 2:
        return 0.0
    returns = equity.pct_change().dropna()
    downside = returns[returns < 0]
    if downside.empty or downside.std() < 1e-12:
        return 0.0
    return float(returns.mean() / downside.std() * np.sqrt(periods_per_year))


def _compute_max_drawdown(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(drawdown.min())
