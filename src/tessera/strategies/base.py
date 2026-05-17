"""Base Nautilus Strategy for Tessera.

Every Tessera strategy inherits from TesseraBaseStrategy, which wires:
  - Bar subscriptions for each configured instrument
  - Kill switch (halt + flatten on max drawdown breach)
  - Fill logging to an in-memory buffer, flushed to Parquet on stop
  - Funding PnL tracking (applied at on_bar when the funding period elapses)

Subclasses must implement _on_bar_impl(bar) with the actual signal logic.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from collections import defaultdict
from typing import Any

import pandas as pd
import structlog
from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import OrderSide
from nautilus_trader.model.events import OrderFilled
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Strategy

logger = structlog.get_logger(__name__)

_NS_PER_HOUR = 3_600_000_000_000


class TesseraStrategyConfig(StrategyConfig, frozen=True):
    """Configuration for TesseraBaseStrategy.

    All fields are immutable after creation (frozen Struct).
    """

    instrument_ids: tuple[str, ...]  # e.g. ("BTC-USDT-PERP.BINANCE",)
    bar_type_suffix: str = "1-MINUTE-LAST-EXTERNAL"

    # Kill switch
    max_drawdown_pct: float = 5.0

    # Order management
    taker_delay_bars: int = 1  # cancel unfilled limit + resubmit market after N bars

    # Position sizing
    kelly_fraction: float = 0.25
    vol_target_pct: float = 0.15  # annualised vol target
    max_position_pct: float = 0.10  # fraction of portfolio per symbol

    # Signal delay in bars (latency ablation parameter)
    signal_delay_bars: int = 0

    # Funding cadence in nanoseconds (8h default)
    funding_period_nanos: int = 8 * _NS_PER_HOUR

    # Parquet log directory (None = skip writing)
    log_dir: str | None = None


class TesseraBaseStrategy(Strategy):
    """Abstract Tessera Strategy.

    Handles lifecycle, kill switch, fill logging, and funding tracking.
    Subclasses implement _on_bar_impl(bar).
    """

    def __init__(self, config: TesseraStrategyConfig) -> None:
        super().__init__(config)
        self._cfg = config
        self._instrument_ids: list[InstrumentId] = []
        self._bar_types: list[BarType] = []

        # Kill switch state
        self._kill_triggered = False
        self._peak_equity: float | None = None

        # Fill log: list[dict] → written to Parquet on stop
        self._fills: list[dict[str, Any]] = []

        # Funding tracker: instrument_id → last funding ts (ns)
        self._last_funding_ns: dict[str, int] = {}
        # Funding PnL accumulator (not in Nautilus account, tracked separately)
        self._funding_pnl: float = 0.0
        self._funding_events: list[dict[str, Any]] = []

        # Signal buffer for delay: deque of (bar, signal) tuples
        from collections import deque

        self._signal_buffers: dict[str, deque[tuple[Bar, int]]] = defaultdict(
            lambda: deque(maxlen=max(1, config.signal_delay_bars + 1))
        )

        # Track pending limit orders for taker fallback
        self._pending_limits: dict[str, list[Any]] = defaultdict(list)

        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_start(self) -> None:
        self._start_time = time.monotonic()
        for id_str in self._cfg.instrument_ids:
            instr_id = InstrumentId.from_str(id_str)
            bar_type = BarType.from_str(f"{id_str}-{self._cfg.bar_type_suffix}")
            self._instrument_ids.append(instr_id)
            self._bar_types.append(bar_type)
            self.subscribe_bars(bar_type)
            self._last_funding_ns[id_str] = 0
        logger.info(
            "strategy_started",
            instruments=list(self._cfg.instrument_ids),
            kill_switch_drawdown_pct=self._cfg.max_drawdown_pct,
        )

    def on_stop(self) -> None:
        if not self._kill_triggered:
            self._flatten_all()
        self._write_fill_log()
        elapsed = time.monotonic() - self._start_time
        logger.info(
            "strategy_stopped",
            n_fills=len(self._fills),
            funding_pnl=round(self._funding_pnl, 4),
            elapsed_s=round(elapsed, 2),
        )

    # ------------------------------------------------------------------
    # Bar dispatch
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> None:
        if self._kill_triggered:
            return
        self._check_kill_switch(bar)
        if self._kill_triggered:
            return

        # Apply funding before processing signal
        self._apply_funding(bar)

        # Buffer the current bar for signal delay
        id_str = str(bar.bar_type.instrument_id)
        if self._cfg.signal_delay_bars > 0:
            self._signal_buffers[id_str].append((bar, 0))  # signal=0 placeholder

        self._on_bar_impl(bar)

    @abstractmethod
    def _on_bar_impl(self, bar: Bar) -> None:
        """Subclass implements the actual signal and order logic."""
        ...

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def _check_kill_switch(self, bar: Bar) -> None:
        account = self.portfolio.account(bar.bar_type.instrument_id.venue)
        if account is None:
            return

        from nautilus_trader.model.currencies import USDT

        balance = account.balance(USDT)
        if balance is None:
            return

        equity = float(balance.total)
        if self._peak_equity is None or equity > self._peak_equity:
            self._peak_equity = equity

        drawdown = (self._peak_equity - equity) / self._peak_equity
        if drawdown > self._cfg.max_drawdown_pct / 100.0:
            logger.warning(
                "kill_switch_triggered",
                drawdown_pct=round(drawdown * 100, 2),
                threshold_pct=self._cfg.max_drawdown_pct,
            )
            self._trigger_kill_switch()

    def _trigger_kill_switch(self) -> None:
        self._kill_triggered = True
        self._flatten_all()

    def _flatten_all(self) -> None:
        for instr_id in self._instrument_ids:
            self.cancel_all_orders(instrument_id=instr_id)
        for instr_id in self._instrument_ids:
            net_qty = self._net_position(instr_id)
            if abs(net_qty) < 1e-9:
                continue
            side = OrderSide.SELL if net_qty > 0 else OrderSide.BUY
            instr = self.cache.instrument(instr_id)
            if instr is None:
                continue
            order = self.order_factory.market(
                instrument_id=instr_id,
                order_side=side,
                quantity=instr.make_qty(abs(net_qty)),
            )
            self.submit_order(order)

    # ------------------------------------------------------------------
    # Funding (tracked separately, not in Nautilus account balance)
    # ------------------------------------------------------------------

    def _apply_funding(self, bar: Bar) -> None:
        """Record a funding payment if the cadence has elapsed."""
        id_str = str(bar.bar_type.instrument_id)
        last_ns = self._last_funding_ns.get(id_str, 0)
        if last_ns == 0:
            self._last_funding_ns[id_str] = bar.ts_event
            return

        elapsed_ns = bar.ts_event - last_ns
        if elapsed_ns < self._cfg.funding_period_nanos:
            return

        # Compute funding: rate × position_value.
        # We use a placeholder 0.01% rate per 8h (typical for BTC).
        # In production, look up the actual funding rate from data.
        funding_rate = 0.0001  # 0.01% per period
        net_qty = self._net_position(bar.bar_type.instrument_id)
        if abs(net_qty) < 1e-9:
            self._last_funding_ns[id_str] = bar.ts_event
            return

        close = float(bar.close)
        position_value = abs(net_qty) * close
        # Long pays funding rate; short receives it
        funding_pnl = -funding_rate * position_value * (1.0 if net_qty > 0 else -1.0)

        self._funding_pnl += funding_pnl
        self._funding_events.append(
            {
                "ts_ns": bar.ts_event,
                "instrument": id_str,
                "net_qty": net_qty,
                "close": close,
                "funding_rate": funding_rate,
                "funding_pnl": funding_pnl,
            }
        )
        self._last_funding_ns[id_str] = bar.ts_event
        logger.debug(
            "funding_applied",
            instrument=id_str,
            funding_pnl=round(funding_pnl, 4),
            ts_ns=bar.ts_event,
        )

    # ------------------------------------------------------------------
    # Order events
    # ------------------------------------------------------------------

    def on_order_filled(self, event: OrderFilled) -> None:
        self._fills.append(
            {
                "ts_ns": event.ts_event,
                "instrument": str(event.instrument_id),
                "side": event.order_side.name,
                "qty": float(event.last_qty),
                "price": float(event.last_px),
                "commission_currency": str(event.commission.currency),
                "commission": float(event.commission),
                "trade_id": str(event.trade_id),
                "order_id": str(event.client_order_id),
            }
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _net_position(self, instr_id: InstrumentId) -> float:
        positions = self.cache.positions(instrument_id=instr_id)
        if not positions:
            return 0.0
        return sum(float(p.quantity) if p.is_long else -float(p.quantity) for p in positions)

    def _current_equity(self, venue_str: str) -> float:
        from nautilus_trader.model.currencies import USDT
        from nautilus_trader.model.identifiers import Venue

        account = self.portfolio.account(Venue(venue_str))
        if account is None:
            return 0.0
        balance = account.balance(USDT)
        return float(balance.total) if balance else 0.0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _write_fill_log(self) -> None:
        if not self._cfg.log_dir or not self._fills:
            return
        from pathlib import Path

        log_dir = Path(self._cfg.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        fills_df = pd.DataFrame(self._fills)
        fills_df.to_parquet(log_dir / "fills.parquet", index=False)

        if self._funding_events:
            funding_df = pd.DataFrame(self._funding_events)
            funding_df.to_parquet(log_dir / "funding.parquet", index=False)

        logger.info("fill_log_written", path=str(log_dir), n_fills=len(self._fills))

    # ------------------------------------------------------------------
    # Public read-only properties (used by BacktestEngine for reporting)
    # ------------------------------------------------------------------

    @property
    def fills(self) -> list[dict[str, Any]]:
        return self._fills

    @property
    def funding_pnl(self) -> float:
        return self._funding_pnl

    @property
    def funding_events(self) -> list[dict[str, Any]]:
        return self._funding_events

    @property
    def kill_triggered(self) -> bool:
        return self._kill_triggered
