"""Kill switch — source of truth for the live trading halt mechanism.

Once engaged, the kill switch stays engaged until explicitly cleared by a
human operator. The live strategy MUST call `is_active` on every cycle and
abort without submitting any orders if it returns True.

Trigger conditions (all flatten + halt):
  DAILY_LOSS       — daily portfolio loss > 3%
  DRAWDOWN         — peak-to-trough drawdown > 8%
  DATA_GAP         — no data tick received for > 30 seconds
  ORDER_REJECT_RATE — order reject rate > 5% over a rolling 5-minute window
  POSITION_MISMATCH — internal and exchange positions disagree beyond tolerance
  MANUAL_SIGTERM   — process received SIGTERM

Thread-safe: `engage()` may be called from any thread concurrently.
"""

from __future__ import annotations

import enum
import signal
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import structlog
from prometheus_client import Counter, Gauge

logger = structlog.get_logger(__name__)

KS_ACTIVE = Gauge(
    "tessera_kill_switch_active",
    "1 if kill switch is engaged (all trading halted), 0 if clear",
)
KS_TRIGGERS = Counter(
    "tessera_kill_switch_triggers_total",
    "Kill switch activations by trigger type",
    ["trigger"],
)


class KSTrigger(enum.StrEnum):
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    DATA_GAP = "data_gap"
    ORDER_REJECT_RATE = "order_reject_rate"
    POSITION_MISMATCH = "position_mismatch"
    MANUAL_SIGTERM = "manual_sigterm"


@dataclass
class KillSwitchConfig:
    daily_loss_threshold: float = 0.03
    drawdown_threshold: float = 0.08
    data_gap_seconds: float = 30.0
    reject_rate_threshold: float = 0.05
    reject_rate_window_seconds: float = 300.0


class KillSwitch:
    """Source-of-truth halt for the live trading system.

    Instantiate once at process start and pass the same instance to the
    strategy, order router, reconcile loop, and data feed handler.

    Example::

        ks = KillSwitch(on_trigger=lambda t, d: flatten_all_positions())
        while True:
            if ks.is_active:
                break
            ks.record_data_tick()
            ks.check_data_gap()
            ks.check_daily_loss(current_equity)
            ks.check_drawdown(current_equity)
            run_strategy_cycle()
    """

    def __init__(
        self,
        config: KillSwitchConfig | None = None,
        on_trigger: Callable[[KSTrigger, str], None] | None = None,
    ) -> None:
        self._cfg = config or KillSwitchConfig()
        self._on_trigger = on_trigger
        self._lock = threading.Lock()
        self._active = False
        self._trigger_reason: tuple[KSTrigger, str] | None = None

        self._day_start_equity: float | None = None
        self._peak_equity: float | None = None
        self._last_data_ts: float = time.monotonic()
        self._order_events: list[tuple[float, bool]] = []  # (monotonic_ts, rejected)

        signal.signal(signal.SIGTERM, self._handle_sigterm)
        KS_ACTIVE.set(0)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def trigger_reason(self) -> tuple[KSTrigger, str] | None:
        return self._trigger_reason

    def engage(self, trigger: KSTrigger, detail: str = "") -> None:
        """Engage the kill switch. Idempotent; first caller wins."""
        with self._lock:
            if self._active:
                return
            self._active = True
            self._trigger_reason = (trigger, detail)

        KS_ACTIVE.set(1)
        KS_TRIGGERS.labels(trigger=trigger.value).inc()
        logger.critical("kill_switch_engaged", trigger=trigger.value, detail=detail)
        _sentry_alert(
            f"KILL SWITCH: {trigger.value} — {detail}",
            extra={"trigger": trigger.value, "detail": detail},
        )

        if self._on_trigger:
            try:
                self._on_trigger(trigger, detail)
            except Exception:
                logger.exception("kill_switch_on_trigger_error")

    def clear(self) -> None:
        """Clear the kill switch after a human operator review."""
        with self._lock:
            self._active = False
            self._trigger_reason = None
        KS_ACTIVE.set(0)
        logger.warning("kill_switch_cleared")

    # ------------------------------------------------------------------
    # Per-cycle checks
    # ------------------------------------------------------------------

    def check_daily_loss(
        self,
        current_equity: float,
        day_start_equity: float | None = None,
    ) -> None:
        if day_start_equity is not None:
            self._day_start_equity = day_start_equity
        if self._day_start_equity is None:
            self._day_start_equity = current_equity
            return
        daily_ret = (current_equity - self._day_start_equity) / self._day_start_equity
        if daily_ret < -self._cfg.daily_loss_threshold:
            self.engage(
                KSTrigger.DAILY_LOSS,
                f"daily loss {daily_ret:.2%} > {self._cfg.daily_loss_threshold:.2%}",
            )

    def check_drawdown(self, current_equity: float) -> None:
        if self._peak_equity is None or current_equity > self._peak_equity:
            self._peak_equity = current_equity
        drawdown = (current_equity - self._peak_equity) / self._peak_equity
        if drawdown < -self._cfg.drawdown_threshold:
            self.engage(
                KSTrigger.DRAWDOWN,
                f"drawdown {drawdown:.2%} > {self._cfg.drawdown_threshold:.2%}",
            )

    def record_data_tick(self) -> None:
        """Call whenever a live market data tick is received."""
        self._last_data_ts = time.monotonic()

    def check_data_gap(self) -> None:
        """Call periodically (e.g. once per bar) to detect stale data feeds."""
        gap = time.monotonic() - self._last_data_ts
        if gap > self._cfg.data_gap_seconds:
            self.engage(
                KSTrigger.DATA_GAP,
                f"data feed silent for {gap:.1f}s (threshold {self._cfg.data_gap_seconds:.1f}s)",
            )

    def record_order_event(self, rejected: bool) -> None:
        """Record an order submission outcome.

        Call this after every order is sent to the exchange. Set `rejected=True`
        if the exchange returned a rejection (not a fill / pending).
        """
        now = time.monotonic()
        cutoff = now - self._cfg.reject_rate_window_seconds
        self._order_events.append((now, rejected))
        self._order_events = [(t, r) for t, r in self._order_events if t >= cutoff]

        # Require a minimum sample before triggering to avoid false positives at startup.
        if len(self._order_events) < 10:
            return
        rate = sum(1 for _, r in self._order_events if r) / len(self._order_events)
        if rate > self._cfg.reject_rate_threshold:
            self.engage(
                KSTrigger.ORDER_REJECT_RATE,
                f"reject rate {rate:.1%} over last {self._cfg.reject_rate_window_seconds:.0f}s",
            )

    def check_position_reconcile(
        self,
        internal: dict[str, float],
        exchange: dict[str, float],
        tolerance: float = 0.01,
    ) -> None:
        """Engage if internal and exchange positions differ beyond `tolerance`.

        `tolerance` is a fraction of the larger of the two quantities.
        A symbol present on only one side with non-negligible size counts
        as a full mismatch.
        """
        for sym in set(internal) | set(exchange):
            internal_qty = internal.get(sym, 0.0)
            exchange_qty = exchange.get(sym, 0.0)
            ref = max(abs(internal_qty), abs(exchange_qty), 1e-9)
            if abs(internal_qty - exchange_qty) / ref > tolerance:
                self.engage(
                    KSTrigger.POSITION_MISMATCH,
                    f"{sym}: internal={internal_qty:.6f} exchange={exchange_qty:.6f}",
                )
                return

    # ------------------------------------------------------------------
    # Signal handler
    # ------------------------------------------------------------------

    def _handle_sigterm(self, signum: int, frame: object) -> None:
        self.engage(KSTrigger.MANUAL_SIGTERM, "received SIGTERM")


def _sentry_alert(message: str, extra: dict[str, Any] | None = None) -> None:
    try:
        import sentry_sdk

        with sentry_sdk.push_scope() as scope:
            scope.level = "critical"  # type: ignore[assignment]
            if extra:
                for k, v in extra.items():
                    scope.set_extra(k, v)
            sentry_sdk.capture_message(message, level="critical")  # type: ignore[arg-type]
    except ImportError:
        logger.debug("sentry_not_available", message=message)
