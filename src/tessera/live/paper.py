"""PaperRunner — Nautilus Live node wired to Binance Testnet + Bybit Demo.

Lifecycle::

    runner = PaperRunner.from_config(live_cfg, settings)
    runner.run()   # blocks until clean stop or restart limit exceeded

Crash recovery:
    On any unhandled exception, PaperRunner restarts the node after
    exponential backoff (1 → 2 → 4 → 8 → 16 → 60 s).  At most 5 restarts
    are allowed per rolling hour window; beyond that the process aborts.

Position reconciliation:
    Every 60 s a background coroutine fetches open positions from each
    exchange via CCXT and diffs them against the Nautilus cache.  A
    KillSwitch.POSITION_MISMATCH is engaged if drift exceeds 1 %.

Heartbeat:
    Every 5 s the tessera_heartbeat_ts Prometheus gauge is stamped with
    `time.time()`, signalling to Grafana that the main loop is alive.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
from collections import deque
from pathlib import Path
from typing import Any

import psycopg2
import structlog

from tessera.config import LiveConfig, TesseraSettings, generate_run_id
from tessera.metrics import (
    EXCHANGE_PING_LATENCY,
    HEARTBEAT_TS,
    RECONCILE_MISMATCH,
    RECONCILE_OK,
    RUNNER_RESTARTS,
)
from tessera.risk.kill_switch import KillSwitch, KillSwitchConfig

from .healthcheck import HealthCheckServer, HealthState

logger = structlog.get_logger(__name__)

_PID_FILE = Path("paper/tessera.pid")
_MAX_RESTARTS_PER_HOUR = 5
_HEARTBEAT_INTERVAL = 5.0  # seconds
_RECONCILE_INTERVAL = 60.0  # seconds
_PING_INTERVAL = 20.0  # seconds; must beat healthcheck threshold of 30 s


class PaperRunner:
    """Orchestrates a Nautilus TradingNode for paper trading.

    Attributes:
        run_id: Unique ID for this runner process (used in Postgres logs).
    """

    def __init__(
        self,
        config: LiveConfig,
        settings: TesseraSettings,
        run_id: str,
        healthcheck_port: int = 8080,
    ) -> None:
        self._config = config
        self._settings = settings
        self.run_id = run_id

        self._kill_switch = KillSwitch(
            config=KillSwitchConfig(
                daily_loss_threshold=config.kill_switch.max_drawdown_pct / 100.0,
                drawdown_threshold=config.kill_switch.max_drawdown_pct / 100.0 * 2,
                data_gap_seconds=60.0,
            ),
            on_trigger=self._on_kill_switch,
        )
        self._health_state = HealthState(
            postgres_dsn=settings.postgres_dsn,
            redis_url=settings.redis_url,
            kill_switch=self._kill_switch,
        )
        self._healthcheck = HealthCheckServer(self._health_state, port=healthcheck_port)
        self._restart_ts: deque[float] = deque()
        self._node: Any = None

    @classmethod
    def from_config(
        cls,
        config: LiveConfig,
        settings: TesseraSettings,
        healthcheck_port: int = 8080,
    ) -> PaperRunner:
        run_id = generate_run_id(settings.random_seed)
        return cls(config, settings, run_id, healthcheck_port)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Block until a clean stop or the restart limit is exhausted."""
        _write_pid_file()
        self._healthcheck.start()
        try:
            asyncio.run(self._main())
        finally:
            self._healthcheck.stop()
            _remove_pid_file()

    # ------------------------------------------------------------------
    # Async core
    # ------------------------------------------------------------------

    async def _main(self) -> None:
        backoff = 1.0
        self._pg_write_state("running")

        while True:
            self._gc_restart_ts()
            if len(self._restart_ts) >= _MAX_RESTARTS_PER_HOUR:
                logger.critical(
                    "restart_limit_exceeded",
                    limit=_MAX_RESTARTS_PER_HOUR,
                    window_hours=1,
                )
                self._pg_write_state("crashed", crash_reason="restart limit exceeded")
                raise RuntimeError("PaperRunner: max restarts/hr exceeded")

            try:
                logger.info("runner_starting", run_id=self.run_id)
                await self._run_node_once()
                logger.info("runner_clean_stop", run_id=self.run_id)
                self._pg_write_state("stopped")
                return
            except asyncio.CancelledError:
                self._pg_write_state("stopped")
                return
            except Exception as exc:
                RUNNER_RESTARTS.inc()
                self._restart_ts.append(time.monotonic())
                logger.error(
                    "runner_crashed",
                    error=str(exc),
                    restart_count=len(self._restart_ts),
                    next_retry_s=backoff,
                )
                self._pg_write_state("crashed", crash_reason=str(exc))
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                self._pg_write_state("running")

    async def _run_node_once(self) -> None:
        """Start the Nautilus node and side-loops; block until node exits."""
        node = self._build_trading_node()
        self._node = node

        async def heartbeat() -> None:
            while True:
                HEARTBEAT_TS.set(time.time())
                await asyncio.sleep(_HEARTBEAT_INTERVAL)

        async def reconcile() -> None:
            await asyncio.sleep(_RECONCILE_INTERVAL)  # warm-up
            while True:
                try:
                    await self._reconcile(node)
                except Exception as exc:
                    logger.warning("reconcile_error", error=str(exc))
                await asyncio.sleep(_RECONCILE_INTERVAL)

        async def ping() -> None:
            while True:
                await self._ping_exchanges()
                await asyncio.sleep(_PING_INTERVAL)

        hb = asyncio.create_task(heartbeat())
        rec = asyncio.create_task(reconcile())
        pg = asyncio.create_task(ping())

        try:
            await node.run_async()
        finally:
            for task in (hb, rec, pg):
                task.cancel()
            await asyncio.gather(hb, rec, pg, return_exceptions=True)
            self._node = None

    # ------------------------------------------------------------------
    # Node construction
    # ------------------------------------------------------------------

    def _build_trading_node(self) -> Any:
        try:
            from nautilus_trader.config import (
                LiveDataEngineConfig,
                LiveExecutionEngineConfig,
                LiveRiskEngineConfig,
                LoggingConfig,
                TradingNodeConfig,
            )
            from nautilus_trader.live.node import TradingNode
        except ImportError as exc:
            raise RuntimeError("nautilus-trader not installed (add backtest extra)") from exc

        cfg = self._config.live
        binance_key = (
            self._settings.binance_api_key.get_secret_value()
            if self._settings.binance_api_key
            else ""
        )
        binance_secret = (
            self._settings.binance_api_secret.get_secret_value()
            if self._settings.binance_api_secret
            else ""
        )
        bybit_key = (
            self._settings.bybit_api_key.get_secret_value() if self._settings.bybit_api_key else ""
        )
        bybit_secret = (
            self._settings.bybit_api_secret.get_secret_value()
            if self._settings.bybit_api_secret
            else ""
        )

        data_clients: dict[str, Any] = {}
        exec_clients: dict[str, Any] = {}
        factories_data: dict[str, Any] = {}
        factories_exec: dict[str, Any] = {}

        if "binance" in cfg.exchange.lower():
            try:
                from nautilus_trader.adapters.binance.factories import (
                    BinanceLiveDataClientFactory,
                    BinanceLiveExecutionClientFactory,
                )
                from nautilus_trader.adapters.binance.futures.config import (
                    BinanceFuturesDataClientConfig,
                    BinanceFuturesExecutionClientConfig,
                )

                data_clients["BINANCE"] = BinanceFuturesDataClientConfig(
                    api_key=binance_key,
                    api_secret=binance_secret,
                    is_testnet=True,
                )
                exec_clients["BINANCE"] = BinanceFuturesExecutionClientConfig(
                    api_key=binance_key,
                    api_secret=binance_secret,
                    is_testnet=True,
                )
                factories_data["BINANCE"] = BinanceLiveDataClientFactory
                factories_exec["BINANCE"] = BinanceLiveExecutionClientFactory
                logger.info("binance_testnet_configured")
            except ImportError:
                logger.warning("binance_adapter_not_available_skipping")

        if "bybit" in cfg.exchange.lower():
            try:
                from nautilus_trader.adapters.bybit.config import (
                    BybitDataClientConfig,
                    BybitExecutionClientConfig,
                )
                from nautilus_trader.adapters.bybit.factories import (
                    BybitLiveDataClientFactory,
                    BybitLiveExecutionClientFactory,
                )

                data_clients["BYBIT"] = BybitDataClientConfig(
                    api_key=bybit_key,
                    api_secret=bybit_secret,
                    is_testnet=True,
                )
                exec_clients["BYBIT"] = BybitExecutionClientConfig(
                    api_key=bybit_key,
                    api_secret=bybit_secret,
                    is_testnet=True,
                )
                factories_data["BYBIT"] = BybitLiveDataClientFactory
                factories_exec["BYBIT"] = BybitLiveExecutionClientFactory
                logger.info("bybit_demo_configured")
            except ImportError:
                logger.warning("bybit_adapter_not_available_skipping")

        node_config = TradingNodeConfig(
            trader_id=f"TESSERA-{self.run_id.upper()}",
            log_level=self._settings.log_level,
            data_engine=LiveDataEngineConfig(debug=False),
            risk_engine=LiveRiskEngineConfig(),
            exec_engine=LiveExecutionEngineConfig(),
            logging=LoggingConfig(log_level=self._settings.log_level),
            data_clients=data_clients,
            exec_clients=exec_clients,
        )

        node = TradingNode(config=node_config)

        for venue, factory in factories_data.items():
            node.add_data_client_factory(venue, factory)
        for venue, factory in factories_exec.items():
            node.add_exec_client_factory(venue, factory)

        strategy = self._build_strategy(cfg.symbols)
        node.add_strategy(strategy)
        node.build()
        return node

    def _build_strategy(self, symbols: list[str]) -> Any:
        from tessera.strategies.ml_directional import MLDirectionalConfig, MLDirectionalStrategy

        # Map CCXT symbol format to Nautilus instrument ID format
        instrument_ids = tuple(
            f"{sym.split('/')[0]}-USDT-PERP.{self._config.live.exchange.upper()}" for sym in symbols
        )
        strategy_config = MLDirectionalConfig(
            instrument_ids=instrument_ids,
            bar_type_suffix="1-MINUTE-LAST-EXTERNAL",
            max_drawdown_pct=self._config.kill_switch.max_drawdown_pct,
            risk_db_dsn=self._settings.postgres_dsn,
        )
        return MLDirectionalStrategy(config=strategy_config)

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    async def _reconcile(self, node: Any) -> None:
        loop = asyncio.get_event_loop()
        exchange_positions = await loop.run_in_executor(None, self._fetch_exchange_positions)
        internal_positions = self._get_internal_positions(node)

        all_symbols = set(internal_positions) | set(exchange_positions)
        mismatch = False
        for sym in all_symbols:
            internal_qty = internal_positions.get(sym, 0.0)
            exchange_qty = exchange_positions.get(sym, 0.0)
            ref = max(abs(internal_qty), abs(exchange_qty), 1e-9)
            drift = abs(internal_qty - exchange_qty) / ref
            if drift > 0.01:
                logger.warning(
                    "position_mismatch",
                    symbol=sym,
                    internal=internal_qty,
                    exchange=exchange_qty,
                    drift_pct=round(drift * 100, 2),
                )
                RECONCILE_MISMATCH.labels(symbol=sym).inc()
                mismatch = True
                self._kill_switch.check_position_reconcile(internal_positions, exchange_positions)

        venue = self._config.live.exchange
        RECONCILE_OK.labels(venue=venue).set(0.0 if mismatch else 1.0)
        logger.debug("reconcile_ok", venue=venue, mismatch=mismatch)

    def _fetch_exchange_positions(self) -> dict[str, float]:
        """Fetch open positions from exchange via CCXT (blocking, run in executor)."""
        try:
            import ccxt

            exchange_id = self._config.live.exchange.lower()
            api_key = (
                self._settings.binance_api_key.get_secret_value()
                if exchange_id == "binance" and self._settings.binance_api_key
                else (
                    self._settings.bybit_api_key.get_secret_value()
                    if exchange_id == "bybit" and self._settings.bybit_api_key
                    else ""
                )
            )
            api_secret = (
                self._settings.binance_api_secret.get_secret_value()
                if exchange_id == "binance" and self._settings.binance_api_secret
                else (
                    self._settings.bybit_api_secret.get_secret_value()
                    if exchange_id == "bybit" and self._settings.bybit_api_secret
                    else ""
                )
            )

            exchange_cls = getattr(ccxt, exchange_id)
            exchange = exchange_cls(
                {
                    "apiKey": api_key,
                    "secret": api_secret,
                    "options": {"defaultType": "future"},
                    "sandbox": True,  # testnet / demo mode
                }
            )
            raw = exchange.fetch_positions()
            return {
                p["symbol"]: float(p.get("contracts", 0) or 0)
                * (1.0 if p.get("side") == "long" else -1.0)
                for p in raw
                if float(p.get("contracts", 0) or 0) != 0
            }
        except Exception as exc:
            logger.warning("fetch_exchange_positions_failed", error=str(exc))
            return {}

    @staticmethod
    def _get_internal_positions(node: Any) -> dict[str, float]:
        """Read net positions from Nautilus's internal cache."""
        result: dict[str, float] = {}
        try:
            for pos in node.cache.positions_open():
                sym = str(pos.instrument_id)
                qty = float(pos.quantity)
                result[sym] = result.get(sym, 0.0) + (qty if pos.is_long else -qty)
        except Exception:
            pass
        return result

    # ------------------------------------------------------------------
    # Exchange ping
    # ------------------------------------------------------------------

    async def _ping_exchanges(self) -> None:
        loop = asyncio.get_event_loop()
        for exchange_id in self._settings.exchanges:
            try:
                t0 = time.monotonic()
                await loop.run_in_executor(None, self._ping_one, exchange_id)
                latency = time.monotonic() - t0
                EXCHANGE_PING_LATENCY.labels(exchange=exchange_id).set(latency)
                self._health_state.record_ping()
            except Exception as exc:
                logger.warning("exchange_ping_failed", exchange=exchange_id, error=str(exc))

    def _ping_one(self, exchange_id: str) -> None:
        import ccxt

        exchange_cls = getattr(ccxt, exchange_id, None)
        if exchange_cls is None:
            return
        exchange = exchange_cls({"options": {"defaultType": "future"}, "sandbox": True})
        exchange.load_markets()

    # ------------------------------------------------------------------
    # Kill switch callback
    # ------------------------------------------------------------------

    def _on_kill_switch(self, trigger: Any, detail: str) -> None:
        logger.critical("kill_switch_flattening", trigger=str(trigger), detail=detail)
        # Actual flatten is handled inside MLDirectionalStrategy._on_kill_switch_trigger

    # ------------------------------------------------------------------
    # Postgres state logging
    # ------------------------------------------------------------------

    def _pg_write_state(self, status: str, crash_reason: str | None = None) -> None:
        try:
            conn = psycopg2.connect(self._settings.postgres_dsn, connect_timeout=3)
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO paper_runner_state (run_id, status, pid, crash_reason)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE
                        SET status = EXCLUDED.status,
                            pid = EXCLUDED.pid,
                            crash_reason = EXCLUDED.crash_reason,
                            updated_at = NOW()
                    """,
                    (self.run_id, status, os.getpid(), crash_reason),
                )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.debug("pg_state_write_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Restart window management
    # ------------------------------------------------------------------

    def _gc_restart_ts(self) -> None:
        cutoff = time.monotonic() - 3600.0
        while self._restart_ts and self._restart_ts[0] < cutoff:
            self._restart_ts.popleft()


# ------------------------------------------------------------------
# PID file helpers
# ------------------------------------------------------------------


def _write_pid_file() -> None:
    _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    _PID_FILE.write_text(str(os.getpid()))


def _remove_pid_file() -> None:
    with contextlib.suppress(FileNotFoundError):
        _PID_FILE.unlink()


def read_pid() -> int | None:
    """Read the runner PID from the PID file, or None if not running."""
    if not _PID_FILE.exists():
        return None
    try:
        return int(_PID_FILE.read_text().strip())
    except ValueError:
        return None


def send_stop_signal() -> bool:
    """Send SIGTERM to the running paper trader.

    Returns True if signal was sent, False if no runner is running.
    """
    pid = read_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        logger.info("sigterm_sent", pid=pid)
        return True
    except ProcessLookupError:
        _remove_pid_file()
        return False
