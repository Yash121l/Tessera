"""Circuit breaker state machine with Postgres-backed persistence.

States (in escalation order):
  OK             → normal operation
  SCALE_DOWN     → halve all position sizes (triggered at -5% MTD)
  HALT_48H       → suspend trading for 48 hours (-10% MTD)
  HALT_INDEFINITE → suspend until manual review (-15% peak-to-trough)

State survives process restarts because it is persisted to Postgres.
Pass dsn=None to use an in-memory backend (useful for tests/backtests).
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import structlog
from prometheus_client import Counter, Gauge

logger = structlog.get_logger(__name__)

CB_STATE = Gauge(
    "tessera_circuit_breaker_state",
    "Current circuit breaker state (0=OK, 1=SCALE_DOWN, 2=HALT_48H, 3=HALT_INDEFINITE)",
)
CB_TRANSITIONS = Counter(
    "tessera_circuit_breaker_transitions_total",
    "Circuit breaker state transitions",
    ["from_state", "to_state"],
)

_MTD_SCALE_DOWN = -0.05
_MTD_HALT_48H = -0.10
_DRAWDOWN_HALT = -0.15
_HALT_48H_SECONDS = 48 * 3600

_STATE_INT = {"OK": 0, "SCALE_DOWN": 1, "HALT_48H": 2, "HALT_INDEFINITE": 3}

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS tessera_circuit_breaker (
    id          SERIAL PRIMARY KEY,
    state       TEXT         NOT NULL DEFAULT 'OK',
    entered_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    halt_until  TIMESTAMPTZ,
    peak_equity DOUBLE PRECISION,
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
INSERT INTO tessera_circuit_breaker (state)
    SELECT 'OK' WHERE NOT EXISTS (SELECT 1 FROM tessera_circuit_breaker);
"""


class CBState(enum.StrEnum):
    OK = "OK"
    SCALE_DOWN = "SCALE_DOWN"
    HALT_48H = "HALT_48H"
    HALT_INDEFINITE = "HALT_INDEFINITE"


@dataclass
class _CBRow:
    state: str = "OK"
    halt_until: float | None = None
    peak_equity: float | None = None


class _Storage(Protocol):
    def load(self) -> _CBRow: ...
    def save(self, state: str, halt_until: float | None, peak_equity: float | None) -> None: ...


class _InMemoryStorage:
    def __init__(self) -> None:
        self._row = _CBRow()

    def load(self) -> _CBRow:
        return _CBRow(
            state=self._row.state,
            halt_until=self._row.halt_until,
            peak_equity=self._row.peak_equity,
        )

    def save(self, state: str, halt_until: float | None, peak_equity: float | None) -> None:
        self._row = _CBRow(state=state, halt_until=halt_until, peak_equity=peak_equity)


class _PostgresStorage:
    def __init__(self, dsn: str) -> None:
        import psycopg2
        import psycopg2.extras

        self._psycopg2 = psycopg2
        self._extras = psycopg2.extras
        self._conn: Any = psycopg2.connect(dsn)
        self._conn.autocommit = True
        with self._conn.cursor() as cur:
            cur.execute(_CREATE_TABLE)

    def load(self) -> _CBRow:
        with self._conn.cursor(cursor_factory=self._extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT state, halt_until, peak_equity "
                "FROM tessera_circuit_breaker ORDER BY id LIMIT 1"
            )
            row = cur.fetchone()
        if row is None:
            return _CBRow()
        halt_until: float | None = None
        if row["halt_until"] is not None:
            halt_until = row["halt_until"].timestamp()
        return _CBRow(
            state=row["state"],
            halt_until=halt_until,
            peak_equity=row["peak_equity"],
        )

    def save(self, state: str, halt_until: float | None, peak_equity: float | None) -> None:
        halt_until_dt: datetime | None = None
        if halt_until is not None:
            halt_until_dt = datetime.fromtimestamp(halt_until, tz=UTC)
        with self._conn.cursor() as cur:
            cur.execute(
                "UPDATE tessera_circuit_breaker "
                "SET state=%s, halt_until=%s, peak_equity=%s, updated_at=NOW()",
                (state, halt_until_dt, peak_equity),
            )

    def close(self) -> None:
        if self._conn:
            self._conn.close()


class CircuitBreaker:
    """Postgres-backed circuit breaker for live trading.

    Usage:
        cb = CircuitBreaker(dsn="postgresql://...")
        # or for tests / backtests:
        cb = CircuitBreaker()

        state = cb.update(current_equity=100_000, mtd_start_equity=105_000)
        multiplier = cb.size_multiplier  # 1.0, 0.5, or 0.0
    """

    def __init__(self, dsn: str | None = None) -> None:
        self._storage: _Storage = _PostgresStorage(dsn) if dsn else _InMemoryStorage()
        row = self._storage.load()
        self._state = CBState(row.state)
        self._halt_until = row.halt_until
        self._peak_equity = row.peak_equity
        self._mtd_start_equity: float | None = None
        CB_STATE.set(_STATE_INT[self._state.value])
        logger.info("circuit_breaker_loaded", state=self._state.value)

    # ------------------------------------------------------------------
    # Public read API
    # ------------------------------------------------------------------

    @property
    def state(self) -> CBState:
        return self._state

    @property
    def is_halted(self) -> bool:
        if self._state == CBState.HALT_INDEFINITE:
            return True
        if self._state == CBState.HALT_48H:
            if self._halt_until is not None and time.time() < self._halt_until:
                return True
            # Auto-recover after the 48-hour window expires.
            self._transition(CBState.OK, "48h halt expired")
            self._halt_until = None
        return False

    @property
    def size_multiplier(self) -> float:
        """1.0 (OK), 0.5 (SCALE_DOWN), 0.0 (any HALT)."""
        if self.is_halted:
            return 0.0
        if self._state == CBState.SCALE_DOWN:
            return 0.5
        return 1.0

    # ------------------------------------------------------------------
    # Main update — call once per bar
    # ------------------------------------------------------------------

    def update(
        self,
        current_equity: float,
        mtd_start_equity: float | None = None,
    ) -> CBState:
        """Evaluate equity against thresholds and advance state if needed.

        Args:
            current_equity: Current portfolio equity in USD.
            mtd_start_equity: First equity value of the current calendar month.
                If None on the first call, the current equity is used as the
                MTD baseline.

        Returns:
            Current CBState after evaluation.
        """
        if self.is_halted:
            return self._state

        if self._peak_equity is None or current_equity > self._peak_equity:
            self._peak_equity = current_equity
            self._persist()

        if mtd_start_equity is not None:
            self._mtd_start_equity = mtd_start_equity
        elif self._mtd_start_equity is None:
            self._mtd_start_equity = current_equity

        mtd_ret = (current_equity - self._mtd_start_equity) / self._mtd_start_equity
        drawdown = (current_equity - self._peak_equity) / self._peak_equity

        if drawdown <= _DRAWDOWN_HALT and self._state != CBState.HALT_INDEFINITE:
            self._transition(CBState.HALT_INDEFINITE, f"peak-to-trough {drawdown:.1%}")
        elif mtd_ret <= _MTD_HALT_48H and self._state not in (
            CBState.HALT_48H,
            CBState.HALT_INDEFINITE,
        ):
            self._halt_until = time.time() + _HALT_48H_SECONDS
            self._transition(CBState.HALT_48H, f"MTD {mtd_ret:.1%}")
        elif mtd_ret <= _MTD_SCALE_DOWN and self._state == CBState.OK:
            self._transition(CBState.SCALE_DOWN, f"MTD {mtd_ret:.1%}")

        return self._state

    def manual_resume(self) -> None:
        """Resume from HALT_INDEFINITE after a human review sign-off."""
        logger.warning("circuit_breaker_manual_resume", from_state=self._state.value)
        self._transition(CBState.OK, "manual_resume")
        self._halt_until = None
        self._mtd_start_equity = None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _transition(self, new_state: CBState, reason: str) -> None:
        old = self._state
        self._state = new_state
        CB_STATE.set(_STATE_INT[new_state.value])
        CB_TRANSITIONS.labels(from_state=old.value, to_state=new_state.value).inc()
        self._persist()
        logger.warning(
            "circuit_breaker_transition",
            from_state=old.value,
            to_state=new_state.value,
            reason=reason,
        )
        _sentry_alert(
            f"CircuitBreaker: {old.value} → {new_state.value} ({reason})",
            level="warning" if new_state == CBState.SCALE_DOWN else "error",
            extra={"from_state": old.value, "to_state": new_state.value, "reason": reason},
        )

    def _persist(self) -> None:
        self._storage.save(self._state.value, self._halt_until, self._peak_equity)


def _sentry_alert(message: str, level: str = "error", extra: dict[str, Any] | None = None) -> None:
    try:
        import sentry_sdk

        with sentry_sdk.push_scope() as scope:
            scope.level = level  # type: ignore[assignment]
            if extra:
                for k, v in extra.items():
                    scope.set_extra(k, v)
            sentry_sdk.capture_message(message, level=level)  # type: ignore[arg-type]
    except ImportError:
        logger.debug("sentry_not_available", message=message)
