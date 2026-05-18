"""Live trading healthcheck HTTP server.

GET /healthz returns 200 OK only when all conditions hold:
  - Last bar received < 60 s ago
  - Last successful exchange ping < 30 s ago
  - Kill switch is not tripped
  - Postgres can execute a trivial SELECT
  - Redis responds to PING

Any failed condition returns 503 with a JSON body listing which checks failed.

Usage::

    state = HealthState(postgres_dsn=settings.postgres_dsn, redis_url=settings.redis_url)
    server = HealthCheckServer(state, port=8080)
    server.start()   # non-blocking (daemon thread)
    ...
    server.stop()
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from tessera.risk.kill_switch import KillSwitch

logger = structlog.get_logger(__name__)

_BAR_AGE_LIMIT = 60.0  # seconds
_PING_AGE_LIMIT = 30.0  # seconds


class HealthState:
    """Shared mutable state read by the healthcheck handler.

    Thread-safe: all attributes are written atomically (single assignment,
    GIL-protected) or via explicit locks for compound updates.
    """

    def __init__(
        self,
        postgres_dsn: str,
        redis_url: str,
        kill_switch: KillSwitch | None = None,
    ) -> None:
        self.postgres_dsn = postgres_dsn
        self.redis_url = redis_url
        self.kill_switch = kill_switch
        # Monotonic timestamps; 0 = never set
        self.last_bar_ts: float = 0.0
        self.last_ping_ts: float = 0.0

    def record_bar(self) -> None:
        self.last_bar_ts = time.monotonic()

    def record_ping(self) -> None:
        self.last_ping_ts = time.monotonic()


class _Handler(BaseHTTPRequestHandler):
    state: HealthState  # injected via server attribute

    def do_GET(self) -> None:  # noqa: N802
        if self.path != "/healthz":
            self._send(404, {"error": "not found"})
            return

        failures: list[str] = []
        now = time.monotonic()

        # 1. Bar freshness
        bar_age = now - self.server.state.last_bar_ts  # type: ignore[attr-defined]
        if self.server.state.last_bar_ts == 0 or bar_age > _BAR_AGE_LIMIT:  # type: ignore[attr-defined]
            failures.append(f"last_bar_age={bar_age:.1f}s > {_BAR_AGE_LIMIT}s")

        # 2. Exchange ping freshness
        ping_age = now - self.server.state.last_ping_ts  # type: ignore[attr-defined]
        if self.server.state.last_ping_ts == 0 or ping_age > _PING_AGE_LIMIT:  # type: ignore[attr-defined]
            failures.append(f"last_ping_age={ping_age:.1f}s > {_PING_AGE_LIMIT}s")

        # 3. Kill switch
        ks = self.server.state.kill_switch  # type: ignore[attr-defined]
        if ks is not None and ks.is_active:
            reason = ks.trigger_reason
            detail = f"{reason[0].value}: {reason[1]}" if reason else "unknown"
            failures.append(f"kill_switch_active: {detail}")

        # 4. Postgres
        pg_err = _check_postgres(self.server.state.postgres_dsn)  # type: ignore[attr-defined]
        if pg_err:
            failures.append(f"postgres: {pg_err}")

        # 5. Redis
        redis_err = _check_redis(self.server.state.redis_url)  # type: ignore[attr-defined]
        if redis_err:
            failures.append(f"redis: {redis_err}")

        if failures:
            self._send(503, {"status": "degraded", "failures": failures})
        else:
            self._send(200, {"status": "ok"})

    def _send(self, code: int, body: dict[str, object]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt: str, *args: object) -> None:
        pass  # silence access logs; structlog handles observability


def _check_postgres(dsn: str) -> str:
    try:
        import psycopg2

        conn = psycopg2.connect(dsn, connect_timeout=3)
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        return ""
    except Exception as exc:
        return str(exc)


def _check_redis(url: str) -> str:
    try:
        import redis as redis_lib

        r = redis_lib.from_url(url, socket_connect_timeout=3, socket_timeout=3)  # type: ignore[no-untyped-call]
        r.ping()
        return ""
    except Exception as exc:
        return str(exc)


class HealthCheckServer:
    """Threaded HTTP server that exposes GET /healthz."""

    def __init__(self, state: HealthState, port: int = 8080) -> None:
        self._state = state
        self._port = port
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        server = ThreadingHTTPServer(("", self._port), _Handler)
        server.state = self._state  # type: ignore[attr-defined]
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("healthcheck_server_started", port=self._port)

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            logger.info("healthcheck_server_stopped")
