"""Unit tests for the live healthcheck module.

No exchange or Nautilus required — tests use mocks and the actual HTTP server.
"""

from __future__ import annotations

import json
import time
import urllib.request
from unittest.mock import MagicMock, patch

import pytest

from tessera.live.healthcheck import HealthCheckServer, HealthState

# ---------------------------------------------------------------------------
# HealthState
# ---------------------------------------------------------------------------


def test_health_state_record_bar():
    state = HealthState(postgres_dsn="", redis_url="")
    assert state.last_bar_ts == 0.0
    state.record_bar()
    assert state.last_bar_ts > 0.0


def test_health_state_record_ping():
    state = HealthState(postgres_dsn="", redis_url="")
    state.record_ping()
    assert state.last_ping_ts > 0.0


# ---------------------------------------------------------------------------
# Healthcheck HTTP server (integration-lite: real server, mocked checks)
# ---------------------------------------------------------------------------


@pytest.fixture
def healthy_state():
    state = HealthState(postgres_dsn="dummy", redis_url="dummy")
    state.record_bar()
    state.record_ping()
    return state


def test_healthcheck_returns_200_when_healthy(healthy_state):
    server = HealthCheckServer(healthy_state, port=19081)

    with (
        patch("tessera.live.healthcheck._check_postgres", return_value=""),
        patch("tessera.live.healthcheck._check_redis", return_value=""),
    ):
        server.start()
        try:
            with urllib.request.urlopen("http://localhost:19081/healthz", timeout=3) as resp:
                assert resp.status == 200
                body = json.loads(resp.read())
                assert body["status"] == "ok"
        finally:
            server.stop()


def test_healthcheck_returns_503_when_bar_stale(healthy_state):
    healthy_state.last_bar_ts = time.monotonic() - 120  # 2 minutes stale
    server = HealthCheckServer(healthy_state, port=19082)

    with (
        patch("tessera.live.healthcheck._check_postgres", return_value=""),
        patch("tessera.live.healthcheck._check_redis", return_value=""),
    ):
        server.start()
        try:
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen("http://localhost:19082/healthz", timeout=3)
            assert exc_info.value.code == 503
            body = json.loads(exc_info.value.read())
            assert body["status"] == "degraded"
            assert any("last_bar_age" in f for f in body["failures"])
        finally:
            server.stop()


def test_healthcheck_returns_503_when_kill_switch_active(healthy_state):
    ks = MagicMock()
    ks.is_active = True
    ks.trigger_reason = None
    healthy_state.kill_switch = ks

    server = HealthCheckServer(healthy_state, port=19083)

    with (
        patch("tessera.live.healthcheck._check_postgres", return_value=""),
        patch("tessera.live.healthcheck._check_redis", return_value=""),
    ):
        server.start()
        try:
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen("http://localhost:19083/healthz", timeout=3)
            assert exc_info.value.code == 503
            body = json.loads(exc_info.value.read())
            assert any("kill_switch" in f for f in body["failures"])
        finally:
            server.stop()


def test_healthcheck_returns_503_when_postgres_down(healthy_state):
    server = HealthCheckServer(healthy_state, port=19084)

    with (
        patch("tessera.live.healthcheck._check_postgres", return_value="connection refused"),
        patch("tessera.live.healthcheck._check_redis", return_value=""),
    ):
        server.start()
        try:
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen("http://localhost:19084/healthz", timeout=3)
            assert exc_info.value.code == 503
            body = json.loads(exc_info.value.read())
            assert any("postgres" in f for f in body["failures"])
        finally:
            server.stop()


def test_healthcheck_404_on_unknown_path(healthy_state):
    server = HealthCheckServer(healthy_state, port=19085)

    with (
        patch("tessera.live.healthcheck._check_postgres", return_value=""),
        patch("tessera.live.healthcheck._check_redis", return_value=""),
    ):
        server.start()
        try:
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen("http://localhost:19085/metrics", timeout=3)
            assert exc_info.value.code == 404
        finally:
            server.stop()


# ---------------------------------------------------------------------------
# Restart token-bucket logic (unit test, no async needed)
# ---------------------------------------------------------------------------


def test_paper_runner_restart_gc():
    """_gc_restart_ts should expire timestamps older than 1 hour."""
    from collections import deque

    from tessera.config import LiveConfig, TesseraSettings
    from tessera.live.paper import PaperRunner

    settings = TesseraSettings()
    cfg = LiveConfig()
    runner = PaperRunner(cfg, settings, run_id="test-unit")

    old_ts = time.monotonic() - 3601  # just over 1 hour ago
    runner._restart_ts = deque([old_ts, old_ts, time.monotonic()])
    runner._gc_restart_ts()
    assert len(runner._restart_ts) == 1  # only the recent one remains


def test_paper_runner_pid_helpers(tmp_path, monkeypatch):
    import tessera.live.paper as paper_mod

    pid_file = tmp_path / "tessera.pid"
    monkeypatch.setattr(paper_mod, "_PID_FILE", pid_file)

    assert paper_mod.read_pid() is None

    paper_mod._write_pid_file()
    assert pid_file.exists()
    assert paper_mod.read_pid() == __import__("os").getpid()

    paper_mod._remove_pid_file()
    assert not pid_file.exists()
    assert paper_mod.read_pid() is None
