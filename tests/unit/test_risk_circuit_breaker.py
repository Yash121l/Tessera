"""Unit tests for tessera.risk.circuit_breaker.

Uses in-memory backend (dsn=None) — no Postgres required.
All drawdown/MTD thresholds match the production constants.
"""

from __future__ import annotations

import time

import pytest

from tessera.risk.circuit_breaker import CBState, CircuitBreaker


@pytest.fixture
def cb() -> CircuitBreaker:
    return CircuitBreaker()  # in-memory


class TestInitialState:
    def test_starts_ok(self, cb: CircuitBreaker) -> None:
        assert cb.state == CBState.OK

    def test_not_halted_initially(self, cb: CircuitBreaker) -> None:
        assert not cb.is_halted

    def test_size_multiplier_one(self, cb: CircuitBreaker) -> None:
        assert cb.size_multiplier == pytest.approx(1.0)


class TestScaleDownTransition:
    def test_triggers_at_minus_5pct_mtd(self, cb: CircuitBreaker) -> None:
        # Month started at 100k; now 94.9k → -5.1% MTD → SCALE_DOWN
        cb.update(current_equity=100_000.0, mtd_start_equity=100_000.0)
        cb.update(current_equity=94_900.0, mtd_start_equity=100_000.0)
        assert cb.state == CBState.SCALE_DOWN

    def test_size_multiplier_halved(self, cb: CircuitBreaker) -> None:
        cb.update(100_000.0, 100_000.0)
        cb.update(94_900.0, 100_000.0)
        assert cb.size_multiplier == pytest.approx(0.5)

    def test_does_not_trigger_above_threshold(self, cb: CircuitBreaker) -> None:
        cb.update(100_000.0, 100_000.0)
        cb.update(96_000.0, 100_000.0)  # -4% MTD — below the -5% threshold
        assert cb.state == CBState.OK


class TestHalt48HTransition:
    def test_triggers_at_minus_10pct_mtd(self, cb: CircuitBreaker) -> None:
        cb.update(100_000.0, 100_000.0)
        cb.update(89_000.0, 100_000.0)  # -11% MTD
        assert cb.state == CBState.HALT_48H

    def test_is_halted_true(self, cb: CircuitBreaker) -> None:
        cb.update(100_000.0, 100_000.0)
        cb.update(89_000.0, 100_000.0)
        assert cb.is_halted

    def test_size_multiplier_zero(self, cb: CircuitBreaker) -> None:
        cb.update(100_000.0, 100_000.0)
        cb.update(89_000.0, 100_000.0)
        assert cb.size_multiplier == pytest.approx(0.0)

    def test_skips_scale_down_when_jumping_straight_to_10pct(self, cb: CircuitBreaker) -> None:
        cb.update(100_000.0, 100_000.0)
        cb.update(89_000.0, 100_000.0)
        # Should jump directly to HALT_48H even if SCALE_DOWN was never set
        assert cb.state == CBState.HALT_48H


class TestHaltIndefiniteTransition:
    def test_triggers_at_minus_15pct_peak_to_trough(self, cb: CircuitBreaker) -> None:
        # Peak = 100k; trough = 84k → -16% drawdown
        cb.update(100_000.0, 100_000.0)
        cb.update(84_000.0, 100_000.0)
        assert cb.state == CBState.HALT_INDEFINITE

    def test_drawdown_beats_mtd(self, cb: CircuitBreaker) -> None:
        # MTD only -5% but peak-to-trough is -16%: HALT_INDEFINITE takes priority
        cb.update(110_000.0, 100_000.0)  # equity grew → new peak
        cb.update(92_400.0, 100_000.0)  # -16% from peak of 110k; MTD only -7.6%
        assert cb.state == CBState.HALT_INDEFINITE

    def test_manual_resume_resets_to_ok(self, cb: CircuitBreaker) -> None:
        cb.update(100_000.0, 100_000.0)
        cb.update(84_000.0, 100_000.0)
        assert cb.state == CBState.HALT_INDEFINITE
        cb.manual_resume()
        assert cb.state == CBState.OK
        assert not cb.is_halted


class TestAutoRecovery48H:
    def test_auto_recovers_after_halt_until_expires(self, cb: CircuitBreaker) -> None:
        cb.update(100_000.0, 100_000.0)
        cb.update(89_000.0, 100_000.0)
        assert cb.state == CBState.HALT_48H
        # Manually expire the halt window
        cb._halt_until = time.time() - 1.0
        assert not cb.is_halted
        assert cb.state == CBState.OK


class TestSyntheticPnLSeries:
    """Drive a synthetic equity curve through all three trigger levels."""

    def test_full_escalation_sequence(self) -> None:
        cb = CircuitBreaker()
        equity = 100_000.0
        start = equity

        # Gradual MTD decline: −5% → SCALE_DOWN
        equity = 94_900.0
        cb.update(equity, start)
        assert cb.state == CBState.SCALE_DOWN, "Expected SCALE_DOWN at -5.1% MTD"

        # Continued decline: −10% → HALT_48H
        equity = 89_000.0
        cb.update(equity, start)
        assert cb.state == CBState.HALT_48H, "Expected HALT_48H at -11% MTD"

        # Force expire and resume
        cb._halt_until = time.time() - 1.0
        _ = cb.is_halted  # triggers auto-recovery
        cb._mtd_start_equity = None  # reset MTD baseline

        # New month, new peak, then catastrophic drawdown: −15% peak-to-trough
        cb._peak_equity = 100_000.0
        cb.update(84_000.0, mtd_start_equity=100_000.0)
        assert cb.state == CBState.HALT_INDEFINITE, "Expected HALT_INDEFINITE at -16% drawdown"
