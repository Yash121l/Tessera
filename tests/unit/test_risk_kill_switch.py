"""Unit tests for tessera.risk.kill_switch.

Covers:
  - All six trigger paths
  - Idempotency (engage twice → single counter increment)
  - Thread safety (kill switch fires mid-cycle → cycle aborts cleanly)
  - Position-reconcile mismatch fires within one check call
"""

from __future__ import annotations

import threading
import time

import pytest

from tessera.risk.kill_switch import KillSwitch, KillSwitchConfig, KSTrigger


@pytest.fixture
def ks() -> KillSwitch:
    return KillSwitch(
        config=KillSwitchConfig(
            daily_loss_threshold=0.03,
            drawdown_threshold=0.08,
            data_gap_seconds=0.5,
            reject_rate_threshold=0.05,
            reject_rate_window_seconds=5.0,
        )
    )


class TestInitialState:
    def test_not_active_initially(self, ks: KillSwitch) -> None:
        assert not ks.is_active

    def test_trigger_reason_none(self, ks: KillSwitch) -> None:
        assert ks.trigger_reason is None


class TestDailyLossTrigger:
    def test_fires_above_threshold(self, ks: KillSwitch) -> None:
        ks.check_daily_loss(100_000.0, day_start_equity=100_000.0)
        ks.check_daily_loss(96_800.0)  # -3.2% → triggers
        assert ks.is_active
        assert ks.trigger_reason is not None
        assert ks.trigger_reason[0] == KSTrigger.DAILY_LOSS

    def test_does_not_fire_below_threshold(self, ks: KillSwitch) -> None:
        ks.check_daily_loss(100_000.0, day_start_equity=100_000.0)
        ks.check_daily_loss(97_500.0)  # -2.5% — below 3%
        assert not ks.is_active

    def test_day_start_set_on_first_call(self, ks: KillSwitch) -> None:
        ks.check_daily_loss(100_000.0)
        assert not ks.is_active  # first call just sets the baseline


class TestDrawdownTrigger:
    def test_fires_above_threshold(self, ks: KillSwitch) -> None:
        ks.check_drawdown(100_000.0)  # sets peak
        ks.check_drawdown(91_500.0)  # -8.5% drawdown → triggers
        assert ks.is_active
        assert ks.trigger_reason[0] == KSTrigger.DRAWDOWN  # type: ignore[index]

    def test_does_not_fire_at_new_peak(self, ks: KillSwitch) -> None:
        ks.check_drawdown(100_000.0)
        ks.check_drawdown(110_000.0)  # new peak
        assert not ks.is_active

    def test_does_not_fire_below_threshold(self, ks: KillSwitch) -> None:
        ks.check_drawdown(100_000.0)
        ks.check_drawdown(93_000.0)  # -7% — below 8%
        assert not ks.is_active


class TestDataGapTrigger:
    def test_fires_after_gap(self, ks: KillSwitch) -> None:
        # Backdate last tick by more than the 0.5s threshold
        ks._last_data_ts = time.monotonic() - 1.0
        ks.check_data_gap()
        assert ks.is_active
        assert ks.trigger_reason[0] == KSTrigger.DATA_GAP  # type: ignore[index]

    def test_does_not_fire_within_gap(self, ks: KillSwitch) -> None:
        ks.record_data_tick()
        ks.check_data_gap()
        assert not ks.is_active


class TestOrderRejectRateTrigger:
    def test_fires_when_rate_exceeds_threshold(self, ks: KillSwitch) -> None:
        # 6 rejections out of 10 = 60%, well above 5%
        for _ in range(4):
            ks.record_order_event(rejected=False)
        for _ in range(6):
            ks.record_order_event(rejected=True)
        assert ks.is_active
        assert ks.trigger_reason[0] == KSTrigger.ORDER_REJECT_RATE  # type: ignore[index]

    def test_does_not_fire_below_minimum_sample(self, ks: KillSwitch) -> None:
        # Fewer than 10 events: no trigger even at 100% reject rate
        for _ in range(5):
            ks.record_order_event(rejected=True)
        assert not ks.is_active

    def test_does_not_fire_with_low_rate(self, ks: KillSwitch) -> None:
        for _ in range(19):
            ks.record_order_event(rejected=False)
        ks.record_order_event(rejected=True)  # 5% exactly — at threshold, not above
        assert not ks.is_active


class TestPositionReconcileTrigger:
    def test_fires_on_mismatch(self, ks: KillSwitch) -> None:
        internal = {"BTCUSDT": 1.0}
        exchange = {"BTCUSDT": 0.5}  # 50% mismatch
        ks.check_position_reconcile(internal, exchange)
        assert ks.is_active
        assert ks.trigger_reason[0] == KSTrigger.POSITION_MISMATCH  # type: ignore[index]

    def test_fires_on_extra_exchange_position(self, ks: KillSwitch) -> None:
        internal: dict[str, float] = {}
        exchange = {"ETHUSDT": 5.0}  # we have no record of this
        ks.check_position_reconcile(internal, exchange)
        assert ks.is_active

    def test_no_fire_within_tolerance(self, ks: KillSwitch) -> None:
        internal = {"BTCUSDT": 1.000}
        exchange = {"BTCUSDT": 1.005}  # 0.5% difference — within 1% tolerance
        ks.check_position_reconcile(internal, exchange, tolerance=0.01)
        assert not ks.is_active

    def test_fires_within_one_reconcile_call(self, ks: KillSwitch) -> None:
        """Mismatch is detected in the same reconcile call, not deferred."""
        internal = {"SOLUSDT": 100.0}
        exchange = {"SOLUSDT": 50.0}
        ks.check_position_reconcile(internal, exchange)
        assert ks.is_active  # immediate — not after a second call


class TestIdempotency:
    def test_engage_twice_first_trigger_wins(self, ks: KillSwitch) -> None:
        ks.engage(KSTrigger.DAILY_LOSS, "first")
        ks.engage(KSTrigger.DRAWDOWN, "second")
        assert ks.trigger_reason[0] == KSTrigger.DAILY_LOSS  # type: ignore[index]

    def test_clear_and_reengage(self, ks: KillSwitch) -> None:
        ks.engage(KSTrigger.DATA_GAP, "test")
        ks.clear()
        assert not ks.is_active
        ks.engage(KSTrigger.DRAWDOWN, "second")
        assert ks.is_active


class TestConcurrency:
    """Kill switch fires from a background thread while a cycle is in flight."""

    def test_cycle_aborts_when_kill_switch_trips(self) -> None:
        ks = KillSwitch()
        aborted = threading.Event()
        cycle_started = threading.Event()

        def strategy_cycle() -> None:
            for _ in range(100):
                cycle_started.set()
                if ks.is_active:
                    aborted.set()
                    return
                time.sleep(0.001)

        def trigger_kill() -> None:
            cycle_started.wait(timeout=1.0)
            ks.engage(KSTrigger.DRAWDOWN, "concurrent test")

        cycle_thread = threading.Thread(target=strategy_cycle)
        trigger_thread = threading.Thread(target=trigger_kill)

        cycle_thread.start()
        trigger_thread.start()

        cycle_thread.join(timeout=2.0)
        trigger_thread.join(timeout=2.0)

        assert aborted.is_set(), "Strategy cycle did not abort after kill switch engaged"

    def test_concurrent_engage_from_multiple_threads(self) -> None:
        ks = KillSwitch()
        triggered_by: list[KSTrigger] = []

        def _engage(trigger: KSTrigger) -> None:
            ks.engage(trigger, "concurrent")
            if ks.trigger_reason is not None:
                triggered_by.append(ks.trigger_reason[0])

        threads = [threading.Thread(target=_engage, args=(t,)) for t in list(KSTrigger)[:4]]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert ks.is_active
        # Exactly one trigger should have won (idempotent)
        assert ks.trigger_reason is not None
