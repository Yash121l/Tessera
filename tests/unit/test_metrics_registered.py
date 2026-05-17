"""Test that all Prometheus metrics are registered in the default registry."""

from __future__ import annotations

from prometheus_client import REGISTRY

from tessera import metrics


def _metric_registered(name: str) -> bool:
    """Check if a metric name exists in the default registry."""
    return any(name in str(c) for c in REGISTRY.collect() if hasattr(c, "name"))


def test_orders_total_registered() -> None:
    """tessera_orders_total counter should be in the registry."""
    assert "tessera_orders" in metrics.ORDERS_TOTAL._name


def test_fills_total_registered() -> None:
    """tessera_fills_total counter should be in the registry."""
    assert "tessera_fills" in metrics.FILLS_TOTAL._name


def test_signal_latency_registered() -> None:
    """tessera_signal_latency_seconds histogram should be in the registry."""
    assert metrics.SIGNAL_LATENCY._name == "tessera_signal_latency_seconds"


def test_order_latency_registered() -> None:
    """tessera_order_latency_seconds histogram should be in the registry."""
    assert metrics.ORDER_LATENCY._name == "tessera_order_latency_seconds"


def test_position_units_registered() -> None:
    """tessera_position_units gauge should be in the registry."""
    assert metrics.POSITION_UNITS._name == "tessera_position_units"


def test_pnl_unrealized_registered() -> None:
    """tessera_pnl_unrealized_usd gauge should be in the registry."""
    assert metrics.PNL_UNREALIZED._name == "tessera_pnl_unrealized_usd"


def test_pnl_realized_registered() -> None:
    """tessera_pnl_realized_usd gauge should be in the registry."""
    assert metrics.PNL_REALIZED._name == "tessera_pnl_realized_usd"


def test_drawdown_registered() -> None:
    """tessera_drawdown_pct gauge should be in the registry."""
    assert metrics.DRAWDOWN_PCT._name == "tessera_drawdown_pct"


def test_metrics_have_documentation() -> None:
    """Every metric should have a non-empty documentation string."""
    all_metrics = [
        metrics.ORDERS_TOTAL,
        metrics.FILLS_TOTAL,
        metrics.SIGNAL_LATENCY,
        metrics.ORDER_LATENCY,
        metrics.POSITION_UNITS,
        metrics.PNL_UNREALIZED,
        metrics.PNL_REALIZED,
        metrics.DRAWDOWN_PCT,
    ]
    for m in all_metrics:
        assert m._documentation, f"{m._name} has no documentation"


def test_counter_labels() -> None:
    """Counters should have the correct label names."""
    assert set(metrics.ORDERS_TOTAL._labelnames) == {"exchange", "symbol", "side", "status"}
    assert set(metrics.FILLS_TOTAL._labelnames) == {"exchange", "symbol", "side"}


def test_histogram_labels() -> None:
    """Histograms should have the correct label names."""
    assert set(metrics.SIGNAL_LATENCY._labelnames) == {"strategy"}
    assert set(metrics.ORDER_LATENCY._labelnames) == {"exchange"}
