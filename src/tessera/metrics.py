"""Prometheus metrics definitions and HTTP server.

All metrics are pre-registered at module import time. Call start_metrics_server()
to expose them on an HTTP endpoint for Prometheus scraping.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# ---------------------------------------------------------------------------
# Order flow metrics
# ---------------------------------------------------------------------------

ORDERS_TOTAL = Counter(
    "tessera_orders_total",
    "Incremented each time an order is submitted to an exchange. "
    "Tracks order creation rate by exchange, symbol, side, and terminal status.",
    ["exchange", "symbol", "side", "status"],
)

FILLS_TOTAL = Counter(
    "tessera_fills_total",
    "Incremented each time a fill (partial or full) is received from the exchange. "
    "Tracks execution rate by exchange, symbol, and side.",
    ["exchange", "symbol", "side"],
)

# ---------------------------------------------------------------------------
# Latency metrics
# ---------------------------------------------------------------------------

SIGNAL_LATENCY = Histogram(
    "tessera_signal_latency_seconds",
    "Time from bar close to signal emission. "
    "Updated after each feature→model→signal cycle completes.",
    ["strategy"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

ORDER_LATENCY = Histogram(
    "tessera_order_latency_seconds",
    "Round-trip time from order submission to exchange acknowledgement. "
    "Updated on each order ACK or rejection.",
    ["exchange"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# ---------------------------------------------------------------------------
# Position metrics
# ---------------------------------------------------------------------------

POSITION_UNITS = Gauge(
    "tessera_position_units",
    "Current net position size in base currency units. "
    "Updated after every fill or position reconciliation.",
    ["exchange", "symbol"],
)

# ---------------------------------------------------------------------------
# PnL metrics
# ---------------------------------------------------------------------------

PNL_UNREALIZED = Gauge(
    "tessera_pnl_unrealized_usd",
    "Mark-to-market unrealized PnL in USD across all positions. "
    "Updated on each price tick or position update.",
)

PNL_REALIZED = Gauge(
    "tessera_pnl_realized_usd",
    "Cumulative realized PnL in USD since process start. "
    "Updated on each position close or partial reduction.",
)

DRAWDOWN_PCT = Gauge(
    "tessera_drawdown_pct",
    "Current drawdown as a fraction of peak equity (0.0 = at peak, 1.0 = total loss). "
    "Updated on each equity update. Used by kill-switch logic.",
)


def start_metrics_server(port: int = 9090) -> None:
    """Start the Prometheus HTTP metrics server.

    Args:
        port: Port to bind the metrics HTTP endpoint.
    """
    start_http_server(port)
