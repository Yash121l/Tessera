"""Prometheus metrics definitions for observability."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

ORDERS_TOTAL = Counter(
    "tessera_orders_total",
    "Total orders placed",
    ["side", "symbol", "exchange"],
)

POSITION_SIZE = Gauge(
    "tessera_position_size",
    "Current position size in base currency",
    ["symbol"],
)

LATENCY_SECONDS = Histogram(
    "tessera_latency_seconds",
    "Operation latency in seconds",
    ["operation"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

PNL_TOTAL = Gauge(
    "tessera_pnl_total",
    "Cumulative realized PnL in USDT",
    ["strategy"],
)
