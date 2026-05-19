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


# ---------------------------------------------------------------------------
# Live trading metrics
# ---------------------------------------------------------------------------

HEARTBEAT_TS = Gauge(
    "tessera_heartbeat_ts",
    "Unix timestamp of last live runner heartbeat. Staleness > 10s indicates a stalled loop.",
)

RUNNER_RESTARTS = Counter(
    "tessera_runner_restarts_total",
    "Number of PaperRunner crash-restarts since process start.",
)

RECONCILE_OK = Gauge(
    "tessera_reconcile_ok",
    "1 if the last position reconciliation matched exchange, 0 on mismatch.",
    ["venue"],
)

RECONCILE_MISMATCH = Counter(
    "tessera_reconcile_mismatch_total",
    "Position reconciliation mismatches (internal vs exchange) by symbol.",
    ["symbol"],
)

FEE_PAID_USD = Counter(
    "tessera_fee_paid_usd_total",
    "Cumulative fees paid in USD.",
    ["exchange", "symbol"],
)

SLIPPAGE_REALIZED_BPS = Histogram(
    "tessera_slippage_realized_bps",
    "Realized slippage vs mid-price in basis points per fill.",
    ["symbol"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0),
)

BAR_ARRIVAL_RATE = Gauge(
    "tessera_bar_arrival_rate_per_min",
    "Rate of bar arrivals over the last minute (bars/min). Gap < 0.5 indicates feed degradation.",
    ["exchange", "symbol"],
)

LAST_BAR_AGE = Gauge(
    "tessera_last_bar_age_seconds",
    "Seconds elapsed since the last bar was received. >60s triggers data-gap kill switch.",
    ["exchange", "symbol"],
)

EXCHANGE_PING_LATENCY = Gauge(
    "tessera_exchange_ping_latency_seconds",
    "Latency of the last successful exchange ping round-trip in seconds.",
    ["exchange"],
)

ADV_FALLBACK = Counter(
    "tessera_adv_fallback_total",
    "Number of times the ADV estimator fell back to the static config default "
    "because live ADV data was unavailable for a symbol.",
    ["symbol"],
)

PORTFOLIO_LEVERAGE_GROSS = Gauge(
    "tessera_portfolio_leverage_gross",
    "Gross portfolio leverage: sum(|notional_i|) / NAV. "
    "Updated after every fill or position reconciliation.",
)


def start_metrics_server(port: int = 9090) -> None:
    """Start the Prometheus HTTP metrics server.

    Args:
        port: Port to bind the metrics HTTP endpoint.
    """
    start_http_server(port)
