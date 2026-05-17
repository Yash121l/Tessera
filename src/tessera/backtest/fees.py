"""Exchange fee schedules for backtest simulation.

Version-controlled fee tiers. To add a new exchange or update VIP tiers, add
an entry to _SCHEDULES. Negative bps = maker rebate.

Sources:
  Binance USDM Futures VIP 0-9 (2024-01)
  Bybit USDT Perpetuals VIP 0-5 (2024-01)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FeeTier:
    maker_bps: float
    taker_bps: float


# exchange → vip_tier → FeeTier
_SCHEDULES: dict[str, dict[int, FeeTier]] = {
    "binance": {
        0: FeeTier(maker_bps=2.0, taker_bps=5.0),
        1: FeeTier(maker_bps=1.6, taker_bps=4.0),
        2: FeeTier(maker_bps=1.4, taker_bps=3.5),
        3: FeeTier(maker_bps=1.2, taker_bps=3.2),
        4: FeeTier(maker_bps=1.0, taker_bps=3.0),
        5: FeeTier(maker_bps=0.8, taker_bps=2.7),
        6: FeeTier(maker_bps=0.6, taker_bps=2.5),
        7: FeeTier(maker_bps=0.4, taker_bps=2.2),
        8: FeeTier(maker_bps=0.2, taker_bps=2.0),
        9: FeeTier(maker_bps=0.0, taker_bps=1.7),
    },
    "bybit": {
        0: FeeTier(maker_bps=2.0, taker_bps=5.5),
        1: FeeTier(maker_bps=1.6, taker_bps=5.0),
        2: FeeTier(maker_bps=1.2, taker_bps=4.5),
        3: FeeTier(maker_bps=1.0, taker_bps=4.0),
        4: FeeTier(maker_bps=0.6, taker_bps=3.5),
        5: FeeTier(maker_bps=0.0, taker_bps=3.0),
    },
}


def effective_fee_bps(
    exchange: str,
    symbol: str,
    side: Literal["buy", "sell"],
    is_maker: bool,
    vip_tier: int = 0,
) -> float:
    """Fee in basis points for one fill leg.

    Args:
        exchange: Exchange name (case-insensitive).
        symbol: Instrument symbol (reserved for future per-symbol overrides).
        side: "buy" or "sell" (not currently differentiated).
        is_maker: True → maker rate; False → taker rate.
        vip_tier: VIP tier (0 = default/lowest).

    Returns:
        Basis points. Positive = cost; negative = rebate.
    """
    tiers = _SCHEDULES.get(exchange.lower())
    if tiers is None:
        return 2.0 if is_maker else 5.0

    max_tier = max(tiers)
    capped = min(vip_tier, max_tier)
    effective = max(t for t in tiers if t <= capped)
    ft = tiers[effective]
    return ft.maker_bps if is_maker else ft.taker_bps


def round_trip_fee_bps(
    exchange: str,
    symbol: str,
    entry_is_maker: bool = True,
    exit_is_maker: bool = False,
    vip_tier: int = 0,
) -> float:
    """Total round-trip cost in bps (entry leg + exit leg)."""
    entry = effective_fee_bps(exchange, symbol, "buy", entry_is_maker, vip_tier)
    exit_ = effective_fee_bps(exchange, symbol, "sell", exit_is_maker, vip_tier)
    return entry + exit_
