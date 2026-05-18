"""Position limit enforcement and correlation-adjusted sizing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PositionLimits:
    """Hard risk limits for a portfolio.

    Default limits:
      - Per-asset:  20% of NAV
      - Gross:     200% of NAV  (sum of |notionals|)
      - Net:       100% of NAV  (|longs − shorts|)
      - Per-sector: configurable via sector_caps
    """

    max_asset_pct: float = 0.20
    max_gross_pct: float = 2.00
    max_net_pct: float = 1.00
    sector_caps: dict[str, float] = field(default_factory=dict)

    def check(
        self,
        positions: dict[str, float],
        nav: float,
        sector_map: dict[str, str] | None = None,
    ) -> dict[str, str]:
        """Return a dict of violated limit names → description. Empty = all clear."""
        if nav <= 0.0:
            return {"nav": "NAV must be positive"}

        violations: dict[str, str] = {}

        for symbol, notional in positions.items():
            pct = abs(notional) / nav
            if pct > self.max_asset_pct:
                violations[symbol] = f"asset limit: {pct:.1%} > {self.max_asset_pct:.1%}"

        gross = sum(abs(v) for v in positions.values())
        if gross / nav > self.max_gross_pct:
            violations["gross"] = f"gross: {gross / nav:.1%} > {self.max_gross_pct:.1%}"

        net = sum(positions.values())
        if abs(net) / nav > self.max_net_pct:
            violations["net"] = f"net: {abs(net) / nav:.1%} > {self.max_net_pct:.1%}"

        if sector_map and self.sector_caps:
            sector_exp: dict[str, float] = {}
            for sym, notional in positions.items():
                sec = sector_map.get(sym)
                if sec:
                    sector_exp[sec] = sector_exp.get(sec, 0.0) + abs(notional)
            for sec, cap in self.sector_caps.items():
                exp = sector_exp.get(sec, 0.0)
                if exp / nav > cap:
                    violations[f"sector:{sec}"] = f"sector {sec}: {exp / nav:.1%} > {cap:.1%}"

        return violations

    def clip_to_limits(
        self,
        positions: dict[str, float],
        nav: float,
        sector_map: dict[str, str] | None = None,
    ) -> dict[str, float]:
        """Return positions proportionally clipped to satisfy all limits.

        Per-asset caps are applied first (hard clip per symbol). Then gross
        and net limits are enforced via proportional scaling of the dominant
        direction. Sector limits are not clipped here — use check() to detect.
        """
        clipped: dict[str, float] = {
            sym: float(np.sign(n)) * min(abs(n), self.max_asset_pct * nav)
            for sym, n in positions.items()
        }

        gross = sum(abs(v) for v in clipped.values())
        if gross > 0.0 and gross / nav > self.max_gross_pct:
            scale = (self.max_gross_pct * nav) / gross
            clipped = {sym: v * scale for sym, v in clipped.items()}

        net = sum(clipped.values())
        if abs(net) / nav > self.max_net_pct:
            excess = abs(net) - self.max_net_pct * nav
            if net > 0.0:
                longs = {s: v for s, v in clipped.items() if v > 0.0}
                total = sum(longs.values())
                if total > 0.0:
                    scale = (total - excess) / total
                    clipped.update({s: v * scale for s, v in longs.items()})
            else:
                shorts = {s: v for s, v in clipped.items() if v < 0.0}
                total = sum(abs(v) for v in shorts.values())
                if total > 0.0:
                    scale = (total - excess) / total
                    clipped.update({s: v * scale for s, v in shorts.items()})

        return clipped


def correlation_limit_check(
    positions: dict[str, float],
    returns: pd.DataFrame,
    returns_window: int = 30,
    corr_threshold: float = 0.7,
) -> dict[str, float]:
    """Compute effective notional by collapsing highly-correlated pairs.

    Pairs with pairwise |correlation| > corr_threshold are treated as a single
    position for sizing purposes: their notionals are summed and attributed to
    the symbol with the largest absolute notional in the group.

    Assets absent from `returns` are passed through unchanged.

    Args:
        positions: symbol → signed notional (USD).
        returns: DataFrame with symbols as columns and timestamps as rows.
        returns_window: Number of trailing rows to use for the correlation window.
        corr_threshold: Pairs above this threshold are merged.

    Returns:
        Dict of symbol → effective notional with correlated pairs collapsed.
    """
    syms = [s for s in positions if s in returns.columns]
    if len(syms) < 2:
        return dict(positions)

    window = returns.tail(returns_window)[syms]
    corr = window.corr()

    # Union-Find grouping of correlated assets
    parent: dict[str, str] = {s: s for s in syms}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: str, y: str) -> None:
        parent[_find(x)] = _find(y)

    for i, s1 in enumerate(syms):
        for s2 in syms[i + 1 :]:
            if abs(corr.loc[s1, s2]) > corr_threshold:
                _union(s1, s2)

    groups: dict[str, list[tuple[str, float]]] = {}
    for sym in syms:
        root = _find(sym)
        groups.setdefault(root, []).append((sym, positions[sym]))

    effective: dict[str, float] = {}
    for members in groups.values():
        total = sum(n for _, n in members)
        rep = max(members, key=lambda x: abs(x[1]))[0]
        effective[rep] = total

    for sym, notional in positions.items():
        if sym not in syms:
            effective[sym] = notional

    return effective
