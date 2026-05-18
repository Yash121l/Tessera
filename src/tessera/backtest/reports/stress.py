"""Stress-window analysis for strategy returns.

Pre-defined windows cover the major crypto tail events since 2020.  For
each window we compute total PnL (cumulative return), maximum intra-window
drawdown, and annualized Sharpe using available returns.

IS/OOS label requires the caller to pass test_start_date (first date of
the hold-out set).  Windows that begin before test_start_date are
in-sample (IS); those on or after are out-of-sample (OOS).  This
distinction is critical: an OOS loss proves the model is being stress-
tested honestly.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

STRESS_WINDOWS: dict[str, tuple[str, str]] = {
    "COVID Crash": ("2020-02-20", "2020-03-13"),
    "China Mining Ban": ("2021-05-12", "2021-05-20"),
    "LUNA Collapse": ("2022-05-08", "2022-05-15"),
    "FTX Collapse": ("2022-11-06", "2022-11-12"),
    "USDC Depeg": ("2023-03-10", "2023-03-13"),
    "Yen Carry Unwind": ("2024-08-02", "2024-08-07"),
}


def compute_stress_pnls(
    returns: pd.Series,
    test_start_date: str | None = None,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    """Compute per-stress-window metrics.

    Args:
        returns: Daily (or per-bar) return series with a DatetimeIndex.
        test_start_date: ISO-8601 date string of the first OOS bar.
            Rows before this date are labelled IS; at or after are OOS.
        annualization_factor: Periods per year for Sharpe annualization.

    Returns:
        DataFrame with columns:
            event, start, end, n_bars, total_return, max_drawdown,
            sharpe, in_sample, coverage
        coverage is 'full' / 'partial' / 'none' based on data availability.
    """
    import numpy as np

    if returns.index.tz is not None:
        returns = returns.copy()
        returns.index = returns.index.tz_localize(None)

    rows: list[dict[str, Any]] = []
    test_dt = pd.Timestamp(test_start_date) if test_start_date else None

    for event, (start_str, end_str) in STRESS_WINDOWS.items():
        start_ts = pd.Timestamp(start_str)
        end_ts = pd.Timestamp(end_str)

        window = returns.loc[(returns.index >= start_ts) & (returns.index <= end_ts)]

        requested_range = (end_ts - start_ts).days + 1
        actual_bars = len(window)

        if actual_bars == 0:
            coverage = "none"
            rows.append(
                {
                    "event": event,
                    "start": start_str,
                    "end": end_str,
                    "n_bars": 0,
                    "total_return": float("nan"),
                    "max_drawdown": float("nan"),
                    "sharpe": float("nan"),
                    "in_sample": _is_label(start_ts, test_dt),
                    "coverage": coverage,
                }
            )
            continue

        coverage = "full" if actual_bars >= max(requested_range // 2, 1) else "partial"

        r: np.ndarray = np.asarray(window.values, dtype=float)
        cum = (1.0 + r).cumprod()
        total_ret = float(cum[-1] - 1.0)
        roll_max = np.maximum.accumulate(cum)
        drawdowns = (cum - roll_max) / roll_max
        max_dd = float(drawdowns.min())

        r_std = float(r.std(ddof=1))
        if actual_bars >= 2 and r_std > 1e-12:
            sharpe = float(r.mean() / r_std * np.sqrt(annualization_factor))
        else:
            sharpe = float("nan")

        rows.append(
            {
                "event": event,
                "start": start_str,
                "end": end_str,
                "n_bars": actual_bars,
                "total_return": total_ret,
                "max_drawdown": max_dd,
                "sharpe": sharpe,
                "in_sample": _is_label(start_ts, test_dt),
                "coverage": coverage,
            }
        )

    return pd.DataFrame(rows)


def _is_label(event_start: pd.Timestamp, test_start: pd.Timestamp | None) -> str:
    if test_start is None:
        return "unknown"
    return "IS" if event_start < test_start else "OOS"
