"""Triple-barrier labeling method (AFML §3).

Assigns labels by tracking which of three barriers a price path touches first:
upper (profit-take), lower (stop-loss), or vertical (time expiry).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd


def compute_volatility(
    prices: pd.Series,  # type: ignore[type-arg]
    span: int = 24 * 60,
    min_periods: int = 60,
) -> pd.Series:  # type: ignore[type-arg]
    """Exponentially weighted return volatility (AFML §3.1).

    Args:
        prices: Close price series with DatetimeIndex.
        span: EWM span in number of bars.
        min_periods: Minimum observations before producing a value.
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.ewm(span=span, min_periods=min_periods).std()


def apply_triple_barrier(
    events: pd.DataFrame,
    close: pd.Series,  # type: ignore[type-arg]
    pt_sl: tuple[float, float] = (2.0, 2.0),
    num_threads: int = 1,
) -> pd.DataFrame:
    """Apply triple-barrier labeling to a set of events (AFML §3.4).

    Args:
        events: DataFrame with DatetimeIndex (event timestamps) and columns:
            - t1: vertical barrier timestamp
            - trgt: volatility target for barrier width
            Optional:
            - side: position side (+1 long, -1 short) for asymmetric barriers
        close: Close price series with DatetimeIndex.
        pt_sl: Multipliers for (profit-take, stop-loss) barriers.
            Set either to 0 to disable that barrier.
        num_threads: Number of parallel workers.

    Returns:
        DataFrame with columns: t_first, t_upper, t_lower, t_vertical.
    """
    out = events[["t1"]].copy()
    if pt_sl[0] > 0:
        out["pt"] = pt_sl[0] * events["trgt"]
    else:
        out["pt"] = pd.Series(np.nan, index=events.index)
    if pt_sl[1] > 0:
        out["sl"] = -pt_sl[1] * events["trgt"]
    else:
        out["sl"] = pd.Series(np.nan, index=events.index)

    func = partial(_apply_barrier_single, close=close)

    if num_threads > 1:
        with ProcessPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(func, [out.loc[idx] for idx in out.index]))
    else:
        results = [func(out.loc[idx]) for idx in out.index]

    result_df = pd.DataFrame(
        results,
        index=out.index,
        columns=["t_upper", "t_lower", "t_vertical"],
    )

    # t_first = earliest of the three barriers
    result_df["t_first"] = result_df[["t_upper", "t_lower", "t_vertical"]].min(axis=1)
    return result_df[["t_first", "t_upper", "t_lower", "t_vertical"]]


def _apply_barrier_single(
    event: pd.Series,  # type: ignore[type-arg]
    close: pd.Series,  # type: ignore[type-arg]
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    """Find barrier touches for a single event."""
    t0 = event.name
    t1 = event["t1"]
    pt = event["pt"]
    sl = event["sl"]

    # Price path from event to vertical barrier
    path = close.loc[t0:t1]
    if len(path) < 2:
        return (pd.NaT, pd.NaT, t1)

    # Returns relative to entry
    returns = (path / path.iloc[0]) - 1.0

    t_upper: pd.Timestamp = pd.NaT  # type: ignore[assignment]
    t_lower: pd.Timestamp = pd.NaT  # type: ignore[assignment]
    t_vertical = t1

    # Upper barrier (profit-take)
    if not np.isnan(pt):
        touches = returns[returns >= pt]
        if not touches.empty:
            t_upper = touches.index[0]

    # Lower barrier (stop-loss)
    if not np.isnan(sl):
        touches = returns[returns <= sl]
        if not touches.empty:
            t_lower = touches.index[0]

    return (t_upper, t_lower, t_vertical)


def get_bins(
    triple_barrier_df: pd.DataFrame,
    close: pd.Series,  # type: ignore[type-arg]
) -> pd.DataFrame:
    """Produce labels {-1, 0, +1} from triple-barrier results (AFML §3.5).

    +1 = upper barrier hit first (long profitable)
    -1 = lower barrier hit first (short profitable)
     0 = vertical barrier hit (no clear direction)
    """
    out = pd.DataFrame(index=triple_barrier_df.index)
    out["t_first"] = triple_barrier_df["t_first"]

    # Return at the first barrier touch
    out["ret"] = close.reindex(out["t_first"]).values / close.reindex(out.index).values - 1.0

    # Label based on which barrier was hit first
    labels = pd.Series(0, index=out.index)

    t_upper = triple_barrier_df["t_upper"]
    t_lower = triple_barrier_df["t_lower"]
    t_first = triple_barrier_df["t_first"]

    labels[t_first == t_upper] = 1
    labels[t_first == t_lower] = -1
    # Vertical barrier hit first → label stays 0 (true AFML §3.5 interpretation)

    out["bin"] = labels
    return out[["ret", "bin"]]


def get_meta_labels(
    triple_barrier_df: pd.DataFrame,
    close: pd.Series,  # type: ignore[type-arg]
    primary_signal: pd.Series,  # type: ignore[type-arg]
) -> pd.DataFrame:
    """Produce meta-labels {0, 1} (AFML §3.6).

    1 = primary signal direction was correct (trade was profitable)
    0 = primary signal direction was wrong (trade was unprofitable)
    """
    bins = get_bins(triple_barrier_df, close)
    primary = primary_signal.reindex(bins.index)

    meta = pd.DataFrame(index=bins.index)
    meta["ret"] = bins["ret"]
    meta["bin"] = (bins["bin"] == np.sign(primary)).astype(int)
    return meta


def make_events(
    close: pd.Series,  # type: ignore[type-arg]
    timestamps: pd.DatetimeIndex,
    vol: pd.Series,  # type: ignore[type-arg]
    vertical_bars: int = 240,
) -> pd.DataFrame:
    """Helper to construct an events DataFrame for apply_triple_barrier.

    Args:
        close: Close price series.
        timestamps: Event timestamps (subset of close.index).
        vol: Volatility series (from compute_volatility).
        vertical_bars: Number of bars for vertical barrier.
    """
    events = pd.DataFrame(index=timestamps)
    events["trgt"] = vol.reindex(timestamps)
    events = events.dropna(subset=["trgt"])

    # Vertical barrier = vertical_bars ahead
    t1 = events.index.map(
        lambda t: (
            close.index[close.index.get_loc(t) + vertical_bars]
            if (close.index.get_loc(t) + vertical_bars) < len(close)
            else close.index[-1]
        )
    )
    events["t1"] = t1
    return events
