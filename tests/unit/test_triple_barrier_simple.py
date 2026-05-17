"""Test triple barrier: synthetic price path where upper barrier is hit at bar 17."""

from __future__ import annotations

import pandas as pd

from tessera.labels.triple_barrier import apply_triple_barrier, get_bins


def test_upper_barrier_hit_at_bar_17() -> None:
    """A price path that rises steadily should hit the upper barrier at the expected bar."""
    n = 50
    idx = pd.date_range("2024-01-01", periods=n, freq="min")

    # Flat at 100 for bars 0-16, then jump above barrier at bar 17
    prices = pd.Series(100.0, index=idx)
    # Set a target of 1% → upper barrier at +1%
    # Bar 17 will be the first to breach 100 * 1.01 = 101.0
    for i in range(17, n):
        prices.iloc[i] = 101.5

    events = pd.DataFrame(
        {"t1": [idx[-1]], "trgt": [0.01]},
        index=[idx[0]],
    )

    result = apply_triple_barrier(events, prices, pt_sl=(1.0, 1.0))

    # Upper barrier should be hit at bar 17
    assert result["t_upper"].iloc[0] == idx[17]
    assert result["t_first"].iloc[0] == idx[17]


def test_lower_barrier_hit() -> None:
    """A price drop should trigger the lower barrier."""
    n = 50
    idx = pd.date_range("2024-01-01", periods=n, freq="min")

    prices = pd.Series(100.0, index=idx)
    # Drop at bar 10
    for i in range(10, n):
        prices.iloc[i] = 97.0

    events = pd.DataFrame(
        {"t1": [idx[-1]], "trgt": [0.02]},
        index=[idx[0]],
    )

    result = apply_triple_barrier(events, prices, pt_sl=(1.0, 1.0))
    assert result["t_lower"].iloc[0] == idx[10]
    assert result["t_first"].iloc[0] == idx[10]


def test_get_bins_labels() -> None:
    """Verify label assignment from triple barrier results."""
    n = 50
    idx = pd.date_range("2024-01-01", periods=n, freq="min")

    prices = pd.Series(100.0, index=idx)
    for i in range(17, n):
        prices.iloc[i] = 102.0

    events = pd.DataFrame(
        {"t1": [idx[-1]], "trgt": [0.01]},
        index=[idx[0]],
    )

    tb = apply_triple_barrier(events, prices, pt_sl=(1.0, 1.0))
    bins = get_bins(tb, prices)
    assert bins["bin"].iloc[0] == 1  # Upper barrier hit → long label
