"""Test triple barrier: flat prices → all events hit the vertical barrier."""

from __future__ import annotations

import pandas as pd

from tessera.labels.triple_barrier import apply_triple_barrier, get_bins


def test_flat_prices_hit_vertical() -> None:
    """With perfectly flat prices, neither upper nor lower barrier is touched."""
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    prices = pd.Series(100.0, index=idx)

    # Create 5 events spread across the series
    event_indices = [idx[0], idx[10], idx[20], idx[30], idx[40]]
    events = pd.DataFrame(
        {
            "t1": [idx[20], idx[30], idx[40], idx[50], idx[60]],
            "trgt": [0.01] * 5,
        },
        index=event_indices,
    )

    result = apply_triple_barrier(events, prices, pt_sl=(1.0, 1.0))

    # All should hit vertical only
    assert result["t_upper"].isna().all()
    assert result["t_lower"].isna().all()
    for i in range(5):
        assert result["t_first"].iloc[i] == events["t1"].iloc[i]


def test_flat_prices_label_zero() -> None:
    """Flat prices should produce label 0 (no trade)."""
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="min")
    prices = pd.Series(100.0, index=idx)

    events = pd.DataFrame(
        {"t1": [idx[50]], "trgt": [0.01]},
        index=[idx[0]],
    )

    tb = apply_triple_barrier(events, prices, pt_sl=(1.0, 1.0))
    bins = get_bins(tb, prices)
    assert bins["bin"].iloc[0] == 0
