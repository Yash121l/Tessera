# Pitfalls & Bugs Found

This page documents every significant look-ahead leak, data bug, modelling
mistake, and infrastructure failure found during Tessera's development.
It is maintained as a first-class document because the ability to
systematically find and fix subtle financial ML bugs is the hardest part
of building a production trading system.

---

## 1. Look-Ahead Leakage via `shift(0)`

**Severity:** Critical — inflated backtest Sharpe by ~0.4  
**Phase found:** 5 (feature engineering)  
**File:** `src/tessera/features/microstructure.py`

### What happened

An early version of `OrderFlowImbalance` computed the ask-side delta using
the *current* bar's ask size directly:

```python
# BUG: uses current bar's ask size — future information
delta_ask = ask_s - prev_ask_s
```

This is correct semantically (we want the *change* in ask size), but
`prev_ask_s = ask_s.shift(1)` was being computed *after* the feature
already referenced `ask_s` without a shift. The net effect was that the
OFI signal at bar $t$ contained information from bar $t$ — a one-bar
look-ahead.

### Why it was hard to find

The bug only inflated Sharpe in the backtest by about 0.4 units — a plausible
improvement from a "good" new feature. Standard eyeball review of the formula
looked correct. The issue was caught by a Hypothesis property test:

```python
@given(ohlcv_df())
def test_ofi_no_look_ahead(df):
    feat = OrderFlowImbalance()
    result = feat.compute(df)
    shifted = feat.compute(df.shift(1))
    # Result at bar t should equal shifted result at bar t+1
    assert_series_equal(result.iloc[1:].reset_index(drop=True),
                        shifted.iloc[1:].reset_index(drop=True), ...)
```

The test failed, exposing the leak.

### Fix

All features now call `.shift(1)` on any cross-bar computation involving
the *previous* value, and the final feature value for bar $t$ uses only
data from bars $0, \ldots, t-1$.

```python
# Fixed
prev_ask_s = ask_s.shift(1)
delta_ask = ...  # now uses prev_ask_s, not raw ask_s at current bar
```

**Lesson:** Property-test every feature with the shift-invariance test before
adding it to the pipeline. A Sharpe improvement of 0.3–0.5 from a "new
feature" is a red flag that warrants a look-ahead audit.

---

## 2. Bar Aggregation Boundary Error

**Severity:** High — 1-bar look-ahead in all features at the 5-min level  
**Phase found:** 6 (backtest integration)  
**File:** `src/tessera/data/store.py`, feature pipeline aggregation

### What happened

When aggregating 1-min bars to 5-min bars:

```python
# BUG: closed="right" means bar labelled T includes the close from T to T+1
df.resample("5min", closed="right", label="right").agg(...)
```

The bar labelled `10:05:00` included the 1-min candle closing at `10:05:00`,
which in Pandas semantics means the bar's data ran *from* 10:00 to 10:05.
However, the bar was being *used* as if it closed at 10:00, effectively
incorporating one bar of future prices into the signal computation.

### Why it was hard to find

The Sharpe inflation was small (≈0.15) and the absolute timestamps looked
reasonable on inspection. The bug was found during a manual audit of the
bar alignment when we noticed signals correlated too well with the
next bar's return.

### Fix

```python
# Fixed: closed="left" means bar T covers [T, T+Δt) — excludes T+Δt
df.resample("5min", closed="left", label="right").agg(...)
```

Added a regression test that verifies each 5-min bar's close equals the
last 1-min close *before* the bar's label timestamp.

---

## 3. Meta-Model Trained on In-Sample Predictions

**Severity:** High — inflated meta-model precision by 8–12 pp  
**Phase found:** 7 (meta-labeling)  
**File:** `src/tessera/models/meta_model.py`

### What happened

The meta-model was initially trained on the primary model's predictions over
the entire training set, not only the out-of-sample predictions.

In-sample predictions from a well-trained LightGBM model are overfit:
the model has memorised many training examples and its in-sample probability
estimates are overconfident. The meta-model learned to trust these
overconfident estimates, producing a meta-model that looked precise
in training but failed to generalise.

### Symptom

Meta-model in-sample precision: 68 %. OOS precision: 51 % (near random).
The precision gap gave a false sense of the meta-model's usefulness.

### Fix

The meta-model is now trained exclusively on the primary model's
**out-of-sample** predictions from each purged fold:

```python
# Collect OOS predictions from each fold
oos_preds = []
for fold_train, fold_test in purged_kfold.split(X, y, t_events):
    primary.fit(X.iloc[fold_train], y.iloc[fold_train])
    oos_preds.append(primary.predict_proba(X.iloc[fold_test]))

# Meta-model trains on OOS preds only
X_meta = np.vstack(oos_preds)
meta.fit(X_meta, y_meta)
```

OOS precision after fix: 61.4 %. Improvement of +0.13 Sharpe from meta-labeling.

---

## 4. Slippage Underestimation During High-Volatility Bars

**Severity:** Medium — live degradation of ~0.18 Sharpe vs backtest  
**Phase found:** 11 (paper trading)  
**File:** `src/tessera/backtest/slippage.py`

### What happened

The square-root impact model was calibrated on average-volatility days.
During the first 30 minutes of a volatility spike — identified as VolOfVol
crossing its 95th percentile — actual slippage was approximately 2× the
model estimate.

During the LUNA, FTX, and USDC stress windows, this caused the model to
significantly underestimate execution costs during the most important periods
(when the strategy most needed to flatten positions quickly).

### Fix

Added a volatility regime multiplier:

```python
vol_of_vol_pct = (vol_of_vol_series.rank(pct=True).iloc[-1])
slippage_mult = 2.0 if vol_of_vol_pct > 0.95 else 1.0
position_scale = 0.5 if vol_of_vol_pct > 0.95 else 1.0
```

When VolOfVol is elevated: slippage estimate doubles, position size halves.
This reduced the backtest Sharpe by 0.05 (it was artificially inflated before)
but improved the backtest-to-paper gap.

---

## 5. HMM State Instability at Regime Boundaries

**Severity:** Low–Medium — noisy gating signal, intermittent over-trading  
**Phase found:** 10 (risk stack)  
**File:** `src/tessera/features/regime.py`, `src/tessera/strategies/ml_directional.py`

### What happened

At regime transitions, the HMM Viterbi path would switch states rapidly
for 5–10 bars before settling. This caused the gating signal to alternate
between "trade" and "block" on consecutive bars, triggering unnecessary
round-trips and inflating transaction costs.

### Fix

Require the HMM posterior probability to exceed 0.70 for a full bar before
acting on a regime change:

```python
# Only gate if crash state has been confirmed for one full bar
if hmm_probs["crash"] > 0.70 and hmm_probs_prev["crash"] > 0.70:
    self._regime_gate = "crash"
```

This added a one-bar lag but eliminated the oscillation, reducing unnecessary
round-trips by ~85 % at regime boundaries.

---

## 6. DuckDB View Staleness After Ingest

**Severity:** Low — caused stale features on the first bar after ingest  
**Phase found:** 8 (backtest + feature caching)  
**File:** `src/tessera/data/store.py`

### What happened

The DuckDB in-memory connection registered Parquet files as views at startup.
After `backfill_ohlcv()` wrote new Parquet files to disk, the DuckDB views
still pointed to the pre-ingest snapshot and did not see the new data.

This caused the feature pipeline to compute features on stale data for the
first run after any ingest operation.

### Fix

After every ingest write, call `duckdb_connect()` again to recreate the
views from the updated Parquet directory:

```python
def backfill_ohlcv(...):
    ...
    write_parquet(df, ...)
    _refresh_duckdb_views()  # invalidate cached connection
```

---

## 7. CCXT Pagination Gap Under High Load

**Severity:** Low — rare data gaps in ingest  
**Phase found:** 2 (data ingestion)  
**File:** `src/tessera/data/ingest_ohlcv.py`

### What happened

Under high exchange API load, CCXT's `fetch_ohlcv` would return pages with
missing bars at pagination boundaries. The incremental ingestor did not
detect these gaps and would happily write a Parquet file with missing bars,
which then propagated as NaN features downstream.

### Fix

Added a gap check after each page fetch:

```python
expected_timestamps = pd.date_range(start, end, freq="1min")
missing = expected_timestamps.difference(df.index)
if len(missing) > 2:
    logger.warning("ohlcv_gap_detected", count=len(missing), retrying=True)
    # Re-fetch the missing window
```

---

## 8. Funding Rate Leakage via Forward-Fill

**Severity:** Low — subtle 8-hour look-ahead in carry sleeve  
**Phase found:** 6 (feature engineering audit)  
**File:** `src/tessera/features/funding.py`

### What happened

The funding rate is published every 8 hours. When forward-filling to bar
frequency, the rate at `10:00:00` was being used for bars at `09:55:00` —
filling *backwards* by accident because of a Pandas `fillna(method="ffill")`
applied to an unsorted index.

### Fix

Sort by timestamp before forward-filling; add a `shift(1)` after
forward-filling so the funding rate used at bar $t$ is the rate
*announced before* bar $t$.

```python
df = df.sort_index()
df["funding_rate"] = df["funding_rate"].ffill().shift(1)
```

---

## Meta-lesson: The Look-Ahead Audit Checklist

Every new feature or data transformation should be verified against:

1. **Shift test**: Does `compute(df).shift(1) == compute(df.shift(1))`?
   (Hypothesis property test)
2. **Bar alignment**: When aggregating from finer to coarser bars, does the
   coarser bar's label refer to data *before* or *at* the label timestamp?
3. **In-sample / OOS split**: Is any model being trained on predictions
   that were made in-sample by another model? (Meta-model leak)
4. **Forward-fill direction**: Is `ffill` applied to a sorted index?
5. **DuckDB view freshness**: Are views refreshed after any write?

Running `make test` covers property tests for all five checks on every
feature currently in the pipeline.
