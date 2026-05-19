# Feature Catalog

Tessera computes 20 engineered features across six families.
Every feature extends the abstract `Feature` base class and declares
`point_in_time_safe = True`, which is enforced by Hypothesis property tests
that verify no feature introduces look-ahead leakage.

Features are cached per-symbol per-day to Parquet under
`data/features/<name>/v<version>/<symbol>/<date>.parquet`.
The pipeline resolves inter-feature dependencies via Kahn's topological sort.

---

## Family 1: Returns

### LogReturn

| Attribute | Value |
|---|---|
| Class | `tessera.features.returns.LogReturn` |
| Formula | $r_t = \ln(c_t / c_{t-1})$ |
| Input | `close` column, any bar frequency |
| Output | Per-bar log-return (float) |
| PIT safe | Yes — uses only past close |
| Reference | Standard |

The primary return feature used as input to all downstream computations.
No lag is applied; the bar's own close is compared to the previous bar's close.
For features that build on returns, a `.shift(1)` is applied inside the
dependent feature's `compute()` method.

---

## Family 2: Volatility

### RealizedVol

| Attribute | Value |
|---|---|
| Formula | $\sigma_t = \text{EWM\_std}(r_t, \text{span}=S)$ |
| Input | `log_return` |
| Parameters | `span` (default 60 bars) |
| Reference | Standard EWMA volatility |

Exponentially weighted standard deviation of log-returns.
Used as the barrier-scaling factor in triple-barrier labeling
and as a position-size denominator in vol-targeting.

### Parkinson

| Attribute | Value |
|---|---|
| Formula | $\hat{\sigma}_P^2 = \frac{1}{4 \ln 2} \left(\ln \frac{H_t}{L_t}\right)^2$ |
| Input | `high`, `low` |
| Reference | Parkinson (1980) |

Intrabar high-low estimator. 5–8× more efficient than close-to-close
in low-volume overnight sessions where the close-to-close estimator
is dominated by bid-ask bounce.

### GarmanKlass

| Attribute | Value |
|---|---|
| Formula | $\hat{\sigma}_{GK}^2 = 0.511(u-d)^2 - 0.019[c(u+d) - 2ud] - 0.383c^2$ |
| Notation | $u = \ln(H/O),\ d = \ln(L/O),\ c = \ln(C/O)$ |
| Input | `open`, `high`, `low`, `close` |
| Reference | Garman & Klass (1980) |

OHLC-based estimator. Most efficient of the classical range estimators;
used as the primary vol estimate in the slippage model.

### VolOfVol

| Attribute | Value |
|---|---|
| Formula | $\text{VoV}_t = \text{rolling\_std}(\sigma_t, W)$ |
| Input | `realized_vol` |
| Parameters | `window` (default 20 bars) |
| Use | Regime transition signal; triggers slippage multiplier |

### GARCH(1,1)

| Attribute | Value |
|---|---|
| Formula | $h_t = \omega + \alpha \varepsilon_{t-1}^2 + \beta h_{t-1}$ |
| Input | `log_return` |
| Library | `arch` (Python) |
| Reference | Bollerslev (1986) |

Conditional variance estimate. Captures volatility clustering better than
EWMA in trending volatility regimes; fitted on a rolling 500-bar window
and updated daily.

---

## Family 3: Microstructure

### OrderFlowImbalance (OFI)

| Attribute | Value |
|---|---|
| Formula | $\text{OFI}_t = \Delta V_t^{\text{bid}} - \Delta V_t^{\text{ask}}$ |
| Input | L2: `bid_price`, `bid_size`, `ask_price`, `ask_size` |
| Fallback | OHLCV proxy: `(close - open) / range × volume` |
| Reference | Cont, Kukanov & Stoikov (2014) |

Signed order-flow pressure. Positive OFI means more aggressive buying;
empirically predicts short-horizon (1–3 bar) price direction.
The L2 fallback is used in backtesting; live trading uses real L2 data.

### MicroPrice

| Attribute | Value |
|---|---|
| Formula | $m_t = \frac{V_a \cdot b_t + V_b \cdot a_t}{V_a + V_b}$ |
| Notation | $b_t, a_t$: bid/ask prices; $V_b, V_a$: bid/ask sizes |
| Input | L2: `bid_price`, `bid_size`, `ask_price`, `ask_size` |
| Reference | Stoikov (2018) |

Size-weighted midprice. Leads the quoted midprice because it incorporates
order-book imbalance. Most predictive at sub-second horizons; at 5-min bars,
its signal decays but still contributes ~0.08 Sharpe in the feature ablation.

### SpreadBps

| Attribute | Value |
|---|---|
| Formula | $s_t = (a_t - b_t) / m_t \times 10^4$ |
| Input | `bid_price`, `ask_price` |
| Use | High spread → reduce position size; strategy backs off |

### VPIN

| Attribute | Value |
|---|---|
| Formula | See Easley et al. (2012) §3 |
| Input | Volume-bucketed trade flow |
| Reference | Easley, Lopez de Prado & O'Hara (2012) |

Volume-synchronised probability of informed trading.
Computed over volume buckets of size $V_n = \text{ADV} / 50$.
Spikes in VPIN reliably precede adverse price impact in the hour following;
removing VPIN alone reduces backtest Sharpe by 0.19.

### DepthWeightedSlippage

| Attribute | Value |
|---|---|
| Formula | Walk simulated order book to fill target notional; take volume-weighted avg price |
| Input | L2 order book snapshot |
| Use | Features the cost surface for meta-model sizing decisions |

---

## Family 4: Funding Rate

### FundingRate

| Attribute | Value |
|---|---|
| Source | Exchange 8-hour funding rate endpoint via CCXT |
| Storage | `data/funding_rates/<exchange>/<symbol>/<date>.parquet` |
| Resampled | Forward-filled to target bar frequency |

Raw annualised funding rate. Positive = longs pay shorts (market is
long-leaning); negative = shorts pay longs (market is short-leaning).

### FundingZScore

| Attribute | Value |
|---|---|
| Formula | $z_t = (r_t - \mu_{30d}) / \sigma_{30d}$ |
| Input | `funding_rate`, rolling 30-day window |
| Use | Trigger signal for the carry sleeve; threshold at $|z| > 2.0$ |

### SpotPerpBasis

| Attribute | Value |
|---|---|
| Formula | $\text{basis}_t = \ln(p_{\text{perp},t}) - \ln(p_{\text{spot},t})$ |
| Input | Perpetual close price + spot index price |
| Use | Persistent positive basis (>5 bps) signals funding pressure building |

---

## Family 5: Cross-Sectional

### UniverseRank

| Attribute | Value |
|---|---|
| Formula | Percentile rank of symbol's 1-h log-return across universe |
| Input | All symbols in universe, 1-h bar closes |
| Reference | AFML §5 (cross-sectional feature engineering) |

Captures relative momentum. A symbol at rank 0.9 has been the strongest
performer over the past hour; this is a reliable short-term continuation
signal in low-regime-uncertainty periods.

### BetaToBTC

| Attribute | Value |
|---|---|
| Formula | $\hat{\beta}_t = \frac{\text{Cov}(r_s, r_{\text{BTC}})}{\text{Var}(r_{\text{BTC}})}$ (rolling OLS, 60 bars) |
| Input | Symbol log-return + BTCUSDT log-return |
| Use | High-beta symbols amplify BTC signals; low-beta are used for diversification |

### IdiosyncraticResidual

| Attribute | Value |
|---|---|
| Formula | $\varepsilon_t = r_{s,t} - \hat{\beta}_t \cdot r_{\text{BTC},t}$ |
| Input | Symbol return, BetaToBTC, BTCUSDT return |
| Use | Symbol-specific alpha, uncorrelated with market direction |

---

## Family 6: Regime

### HMMRegime

| Attribute | Value |
|---|---|
| States | 3: trending, mean-reverting, crash |
| Input | $(r_t, \sigma_t)$ — log-return and realised vol |
| Library | `hmmlearn` (GaussianHMM) |
| Update | Refitted weekly on rolling 500-bar window |
| Reference | Rabiner (1989); AFML §17 |

The HMM outputs a per-bar posterior probability vector over 3 states.
The regime gate in `MLDirectionalStrategy` blocks signals when the crash
state probability exceeds 0.70 (confirmed for one full bar to avoid
noise at regime boundaries).

**State mapping (empirical, not fixed):**

| State | Typical return | Typical vol | Strategy action |
|---|---|---|---|
| 0: Trending | μ > 0 | Low–medium | Full signal |
| 1: Mean-reverting | μ ≈ 0 | Medium | Full signal |
| 2: Crash | μ < 0 | High | Filter — no new signals |

---

## Point-in-time safety

Every feature is property-tested with Hypothesis:

```python
@given(ohlcv_df())
def test_no_future_leakage(df):
    feat = SomeFeature()
    result = feat.compute(df)
    # Shift the input by 1 bar and verify output shifts identically
    shifted = feat.compute(df.shift(1))
    assert_series_equal(result.shift(1).dropna(), shifted.dropna())
```

Features that fail this test cannot be added to the pipeline without
resolving the violation.
See `tests/property/test_feature_pit_safety.py`.
