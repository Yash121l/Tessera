# Research Methodology

## Triple-Barrier Labeling (AFML §3)

Standard fixed-horizon returns create noisy labels because they ignore the
path taken by prices. The triple-barrier method assigns labels based on
which of three barriers is touched first:

```
                  ┌─── Upper barrier (profit-take): entry × (1 + pt × σ)
                  │
    Price ───────►├─── Vertical barrier (time expiry): t₀ + Δt
                  │
                  └─── Lower barrier (stop-loss):  entry × (1 - sl × σ)
```

- **Upper hit first → +1 (long)**: price moved up enough to take profit.
- **Lower hit first → -1 (short)**: price moved down past the stop.
- **Vertical hit → 0 (no trade)**: price didn't move enough.

### Volatility scaling (§3.1)

Barriers are scaled by local volatility σ computed as the exponentially
weighted standard deviation of log returns:

    σ_t = EWM_std(ln(p_t / p_{t-1}), span=S)

This makes labels adaptive: the same event in a calm vs. volatile market
gets differently-sized barriers.

### Meta-labeling (§3.6)

A secondary model predicts whether the primary model's *direction* was
correct (binary: 0 or 1), rather than predicting direction itself. This
separates the "when to trade" question from "which direction", and allows
the meta-model to focus on sizing and filtering.

![Label Distribution](figures/label_distribution.png)

## Sample Weights (AFML §4)

Labels with overlapping time windows share information. If label A spans
bars 10–20 and label B spans bars 15–25, training on both gives redundant
signal. The sample weighting scheme:

1. **Concurrency (§4.2)**: Count how many label windows overlap each bar.
2. **Uniqueness**: Each event's weight is inversely proportional to the
   average concurrency over its window.
3. **Return-weighted (§4.4)**: Scale by absolute return over the window —
   large moves are more informative.
4. **Time decay (§4.10)**: Optionally down-weight older samples.

## Purged K-Fold Cross-Validation (AFML §7.4)

Standard k-fold CV leaks information in financial data because labels have
overlapping time windows that span across folds:

```
  Fold 1 (train)    │ Fold 2 (test)     │ Fold 3 (train)
  ─────────────────►│──────────────────►│────────────────►
       ┌──── Label A ────┐                      Time →
             Window spans into test fold!
```

**PurgedKFold** fixes this by:

1. **Purging**: Any training sample whose label window [t₀, t₁] overlaps
   the test fold's time range is removed from training.
2. **Embargo**: An additional buffer of `pct_embargo × N` samples after
   each test fold is also excluded, guarding against serial correlation
   in residuals.

### sklearn CV traps this code avoids

| Trap | Standard KFold | PurgedKFold |
|------|---------------|-------------|
| Label leakage via overlapping windows | Yes — labels spanning fold boundaries leak test info | Purged — overlapping samples removed |
| Serial correlation after test fold | Yes — adjacent bars are correlated | Embargo period excludes post-test samples |
| Non-IID assumption | Assumes IID samples | Accounts for temporal dependence |
| Survivorship bias in time series | Shuffles time order | Preserves chronological order |
| Redundant samples dominating | Equal weight | Pairs with concurrency-based weighting |

## Combinatorial Purged K-Fold (AFML §12)

Standard purged k-fold produces N backtest paths (one per fold). CPCV
generates C(N, k) splits by designating k out of N folds as test in each
combination, producing C(N−1, k−1) independent backtest paths.

For the default (N=6, k=2):
- **15 splits** = C(6, 2)
- **5 backtest paths** = C(5, 1)

Each path is a complete walk through the data using only out-of-sample
predictions. The distribution of path-level Sharpe ratios feeds into the
deflated Sharpe ratio test (Bailey & Lopez de Prado, 2014), which adjusts
for multiple testing.

## Walk-Forward Validation

For non-overlapping features or as a simpler baseline:

- **Expanding window**: Train on [0, t), test on [t, t+Δ), advance by step.
  Training set grows over time.
- **Rolling window**: Train on [t-W, t), test on [t, t+Δ). Fixed-size
  training window slides forward.

## References

- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning* (AFML)
    - §3: Triple-barrier labeling and meta-labeling
    - §4: Sample weights and concurrency
    - §7.4: Purged k-fold cross-validation
    - §12: Combinatorial purged cross-validation
- Bailey, D. & Lopez de Prado, M. (2014). *The Deflated Sharpe Ratio*
- Chan, E. (2013). *Algorithmic Trading*
