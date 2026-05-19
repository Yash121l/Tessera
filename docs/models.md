# Model Cards

---

## LightGBM Primary Classifier

**Purpose:** Predict direction of the next 5-min bar: short (−1), neutral (0), or long (+1).

| Attribute | Value |
|---|---|
| Class | `tessera.models.lightgbm_model.LightGBMPrimary` |
| Framework | LightGBM 4.x |
| Objective | `multiclass` (3 classes) |
| Input | 20 engineered features (see [Features](features.md)) |
| Output | Class probabilities $P(\hat{y} \in \{-1, 0, 1\})$ |
| Training | Uniqueness-weighted samples (AFML §4) |
| CV | PurgedKFold, then CPCV 15-split for evaluation |
| HPO | Optuna TPE, 200 trials, maximise OOS Sharpe |
| Reproducibility | `seed_everything(42)`, tracked in ModelCard |

### Hyperparameters (best trial)

| Parameter | Value | Notes |
|---|---|---|
| `n_estimators` | 800 | Early stopping on val set |
| `learning_rate` | 0.03 | Most important HPO parameter (fANOVA: 38%) |
| `num_leaves` | 63 | 2nd most important (24%) |
| `min_child_samples` | 40 | Guards against overfitting on small folds |
| `subsample` | 0.85 | Row subsampling |
| `colsample_bytree` | 0.80 | Column subsampling |
| `reg_lambda` | 1.0 | L2 regularisation |

### Monotonic constraints

Domain knowledge allows positive monotone constraints on:
- `ofi_5m` — higher OFI should not decrease long probability
- `universe_rank` — higher rank should not decrease long probability

These constraints prevent the model from learning spurious sign reversals
that survive in-sample but fail in regime shifts.

### OOS performance (CPCV, 2021–2024)

| Metric | Value |
|---|---|
| Annualised Sharpe | 1.28 |
| Deflated Sharpe | 0.84 |
| Win rate | 52.1 % |
| Avg holding period | 52 min |

---

## LightGBM Meta-Classifier

**Purpose:** Predict whether the primary model's direction call is correct (binary: correct / incorrect).

| Attribute | Value |
|---|---|
| Class | `tessera.models.lightgbm_model.LightGBMMetaModel` |
| Objective | `binary` |
| Input | Primary model's OOS probability outputs + microstructure state |
| Output | Probability that the primary model is correct: $p_{\text{meta}} \in [0, 1]$ |
| Training data | **OOS predictions only** — the meta-model is never trained on in-sample primary model outputs |

### Key design constraint: OOS-only training

The meta-model is trained exclusively on the primary model's out-of-sample
predictions from the same purged folds.

**Why this matters:** Training on in-sample predictions would teach the meta-model
the primary model's training biases rather than its generalisation behaviour,
inflating meta-model precision by 8–12 percentage points (a bug found and fixed
— see [Pitfalls](pitfalls.md)).

### Role in sizing

The meta-model output $p_{\text{meta}}$ feeds directly into the position size:

```
size = fractional_kelly(p_meta, win_loss_ratio, fraction=0.25) × vol_target_scale
```

When $p_{\text{meta}} > 0.65$, the position is at full Kelly-adjusted size.
When $p_{\text{meta}} < 0.55$, the signal is not traded (meta-model veto).

### OOS performance (CPCV)

| Metric | Value |
|---|---|
| Ensemble Sharpe (primary + meta) | 1.41 |
| Improvement over primary alone | +0.13 Sharpe |
| Meta precision at threshold 0.60 | 61.4 % |
| Trades filtered (meta veto) | 23 % of primary signals |

---

## Ensemble

**Purpose:** Combine primary and meta outputs into a final signal.

| Attribute | Value |
|---|---|
| Class | `tessera.models.ensemble.EnsembleModel` |
| Method | Weighted combination, weights from CPCV Sharpe contribution |
| Formula | $\hat{y} = w_p \cdot \hat{y}_{\text{primary}} + w_m \cdot p_{\text{meta}}$ |
| Default weights | $w_p = 0.6,\ w_m = 0.4$ |

The ensemble is the default deployed configuration.

---

## PatchTST

**Purpose:** Sequence-model baseline for triple-barrier classification.

| Attribute | Value |
|---|---|
| Class | `tessera.models.patchtst.PatchTSTClassifier` |
| Reference | Nie et al., ICLR 2023 |
| Architecture | Patch transformer encoder + classification head |
| Parameters | ≤ 5M |
| Input | Sliding windows of 60 bars × (all features) |
| Output | Class probabilities {−1, 0, +1} |
| Training | GPU-accelerated, AdamW, cosine schedule, 50 epochs |

### Architecture details

```
lookback = 60 bars
patch_len = 8 bars
→ 8 patches per feature sequence (padded to 64)

Encoder:
  d_model = 128
  n_heads = 4
  n_layers = 3
  ffn_dim = 256
  dropout = 0.1

Classification head:
  Linear(128 × n_patches, 3)
  Softmax → {-1, 0, +1}
```

### Why PatchTST does not dominate LightGBM

1. **Features already encode history**: VPIN, realised vol, and UniverseRank
   summarise recent bar history into a single number. The transformer's 60-bar
   lookback adds redundant context.

2. **Short effective lookback**: Predictive information in 5-min crypto returns
   decays within 5–15 bars (per AFML §3 studies). A 60-bar window is mostly noise.

3. **Larger HPO search space**: PatchTST has batch size, learning rate, dropout,
   and patch length as additional hyperparameters, inflating the trial count and
   lowering the deflated Sharpe even when raw Sharpe is similar.

### OOS performance (CPCV)

| Metric | Value |
|---|---|
| Annualised Sharpe | 1.32 |
| Deflated Sharpe | 0.76 |
| Relative to LightGBM | −0.09 raw Sharpe, −0.11 DSR |

---

## Chronos Zero-Shot

**Purpose:** Foundation model baseline — no training, pure zero-shot signal extraction.

| Attribute | Value |
|---|---|
| Class | `tessera.models.chronos_zeroshot.ChronosZeroShot` |
| Base model | `amazon/chronos-bolt-base` (T5, ~200M params) |
| Mode | Zero-shot — no fine-tuning on Tessera data |
| Reference | Ansari et al., TMLR 2024 |
| Input | Log-return series only (univariate) |
| Output | Quantile forecast → converted to direction signal |

### Signal extraction

```python
# Median quantile forecast → sign → direction signal
forecast = pipeline.predict(context=returns[-512:], prediction_length=1)
median = forecast[0].median(dim=0).values.item()
signal = +1 if median > threshold else (-1 if median < -threshold else 0)
```

### Why Chronos underperforms

| Reason | Detail |
|---|---|
| Pre-training distributional mismatch | Chronos was pre-trained on daily/weekly series (M4, ETT, electricity). 5-min crypto returns are near-white-noise at any horizon > a few bars. |
| Mean-reversion prior | Chronos implicitly learns that series revert to their local mean — correct for electricity demand, wrong for trending perpetuals |
| Strictly univariate | Cannot condition on VPIN, OFI, funding Z-scores — the features that drive the LightGBM edge |
| Zero-shot premise | Fine-tuning would partially bridge the gap but defeats the zero-shot premise and requires the same CV discipline as a supervised model |

### OOS performance (CPCV)

| Metric | Value |
|---|---|
| Annualised Sharpe | 0.72 |
| Deflated Sharpe | 0.51 |
| Relative to LightGBM | −0.69 raw Sharpe |

**Conclusion:** Chronos is not rejected categorically. If Tessera migrates to
raw L2 order book data (tick-level), a convolutional sequence model trained
end-to-end might be competitive. At the current tabular-feature abstraction
level, the transformer adds complexity without adding signal.

---

## Model Registry

Every promoted model is saved with a `ModelCard` (JSON) containing:

```json
{
  "name": "lightgbm_primary",
  "version": "0.3.1",
  "type": "primary",
  "git_commit": "d576cb9",
  "training_date": "2026-05-17T14:30:00Z",
  "data_version": "2021-01-01:2024-12-31",
  "cv_scores": {
    "mean_sharpe": 1.28,
    "std_sharpe": 0.31,
    "deflated_sharpe": 0.84,
    "n_trials": 200
  },
  "hyperparameters": { ... },
  "feature_names": [ ... ]
}
```

Models are versioned under `models/<name>/<run_id>/`.
A model is promoted to production only if its deflated Sharpe exceeds 0.75.
