# Tessera Phase 8 Report — Statistical Evaluation & Reporting Layer

## Overview

Phase 8 builds the reporting layer that separates honest performance
evaluation from lucky draws.  The key addition: every headline number is now
accompanied by a statistical test that accounts for how many models were tried
before settling on the reported one.

---

## Deliverables

| Artefact | Location |
|---|---|
| Probabilistic Sharpe Ratio | `src/tessera/backtest/reports/probabilistic_sharpe.py` |
| Deflated Sharpe Ratio + trial count | `src/tessera/backtest/reports/deflated_sharpe.py` |
| Stationary block bootstrap CI | `src/tessera/backtest/reports/bootstrap.py` |
| Stress-window analysis | `src/tessera/backtest/reports/stress.py` |
| Enhanced QuantStats tearsheet | `src/tessera/backtest/reports/tearsheet.py` |
| CLI command | `tessera report backtest --run-id <id> --output docs/figures/` |
| Phase 8 generation script | `scripts/generate_phase8_tearsheet.py` |
| Unit tests (22) | `tests/unit/test_deflated_sharpe.py`, `test_bootstrap_coverage.py` |
| Updated ablation notebook | `notebooks/11_ablation_study.ipynb` |
| Methodology docs | `docs/methodology.md` |
| **Phase 8 tearsheet** | **`docs/figures/phase8_tearsheet.html`** |

---

## Benchmark Run (Phase 7 oracle on 1-year synthetic data)

Instrument: BTC-USDT-PERP.BINANCE | Synthetic 2023-01-01 → 2023-12-31 | seed=7

| Metric | Value |
|---|---|
| Bars | 525 600 |
| Trades | 31 183 |
| Sharpe (annualised) | **1.147** |
| Max drawdown | −1.03% |

---

## Phase 8 Evaluation Results

### Deflated Sharpe (DSR)

| Parameter | Value | Rationale |
|---|---|---|
| Trial count N | 180 | 20 (nb 04) + 20 (nb 05) + 40 (ablation variants) + 100 (CLI budget) |
| σ_SR (cross-trial SR std) | 0.5 | Conservative; actual fold std from nb 04 was ~0.033 — using 0.5 is intentionally pessimistic |
| Expected max SR under H₀ (SR*) | ≈ 1.19 | σ_SR × Euler-Mascheroni formula with N=180 |
| PSR (vs SR₀=0) | 0.974 | P(true SR > 0) before multiple-testing adjustment |
| **DSR (vs SR*=1.19)** | **0.836** | P(true SR > SR*) after 180-trial penalty |
| T (daily observations) | 365 | Return bars from 1-year equity curve |

**Verdict: MODERATE (0.75 ≤ DSR < 0.95).** The oracle strategy on synthetic
alternating-trend data passes a moderate bar.  On real data with a non-oracle
ML model (notebook 04 DSR = 0.000), the result is much weaker — expected, and
honest.

### Bootstrap 95% CI for Sharpe

| | Value |
|---|---|
| Point estimate | 1.147 |
| Bootstrap 95% CI | see tearsheet |
| Block size | 20 bars (≈ mean holding period for oracle) |
| Method | Politis-Romano stationary bootstrap (arch 7.2) |

### Stress-Window Table

All 6 named stress windows fall outside the 2023 synthetic backtest window.
Coverage = "none" for all events — which is the correct and expected result.
This makes explicit what was previously hidden: the strategy has never been
tested through a real tail event.

| Event | Window | Coverage | OOS? |
|---|---|---|---|
| COVID Crash | 2020-02-20 → 2020-03-13 | none | OOS |
| China Mining Ban | 2021-05-12 → 2021-05-20 | none | OOS |
| LUNA Collapse | 2022-05-08 → 2022-05-15 | none | OOS |
| FTX Collapse | 2022-11-06 → 2022-11-12 | none | OOS |
| USDC Depeg | 2023-03-10 → 2023-03-13 | none | OOS |
| Yen Carry Unwind | 2024-08-02 → 2024-08-07 | none | OOS |

Once real historical data is ingested (Phase 9 target), stress coverage will
become the primary credibility signal.

---

## Tearsheet Content

`docs/figures/phase8_tearsheet.html` contains:

1. **Tessera Evaluation Metrics panel** (injected at the top):
   - Point SR, Bootstrap 95% CI, PSR, DSR, trial count, σ_SR, skew/kurt, T
   - Stress-window table with IS/OOS labels
   - Interpretation guide with traffic-light thresholds

2. **QuantStats full report** below the panel:
   - Cumulative returns chart
   - BTC buy-and-hold synthetic benchmark (same price path, buy-and-hold)
   - Rolling Sharpe, Sortino, drawdown
   - Monthly returns heatmap
   - Full statistics table

---

## Fixes Applied vs Phase 8 Audit

| Audit finding | Fix |
|---|---|
| DSR formula misapplied (fold SRs ≠ trial distribution) | New `deflated_sharpe()` takes `(observed_sr, sr_std, n_trials, T, skew, kurt)` — caller provides cross-trial SR std explicitly |
| n_trials undercounted (notebooks not counted) | `compute_trial_count(optuna_study, manual_configs)` aggregates all sources; Phase 8 uses N=180 |
| Ablation Sharpe = 0 (600 bars too short) | Ablation notebook still shows friction analysis; Phase 8 adds separate 1-year oracle run for statistical tests |
| No benchmark in tearsheet | `benchmark_returns` parameter added; synthetic BTC buy-and-hold plotted |
| DSR not visible in tearsheet | DSR/PSR/bootstrap CI/stress table all injected as first panel in HTML |
| No bootstrap CI | `block_bootstrap_sharpe()` implemented with `arch.StationaryBootstrap` |
| No stress windows | 6 defined windows in `STRESS_WINDOWS`, `compute_stress_pnls()` reports all with IS/OOS labels |
| Plot titles missing context | tearsheet title now includes date range, universe, version |

---

## Test Suite

22 new unit tests, all passing:

| File | Tests | Validates |
|---|---|---|
| `test_deflated_sharpe.py` | 17 | PSR monotonicity, DSR < single-trial, DSR decreases with N, trial count aggregation |
| `test_bootstrap_coverage.py` | 5 | CI ordering, reproducibility, flat/short series, plus `@slow` coverage meta-test |

Bootstrap coverage meta-test (200 simulations, 500 resamples each) is
marked `@pytest.mark.slow` and excluded from the default run.  Run with
`uv run pytest -m slow` to verify ~95% empirical coverage.
