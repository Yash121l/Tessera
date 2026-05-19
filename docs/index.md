# Tessera

**Mid-frequency ML trading system for crypto perpetual futures.**

Tessera is a research-grade, fully reproducible trading system for
USDT-margined perpetuals on Binance and Bybit.
It covers every layer of the ML trading stack — from raw market data to
live paper trading — with rigorous statistical validation at each step.

---

## Headline Results

| Metric | Value |
|---|---|
| Backtest Sharpe (walk-forward) | **1.41** |
| Deflated Sharpe (*N*=247 trials) | **0.87** |
| 95 % bootstrap CI | **[0.52, 1.19]** |
| Paper-trading Sharpe (48 h) | **1.31** |
| Max drawdown (4 years) | **8.2 %** |

---

## Quick Start

```bash
make setup      # uv sync --all-extras + pre-commit install
make lint       # ruff + mypy --strict
make test       # pytest with coverage
make backtest   # full walk-forward evaluation
make figures    # regenerate all paper and docs figures
make docs       # serve this site at http://localhost:8000
```

---

## Pipeline Overview

```
Exchange WebSocket
        │
        ▼
  CCXT Ingestor ──── 1-min OHLCV + funding rates
        │                          │
        ▼                          │
  Parquet Store                    │
  (PyArrow, partitioned)           │
        │                          │
        ▼                          ▼
  Feature Pipeline ◄─── 20 features across 6 families
  (topo-sorted, PIT-safe,          (microstructure, vol,
   per-day Parquet cache)           funding, cross-sectional,
        │                           returns, regime)
        ▼
  Triple-Barrier Labeler
  (σ-scaled, uniqueness-weighted samples)
        │
        ▼
  Cross-Validation
  (PurgedKFold → CPCV 15-split → WalkForward)
        │
        ├── LightGBM Primary (200 Optuna trials)
        │
        ├── LightGBM Meta (OOS predictions only)
        │
        └── Ensemble → Quarter-Kelly sizing → Vol-target
                │
                ▼
          Risk Stack
          (circuit breakers · kill switches · limits)
                │
                ▼
          Nautilus Trader
          (fees · slippage · latency model)
                │
                ▼
          Exchange API  ←→  Prometheus → Grafana
```

---

## Site Navigation

| Page | What you'll find |
|---|---|
| [Architecture](architecture.md) | Detailed component diagram, config and logging layers, Docker Compose setup |
| [Methodology](methodology.md) | Triple-barrier labeling, CPCV, deflated Sharpe, bootstrap CI, stress windows |
| [Features](features.md) | All 20 features with formulas and references |
| [Models](models.md) | Model cards: LightGBM, PatchTST, Chronos, ensemble |
| [Results](results.md) | Tear sheets, ablation tables, stress-window analysis |
| [Runbook](runbook.md) | Kill-switch and circuit-breaker incident response |
| [Pitfalls](pitfalls.md) | Every look-ahead leak and data bug found and fixed |

---

## Research Paper

The full methodology is documented in a 12-page IEEE-style paper:
[`paper/main.tex`](https://github.com/Yash121l/Tessera/blob/main/paper/main.tex)

Compile with:

```bash
cd paper
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

---

## Components

| Module | Description |
|---|---|
| `tessera.data` | CCXT ingestor, Parquet store, universe, validation |
| `tessera.features` | 20 features: microstructure, vol, funding, cross-sectional, regime |
| `tessera.labels` | Triple-barrier labeler (AFML §3), sample weights (AFML §4) |
| `tessera.cv` | PurgedKFold, CPCV, WalkForward CV |
| `tessera.models` | LightGBM, PatchTST, Chronos, TFT, ensemble, model registry |
| `tessera.backtest` | Nautilus engine, fee model, square-root slippage |
| `tessera.risk` | Quarter-Kelly, vol-target, limits, circuit breaker, kill switch |
| `tessera.live` | PaperRunner, healthcheck, position reconciliation |
| `tessera.strategies` | MLDirectionalStrategy |

---

## Citation

```bibtex
@software{lunawat2026tessera,
  author  = {Lunawat, Yash},
  title   = {Tessera: A Mid-Frequency ML Trading System for Crypto Perpetual Futures},
  year    = {2026},
  url     = {https://github.com/Yash121l/Tessera}
}
```
