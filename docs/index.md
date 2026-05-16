# Tessera

Mid-frequency ML trading system for crypto perpetual futures (Binance + Bybit, USDT-margined).

## Overview

Tessera is a research-grade trading system targeting holding periods of 15 minutes to 4 hours,
with a secondary delta-neutral funding-rate carry sleeve.

## Quick Start

```bash
make setup    # install deps + pre-commit hooks
make lint     # ruff + mypy --strict
make test     # pytest with coverage
```

## Components

- **Data**: Market data ingestion via CCXT + Tardis.dev
- **Features**: Microstructure features (AFML Ch. 5, 18, 19)
- **Models**: LightGBM, PatchTST, RL (PPO/SAC)
- **Backtest**: Nautilus Trader with realistic cost modeling
- **Risk**: Kelly sizing, drawdown limits, kill switches
- **Execution**: Smart order routing with latency monitoring
